#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "../src/common.h"

// Kernel prototypes from persistent kernel unit
extern "C" __global__ void init_builtin_ops();
extern "C" __global__ void persistent_worker(WorkQueue q);

#define CUDA_RT_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA Runtime error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
    std::exit(1); \
  } \
} while(0)

static const char* cu_errstr(CUresult r) {
  const char* s = nullptr; cuGetErrorString(r, &s); return s ? s : "<unknown>";
}

#define CUDA_DRV_CHECK(expr) do { \
  CUresult _res = (expr); \
  if (_res != CUDA_SUCCESS) { \
    fprintf(stderr, "CUDA Driver error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cu_errstr(_res)); \
    std::exit(2); \
  } \
} while(0)

#define NVRTC_CHECK(expr) do { \
  nvrtcResult _res = (expr); \
  if (_res != NVRTC_SUCCESS) { \
    fprintf(stderr, "NVRTC error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, nvrtcGetErrorString(_res)); \
    std::exit(3); \
  } \
} while(0)

namespace gpuos_ext {

// Runtime state
static WorkQueue g_q{};
static bool g_started = false;
static int g_capacity = 0;
static cudaStream_t g_kernel_stream = nullptr;
static cudaStream_t g_ctrl_stream = nullptr;
static std::mutex g_mu;

// Pending micro-requests to be aggregated
static std::vector<Task> g_pending;
static const int kBatchSlot = 10; // slot id for aggregated operator
static bool g_batch_compiled = false;

static unsigned long long get_processed_count() {
  unsigned long long c = 0;
  CUDA_RT_CHECK(cudaMemcpyFromSymbol(&c, g_processed_count, sizeof(c), 0, cudaMemcpyDeviceToHost));
  return c;
}

static std::string arch_opt() {
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

// Aggregator operator: handles a batch of micro-Tasks (reuses Task layout)
static std::string build_batch_op_src() {
  return R"(
  extern "C" {
    struct Task { int op; int n; void* in0; void* in1; void* out0; };
    typedef void(*OpFn)(const Task&);
    __device__ void op_batch(const Task& t) {
      const Task* req = (const Task*)t.in0;
      int m = t.n;
      for (int k = 0; k < m; ++k) {
        const Task& u = req[k];
        if (u.op == 0) {
          const float* a = (const float*)u.in0;
          const float* b = (const float*)u.in1;
          float* c = (float*)u.out0;
          for (int i = threadIdx.x; i < u.n; i += blockDim.x) c[i] = a[i] + b[i];
        } else if (u.op == 1) {
          const float* a = (const float*)u.in0;
          const float* b = (const float*)u.in1;
          float* c = (float*)u.out0;
          for (int i = threadIdx.x; i < u.n; i += blockDim.x) c[i] = a[i] * b[i];
        }
      }
    }
    __device__ void* op_batch_ptr = (void*)op_batch;
  }
  )";
}

static std::vector<char> nvrtc_compile_ptx(const std::string& src) {
  nvrtcProgram prog; NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "batch.cu", 0, nullptr, nullptr));
  std::string arch = arch_opt();
  const char* opts[] = { arch.c_str(), "--std=c++17", "--relocatable-device-code=true", "-rdc=true", "--device-as-default-execution-space" };
  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);
  size_t logSize = 0; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) { std::string log(logSize, '\0'); NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data())); if (res != NVRTC_SUCCESS) fprintf(stderr, "NVRTC log:\n%s\n", log.c_str()); }
  if (res != NVRTC_SUCCESS) std::exit(4);
  size_t ptxSize = 0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize)); std::vector<char> ptx(ptxSize); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data())); NVRTC_CHECK(nvrtcDestroyProgram(&prog)); return ptx;
}

static OpPtrInt load_ptr_from_ptx(const std::vector<char>& ptx, const char* sym_name) {
  CUDA_DRV_CHECK(cuInit(0)); CUDA_RT_CHECK(cudaFree(0)); CUcontext ctx = nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx)); if (!ctx) { fprintf(stderr, "No CUDA context\n"); std::exit(5);} CUmodule mod=nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUdeviceptr sym=0; size_t bytes=0; CUDA_DRV_CHECK(cuModuleGetGlobal(&sym, &bytes, mod, sym_name)); if(bytes < sizeof(OpPtrInt)) { fprintf(stderr, "bad sym size %zu\n", bytes); std::exit(6);} OpPtrInt addr=0; CUDA_DRV_CHECK(cuMemcpyDtoH(&addr, sym, sizeof(addr))); return addr;
}

static void set_table_slot_async(int index, OpPtrInt fn_addr) {
  CUDA_RT_CHECK(cudaMemcpyToSymbolAsync(g_op_table, &fn_addr, sizeof(fn_addr), index * sizeof(OpFn), cudaMemcpyHostToDevice, g_ctrl_stream));
}

static void compile_batch_if_needed() {
  if (g_batch_compiled) return;
  auto ptx = nvrtc_compile_ptx(build_batch_op_src());
  OpPtrInt addr = load_ptr_from_ptx(ptx, "op_batch_ptr");
  set_table_slot_async(kBatchSlot, addr);
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  g_batch_compiled = true;
}

// Python API
void init(int capacity, int threads_per_block) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_started) return;
  CUDA_RT_CHECK(cudaSetDevice(0));
  CUDA_RT_CHECK(cudaFree(0));
  g_capacity = capacity;
  // Allocate queue
  CUDA_RT_CHECK(cudaMallocManaged(&g_q.tasks, (size_t)capacity * sizeof(Task)));
  CUDA_RT_CHECK(cudaMallocManaged(&g_q.head, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&g_q.tail, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&g_q.quit, sizeof(int)));
  g_q.capacity = capacity;
  CUDA_RT_CHECK(cudaMemset(g_q.head, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(g_q.tail, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(g_q.quit, 0, sizeof(int)));
  // Streams
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&g_kernel_stream, cudaStreamNonBlocking));
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&g_ctrl_stream, cudaStreamNonBlocking));
  // Init builtins
  {
    dim3 blk(128); dim3 grd((GPUOS_MAX_OPS + blk.x - 1) / blk.x);
    init_builtin_ops<<<grd, blk, 0, g_ctrl_stream>>>();
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  }
  // Launch persistent worker
  int sm = 0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  persistent_worker<<<sm, threads_per_block, 0, g_kernel_stream>>>(g_q);
  CUDA_RT_CHECK(cudaGetLastError());
  g_started = true;
}

void shutdown() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_started) return;
  int one = 1;
  CUDA_RT_CHECK(cudaMemcpyAsync(g_q.quit, &one, sizeof(one), cudaMemcpyHostToDevice, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  CUDA_RT_CHECK(cudaDeviceSynchronize());
  // Free queue
  CUDA_RT_CHECK(cudaFree(g_q.tasks));
  CUDA_RT_CHECK(cudaFree(g_q.head));
  CUDA_RT_CHECK(cudaFree(g_q.tail));
  CUDA_RT_CHECK(cudaFree(g_q.quit));
  CUDA_RT_CHECK(cudaStreamDestroy(g_kernel_stream));
  CUDA_RT_CHECK(cudaStreamDestroy(g_ctrl_stream));
  g_started = false;
}

// Enqueue a micro-request into host-side pending list
void submit_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32 && out.dtype() == torch::kFloat32, "dtype must be float32");
  TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "size mismatch");
  Task t{}; t.op = 0; t.n = (int)a.numel(); t.in0 = a.data_ptr(); t.in1 = b.data_ptr(); t.out0 = out.data_ptr();
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

void submit_mul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  TORCH_CHECK(a.is_cuda() && b.is_cuda() && out.is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && out.is_contiguous(), "tensors must be contiguous");
  TORCH_CHECK(a.dtype() == torch::kFloat32 && b.dtype() == torch::kFloat32 && out.dtype() == torch::kFloat32, "dtype must be float32");
  TORCH_CHECK(a.numel() == b.numel() && a.numel() == out.numel(), "size mismatch");
  Task t{}; t.op = 1; t.n = (int)a.numel(); t.in0 = a.data_ptr(); t.in1 = b.data_ptr(); t.out0 = out.data_ptr();
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

// Flush pending micro-requests by aggregating into one batch Task processed by JIT batch op
void flush(bool sync) {
  std::vector<Task> local;
  {
    std::lock_guard<std::mutex> lock(g_mu);
    if (g_pending.empty()) return;
    local.swap(g_pending);
  }
  compile_batch_if_needed();
  // Copy micro-tasks to device array
  Task* d_subs = nullptr;
  CUDA_RT_CHECK(cudaMallocManaged(&d_subs, local.size() * sizeof(Task)));
  std::memcpy(d_subs, local.data(), local.size() * sizeof(Task));
  // Publish a batch Task to queue
  Task batch{}; batch.op = kBatchSlot; batch.n = (int)local.size(); batch.in0 = d_subs; batch.in1 = nullptr; batch.out0 = nullptr;
  int tail = 0;
  // Enqueue into ring buffer (single producer)
  tail = *g_q.tail; g_q.tasks[tail % g_q.capacity] = batch; *g_q.tail = tail + 1;
  // Prefetch
  int dev = 0; CUDA_RT_CHECK(cudaGetDevice(&dev));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(g_q.tasks, g_q.capacity * sizeof(Task), dev, g_ctrl_stream));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(g_q.tail, sizeof(int), dev, g_ctrl_stream));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(d_subs, local.size() * sizeof(Task), dev, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  if (sync) {
    unsigned long long before = get_processed_count();
    while (get_processed_count() < before + 1ULL) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    CUDA_RT_CHECK(cudaFree(d_subs));
  } else {
    // Leak for simplicity in async mode; production should track and free later
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "Initialize GPUOS persistent runtime", py::arg("capacity")=4096, py::arg("threads_per_block")=256);
  m.def("shutdown", &shutdown, "Shutdown GPUOS runtime");
  m.def("submit_add", &submit_add, "Submit add micro-op");
  m.def("submit_mul", &submit_mul, "Submit mul micro-op");
  m.def("flush", &flush, "Flush pending micro-ops", py::arg("sync")=false);
}

} // namespace gpuos_ext

