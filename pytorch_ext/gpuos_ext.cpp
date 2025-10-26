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
#include <unordered_map>

#include "../src/common.h"

// Wrapper function prototypes from persistent_kernel.cu
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t stream);
extern "C" cudaError_t launch_persistent_worker(WorkQueue q, int blocks, int threads, cudaStream_t stream);
extern "C" cudaError_t gpu_get_processed_count_async(unsigned long long* out, cudaStream_t s);
extern "C" cudaError_t gpu_set_op_table_async(int index, OpPtrInt fn, cudaStream_t s);
extern "C" cudaError_t gpu_set_alias_async(int logical_id, int physical_slot, cudaStream_t s);

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

// Generic elementwise JIT operator registry
static std::mutex g_reg_mu;
static std::unordered_map<std::string, int> g_op_slots; // key -> slot
static int g_next_slot = 20; // reserve [0] builtin add, [10] batch

static unsigned long long get_processed_count() {
  unsigned long long c = 0;
  CUDA_RT_CHECK(gpu_get_processed_count_async(&c, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  return c;
}

// ---- Helpers to build Task from torch::Tensor ----
static int dtype_code(const torch::Tensor& t) {
  switch (t.scalar_type()) {
    case torch::kFloat: return (int)kF32;
    case torch::kHalf: return (int)kF16;
    case torch::kBFloat16: return (int)kBF16;
    default: return (int)kF32; // fallback
  }
}

static void fill_tensorref(TensorRef& tr, const torch::Tensor& ten, int out_ndim, const std::vector<int64_t>& out_sizes) {
  tr.data = (void*)ten.data_ptr();
  tr.dtype = dtype_code(ten);
  tr.ndim = out_ndim;
  // Align input dims to out dims (right-aligned)
  int in_nd = (int)ten.dim();
  auto sizes_in = ten.sizes();
  auto strides_in = ten.strides();
  for (int d = 0; d < out_ndim; ++d) {
    int od = out_ndim - 1 - d;
    int id = in_nd - 1 - d;
    int64_t out_size = out_sizes[od];
    int64_t size_in = (id >= 0) ? sizes_in[id] : 1;
    int64_t stride_in = (id >= 0) ? strides_in[id] : 0;
    tr.sizes[od] = size_in;
    // Broadcast: if size_in == 1, set stride 0
    tr.strides[od] = (size_in == 1) ? 0 : stride_in;
  }
}

static void build_binary_task(Task& t, int op, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& out) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda() && out.device().is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(a.device() == out.device() && b.device() == out.device(), "device mismatch");
  // Compute out shape
  std::vector<int64_t> out_sizes(out.sizes().begin(), out.sizes().end());
  int out_ndim = (int)out_sizes.size();
  int64_t numel = out.numel();
  t.op = op; t.flags = 0; t.ndim = out_ndim; t.numel = numel;
  // out0
  t.out0.data = (void*)out.data_ptr(); t.out0.dtype = dtype_code(out); t.out0.ndim = out_ndim;
  for (int i = 0; i < out_ndim; ++i) { t.out0.sizes[i] = out_sizes[i]; }
  // contiguous out strides in elements
  int64_t stride = 1;
  for (int d = out_ndim - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
  // inputs
  fill_tensorref(t.in0, a, out_ndim, out_sizes);
  fill_tensorref(t.in1, b, out_ndim, out_sizes);
}

static void build_unary_task(Task& t, int op, const torch::Tensor& x, const torch::Tensor& out) {
  TORCH_CHECK(x.device().is_cuda() && out.device().is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.device() == out.device(), "device mismatch");
  std::vector<int64_t> out_sizes(out.sizes().begin(), out.sizes().end());
  int out_ndim = (int)out_sizes.size();
  int64_t numel = out.numel();
  t.op = op; t.flags = 0; t.ndim = out_ndim; t.numel = numel;
  // out0
  t.out0.data = (void*)out.data_ptr(); t.out0.dtype = dtype_code(out); t.out0.ndim = out_ndim;
  for (int i = 0; i < out_ndim; ++i) { t.out0.sizes[i] = out_sizes[i]; }
  int64_t stride = 1; for (int d = out_ndim - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
  // input
  fill_tensorref(t.in0, x, out_ndim, out_sizes);
  // make in1 dummy
  t.in1.data = nullptr; t.in1.dtype = t.in0.dtype; t.in1.ndim = out_ndim; for (int i = 0; i < out_ndim; ++i) { t.in1.sizes[i] = 1; t.in1.strides[i] = 0; }
}

static std::string arch_opt() {
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

// Aggregator operator: handles a batch of micro-Tasks (reuses Task layout)
static std::string build_batch_op_src() {
  return R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #include <math.h>
  #include <stdint.h>
  extern "C" {
    enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
    const int MAX_NDIM = 8;
    struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
    struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };

    __device__ inline int64_t linear_to_offset(const TensorRef& tr, int64_t idx) {
      int64_t off = 0; int nd = tr.ndim; for (int d = nd - 1; d >= 0; --d) { long long dim = tr.sizes[d] > 0 ? tr.sizes[d] : 1; long long i = idx % dim; idx /= dim; off += i * tr.strides[d]; } return off;
    }
    __device__ inline float ld_as_float(const TensorRef& tr, int64_t off_elems) {
      char* base = (char*)tr.data;
      switch (tr.dtype) {
        case kF32: return ((float*)base)[off_elems];
        case kF16: return __half2float(((const __half*)base)[off_elems]);
        case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off_elems]);
        default: return ((float*)base)[off_elems];
      }
    }
    __device__ inline void st_from_float(const TensorRef& tr, int64_t off_elems, float v) {
      char* base = (char*)tr.data;
      switch (tr.dtype) {
        case kF32: ((float*)base)[off_elems] = v; break;
        case kF16: ((__half*)base)[off_elems] = __float2half_rn(v); break;
        case kBF16: ((__nv_bfloat16*)base)[off_elems] = __float2bfloat16(v); break;
        default: ((float*)base)[off_elems] = v; break;
      }
    }

    __device__ void op_batch(const Task& t) {
      const Task* req = (const Task*)t.in0.data;
      int m = (int)t.numel; // using numel to carry count of sub-tasks
      for (int k = 0; k < m; ++k) {
        const Task& u = req[k];
        long long N = u.numel;
        for (long long li = threadIdx.x; li < N; li += blockDim.x) {
          long long oa = linear_to_offset(u.in0, li);
          long long ob = linear_to_offset(u.in1, li);
          long long oc = linear_to_offset(u.out0, li);
          float A = ld_as_float(u.in0, oa);
          float B = ld_as_float(u.in1, ob);
          float R = (u.op == 0) ? (A + B) : (A * B);
          st_from_float(u.out0, oc, R);
        }
        __syncthreads();
      }
    }
  }
  )";
}

static std::vector<char> nvrtc_compile_ptx(const std::string& src) {
  nvrtcProgram prog; NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "batch.cu", 0, nullptr, nullptr));
  std::string arch = arch_opt();
  const char* opts[] = {
    arch.c_str(),
    "--std=c++17",
    "--relocatable-device-code=true",
    "-rdc=true",
    "--device-as-default-execution-space",
    "-I/usr/local/cuda/include","-I/usr/include"
  };
  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);
  size_t logSize = 0; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) { std::string log(logSize, '\0'); NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data())); if (res != NVRTC_SUCCESS) fprintf(stderr, "NVRTC log:\n%s\n", log.c_str()); }
  if (res != NVRTC_SUCCESS) std::exit(4);
  size_t ptxSize = 0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize)); std::vector<char> ptx(ptxSize); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data())); NVRTC_CHECK(nvrtcDestroyProgram(&prog)); return ptx;
}

static OpPtrInt load_function_ptr_from_ptx(const std::vector<char>& ptx, const char* fn_name) {
  CUDA_DRV_CHECK(cuInit(0)); CUDA_RT_CHECK(cudaFree(0));
  CUcontext ctx = nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) { fprintf(stderr, "No CUDA context\n"); std::exit(5);}
  CUmodule mod=nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUfunction fn=nullptr; CUDA_DRV_CHECK(cuModuleGetFunction(&fn, mod, fn_name));
  return (OpPtrInt)fn;
}

static OpPtrInt load_ptr_from_ptx(const std::vector<char>& ptx, const char* sym_name) {
  CUDA_DRV_CHECK(cuInit(0)); CUDA_RT_CHECK(cudaFree(0)); CUcontext ctx = nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx)); if (!ctx) { fprintf(stderr, "No CUDA context\n"); std::exit(5);} CUmodule mod=nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUdeviceptr sym=0; size_t bytes=0; CUDA_DRV_CHECK(cuModuleGetGlobal(&sym, &bytes, mod, sym_name)); if(bytes < sizeof(OpPtrInt)) { fprintf(stderr, "bad sym size %zu\n", bytes); std::exit(6);} OpPtrInt addr=0; CUDA_DRV_CHECK(cuMemcpyDtoH(&addr, sym, sizeof(addr))); return addr;
}

static void set_table_slot_async(int index, OpPtrInt fn_addr) {
  CUDA_RT_CHECK(gpu_set_op_table_async(index, fn_addr, g_ctrl_stream));
}

static void compile_batch_if_needed() {
  if (g_batch_compiled) return;
  auto ptx = nvrtc_compile_ptx(build_batch_op_src());
  OpPtrInt addr = load_function_ptr_from_ptx(ptx, "op_batch");
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
  CUDA_RT_CHECK(launch_init_builtin_ops(g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));

  // Launch persistent worker
  int sm = 0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  CUDA_RT_CHECK(launch_persistent_worker(g_q, sm, threads_per_block, g_kernel_stream));
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
  Task t{}; build_binary_task(t, /*op=*/0, a, b, out);
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

void submit_mul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, /*op=*/1, a, b, out);
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
  // Publish a batch Task to queue; carry sub-task count in numel, and pass pointer via in0.data
  Task batch{}; batch.op = kBatchSlot; batch.flags = 0; batch.ndim = 1; batch.numel = (long long)local.size();
  batch.in0.data = d_subs; batch.in0.dtype = kF32; batch.in0.ndim = 1; batch.in0.sizes[0] = (long long)local.size(); batch.in0.strides[0] = 1;
  batch.in1.data = nullptr; batch.out0.data = nullptr;
  int tail = 0;
  // Enqueue into ring buffer (single producer)
  tail = *g_q.tail; g_q.tasks[tail % g_q.capacity] = batch; *g_q.tail = tail + 1;
  // Prefetch optional; skipped for portability across CUDA versions
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  if (sync) {
    unsigned long long before = get_processed_count();
    while (get_processed_count() < before + 1ULL) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
    CUDA_RT_CHECK(cudaFree(d_subs));
  } else {
    // Leak for simplicity in async mode; production should track and free later
  }
}

// -------- Generic elementwise JIT (float32 contiguous, unary/binary) --------
static std::string build_elementwise_src(const std::string& expr, int arity) {
  // Generate a generic elementwise kernel over TensorRef with dtype/broadcast/strides support.
  std::string src;
  src += R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #include <math.h>
  extern "C" {
    enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
    const int MAX_NDIM = 8;
    struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
    struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
    __device__ inline long long linear_to_offset(const TensorRef& tr, long long idx) {
      long long off = 0; int nd = tr.ndim; for (int d = nd - 1; d >= 0; --d) { long long dim = tr.sizes[d] > 0 ? tr.sizes[d] : 1; long long i = idx % dim; idx /= dim; off += i * tr.strides[d]; } return off;
    }
    __device__ inline float ld_as_float(const TensorRef& tr, long long off_elems) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: return ((float*)base)[off_elems];
        case kF16: return __half2float(((const __half*)base)[off_elems]);
        case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off_elems]);
        default: return ((float*)base)[off_elems];
      }
    }
    __device__ inline void st_from_float(const TensorRef& tr, long long off_elems, float v) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: ((float*)base)[off_elems] = v; break;
        case kF16: ((__half*)base)[off_elems] = __float2half_rn(v); break;
        case kBF16: ((__nv_bfloat16*)base)[off_elems] = __float2bfloat16(v); break;
        default: ((float*)base)[off_elems] = v; break;
      }
    }
    __device__ void op_impl(const Task& t) {
      long long N = t.numel;
      for (long long li = threadIdx.x; li < N; li += blockDim.x) {
        long long oa = linear_to_offset(t.in0, li);
        float A = ld_as_float(t.in0, oa);
  )";
  if (arity == 2) {
    src += R"(
        long long ob = linear_to_offset(t.in1, li);
        float B = ld_as_float(t.in1, ob);
    )";
  }
  src += "        float R = " + (arity == 1 ? std::string("(") + expr + ")" : std::string("(") + expr + ")") + ";\n";
  src += R"(
        long long oc = linear_to_offset(t.out0, li);
        st_from_float(t.out0, oc, R);
      }
    }
  }
  )";
  return src;
}

static int ensure_elementwise_registered(const std::string& key, const std::string& expr, int arity) {
  std::lock_guard<std::mutex> lock(g_reg_mu);
  auto it = g_op_slots.find(key);
  if (it != g_op_slots.end()) return it->second;
  auto ptx = nvrtc_compile_ptx(build_elementwise_src(expr, arity));
  OpPtrInt addr = load_function_ptr_from_ptx(ptx, "op_impl");
  int slot = g_next_slot++;
  set_table_slot_async(slot, addr);
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  g_op_slots.emplace(key, slot);
  return slot;
}

// Submitters for generic elementwise ops
void submit_unary(int slot, torch::Tensor a, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_unary_task(t, slot, a, out);
  std::lock_guard<std::mutex> lock(g_mu);
  int tail = *g_q.tail; g_q.tasks[tail % g_q.capacity] = t; *g_q.tail = tail + 1;
}

void submit_binary(int slot, torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, slot, a, b, out);
  std::lock_guard<std::mutex> lock(g_mu);
  int tail = *g_q.tail; g_q.tasks[tail % g_q.capacity] = t; *g_q.tail = tail + 1;
}

// -------- Reduction (sum/mean) along a single axis (rrank==1) --------
static std::string build_reduce_src(const std::string& op_name) {
  std::string src;
  src += R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  #include <math.h>
  extern "C" {
    enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
    const int MAX_NDIM = 8;
    struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
    struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
    __device__ inline float ld_as_float(const TensorRef& tr, long long off) {
      char* base=(char*)tr.data; switch(tr.dtype){ case kF32: return ((float*)base)[off]; case kF16: return __half2float(((const __half*)base)[off]); case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off]); default: return ((float*)base)[off]; }
    }
    __device__ inline void st_from_float(const TensorRef& tr, long long off, float v) {
      char* base=(char*)tr.data; switch(tr.dtype){ case kF32: ((float*)base)[off]=v; break; case kF16: ((__half*)base)[off]=__float2half_rn(v); break; case kBF16: ((__nv_bfloat16*)base)[off]=__float2bfloat16(v); break; default: ((float*)base)[off]=v; break; }
    }
    __device__ void op_reduce(const Task& t) {
      const int in_nd = t.in0.ndim;
      const int out_nd = t.out0.ndim;
      const int rrank = t.rrank;
      // Fast path: single-axis reduce over last dim with contiguous stride
      if (rrank == 1) {
        int axis = t.r_axes[0];
        long long red_N = t.in0.sizes[axis] > 0 ? t.in0.sizes[axis] : 1;
        if (axis == in_nd - 1 && t.in0.strides[axis] == 1) {
          // Iterate outputs (all dims except last), each reduced by the whole block
          long long outer = t.numel; // out elements count
          for (long long li = 0; li < outer; ++li) {
            // Decode out coords
            long long coord_out[MAX_NDIM]; long long tmp = li;
            for (int d = out_nd - 1; d >= 0; --d) { long long dim = t.out0.sizes[d] > 0 ? t.out0.sizes[d] : 1; coord_out[d] = tmp % dim; tmp /= dim; }
            // Map to input base offset (excluding last dim)
            long long off_in_base = 0; int out_ptr = 0;
            for (int d_in = 0; d_in < in_nd; ++d_in) {
              if (d_in == axis) continue;
              long long idx = coord_out[out_ptr++]; off_in_base += idx * t.in0.strides[d_in];
            }
            // Parallel accumulate over last dim
            float acc = 0.0f;
            for (long long r = threadIdx.x; r < red_N; r += blockDim.x) {
              long long off_in = off_in_base + r; // stride 1 along last dim
              acc += ld_as_float(t.in0, off_in);
            }
            __shared__ float shm[1024]; // assumes blockDim.x <= 1024
            int tid = threadIdx.x;
            shm[tid] = acc;
            __syncthreads();
            for (int step = blockDim.x >> 1; step > 0; step >>= 1) {
              if (tid < step) shm[tid] += shm[tid + step];
              __syncthreads();
            }
            if (tid == 0) {
              float outv = shm[0];
  )";
  if (op_name == "mean") {
    src += "              outv = outv / (float)red_N;\n";
  }
  src += R"(
              // Output offset
              long long off_out = 0; for (int d = 0; d < out_nd; ++d) off_out += coord_out[d] * t.out0.strides[d];
              st_from_float(t.out0, off_out, outv);
            }
            __syncthreads();
          }
          return;
        }
      }
      // Compute product of reduced sizes
      long long red_N = 1;
      for (int j = 0; j < rrank; ++j) {
        int ax = t.r_axes[j]; long long dim = t.in0.sizes[ax] > 0 ? t.in0.sizes[ax] : 1; red_N *= dim;
      }
      for (long long li = threadIdx.x; li < t.numel; li += blockDim.x) {
        long long coord_out[MAX_NDIM]; long long tmp = li;
        for (int d = out_nd - 1; d >= 0; --d) { long long dim = t.out0.sizes[d] > 0 ? t.out0.sizes[d] : 1; coord_out[d] = tmp % dim; tmp /= dim; }
        // Map output coords to input base offset, skipping reduced axes when keepdim==0
        long long off_in_base = 0; int out_ptr = 0;
        for (int d_in = 0; d_in < in_nd; ++d_in) {
          bool is_reduced = false; for (int j = 0; j < rrank; ++j) if (t.r_axes[j] == d_in) { is_reduced = true; break; }
          if (is_reduced) { if (t.r_keepdim) { /*coord_out[out_ptr] should be 0*/ out_ptr++; } continue; }
          long long idx = coord_out[out_ptr++]; off_in_base += idx * t.in0.strides[d_in];
        }
        long long off_out = 0; for (int d = 0; d < out_nd; ++d) off_out += coord_out[d] * t.out0.strides[d];
        float acc = 0.0f;
        for (long long rv = 0; rv < red_N; ++rv) {
          long long ttmp = rv; long long off_add = 0;
          for (int j = 0; j < rrank; ++j) { int ax = t.r_axes[j]; long long dim = t.in0.sizes[ax] > 0 ? t.in0.sizes[ax] : 1; long long idx = ttmp % dim; ttmp /= dim; off_add += idx * t.in0.strides[ax]; }
          long long off_in = off_in_base + off_add; acc += ld_as_float(t.in0, off_in);
        }
  )";
  if (op_name == "mean") {
    src += "        acc = acc / (float)red_N;\n";
  }
  src += R"(
        st_from_float(t.out0, off_out, acc);
      }
    }
  }
  )";
  return src;
}

static int ensure_reduce_registered(const std::string& key, const std::string& op_name) {
  std::lock_guard<std::mutex> lock(g_reg_mu);
  auto it = g_op_slots.find(key);
  if (it != g_op_slots.end()) return it->second;
  auto ptx = nvrtc_compile_ptx(build_reduce_src(op_name));
  OpPtrInt addr = load_function_ptr_from_ptx(ptx, "op_reduce");
  int slot = g_next_slot++;
  set_table_slot_async(slot, addr);
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  g_op_slots.emplace(key, slot);
  return slot;
}

static void build_reduce_task(Task& t, int op, const torch::Tensor& x, const torch::Tensor& out, const std::vector<int>& axes, bool keepdim) {
  TORCH_CHECK(x.device().is_cuda() && out.device().is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.device() == out.device(), "device mismatch");
  int in_nd = (int)x.dim();
  std::vector<int64_t> out_sizes(out.sizes().begin(), out.sizes().end());
  int out_nd = (int)out_sizes.size();
  t.op = op; t.flags = 0; t.ndim = out_nd; t.numel = out.numel();
  t.rrank = (int)axes.size(); t.r_keepdim = keepdim ? 1 : 0;
  for (int i = 0; i < t.rrank && i < MAX_NDIM; ++i) t.r_axes[i] = axes[i];
  // in0: original shape/strides
  t.in0.data = (void*)x.data_ptr(); t.in0.dtype = dtype_code(x); t.in0.ndim = in_nd;
  for (int i = 0; i < in_nd; ++i) { t.in0.sizes[i] = x.size(i); t.in0.strides[i] = x.stride(i); }
  // out0: contiguous
  t.out0.data = (void*)out.data_ptr(); t.out0.dtype = dtype_code(out); t.out0.ndim = out_nd;
  for (int i = 0; i < out_nd; ++i) { t.out0.sizes[i] = out_sizes[i]; }
  int64_t stride = 1; for (int d = out_nd - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
  // in1 unused
  t.in1.data = nullptr; t.in1.ndim = out_nd; for (int i = 0; i < out_nd; ++i) { t.in1.sizes[i] = 1; t.in1.strides[i] = 0; }
}

int register_reduce(const std::string& key, const std::string& op_name) {
  return ensure_reduce_registered(key, op_name);
}

void submit_reduce(int slot, torch::Tensor x, torch::Tensor out, std::vector<int> axes, bool keepdim) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_reduce_task(t, slot, x, out, axes, keepdim);
  std::lock_guard<std::mutex> lock(g_mu);
  int tail = *g_q.tail; g_q.tasks[tail % g_q.capacity] = t; *g_q.tail = tail + 1;
}

} // namespace gpuos_ext

// Pybind11 module must be at global scope
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace gpuos_ext;
  m.def("init", &init, "Initialize GPUOS persistent runtime", py::arg("capacity")=4096, py::arg("threads_per_block")=256);
  m.def("shutdown", &shutdown, "Shutdown GPUOS runtime");
  m.def("submit_add", &submit_add, "Submit add micro-op");
  m.def("submit_mul", &submit_mul, "Submit mul micro-op");
  m.def("flush", &flush, "Flush pending micro-ops", py::arg("sync")=false);
  m.def("register_elementwise", [](const std::string& key, const std::string& expr, int arity){
      return ensure_elementwise_registered(key, expr, arity);
    }, "Register JIT elementwise op and return slot");
  m.def("submit_unary", &submit_unary, "Submit unary op for a slot");
  m.def("submit_binary", &submit_binary, "Submit binary op for a slot");
  m.def("register_reduce", &register_reduce, "Register reduce op (sum/mean) and return slot");
  m.def("submit_reduce", &submit_reduce, "Submit reduce task (axes, keepdim)");
}
