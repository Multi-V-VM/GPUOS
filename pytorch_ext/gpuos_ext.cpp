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
extern "C" cudaError_t gpu_set_debug_level_async(int level, cudaStream_t s);
extern "C" cudaError_t gpu_get_heartbeat_async(unsigned long long* out, cudaStream_t s);
extern "C" cudaError_t gpu_set_yield_every_async(unsigned long long every, cudaStream_t s);

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
static int g_verbose_level = 0;
enum class StagingMode { DeviceMemcpy = 0, MappedHost = 1 };
static StagingMode g_staging = StagingMode::DeviceMemcpy;
static int g_shard_size = 512; // number of sub-tasks per batch shard
static int g_blocks = 0;
static int g_threads_per_block = 0;
static int g_tail_shadow = 0;

// Pending micro-requests to be aggregated
static std::vector<Task> g_pending;
static const int kBatchSlot = 10; // slot id for aggregated operator
static Task* g_batch_buffer = nullptr;
static size_t g_batch_capacity = 0;
static std::vector<void*> g_async_buffers;
// Mapped host staging (pinned host memory with device alias)
static Task* g_batch_host = nullptr;            // host pinned buffer
static Task* g_batch_host_dev = nullptr;        // device alias of pinned host buffer
static size_t g_batch_host_capacity = 0;
static std::vector<void*> g_async_host_buffers; // host pinned blocks to free at shutdown
static size_t g_cached_bytes = 0;
static unsigned long long g_cached_hash = 0;
static bool g_cached_valid = false;

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

static unsigned long long get_heartbeat() {
  unsigned long long h = 0;
  CUDA_RT_CHECK(gpu_get_heartbeat_async(&h, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  return h;
}

static void ensure_worker_alive() {
  if (!g_started) return;
  cudaError_t st = cudaStreamQuery(g_kernel_stream);
  if (st == cudaSuccess) {
    // Persistent kernel ended; relaunch with stored blocks/threads
    int zero = 0;
    CUDA_RT_CHECK(cudaMemcpyAsync(g_q.quit, &zero, sizeof(zero), cudaMemcpyHostToDevice, g_ctrl_stream));
    CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
    CUDA_RT_CHECK(launch_persistent_worker(g_q, g_blocks, g_threads_per_block, g_kernel_stream));
    if (g_verbose_level > 0) {
      fprintf(stderr, "[host] worker relaunched (blocks=%d threads=%d)\n", g_blocks, g_threads_per_block);
    }
  } else if (st != cudaErrorNotReady) {
    CUDA_RT_CHECK(st);
  }
}

bool worker_alive() {
  if (!g_started) return false;
  cudaError_t st = cudaStreamQuery(g_kernel_stream);
  if (st == cudaErrorNotReady) return true;
  if (st == cudaSuccess) return false;
  CUDA_RT_CHECK(st);
  return false;
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

static void fill_tensorref(TensorRef& tr, const torch::Tensor& ten, int out_ndim, const std::vector<long>& out_sizes) {
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
    long out_size = out_sizes[od];
    long size_in = (id >= 0) ? sizes_in[id] : 1;
    long stride_in = (id >= 0) ? strides_in[id] : 0;
    tr.sizes[od] = size_in;
    // Broadcast: if size_in == 1, set stride 0
    tr.strides[od] = (size_in == 1) ? 0 : stride_in;
  }
}

static void build_binary_task(Task& t, int op, const torch::Tensor& a, const torch::Tensor& b, const torch::Tensor& out) {
  TORCH_CHECK(a.device().is_cuda() && b.device().is_cuda() && out.device().is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(a.device() == out.device() && b.device() == out.device(), "device mismatch");
  // Compute out shape
  std::vector<long> out_sizes(out.sizes().begin(), out.sizes().end());
  int out_ndim = (int)out_sizes.size();
  long numel = out.numel();
  t.op = op; t.flags = 0; t.ndim = out_ndim; t.numel = numel;
  // out0
  t.out0.data = (void*)out.data_ptr(); t.out0.dtype = dtype_code(out); t.out0.ndim = out_ndim;
  for (int i = 0; i < out_ndim; ++i) { t.out0.sizes[i] = out_sizes[i]; }
  // contiguous out strides in elements
  long stride = 1;
  for (int d = out_ndim - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
  // inputs
  fill_tensorref(t.in0, a, out_ndim, out_sizes);
  fill_tensorref(t.in1, b, out_ndim, out_sizes);
}

static void build_unary_task(Task& t, int op, const torch::Tensor& x, const torch::Tensor& out) {
  TORCH_CHECK(x.device().is_cuda() && out.device().is_cuda(), "tensors must be CUDA");
  TORCH_CHECK(x.device() == out.device(), "device mismatch");
  std::vector<long> out_sizes(out.sizes().begin(), out.sizes().end());
  int out_ndim = (int)out_sizes.size();
  long numel = out.numel();
  t.op = op; t.flags = 0; t.ndim = out_ndim; t.numel = numel;
  // out0
  t.out0.data = (void*)out.data_ptr(); t.out0.dtype = dtype_code(out); t.out0.ndim = out_ndim;
  for (int i = 0; i < out_ndim; ++i) { t.out0.sizes[i] = out_sizes[i]; }
  long stride = 1; for (int d = out_ndim - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
  // input
  fill_tensorref(t.in0, x, out_ndim, out_sizes);
  // make in1 dummy
  t.in1.data = nullptr; t.in1.dtype = t.in0.dtype; t.in1.ndim = out_ndim; for (int i = 0; i < out_ndim; ++i) { t.in1.sizes[i] = 1; t.in1.strides[i] = 0; }
}

static std::string arch_opt() {
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  int dev = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    char buf[64];
    snprintf(buf, sizeof(buf), "--gpu-architecture=compute_%d%d", prop.major, prop.minor);
    return std::string(buf);
  }
  return std::string("--gpu-architecture=compute_90");
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

static Task* ensure_batch_buffer(size_t count) {
  if (count == 0) return nullptr;
  if (count > g_batch_capacity) {
    if (g_batch_buffer) {
      CUDA_RT_CHECK(cudaFree(g_batch_buffer));
    }
    CUDA_RT_CHECK(cudaMalloc(&g_batch_buffer, count * sizeof(Task)));
    g_batch_capacity = count;
    if (g_verbose_level > 0) {
      fprintf(stderr, "[host] resized batch buffer to %zu tasks (bytes=%zu)\n", count, count * sizeof(Task));
    }
  }
  return g_batch_buffer;
}

static Task* ensure_batch_host_buffer(size_t count) {
  if (count == 0) return nullptr;
  if (count > g_batch_host_capacity) {
    if (g_batch_host) {
      CUDA_RT_CHECK(cudaFreeHost(g_batch_host));
      g_batch_host = nullptr;
      g_batch_host_dev = nullptr;
    }
    size_t bytes = count * sizeof(Task);
    CUDA_RT_CHECK(cudaHostAlloc((void**)&g_batch_host, bytes, cudaHostAllocMapped));
    CUDA_RT_CHECK(cudaHostGetDevicePointer((void**)&g_batch_host_dev, (void*)g_batch_host, 0));
    g_batch_host_capacity = count;
    if (g_verbose_level > 0) {
      fprintf(stderr, "[host] resized mapped host buffer to %zu tasks (bytes=%zu) host=%p dev=%p\n",
              count, bytes, (void*)g_batch_host, (void*)g_batch_host_dev);
    }
  }
  return g_batch_host;
}

static inline unsigned long long fnv1a64(const void* data, size_t n) {
  const unsigned char* p = reinterpret_cast<const unsigned char*>(data);
  unsigned long long h = 1469598103934665603ull; // FNV offset basis
  for (size_t i = 0; i < n; ++i) {
    h ^= (unsigned long long)p[i];
    h *= 1099511628211ull; // FNV prime
  }
  return h;
}

static inline void enqueue_task_host_to_device(const Task& t) {
  int tail = g_tail_shadow;
  // Backpressure: avoid overwriting unconsumed entries
  for (;;) {
    int head = -1;
    (void)cudaMemcpy((void*)&head, (const void*)g_q.head, sizeof(int), cudaMemcpyDeviceToHost);
    int in_flight = tail - head;
    if (in_flight < g_q.capacity) break;
    if (g_verbose_level > 0) {
      fprintf(stderr, "[host] backpressure: head=%d tail=%d (waiting for space)\n", head, tail);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  int idx = tail % g_q.capacity;
  if (g_verbose_level > 1) {
    fprintf(stderr, "[host] enqueue idx=%d tail_before=%d op=%d numel=%lld in0=%p in1=%p out0=%p\n",
            idx, tail, t.op, (long long)t.numel, t.in0.data, t.in1.data, t.out0.data);
  }
  CUDA_RT_CHECK(cudaMemcpyAsync(&g_q.tasks[idx], &t, sizeof(Task), cudaMemcpyHostToDevice, g_ctrl_stream));
  int new_tail = tail + 1;
  CUDA_RT_CHECK(cudaMemcpyAsync(g_q.tail, &new_tail, sizeof(new_tail), cudaMemcpyHostToDevice, g_ctrl_stream));
  g_tail_shadow = new_tail;
}

static inline int read_dev_int(const int* dev_ptr) {
  int v = -1;
  (void)cudaMemcpy((void*)&v, (const void*)dev_ptr, sizeof(int), cudaMemcpyDeviceToHost);
  return v;
}

// Python API
void init(int capacity, int threads_per_block) {
  std::lock_guard<std::mutex> lock(g_mu);
  if (g_started) return;
  CUDA_RT_CHECK(cudaSetDevice(0));
  CUDA_RT_CHECK(cudaFree(0));
  g_capacity = capacity;
  // Allocate queue in device memory
  CUDA_RT_CHECK(cudaMalloc(&g_q.tasks, (size_t)capacity * sizeof(Task)));
  CUDA_RT_CHECK(cudaMalloc(&g_q.head, sizeof(int)));
  CUDA_RT_CHECK(cudaMalloc(&g_q.tail, sizeof(int)));
  CUDA_RT_CHECK(cudaMalloc(&g_q.quit, sizeof(int)));
  g_q.capacity = capacity;
  CUDA_RT_CHECK(cudaMemset(g_q.head, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(g_q.tail, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(g_q.quit, 0, sizeof(int)));
  g_tail_shadow = 0;
  // Streams
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&g_kernel_stream, cudaStreamNonBlocking));
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&g_ctrl_stream, cudaStreamNonBlocking));
  // Init builtins
  CUDA_RT_CHECK(launch_init_builtin_ops(g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));

  // Launch persistent worker
  int sm = 0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  g_blocks = sm;
  g_threads_per_block = threads_per_block;
  CUDA_RT_CHECK(launch_persistent_worker(g_q, g_blocks, g_threads_per_block, g_kernel_stream));
  g_started = true;
  // Optional: set debug level from env
  const char* dbg = std::getenv("GPUOS_DEBUG");
  if (dbg && *dbg) {
    int lvl = std::atoi(dbg);
    CUDA_RT_CHECK(gpu_set_debug_level_async(lvl, g_ctrl_stream));
    CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  }
  const char* verb = std::getenv("GPUOS_VERBOSE");
  if (verb && *verb) { g_verbose_level = std::atoi(verb); }
  const char* st = std::getenv("GPUOS_STAGING");
  if (st && *st) {
    if (!strcmp(st, "device") || !strcmp(st, "memcpy")) g_staging = StagingMode::DeviceMemcpy;
    else if (!strcmp(st, "mapped")) g_staging = StagingMode::MappedHost;
  }
  const char* shard = std::getenv("GPUOS_BATCH_SHARD");
  if (shard && *shard) {
    int v = std::atoi(shard);
    if (v > 0) g_shard_size = v;
  }
  if (g_verbose_level > 0) {
    fprintf(stderr, "[host] init capacity=%d threads_per_block=%d\n", capacity, threads_per_block);
    fprintf(stderr, "[host] queue tasks=%p head=%p tail=%p quit=%p\n", (void*)g_q.tasks, (void*)g_q.head, (void*)g_q.tail, (void*)g_q.quit);
    fprintf(stderr, "[host] staging=%s\n", g_staging == StagingMode::MappedHost ? "mapped" : "device");
    fprintf(stderr, "[host] shard_size=%d\n", g_shard_size);
  }
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
  for (void* buf : g_async_buffers) {
    CUDA_RT_CHECK(cudaFree(buf));
  }
  g_async_buffers.clear();
  for (void* hbuf : g_async_host_buffers) {
    CUDA_RT_CHECK(cudaFreeHost(hbuf));
  }
  g_async_host_buffers.clear();
  if (g_batch_buffer) {
    CUDA_RT_CHECK(cudaFree(g_batch_buffer));
    g_batch_buffer = nullptr;
    g_batch_capacity = 0;
  }
  if (g_batch_host) {
    CUDA_RT_CHECK(cudaFreeHost(g_batch_host));
    g_batch_host = nullptr;
    g_batch_host_dev = nullptr;
    g_batch_host_capacity = 0;
  }
  g_cached_valid = false;
  g_cached_bytes = 0;
  g_cached_hash = 0;
  g_started = false;
}

void set_debug_level(int level) {
  if (!g_started) return;
  CUDA_RT_CHECK(gpu_set_debug_level_async(level, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
}

void set_yield_every(unsigned long long every) {
  // 0 means never yield; otherwise kernel blocks exit after every `every` tasks
  CUDA_RT_CHECK(gpu_set_yield_every_async(every, g_ctrl_stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
}

py::dict peek_queue() {
  std::lock_guard<std::mutex> lock(g_mu);
  py::dict d;
  int head = read_dev_int(g_q.head);
  int tail = read_dev_int(g_q.tail);
  int quit = read_dev_int(g_q.quit);
  unsigned long long proc = get_processed_count();
  int inflight = tail - head;
  if (inflight < 0) inflight = 0;
  d["head"] = head;
  d["tail"] = tail;
  d["in_flight"] = inflight;
  d["capacity"] = g_q.capacity;
  d["processed"] = proc;
  d["heartbeat"] = get_heartbeat();
  d["quit"] = quit;
  d["host_tail_shadow"] = g_tail_shadow;
  d["host_pending"] = (int)g_pending.size();
  d["staging"] = (g_staging == StagingMode::MappedHost) ? "mapped" : "device";
  d["shard_size"] = g_shard_size;
  return d;
}

// Enqueue a micro-request into host-side pending list
void submit_add(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, /*op=*/0, a, b, out);
  if (g_verbose_level > 1) {
    fprintf(stderr, "[host] pending add numel=%lld in0=%p in1=%p out0=%p\n", (long long)t.numel, t.in0.data, t.in1.data, t.out0.data);
  }
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

void submit_mul(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, /*op=*/1, a, b, out);
  if (g_verbose_level > 1) {
    fprintf(stderr, "[host] pending mul numel=%lld in0=%p in1=%p out0=%p\n", (long long)t.numel, t.in0.data, t.in1.data, t.out0.data);
  }
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

void submit_sub(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, /*op=*/2, a, b, out);
  if (g_verbose_level > 1) {
    fprintf(stderr, "[host] pending sub numel=%lld in0=%p in1=%p out0=%p\n", (long long)t.numel, t.in0.data, t.in1.data, t.out0.data);
  }
  std::lock_guard<std::mutex> lock(g_mu); g_pending.push_back(t);
}

void submit_div(torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, /*op=*/3, a, b, out);
  if (g_verbose_level > 1) {
    fprintf(stderr, "[host] pending div numel=%lld in0=%p in1=%p out0=%p\n", (long long)t.numel, t.in0.data, t.in1.data, t.out0.data);
  }
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
  // Ensure worker is running (it may have yielded/exited previously)
  ensure_worker_alive();
  if (g_verbose_level > 0) {
    fprintf(stderr, "[host] flush starting count=%zu sync=%d\n", local.size(), (int)sync);
  }
  // Copy micro-tasks to device array
  Task* d_subs = nullptr; // device-visible pointer carrying the Task array
  size_t bytes = local.size() * sizeof(Task);
  if (g_staging == StagingMode::MappedHost) {
    // Use pinned mapped host memory: CPU memcpy to host buffer, pass device alias to GPU
    Task* h_subs = nullptr;
    void* host_block_to_free = nullptr;
    if (sync) {
      h_subs = ensure_batch_host_buffer(local.size());
      d_subs = g_batch_host_dev;
    } else {
      CUDA_RT_CHECK(cudaHostAlloc((void**)&h_subs, bytes, cudaHostAllocMapped));
      void* dev_alias = nullptr;
      CUDA_RT_CHECK(cudaHostGetDevicePointer(&dev_alias, h_subs, 0));
      d_subs = (Task*)dev_alias;
      host_block_to_free = h_subs;
      g_async_host_buffers.push_back(host_block_to_free);
    }
    if (bytes > 0) {
      // CPU memcpy to pinned buffer
      std::memcpy(h_subs, local.data(), bytes);
      // Attach memory to control stream to enforce ordering
      (void)cudaStreamAttachMemAsync(g_ctrl_stream, h_subs, 0, cudaMemAttachGlobal);
    }
  } else {
    // Device memcpy staging
    if (sync) {
      d_subs = ensure_batch_buffer(local.size());
    } else {
      CUDA_RT_CHECK(cudaMalloc(&d_subs, bytes));
      g_async_buffers.push_back(d_subs);
    }
    if (bytes > 0) {
      bool need_copy = true;
      unsigned long long h = 0;
      if (sync) {
        // Cache device sub-task array across sync flushes if content is identical
        if (bytes == g_cached_bytes) {
          h = fnv1a64(local.data(), bytes);
          if (g_cached_valid && h == g_cached_hash) {
            need_copy = false;
            if (g_verbose_level > 0) {
              fprintf(stderr, "[host] reuse cached sub-task array (bytes=%zu)\n", bytes);
            }
          }
        }
      }
      if (need_copy) {
        CUDA_RT_CHECK(cudaMemcpy(d_subs, local.data(), bytes, cudaMemcpyHostToDevice));
        if (sync) {
          if (h == 0) h = fnv1a64(local.data(), bytes);
          g_cached_bytes = bytes;
          g_cached_hash = h;
          g_cached_valid = true;
        }
      }
    }
  }
  if (g_verbose_level > 0) {
    fprintf(stderr, "[host] staged %zu sub-tasks at %p (%zu bytes, %s)\n", local.size(), (void*)d_subs, bytes, g_staging == StagingMode::MappedHost ? "mapped" : "device");
    if (g_verbose_level > 1) {
      size_t lim = local.size() < 8 ? local.size() : 8;
      for (size_t i = 0; i < lim; ++i) {
        const Task& u = local[i];
        fprintf(stderr, "[host] sub[%zu] op=%d numel=%lld in0=%p in1=%p out0=%p\n", i, u.op, (long long)u.numel, u.in0.data, u.in1.data, u.out0.data);
      }
    }
  }
  // Publish batch shards: split sub-task array to allow concurrent processing
  const long long total = (long long)local.size();
  int shards = (g_shard_size > 0) ? (int)((total + g_shard_size - 1) / g_shard_size) : 1;
  if (shards < 1) shards = 1;
  if (g_verbose_level > 0) {
    fprintf(stderr, "[host] enqueue %d batch shards (shard_size=%d, total=%lld)\n", shards, g_shard_size, total);
  }
  unsigned long long before = 0ULL;
  if (sync) before = get_processed_count();
  long long start = 0;
  for (int s = 0; s < shards; ++s) {
    long long cnt = std::min<long long>(g_shard_size, total - start);
    Task batch{}; batch.op = kBatchSlot; batch.flags = 0; batch.ndim = 1; batch.numel = cnt;
    batch.in0.data = (void*)((char*)d_subs + start * sizeof(Task));
    batch.in0.dtype = kF32; batch.in0.ndim = 1; batch.in0.sizes[0] = cnt; batch.in0.strides[0] = 1;
    batch.in1.data = nullptr; batch.out0.data = nullptr;
    if (g_verbose_level > 1) {
      int tail = g_tail_shadow; int idx = tail % g_q.capacity;
      fprintf(stderr, "[host] enqueue shard %d/%d idx=%d tail_before=%d start=%lld cnt=%lld ptr=%p\n",
              s+1, shards, idx, tail, start, cnt, batch.in0.data);
    }
    enqueue_task_host_to_device(batch);
    start += cnt;
  }
  CUDA_RT_CHECK(cudaStreamSynchronize(g_ctrl_stream));
  if (sync) {
    const unsigned long long target = before + (unsigned long long)shards;
    int spins = 0;
    while (true) {
      unsigned long long cur = get_processed_count();
      if (cur >= target) break;
      // If worker yielded/ended mid-flush, relaunch it
      if (!worker_alive()) {
        if (g_verbose_level > 0) fprintf(stderr, "[host] worker not alive; relaunching...\n");
        ensure_worker_alive();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      if (g_verbose_level > 0 && (++spins % 40) == 0) { // ~200ms
        int h = read_dev_int(g_q.head);
        int t = read_dev_int(g_q.tail);
        unsigned long long hb = get_heartbeat();
        fprintf(stderr, "[host] wait... head=%d tail=%d processed=%llu target=%llu (before=%llu) hb=%llu\n",
                h, t, (unsigned long long)cur, target, before, hb);
      }
    }
  }
}

// -------- Generic elementwise JIT (float32 contiguous, unary/binary) --------
static std::string build_elementwise_src(const std::string& expr, int arity) {
  // Generate a generic elementwise kernel over TensorRef with dtype/broadcast/strides support.
  std::string src;
  src += R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
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
    // Export device pointer so host can fetch true device function address (typed)
    typedef void (*OpFnT)(const Task&);
    __device__ OpFnT op_impl_ptr = op_impl;
  }
  )";
  return src;
}

static int ensure_elementwise_registered(const std::string& key, const std::string& expr, int arity) {
  std::lock_guard<std::mutex> lock(g_reg_mu);
  auto it = g_op_slots.find(key);
  if (it != g_op_slots.end()) return it->second;
  auto ptx = nvrtc_compile_ptx(build_elementwise_src(expr, arity));
  OpPtrInt addr = load_ptr_from_ptx(ptx, "op_impl_ptr");
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
  enqueue_task_host_to_device(t);
}

void submit_binary(int slot, torch::Tensor a, torch::Tensor b, torch::Tensor out) {
  TORCH_CHECK(g_started, "gpuos not initialized");
  Task t{}; build_binary_task(t, slot, a, b, out);
  std::lock_guard<std::mutex> lock(g_mu);
  enqueue_task_host_to_device(t);
}

// -------- Reduction (sum/mean) along a single axis (rrank==1) --------
static std::string build_reduce_src(const std::string& op_name) {
  std::string src;
  src += R"(
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
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
    typedef void (*OpFnT)(const Task&);
    __device__ OpFnT op_reduce_ptr = op_reduce;
  }
  )";
  return src;
}

static int ensure_reduce_registered(const std::string& key, const std::string& op_name) {
  std::lock_guard<std::mutex> lock(g_reg_mu);
  auto it = g_op_slots.find(key);
  if (it != g_op_slots.end()) return it->second;
  auto ptx = nvrtc_compile_ptx(build_reduce_src(op_name));
  OpPtrInt addr = load_ptr_from_ptx(ptx, "op_reduce_ptr");
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
  std::vector<long> out_sizes(out.sizes().begin(), out.sizes().end());
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
  long stride = 1; for (int d = out_nd - 1; d >= 0; --d) { t.out0.strides[d] = stride; stride *= out_sizes[d]; }
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
  enqueue_task_host_to_device(t);
}

} // namespace gpuos_ext

// Pybind11 module must be at global scope
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace gpuos_ext;
  m.def("init", &init, "Initialize GPUOS persistent runtime", py::arg("capacity")=4096, py::arg("threads_per_block")=256);
  m.def("shutdown", &shutdown, "Shutdown GPUOS runtime");
  m.def("abi_version", [](){ return 1; }, "Extension ABI/version for compatibility checks");
  m.def("set_debug_level", &set_debug_level, "Set device debug print level (0-2)");
  m.def("set_verbose_level", [](int lvl){ g_verbose_level = lvl; }, "Set host verbose level (0-2)");
  m.def("peek_queue", &peek_queue, "Peek GPUOS queue and control state");
  m.def("set_yield_every", &set_yield_every, "Yield persistent worker after every N processed tasks (0=never)");
  m.def("worker_alive", &worker_alive, "Return True if persistent worker kernel is running");
  m.def("submit_add", &submit_add, "Submit add micro-op");
  m.def("submit_mul", &submit_mul, "Submit mul micro-op");
  m.def("submit_sub", &submit_sub, "Submit sub micro-op");
  m.def("submit_div", &submit_div, "Submit div micro-op");
  m.def("flush", &flush, "Flush pending micro-ops", py::arg("sync")=false);
  m.def("register_elementwise", [](const std::string& key, const std::string& expr, int arity){
      return ensure_elementwise_registered(key, expr, arity);
    }, "Register JIT elementwise op and return slot");
  m.def("submit_unary", &submit_unary, "Submit unary op for a slot");
  m.def("submit_binary", &submit_binary, "Submit binary op for a slot");
  m.def("register_reduce", &register_reduce, "Register reduce op (sum/mean) and return slot");
  m.def("submit_reduce", &submit_reduce, "Submit reduce task (axes, keepdim)");
}
