#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cassert>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "common.h"

// Error handling helpers
#define CUDA_RT_CHECK(expr) do { \
  cudaError_t _err = (expr); \
  if (_err != cudaSuccess) { \
    fprintf(stderr, "CUDA Runtime error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__, cudaGetErrorString(_err)); \
    std::exit(1); \
  } \
} while(0)

static const char* cu_errstr(CUresult r) {
  const char* s = nullptr;
  cuGetErrorString(r, &s);
  return s ? s : "<unknown>";
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

// Build JIT operator source (op_mul) with a pointer-bridge symbol
static std::string build_operator_source_mul() {
  static const char* src = R"(
  extern "C" {
    struct Task { int op; int n; void* in0; void* in1; void* out0; };
    typedef void(*OpFn)(const Task&);
    __device__ void op_mul(const Task& t) {
      const float* a = (const float*)t.in0;
      const float* b = (const float*)t.in1;
      float* c = (float*)t.out0;
      int n = t.n;
      for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
        c[i] = a[i] * b[i];
      }
    }
    // Bridge variable to expose the device function pointer value
    __device__ void* op_mul_ptr = (void*)op_mul;
  }
  )";
  return std::string(src);
}

static std::string compute_arch_option_default90() {
  // Prefer explicit compute_90 (as per spec example); allow override by env
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

static std::vector<char> nvrtc_compile_to_ptx(const std::string& src) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), /*name*/"op.cu", 0, nullptr, nullptr));

  std::string arch = compute_arch_option_default90();
  const char* opts[] = {
    arch.c_str(),
    "--std=c++17",
    "--relocatable-device-code=true",
    "-rdc=true",
    "--device-as-default-execution-space",
  };

  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);

  // Print log on failure or if non-empty
  size_t logSize = 0;
  NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) {
    std::string log(logSize, '\0');
    NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
    if (res != NVRTC_SUCCESS) {
      fprintf(stderr, "NVRTC compile log:\n%s\n", log.c_str());
    }
  }

  if (res != NVRTC_SUCCESS) {
    fprintf(stderr, "NVRTC compilation failed\n");
    std::exit(4);
  }

  size_t ptxSize = 0;
  NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize);
  NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
}

// Load module and extract the device function pointer stored in symbol op_mul_ptr
static OpPtrInt load_op_mul_ptr_from_ptx(const std::vector<char>& ptx) {
  CUDA_DRV_CHECK(cuInit(0));
  // Ensure runtime context is created
  CUDA_RT_CHECK(cudaFree(0));

  CUcontext ctx = nullptr;
  CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    fprintf(stderr, "No current CUDA context found.\n");
    std::exit(5);
  }

  CUmodule mod = nullptr;
  CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));

  CUdeviceptr d_sym = 0;
  size_t bytes = 0;
  CUDA_DRV_CHECK(cuModuleGetGlobal(&d_sym, &bytes, mod, "op_mul_ptr"));
  if (bytes < sizeof(OpPtrInt)) {
    fprintf(stderr, "Unexpected op_ptr size: %zu\n", bytes);
    std::exit(6);
  }
  OpPtrInt fn_addr = 0;
  CUDA_DRV_CHECK(cuMemcpyDtoH(&fn_addr, d_sym, sizeof(fn_addr)));

  // Keep module alive as long as operator is in use. For this demo, we leak it
  // on purpose; in a real system, store CUmodule and unload when replacing.
  return fn_addr;
}

// Update the device jump table at given index with the device function pointer value
static void update_jump_table_async(int index, OpPtrInt fn_addr, cudaStream_t stream) {
  if (index < 0 || index >= GPUOS_MAX_OPS) {
    fprintf(stderr, "Invalid op index %d\n", index);
    std::exit(7);
  }
  // __managed__ symbol: use symbol copy with offset
  CUDA_RT_CHECK(cudaMemcpyToSymbolAsync(g_op_table, &fn_addr, sizeof(fn_addr), index * sizeof(OpFn), cudaMemcpyHostToDevice, stream));
}

static unsigned long long get_processed_count(cudaStream_t stream) {
  unsigned long long c = 0;
  CUDA_RT_CHECK(cudaMemcpyFromSymbolAsync(&c, g_processed_count, sizeof(c), 0, cudaMemcpyDeviceToHost, stream));
  CUDA_RT_CHECK(cudaStreamSynchronize(stream));
  return c;
}

int main() {
  // Basic device init
  CUDA_RT_CHECK(cudaSetDevice(0));
  CUDA_RT_CHECK(cudaFree(0));

  // Create queue and buffers (Unified Memory for simplicity)
  const int capacity = 1024;         // ring size
  const int N = 1 << 16;             // elements per task
  const int num_tasks = 256;         // number of tasks to feed

  WorkQueue q{};
  CUDA_RT_CHECK(cudaMallocManaged(&q.tasks, capacity * sizeof(Task)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.head, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.tail, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.quit, sizeof(int)));
  q.capacity = capacity;

  // Data buffers A, B, C
  float *A = nullptr, *B = nullptr, *C = nullptr;
  CUDA_RT_CHECK(cudaMallocManaged(&A, (size_t)N * sizeof(float)));
  CUDA_RT_CHECK(cudaMallocManaged(&B, (size_t)N * sizeof(float)));
  CUDA_RT_CHECK(cudaMallocManaged(&C, (size_t)N * sizeof(float)));
  for (int i = 0; i < N; ++i) { A[i] = (float)i * 0.5f; B[i] = 2.0f + (float)(i % 7); C[i] = 0.0f; }

  // Initialize queue indices
  CUDA_RT_CHECK(cudaMemset(q.head, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(q.tail, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(q.quit, 0, sizeof(int)));

  // Initialize jump table with built-in op_add at slot 0
  {
    dim3 blk(128);
    dim3 grd((GPUOS_MAX_OPS + blk.x - 1) / blk.x);
    init_builtin_ops<<<grd, blk>>>();
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaDeviceSynchronize());
  }

  // Launch persistent kernel (one block per SM, 128 threads each) on a non-default stream
  int sm = 0;
  CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  dim3 blocks(sm);
  dim3 threads(128);
  cudaStream_t s_kernel, s_ctrl;
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_kernel, cudaStreamNonBlocking));
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_ctrl, cudaStreamNonBlocking));
  persistent_worker<<<blocks, threads, 0, s_kernel>>>(q);
  CUDA_RT_CHECK(cudaGetLastError());

  // JIT-compile op_mul and install into slot 1 (keeping built-in add at 0)
  {
    std::string src = build_operator_source_mul();
    auto ptx = nvrtc_compile_to_ptx(src);
    OpPtrInt addr = load_op_mul_ptr_from_ptx(ptx);
    update_jump_table_async(/*index=*/1, addr, s_ctrl);
    CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
    std::cout << "Updated op[1] via JIT to op_mul (C=A*B)" << std::endl;
  }

  // Prepare and publish tasks (use op 1 = mul)
  {
    for (int t = 0; t < num_tasks; ++t) {
      Task tk;
      tk.op = 1; // op_mul
      tk.n = N;
      tk.in0 = A;
      tk.in1 = B;
      tk.out0 = C;
      q.tasks[t % q.capacity] = tk;
    }
    // Publish tail after writing tasks
    *q.tail = num_tasks;
    int dev = 0;
    CUDA_RT_CHECK(cudaGetDevice(&dev));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tasks, q.capacity * sizeof(Task), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(q.head, sizeof(int), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail, sizeof(int), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(q.quit, sizeof(int), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(A, (size_t)N * sizeof(float), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(B, (size_t)N * sizeof(float), dev, s_ctrl));
    CUDA_RT_CHECK(cudaMemPrefetchAsync(C, (size_t)N * sizeof(float), dev, s_ctrl));
    CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
  }

  // Wait for all tasks to be processed
  unsigned long long target = num_tasks;
  while (true) {
    unsigned long long done = get_processed_count(s_ctrl);
    if (done >= target) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }

  // Signal stop and join
  int one = 1;
  CUDA_RT_CHECK(cudaMemcpyAsync(q.quit, &one, sizeof(one), cudaMemcpyHostToDevice, s_ctrl));
  CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
  CUDA_RT_CHECK(cudaDeviceSynchronize());

  // Verify a few results C = A * B
  bool ok = true;
  for (int i = 0; i < 5; ++i) {
    float expect = A[i] * B[i];
    float got = C[i];
    if (std::abs(expect - got) > 1e-4f) ok = false;
    std::cout << "C[" << i << "] = " << got << " (expect " << expect << ")\n";
  }
  std::cout << (ok ? "Verification OK" : "Verification FAILED") << std::endl;

  // Cleanup
  CUDA_RT_CHECK(cudaFree(A));
  CUDA_RT_CHECK(cudaFree(B));
  CUDA_RT_CHECK(cudaFree(C));
  CUDA_RT_CHECK(cudaFree(q.tasks));
  CUDA_RT_CHECK(cudaFree(q.head));
  CUDA_RT_CHECK(cudaFree(q.tail));
  CUDA_RT_CHECK(cudaFree(q.quit));
  CUDA_RT_CHECK(cudaStreamDestroy(s_kernel));
  CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));
  return ok ? 0 : 1;
}
