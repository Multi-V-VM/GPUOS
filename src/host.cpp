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

// Kernel launch/symbol wrappers defined in persistent_kernel.cu
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t stream);
extern "C" cudaError_t launch_persistent_worker(WorkQueue q, int blocks, int threads, cudaStream_t stream);
extern "C" cudaError_t gpu_get_processed_count_async(unsigned long long* out, cudaStream_t s);
extern "C" cudaError_t gpu_set_op_table_async(int index, OpPtrInt fn, cudaStream_t s);

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
  #include <cuda_fp16.h>
  #include <cuda_bf16.h>
  extern "C" {
    enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
    const int MAX_NDIM = 8;
    struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
    struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
    __device__ inline long long linear_to_offset(const TensorRef& tr, long long idx) {
      long long off = 0; int nd = tr.ndim; for (int d = nd - 1; d >= 0; --d) { long long dim = tr.sizes[d] > 0 ? tr.sizes[d] : 1; long long i = idx % dim; idx /= dim; off += i * tr.strides[d]; } return off; }
    __device__ inline float ld_as_float(const TensorRef& tr, long long off_elems) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: return ((float*)base)[off_elems];
        case kF16: return __half2float(((const __half*)base)[off_elems]);
        case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off_elems]);
        default: return ((float*)base)[off_elems]; } }
    __device__ inline void st_from_float(const TensorRef& tr, long long off_elems, float v) {
      char* base = (char*)tr.data; switch (tr.dtype) {
        case kF32: ((float*)base)[off_elems] = v; break;
        case kF16: ((__half*)base)[off_elems] = __float2half_rn(v); break;
        case kBF16: ((__nv_bfloat16*)base)[off_elems] = __float2bfloat16(v); break;
        default: ((float*)base)[off_elems] = v; break; } }
    __device__ void op_mul(const Task& t) {
      long long N = t.numel;
      for (long long li = threadIdx.x; li < N; li += blockDim.x) {
        long long oa = linear_to_offset(t.in0, li);
        long long ob = linear_to_offset(t.in1, li);
        long long oc = linear_to_offset(t.out0, li);
        float A = ld_as_float(t.in0, oa);
        float B = ld_as_float(t.in1, ob);
        float R = A * B;
        st_from_float(t.out0, oc, R);
      }
    }
    __global__ void get_op_mul_ptr(void** out) { *out = (void*)op_mul; }
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
    "-I/usr/local/cuda/include",
    "-I/opt/spack/opt/spack/linux-sapphirerapids/cuda-12.9.0-3eylvnf4bglzu4xuvf4iqvqv5fq7bjpt/targets/x86_64-linux/include",
    "-I/usr/include/"
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

// Load module and extract the device function pointer using a helper kernel
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

  CUfunction kernel = nullptr;
  CUDA_DRV_CHECK(cuModuleGetFunction(&kernel, mod, "get_op_mul_ptr"));

  void** d_out = nullptr;
  CUDA_RT_CHECK(cudaMalloc(&d_out, sizeof(void*)));

  void* args[] = { &d_out };
  CUDA_DRV_CHECK(cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0, nullptr, args, nullptr));
  CUDA_RT_CHECK(cudaDeviceSynchronize());

  OpPtrInt fn_addr = 0;
  CUDA_RT_CHECK(cudaMemcpy(&fn_addr, d_out, sizeof(fn_addr), cudaMemcpyDeviceToHost));
  CUDA_RT_CHECK(cudaFree(d_out));

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
  CUDA_RT_CHECK(gpu_set_op_table_async(index, fn_addr, stream));
}

static unsigned long long get_processed_count(cudaStream_t stream) {
  unsigned long long c = 0;
  CUDA_RT_CHECK(gpu_get_processed_count_async(&c, stream));
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
    CUDA_RT_CHECK(launch_init_builtin_ops(0));
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
  CUDA_RT_CHECK(launch_persistent_worker(q, blocks.x, threads.x, s_kernel));

  // JIT-compile op_mul and install into slot 1 (keeping built-in add at 0)
  {
    std::string src = build_operator_source_mul();
    auto ptx = nvrtc_compile_to_ptx(src);
    OpPtrInt addr = load_op_mul_ptr_from_ptx(ptx);
    update_jump_table_async(/*index=*/1, addr, s_ctrl);
    CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
    std::cout << "Updated op[1] via JIT to op_mul (C=A*B)" << std::endl;
  }

  // Prepare and publish tasks (use op 1 = mul) with new Task layout
  {
    for (int t = 0; t < num_tasks; ++t) {
      Task tk{};
      tk.op = 1; // op_mul
      tk.flags = 0;
      tk.ndim = 1;
      tk.numel = N;
      // in0
      tk.in0.data = A; tk.in0.dtype = kF32; tk.in0.ndim = 1; tk.in0.sizes[0] = N; tk.in0.strides[0] = 1;
      // in1
      tk.in1.data = B; tk.in1.dtype = kF32; tk.in1.ndim = 1; tk.in1.sizes[0] = N; tk.in1.strides[0] = 1;
      // out0
      tk.out0.data = C; tk.out0.dtype = kF32; tk.out0.ndim = 1; tk.out0.sizes[0] = N; tk.out0.strides[0] = 1;
      q.tasks[t % q.capacity] = tk;
    }
    // Publish tail after writing tasks
    *q.tail = num_tasks;
    // Prefetch optional for portability; skip
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
