#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "../src/common.h"

// Kernel prototypes (defined in src/persistent_kernel.cu)
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

// JIT source: op_mul with pointer bridge
static std::string build_op_mul_src() {
  return R"(
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    #include <math.h>
    extern "C" {
      enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
      const int MAX_NDIM = 8;
      struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
      struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
      __device__ inline long long linear_to_offset(const TensorRef& tr, long long idx) { long long off=0; int nd=tr.ndim; for (int d=nd-1; d>=0; --d){ long long dim=tr.sizes[d]>0?tr.sizes[d]:1; long long i=idx%dim; idx/=dim; off+=i*tr.strides[d]; } return off; }
      __device__ inline float ld_as_float(const TensorRef& tr, long long off) { char* base=(char*)tr.data; switch(tr.dtype){ case kF32: return ((float*)base)[off]; case kF16: return __half2float(((const __half*)base)[off]); case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off]); default: return ((float*)base)[off]; } }
      __device__ inline void st_from_float(const TensorRef& tr, long long off, float v) { char* base=(char*)tr.data; switch(tr.dtype){ case kF32: ((float*)base)[off]=v; break; case kF16: ((__half*)base)[off]=__float2half_rn(v); break; case kBF16: ((__nv_bfloat16*)base)[off]=__float2bfloat16(v); break; default: ((float*)base)[off]=v; break; } }
      __device__ void op_mul(const Task& t) { long long N=t.numel; for(long long li=threadIdx.x; li<N; li+=blockDim.x){ long long oa=linear_to_offset(t.in0,li); long long ob=linear_to_offset(t.in1,li); long long oc=linear_to_offset(t.out0,li); float R = ld_as_float(t.in0,oa) * ld_as_float(t.in1,ob); st_from_float(t.out0,oc,R);} }
      __device__ void* op_mul_ptr = (void*)op_mul;
    }
  )";
}

static std::string arch_opt() {
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

static std::vector<char> nvrtc_compile_ptx(const std::string& src) {
  nvrtcProgram prog;
  NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "op.cu", 0, nullptr, nullptr));
  std::string arch = arch_opt();
  const char* opts[] = {
    arch.c_str(), "--std=c++17", "--relocatable-device-code=true", "-rdc=true", "--device-as-default-execution-space"
  };
  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);
  size_t logSize = 0; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize > 1) {
    std::string log(logSize, '\0'); NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data()));
    if (res != NVRTC_SUCCESS) fprintf(stderr, "NVRTC log:\n%s\n", log.c_str());
  }
  if (res != NVRTC_SUCCESS) { std::exit(4); }
  size_t ptxSize = 0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
  std::vector<char> ptx(ptxSize); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data()));
  NVRTC_CHECK(nvrtcDestroyProgram(&prog));
  return ptx;
}

static OpPtrInt load_op_mul_ptr(const std::vector<char>& ptx) {
  CUDA_DRV_CHECK(cuInit(0));
  CUDA_RT_CHECK(cudaFree(0));
  CUcontext ctx = nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) { fprintf(stderr, "No CUDA context\n"); std::exit(5); }
  CUmodule mod = nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUdeviceptr sym = 0; size_t bytes = 0; CUDA_DRV_CHECK(cuModuleGetGlobal(&sym, &bytes, mod, "op_mul_ptr"));
  if (bytes < sizeof(OpPtrInt)) { fprintf(stderr, "bad op_mul_ptr size %zu\n", bytes); std::exit(6); }
  OpPtrInt addr = 0; CUDA_DRV_CHECK(cuMemcpyDtoH(&addr, sym, sizeof(addr)));
  // module lifetime should exceed usage; we intentionally keep it loaded
  return addr;
}

static void set_table_slot_async(int index, OpPtrInt fn_addr, cudaStream_t stream) {
  CUDA_RT_CHECK(cudaMemcpyToSymbolAsync(g_op_table, &fn_addr, sizeof(fn_addr), index * sizeof(OpFn), cudaMemcpyHostToDevice, stream));
}

static unsigned long long get_done(cudaStream_t s) {
  unsigned long long c = 0;
  CUDA_RT_CHECK(cudaMemcpyFromSymbolAsync(&c, g_processed_count, sizeof(c), 0, cudaMemcpyDeviceToHost, s));
  CUDA_RT_CHECK(cudaStreamSynchronize(s));
  return c;
}

int main() {
  CUDA_RT_CHECK(cudaSetDevice(0));
  CUDA_RT_CHECK(cudaFree(0));

  // Queue
  const int capacity = 1024;
  WorkQueue q{};
  CUDA_RT_CHECK(cudaMallocManaged(&q.tasks, capacity * sizeof(Task)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.head, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.tail, sizeof(int)));
  CUDA_RT_CHECK(cudaMallocManaged(&q.quit, sizeof(int)));
  q.capacity = capacity;
  CUDA_RT_CHECK(cudaMemset(q.head, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(q.tail, 0, sizeof(int)));
  CUDA_RT_CHECK(cudaMemset(q.quit, 0, sizeof(int)));

  // Data
  const int N = 1 << 15;
  float *A=nullptr, *B=nullptr, *C_add=nullptr, *C_mul=nullptr;
  CUDA_RT_CHECK(cudaMallocManaged(&A, (size_t)N * sizeof(float)));
  CUDA_RT_CHECK(cudaMallocManaged(&B, (size_t)N * sizeof(float)));
  CUDA_RT_CHECK(cudaMallocManaged(&C_add, (size_t)N * sizeof(float)));
  CUDA_RT_CHECK(cudaMallocManaged(&C_mul, (size_t)N * sizeof(float)));
  for (int i = 0; i < N; ++i) { A[i] = 0.5f * i; B[i] = 1.0f + (float)(i % 5); C_add[i] = 0.0f; C_mul[i] = 0.0f; }

  // Streams
  cudaStream_t s_kernel, s_ctrl;
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_kernel, cudaStreamNonBlocking));
  CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_ctrl, cudaStreamNonBlocking));

  // Initialize builtin ops table
  {
    dim3 blk(128); dim3 grd((GPUOS_MAX_OPS + blk.x - 1) / blk.x);
    init_builtin_ops<<<grd, blk, 0, s_ctrl>>>();
    CUDA_RT_CHECK(cudaGetLastError());
    CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
  }

  // Launch persistent kernel
  int sm = 0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, 0));
  persistent_worker<<<sm, 128, 0, s_kernel>>>(q);
  CUDA_RT_CHECK(cudaGetLastError());

  // Batch 1: op 0 (add) -> C_add
  const int batch1 = 128, batch2 = 128;
  for (int t = 0; t < batch1; ++t) {
    Task tk{}; tk.op = 0; tk.flags=0; tk.ndim=1; tk.numel=N;
    tk.in0 = {A, kF32, 1, {N,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}};
    tk.in1 = {B, kF32, 1, {N,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}};
    tk.out0= {C_add, kF32, 1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}};
    q.tasks[t % q.capacity] = tk;
  }
  *q.tail = batch1;
  int device_ordinal = 0; CUDA_RT_CHECK(cudaGetDevice(&device_ordinal));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tasks, static_cast<size_t>(q.capacity) * sizeof(Task), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(q.head, sizeof(int), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail, sizeof(int), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(q.quit, sizeof(int), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(A, static_cast<size_t>(N) * sizeof(float), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(B, static_cast<size_t>(N) * sizeof(float), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(C_add, static_cast<size_t>(N) * sizeof(float), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Wait completion of batch1
  while (get_done(s_ctrl) < (unsigned long long)batch1) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Hot-swap: JIT op_mul and write into slot 0
  {
    auto ptx = nvrtc_compile_ptx(build_op_mul_src());
    OpPtrInt addr = load_op_mul_ptr(ptx);
    set_table_slot_async(0, addr, s_ctrl); // swap slot 0 in-place
    CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
  }

  // Batch 2: still op 0 but now mul -> C_mul
  for (int t = 0; t < batch2; ++t) {
    Task tk{}; tk.op = 0; tk.flags=0; tk.ndim=1; tk.numel=N;
    tk.in0 = {A, kF32, 1, {N,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}};
    tk.in1 = {B, kF32, 1, {N,0,0,0,0,0,0,0}, {1,0,0,0,0,0,0,0}};
    tk.out0= {C_mul, kF32, 1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}};
    q.tasks[(batch1 + t) % q.capacity] = tk;
  }
  *q.tail = batch1 + batch2;
  CUDA_RT_CHECK(cudaMemPrefetchAsync(C_mul, static_cast<size_t>(N) * sizeof(float), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail, sizeof(int), device_ordinal, s_ctrl));
  CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Wait all complete
  while (get_done(s_ctrl) < (unsigned long long)(batch1 + batch2)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  // Quit
  int one = 1; CUDA_RT_CHECK(cudaMemcpyAsync(q.quit, &one, sizeof(one), cudaMemcpyHostToDevice, s_ctrl));
  CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));
  CUDA_RT_CHECK(cudaDeviceSynchronize());

  // Verify: add then mul
  bool ok = true;
  for (int i = 0; i < 5; ++i) {
    float e_add = A[i] + B[i];
    float e_mul = A[i] * B[i];
    if (std::abs(C_add[i] - e_add) > 1e-4f) ok = false;
    if (std::abs(C_mul[i] - e_mul) > 1e-4f) ok = false;
    std::cout << "C_add[" << i << "]=" << C_add[i] << " expect " << e_add
              << "; C_mul[" << i << "]=" << C_mul[i] << " expect " << e_mul << "\n";
  }
  std::cout << (ok ? "Online switch OK" : "Online switch FAILED") << std::endl;

  CUDA_RT_CHECK(cudaFree(A));
  CUDA_RT_CHECK(cudaFree(B));
  CUDA_RT_CHECK(cudaFree(C_add));
  CUDA_RT_CHECK(cudaFree(C_mul));
  CUDA_RT_CHECK(cudaFree(q.tasks));
  CUDA_RT_CHECK(cudaFree(q.head));
  CUDA_RT_CHECK(cudaFree(q.tail));
  CUDA_RT_CHECK(cudaFree(q.quit));
  CUDA_RT_CHECK(cudaStreamDestroy(s_kernel));
  CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));
  return ok ? 0 : 1;
}
