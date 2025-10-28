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

#include "../src/common.h"
#include <vector>
#include <cuda_runtime_api.h>

// Portable wrapper for cudaMemPrefetchAsync across CUDA versions
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12000)
static inline cudaError_t gpuos_prefetch_async(void* ptr, size_t bytes, int device, cudaStream_t stream) {
  cudaMemLocation loc{}; loc.type = cudaMemLocationTypeDevice; loc.id = device;
  return cudaMemPrefetchAsync(ptr, bytes, loc, 0u, stream);
}
#define cudaMemPrefetchAsync(PTR, BYTES, DEV, STREAM) gpuos_prefetch_async((void*)(PTR), (size_t)(BYTES), (int)(DEV), (cudaStream_t)(STREAM))
#endif

// As a final fallback for environments where cudaMemPrefetchAsync signature
// differs or is unavailable, turn prefetch into a no-op to unblock builds.
#ifndef GPUOS_ENABLE_PREFETCH
#undef cudaMemPrefetchAsync
#define cudaMemPrefetchAsync(...) (cudaSuccess)
#endif

// Kernel/symbol wrappers (defined in src/persistent_kernel.cu)
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

// JIT source: op_mul with pointer bridge
static std::string build_op_mul_src() {
  return R"(
    #include <cuda_fp16.h>
    #include <cuda_bf16.h>
    extern "C" {
      enum DType { kF32=0, kF16=1, kBF16=2, kI32=3, kF64=4 };
      const int MAX_NDIM = 8;
      struct TensorRef { void* data; int dtype; int ndim; long long sizes[MAX_NDIM]; long long strides[MAX_NDIM]; };
      struct Task { int op; int flags; int ndim; long long numel; int rrank; int r_axes[MAX_NDIM]; int r_keepdim; TensorRef in0; TensorRef in1; TensorRef out0; };
      __device__ inline long long linear_to_offset(const TensorRef& tr, long long idx) { long long off=0; int nd=tr.ndim; for (int d=nd-1; d>=0; --d){ long long dim=tr.sizes[d]>0?tr.sizes[d]:1; long long i=idx%dim; idx/=dim; off+=i*tr.strides[d]; } return off; }
      __device__ inline float ld_as_float(const TensorRef& tr, long long off) { char* base=(char*)tr.data; switch(tr.dtype){ case kF32: return ((float*)base)[off]; case kF16: return __half2float(((const __half*)base)[off]); case kBF16: return __bfloat162float(((const __nv_bfloat16*)base)[off]); default: return ((float*)base)[off]; } }
      __device__ inline void st_from_float(const TensorRef& tr, long long off, float v) { char* base=(char*)tr.data; switch(tr.dtype){ case kF32: ((float*)base)[off]=v; break; case kF16: ((__half*)base)[off]=__float2half_rn(v); break; case kBF16: ((__nv_bfloat16*)base)[off]=__float2bfloat16(v); break; default: ((float*)base)[off]=v; break; } }
      __device__ void op_mul(const Task& t) { long long N=t.numel; for(long long li=threadIdx.x; li<N; li+=blockDim.x){ long long oa=linear_to_offset(t.in0,li); long long ob=linear_to_offset(t.in1,li); long long oc=linear_to_offset(t.out0,li); float R = ld_as_float(t.in0,oa) * ld_as_float(t.in1,ob); st_from_float(t.out0,oc,R);} }
      __global__ void get_op_mul_ptr(void** out) { *out = (void*)op_mul; }
    }
  )";
}

static std::string arch_opt() {
  const char* env = std::getenv("GPUOS_NVRTC_ARCH");
  if (env && *env) return std::string("--gpu-architecture=") + env;
  return std::string("--gpu-architecture=compute_90");
}

static std::vector<char> nvrtc_compile_ptx(const std::string& src) {
  nvrtcProgram prog; NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "op.cu", 0, nullptr, nullptr));
  std::string arch = arch_opt();
  const char* opts[] = { arch.c_str(), "--std=c++17", "--relocatable-device-code=true", "-rdc=true", "--device-as-default-execution-space",     "-I/opt/spack/opt/spack/linux-sapphirerapids/cuda-12.9.0-3eylvnf4bglzu4xuvf4iqvqv5fq7bjpt/targets/x86_64-linux/include",
    "-I/usr/include/"};
  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);
  size_t logSize=0; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize>1){ std::string log(logSize,'\0'); NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data())); if(res!=NVRTC_SUCCESS) fprintf(stderr,"NVRTC compile log:\n%s\n",log.c_str()); }
  if (res != NVRTC_SUCCESS) std::exit(4);
  size_t ptxSize=0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize)); std::vector<char> ptx(ptxSize); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data())); NVRTC_CHECK(nvrtcDestroyProgram(&prog)); return ptx;
}

static OpPtrInt load_op_mul_ptr(const std::vector<char>& ptx) {
  CUDA_DRV_CHECK(cuInit(0)); CUDA_RT_CHECK(cudaFree(0)); CUcontext ctx=nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx)); if(!ctx){fprintf(stderr,"No CUDA ctx\n"); std::exit(5);} CUmodule mod=nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUfunction kernel=nullptr; CUDA_DRV_CHECK(cuModuleGetFunction(&kernel, mod, "get_op_mul_ptr"));
  void** d_out=nullptr; CUDA_RT_CHECK(cudaMalloc(&d_out, sizeof(void*)));
  void* args[] = { &d_out }; CUDA_DRV_CHECK(cuLaunchKernel(kernel, 1,1,1, 1,1,1, 0, nullptr, args, nullptr)); CUDA_RT_CHECK(cudaDeviceSynchronize());
  OpPtrInt addr=0; CUDA_RT_CHECK(cudaMemcpy(&addr, d_out, sizeof(addr), cudaMemcpyDeviceToHost)); CUDA_RT_CHECK(cudaFree(d_out)); return addr;
}

static void set_table_slot_async(int index, OpPtrInt fn_addr, cudaStream_t s) {
  CUDA_RT_CHECK(gpu_set_op_table_async(index, fn_addr, s));
}

static void set_alias_async(int logical_id, int physical_slot, cudaStream_t s) {
  CUDA_RT_CHECK(gpu_set_alias_async(logical_id, physical_slot, s));
}

static unsigned long long get_done(cudaStream_t s) {
  unsigned long long c = 0;
  CUDA_RT_CHECK(gpu_get_processed_count_async(&c, s));
  CUDA_RT_CHECK(cudaStreamSynchronize(s));
  return c;
}

int main(){
  CUDA_RT_CHECK(cudaSetDevice(0)); CUDA_RT_CHECK(cudaFree(0));

  // Queue
  const int capacity = 1024;
  WorkQueue q{}; CUDA_RT_CHECK(cudaMallocManaged(&q.tasks, capacity * sizeof(Task))); CUDA_RT_CHECK(cudaMallocManaged(&q.head, sizeof(int))); CUDA_RT_CHECK(cudaMallocManaged(&q.tail, sizeof(int))); CUDA_RT_CHECK(cudaMallocManaged(&q.quit, sizeof(int))); q.capacity = capacity; CUDA_RT_CHECK(cudaMemset(q.head, 0, sizeof(int))); CUDA_RT_CHECK(cudaMemset(q.tail, 0, sizeof(int))); CUDA_RT_CHECK(cudaMemset(q.quit, 0, sizeof(int)));

  // Data
  const int N = 1<<15; float *A=nullptr,*B=nullptr,*C1=nullptr,*C2=nullptr,*C3=nullptr; CUDA_RT_CHECK(cudaMallocManaged(&A,(size_t)N*sizeof(float))); CUDA_RT_CHECK(cudaMallocManaged(&B,(size_t)N*sizeof(float))); CUDA_RT_CHECK(cudaMallocManaged(&C1,(size_t)N*sizeof(float))); CUDA_RT_CHECK(cudaMallocManaged(&C2,(size_t)N*sizeof(float))); CUDA_RT_CHECK(cudaMallocManaged(&C3,(size_t)N*sizeof(float)));
  for(int i=0;i<N;++i){ A[i]=0.3f*i; B[i]=2.0f+(float)(i%9); C1[i]=C2[i]=C3[i]=0.0f; }

  // Streams
  cudaStream_t s_kernel,s_ctrl; CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_kernel,cudaStreamNonBlocking)); CUDA_RT_CHECK(cudaStreamCreateWithFlags(&s_ctrl,cudaStreamNonBlocking));

  // Init builtin (alias identity, table[0]=add)
  { CUDA_RT_CHECK(launch_init_builtin_ops(s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); }

  // Launch persistent kernel
  int sm=0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0)); CUDA_RT_CHECK(launch_persistent_worker(q, sm, 128, s_kernel));

  const int b1=96,b2=96,b3=96;

  auto prefetch_q = [&](){ /* optional UM prefetch skipped for portability */ };

  // Batch1: logical op L=0 routed to slot 0 (add) -> C1
  for(int t=0;t<b1;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C1,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[t%q.capacity]=tk; }
  *q.tail = b1; prefetch_q(); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)b1) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Prepare backup: JIT mul into physical slot 1 (backup)
  { auto ptx = nvrtc_compile_ptx(build_op_mul_src()); OpPtrInt addr = load_op_mul_ptr(ptx); set_table_slot_async(1, addr, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); }

  // Switch alias: route logical 0 -> physical 1 (mul)
  set_alias_async(/*logical=*/0, /*physical=*/1, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Batch2: logical op 0 now uses mul -> C2
  for(int t=0;t<b2;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C2,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[(b1+t)%q.capacity]=tk; }
  *q.tail = b1 + b2; prefetch_q(); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)(b1+b2)) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Rollback: alias logical 0 -> physical 0 (add)
  set_alias_async(/*logical=*/0, /*physical=*/0, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Batch3: logical op 0 now back to add -> C3
  for(int t=0;t<b3;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C3,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[(b1+b2+t)%q.capacity]=tk; }
  *q.tail = b1 + b2 + b3; prefetch_q(); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)(b1+b2+b3)) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Stop
  int one=1; CUDA_RT_CHECK(cudaMemcpyAsync(q.quit,&one,sizeof(one),cudaMemcpyHostToDevice,s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); CUDA_RT_CHECK(cudaDeviceSynchronize());

  // Verify few values
  bool ok=true; for(int i=0;i<5;++i){ float e_add=A[i]+B[i]; float e_mul=A[i]*B[i]; if (std::abs(C1[i]-e_add)>1e-4f) ok=false; if (std::abs(C2[i]-e_mul)>1e-4f) ok=false; if (std::abs(C3[i]-e_add)>1e-4f) ok=false; std::cout << "C1["<<i<<"]="<<C1[i]<<" add="<<e_add<<"; C2["<<i<<"]="<<C2[i]<<" mul="<<e_mul<<"; C3["<<i<<"]="<<C3[i]<<" add="<<e_add<<"\n"; }
  std::cout << (ok?"Dual-slot switch OK":"Dual-slot switch FAILED") << std::endl;

  CUDA_RT_CHECK(cudaFree(A)); CUDA_RT_CHECK(cudaFree(B)); CUDA_RT_CHECK(cudaFree(C1)); CUDA_RT_CHECK(cudaFree(C2)); CUDA_RT_CHECK(cudaFree(C3)); CUDA_RT_CHECK(cudaFree(q.tasks)); CUDA_RT_CHECK(cudaFree(q.head)); CUDA_RT_CHECK(cudaFree(q.tail)); CUDA_RT_CHECK(cudaFree(q.quit)); CUDA_RT_CHECK(cudaStreamDestroy(s_kernel)); CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));
  return ok?0:1;
}
