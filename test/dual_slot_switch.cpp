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
    extern "C" {
      struct Task { int op; int n; void* in0; void* in1; void* out0; };
      typedef void(*OpFn)(const Task&);
      __device__ void op_mul(const Task& t) {
        const float* a = (const float*)t.in0;
        const float* b = (const float*)t.in1;
        float* c = (float*)t.out0;
        int n = t.n;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
          c[i] = a[i] * b[i];
        }
      }
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
  nvrtcProgram prog; NVRTC_CHECK(nvrtcCreateProgram(&prog, src.c_str(), "op.cu", 0, nullptr, nullptr));
  std::string arch = arch_opt();
  const char* opts[] = { arch.c_str(), "--std=c++17", "--relocatable-device-code=true", "-rdc=true", "--device-as-default-execution-space" };
  nvrtcResult res = nvrtcCompileProgram(prog, (int)(sizeof(opts)/sizeof(opts[0])), opts);
  size_t logSize=0; NVRTC_CHECK(nvrtcGetProgramLogSize(prog, &logSize));
  if (logSize>1){ std::string log(logSize,'\0'); NVRTC_CHECK(nvrtcGetProgramLog(prog, log.data())); if(res!=NVRTC_SUCCESS) fprintf(stderr,"NVRTC log:\n%s\n",log.c_str()); }
  if (res != NVRTC_SUCCESS) std::exit(4);
  size_t ptxSize=0; NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize)); std::vector<char> ptx(ptxSize); NVRTC_CHECK(nvrtcGetPTX(prog, ptx.data())); NVRTC_CHECK(nvrtcDestroyProgram(&prog)); return ptx;
}

static OpPtrInt load_op_mul_ptr(const std::vector<char>& ptx) {
  CUDA_DRV_CHECK(cuInit(0)); CUDA_RT_CHECK(cudaFree(0)); CUcontext ctx=nullptr; CUDA_DRV_CHECK(cuCtxGetCurrent(&ctx)); if(!ctx){fprintf(stderr,"No CUDA ctx\n"); std::exit(5);} CUmodule mod=nullptr; CUDA_DRV_CHECK(cuModuleLoadDataEx(&mod, ptx.data(), 0, nullptr, nullptr));
  CUdeviceptr sym=0; size_t bytes=0; CUDA_DRV_CHECK(cuModuleGetGlobal(&sym, &bytes, mod, "op_mul_ptr")); if(bytes<sizeof(OpPtrInt)){fprintf(stderr,"bad op_mul_ptr size %zu\n",bytes); std::exit(6);} OpPtrInt addr=0; CUDA_DRV_CHECK(cuMemcpyDtoH(&addr, sym, sizeof(addr))); return addr;
}

static void set_table_slot_async(int index, OpPtrInt fn_addr, cudaStream_t s) {
  CUDA_RT_CHECK(cudaMemcpyToSymbolAsync(g_op_table, &fn_addr, sizeof(fn_addr), index * sizeof(OpFn), cudaMemcpyHostToDevice, s));
}

static void set_alias_async(int logical_id, int physical_slot, cudaStream_t s) {
  CUDA_RT_CHECK(cudaMemcpyToSymbolAsync(g_op_alias, &physical_slot, sizeof(physical_slot), logical_id * sizeof(int), cudaMemcpyHostToDevice, s));
}

static unsigned long long get_done(cudaStream_t s) {
  unsigned long long c = 0; CUDA_RT_CHECK(cudaMemcpyFromSymbolAsync(&c, g_processed_count, sizeof(c), 0, cudaMemcpyDeviceToHost, s)); CUDA_RT_CHECK(cudaStreamSynchronize(s)); return c;
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
  { dim3 blk(128); dim3 grd((GPUOS_MAX_OPS+blk.x-1)/blk.x); init_builtin_ops<<<grd,blk,0,s_ctrl>>>(); CUDA_RT_CHECK(cudaGetLastError()); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); }

  // Launch persistent kernel
  int sm=0; CUDA_RT_CHECK(cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,0)); persistent_worker<<<sm,128,0,s_kernel>>>(q); CUDA_RT_CHECK(cudaGetLastError());

  const int b1=96,b2=96,b3=96; int dev=0; CUDA_RT_CHECK(cudaGetDevice(&dev));

  auto prefetch_q = [&](){ CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tasks,q.capacity*sizeof(Task),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(q.head,sizeof(int),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail,sizeof(int),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(q.quit,sizeof(int),dev,s_ctrl)); };

  // Batch1: logical op L=0 routed to slot 0 (add) -> C1
  for(int t=0;t<b1;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C1,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[t%q.capacity]=tk; }
  *q.tail = b1; prefetch_q(); CUDA_RT_CHECK(cudaMemPrefetchAsync(A,(size_t)N*sizeof(float),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(B,(size_t)N*sizeof(float),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(C1,(size_t)N*sizeof(float),dev,s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)b1) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Prepare backup: JIT mul into physical slot 1 (backup)
  { auto ptx = nvrtc_compile_ptx(build_op_mul_src()); OpPtrInt addr = load_op_mul_ptr(ptx); set_table_slot_async(1, addr, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); }

  // Switch alias: route logical 0 -> physical 1 (mul)
  set_alias_async(/*logical=*/0, /*physical=*/1, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Batch2: logical op 0 now uses mul -> C2
  for(int t=0;t<b2;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C2,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[(b1+t)%q.capacity]=tk; }
  *q.tail = b1 + b2; prefetch_q(); CUDA_RT_CHECK(cudaMemPrefetchAsync(C2,(size_t)N*sizeof(float),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail,sizeof(int),dev,s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)(b1+b2)) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Rollback: alias logical 0 -> physical 0 (add)
  set_alias_async(/*logical=*/0, /*physical=*/0, s_ctrl); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl));

  // Batch3: logical op 0 now back to add -> C3
  for(int t=0;t<b3;++t){ Task tk{}; tk.op=0; tk.flags=0; tk.ndim=1; tk.numel=N; tk.in0={A,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.in1={B,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; tk.out0={C3,kF32,1,{N,0,0,0,0,0,0,0},{1,0,0,0,0,0,0,0}}; q.tasks[(b1+b2+t)%q.capacity]=tk; }
  *q.tail = b1 + b2 + b3; prefetch_q(); CUDA_RT_CHECK(cudaMemPrefetchAsync(C3,(size_t)N*sizeof(float),dev,s_ctrl)); CUDA_RT_CHECK(cudaMemPrefetchAsync(q.tail,sizeof(int),dev,s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); while(get_done(s_ctrl) < (unsigned long long)(b1+b2+b3)) std::this_thread::sleep_for(std::chrono::milliseconds(5));

  // Stop
  int one=1; CUDA_RT_CHECK(cudaMemcpyAsync(q.quit,&one,sizeof(one),cudaMemcpyHostToDevice,s_ctrl)); CUDA_RT_CHECK(cudaStreamSynchronize(s_ctrl)); CUDA_RT_CHECK(cudaDeviceSynchronize());

  // Verify few values
  bool ok=true; for(int i=0;i<5;++i){ float e_add=A[i]+B[i]; float e_mul=A[i]*B[i]; if (std::abs(C1[i]-e_add)>1e-4f) ok=false; if (std::abs(C2[i]-e_mul)>1e-4f) ok=false; if (std::abs(C3[i]-e_add)>1e-4f) ok=false; std::cout << "C1["<<i<<"]="<<C1[i]<<" add="<<e_add<<"; C2["<<i<<"]="<<C2[i]<<" mul="<<e_mul<<"; C3["<<i<<"]="<<C3[i]<<" add="<<e_add<<"\n"; }
  std::cout << (ok?"Dual-slot switch OK":"Dual-slot switch FAILED") << std::endl;

  CUDA_RT_CHECK(cudaFree(A)); CUDA_RT_CHECK(cudaFree(B)); CUDA_RT_CHECK(cudaFree(C1)); CUDA_RT_CHECK(cudaFree(C2)); CUDA_RT_CHECK(cudaFree(C3)); CUDA_RT_CHECK(cudaFree(q.tasks)); CUDA_RT_CHECK(cudaFree(q.head)); CUDA_RT_CHECK(cudaFree(q.tail)); CUDA_RT_CHECK(cudaFree(q.quit)); CUDA_RT_CHECK(cudaStreamDestroy(s_kernel)); CUDA_RT_CHECK(cudaStreamDestroy(s_ctrl));
  return ok?0:1;
}
