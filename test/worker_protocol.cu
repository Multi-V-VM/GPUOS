// Regression: block-uniform dispatch while an alias changes; block-uniform exit.
#include "common.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t);
extern "C" cudaError_t launch_persistent_worker(WorkQueue, int, int, cudaStream_t);
extern "C" cudaError_t gpu_set_alias_async(int, int, cudaStream_t);
extern "C" cudaError_t gpu_get_processed_count_async(unsigned long long*, cudaStream_t);
extern __device__ __managed__ unsigned long long g_processed_count;
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { \
  fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); std::exit(1); } } while(0)
__global__ void stop_when_done(int* quit, unsigned long long target) {
  auto start=clock64();
  while (atomicAdd(&g_processed_count,0ULL)<target) {
    __nanosleep(100);
    if(clock64()-start>3000000000ULL) {
      printf("Timed out on device: processed=%llu target=%llu\n",g_processed_count,target);
      break;
    }
  }
  atomicExch(quit,1);
}
int main() {
  constexpr int count=2048, n=2049;
  WorkQueue q{}; q.capacity=count;
  CK(cudaMalloc(&q.tasks,count*sizeof(Task)));
  CK(cudaMalloc(&q.head,sizeof(int))); CK(cudaMalloc(&q.tail,sizeof(int)));
  CK(cudaMalloc(&q.quit,sizeof(int)));
  CK(cudaMemset(q.head,0,sizeof(int))); CK(cudaMemset(q.quit,0,sizeof(int)));
  CK(cudaMemcpy(q.tail,&count,sizeof(int),cudaMemcpyHostToDevice));
  float *a,*b,*out;
  CK(cudaMalloc(&a,n*sizeof(float))); CK(cudaMalloc(&b,n*sizeof(float)));
  CK(cudaMalloc(&out,size_t(count)*n*sizeof(float)));
  std::vector<float> ha(n,2.0f),hb(n,3.0f),result(size_t(count)*n);
  CK(cudaMemcpy(a,ha.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b,hb.data(),n*sizeof(float),cudaMemcpyHostToDevice));
  CK(cudaMemset(out,0xff,result.size()*sizeof(float)));
  std::vector<Task> tasks(count);
  for(int i=0;i<count;++i) {
    Task& t=tasks[i]; t.op=2; t.ndim=1; t.numel=n;
    t.in0.data=a; t.in1.data=b; t.out0.data=out+size_t(i)*n;
    for(TensorRef* ref : {&t.in0,&t.in1,&t.out0}) {
      ref->dtype=kF32; ref->ndim=1; ref->sizes[0]=n; ref->strides[0]=1;
    }
  }
  CK(cudaMemcpy(q.tasks,tasks.data(),count*sizeof(Task),cudaMemcpyHostToDevice));
  cudaStream_t worker,ctrl;
  CK(cudaStreamCreateWithFlags(&worker,cudaStreamNonBlocking));
  CK(cudaStreamCreateWithFlags(&ctrl,cudaStreamNonBlocking));
  // Lazy loading a new control kernel while a persistent kernel is running can
  // synchronize the context. Resolve it before starting the resident workers.
  cudaFuncAttributes attributes{};
  CK(cudaFuncGetAttributes(&attributes,stop_when_done));
  CK(launch_init_builtin_ops(ctrl)); CK(gpu_set_alias_async(2,0,ctrl));
  CK(cudaStreamSynchronize(ctrl));
  fprintf(stderr,"Initialized operators\n");
  if(gpu_set_alias_async(-1,0,ctrl)!=cudaErrorInvalidValue) return 1;
  CK(launch_persistent_worker(q,4,128,worker));
  fprintf(stderr,"Launched workers\n");
  // Flip during execution. Every task may choose add OR mul, but all elements
  // within one task must come from the same function snapshot.
  for(int flip=0;flip<200;++flip) {
    CK(gpu_set_alias_async(2,flip%2,ctrl)); CK(cudaStreamSynchronize(ctrl));
    if(flip==0) fprintf(stderr,"First alias flip complete\n");
  }
  fprintf(stderr,"All alias flips complete\n");
  stop_when_done<<<1,1,0,ctrl>>>(q.quit,count); CK(cudaStreamSynchronize(ctrl));
  CK(cudaStreamSynchronize(worker));
  CK(cudaMemcpy(result.data(),out,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
  int adds=0,muls=0;
  for(int t=0;t<count;++t) {
    float first=result[size_t(t)*n];
    if(first!=5.0f && first!=6.0f) return 3;
    for(int i=0;i<n;++i) if(result[size_t(t)*n+i]!=first) return 4;
    adds+=first==5.0f; muls+=first==6.0f;
  }
  printf("PASS: %d tasks, %zu elements, add=%d mul=%d, 200 alias flips, clean shutdown\n",
         count,result.size(),adds,muls);
  if(!adds || !muls) { fprintf(stderr,"Alias stress did not exercise both versions\n"); return 5; }
  for(void* p : {static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),
      static_cast<void*>(q.tasks),static_cast<void*>(q.head),static_cast<void*>(q.tail),
      static_cast<void*>(q.quit)}) CK(cudaFree(p));
  CK(cudaStreamDestroy(worker)); CK(cudaStreamDestroy(ctrl));
}
