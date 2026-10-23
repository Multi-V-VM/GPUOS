// Finite resident-worker regression: TMA full tiles, tails, stage reuse, fallback.
#include "common.h"
#include "tma_copy.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t);
extern "C" cudaError_t launch_persistent_worker(WorkQueue,int,int,cudaStream_t);
extern __device__ __managed__ unsigned long long g_processed_count;
extern __device__ gpuos::TmaState<1024,2>& worker_tma_state();
extern "C" __device__ void op_tma_add(const Task&);
#define CK(x) do { auto e=(x); if(e!=cudaSuccess) { fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); std::exit(1); } } while(0)
__global__ void stop_tma(int* quit,int count) {
  auto begin=clock64();
  while(atomicAdd(&g_processed_count,0ULL)<count && clock64()-begin<3000000000ULL) __nanosleep(100);
  atomicExch(quit,1);
}
// Finite invocation path for instrumentation that cannot schedule a concurrent
// stop kernel. It uses the exact runtime handler and shared-state accessor.
__global__ void bounded_tma(const Task* tasks,int count) {
  gpuos::tma_init(worker_tma_state());
  for(int i=blockIdx.x;i<count;i+=gridDim.x) {
    op_tma_add(tasks[i]);__syncthreads();
    if(threadIdx.x==0)atomicAdd(&g_processed_count,1ULL);
  }
  gpuos::tma_destroy(worker_tma_state());
}
int main(int argc,char** argv) {
  bool bounded=argc==2 && std::strcmp(argv[1],"--bounded")==0;
  if(argc>1 && !bounded)return 1;
  constexpr int count=256,span=32768;
  const int lengths[]={0,1,67,1024,1025,4103,8192,16385};
  std::vector<float> input(size_t(count)*span,2.0f), result(input.size());
  float *a,*b,*out;
  CK(cudaMalloc(&a,input.size()*4)); CK(cudaMalloc(&b,input.size()*4)); CK(cudaMalloc(&out,input.size()*4));
  CK(cudaMemcpy(a,input.data(),input.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(b,input.data(),input.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemset(out,0xff,input.size()*4));
  std::vector<Task> tasks(count);
  for(int i=0;i<count;++i) {
    int n=lengths[i%8],stride=(i%3==1 && n<=8192)?2:1,shift=i%3==2?1:0;
    Task& t=tasks[i]; t.op=11; t.ndim=1; t.numel=n;
    t.in0.data=a+size_t(i)*span+shift; t.in1.data=b+size_t(i)*span+shift; t.out0.data=out+size_t(i)*span+shift;
    for(TensorRef* r:{&t.in0,&t.in1,&t.out0}) { r->dtype=kF32; r->ndim=1; r->sizes[0]=n; r->strides[0]=stride; }
  }
  WorkQueue q{}; q.capacity=count;
  CK(cudaMalloc(&q.tasks,count*sizeof(Task))); CK(cudaMalloc(&q.head,4)); CK(cudaMalloc(&q.tail,4)); CK(cudaMalloc(&q.quit,4));
  CK(cudaMemcpy(q.tasks,tasks.data(),count*sizeof(Task),cudaMemcpyHostToDevice));
  CK(cudaMemset(q.head,0,4)); CK(cudaMemset(q.quit,0,4)); CK(cudaMemcpy(q.tail,&count,4,cudaMemcpyHostToDevice));
  cudaStream_t worker,ctrl; CK(cudaStreamCreateWithFlags(&worker,cudaStreamNonBlocking)); CK(cudaStreamCreateWithFlags(&ctrl,cudaStreamNonBlocking));
  cudaFuncAttributes attr{}; CK(cudaFuncGetAttributes(&attr,stop_tma));
  CK(launch_init_builtin_ops(ctrl)); CK(cudaStreamSynchronize(ctrl));
  if(bounded) { bounded_tma<<<4,128,0,worker>>>(q.tasks,count);CK(cudaGetLastError()); }
  else {
    CK(launch_persistent_worker(q,4,128,worker));
    stop_tma<<<1,1,0,ctrl>>>(q.quit,count); CK(cudaStreamSynchronize(ctrl));
  }
  CK(cudaStreamSynchronize(worker));
  if(g_processed_count!=count) { fprintf(stderr,"Completion mismatch\n"); return 2; }
  CK(cudaMemcpy(result.data(),out,result.size()*4,cudaMemcpyDeviceToHost));
  for(int t=0;t<count;++t) {
    int n=lengths[t%8],stride=tasks[t].out0.strides[0],shift=t%3==2?1:0;
    for(int i=0;i<span;++i) {
      bool valid=i>=shift && (i-shift)%stride==0 && (i-shift)/stride<n;
      float x=result[size_t(t)*span+i];
      if(valid?x!=4.0f:!std::isnan(x)) { fprintf(stderr,"FAIL task=%d index=%d\n",t,i); return 3; }
    }
  }
  printf("PASS: 256 tasks; lengths 0..16385; full tiles, tails, stride/alignment fallback, padding and shutdown\n");
  for(void* p:{static_cast<void*>(a),static_cast<void*>(b),static_cast<void*>(out),static_cast<void*>(q.tasks),static_cast<void*>(q.head),static_cast<void*>(q.tail),static_cast<void*>(q.quit)}) CK(cudaFree(p));
  CK(cudaStreamDestroy(worker)); CK(cudaStreamDestroy(ctrl));
}
