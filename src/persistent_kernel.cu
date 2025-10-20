#include "common.h"

#include <cuda.h>
#include <cuda_runtime.h>

// Global jump table and counters
__device__ __managed__ OpFn g_op_table[GPUOS_MAX_OPS];
__device__ __managed__ int  g_op_alias[GPUOS_MAX_OPS];
__device__ __managed__ unsigned long long g_processed_count = 0ULL;

// Built-in default operator: C = A + B
extern "C" __device__ void op_add(const Task& t) {
  const float* a = static_cast<const float*>(t.in0);
  const float* b = static_cast<const float*>(t.in1);
  float* c = static_cast<float*>(t.out0);
  int n = t.n;
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n; i += blockDim.x * gridDim.x) {
    c[i] = a[i] + b[i];
  }
}

// Initialize jump table: null everything, install built-in op at slot 0
extern "C" __global__ void init_builtin_ops() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < GPUOS_MAX_OPS) {
    g_op_table[idx] = nullptr;
    g_op_alias[idx] = idx; // identity mapping
  }
  if (idx == 0) {
    g_op_table[0] = op_add;
  }
}

// Persistent worker kernel: each thread acts as a consumer
extern "C" __global__ void persistent_worker(WorkQueue q) {
  if (q.capacity == 0) return;
  while (atomicAdd(q.quit, 0) == 0) {
    int idx = atomicAdd(q.head, 1);
    if (idx >= atomicAdd(q.tail, 0)) {
      __nanosleep(1000);
      continue;
    }

    Task t = q.tasks[idx % q.capacity];

    // Ensure we observe the latest function table contents
    __threadfence();
    OpFn fn = nullptr;
    if (t.op >= 0 && t.op < GPUOS_MAX_OPS) {
      int phys = atomicAdd(&g_op_alias[t.op], 0); // atomic read alias
      if (phys >= 0 && phys < GPUOS_MAX_OPS) {
        unsigned long long p = atomicAdd((unsigned long long*)&g_op_table[phys], 0ULL);
        fn = (OpFn)p;
      }
    }
    if (fn) {
      fn(t);
      atomicAdd(&g_processed_count, 1ULL);
    }
  }
}
