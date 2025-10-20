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
  // Block-local striding: one block handles one Task
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
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
  __shared__ Task s_task;
  __shared__ int s_has_work;
  while (atomicAdd(q.quit, 0) == 0) {
    if (threadIdx.x == 0) {
      int idx = atomicAdd(q.head, 1);
      int tail = atomicAdd(q.tail, 0);
      if (idx < tail) {
        s_task = q.tasks[idx % q.capacity];
        s_has_work = 1;
      } else {
        s_has_work = 0;
      }
    }
    __syncthreads();
    if (!s_has_work) {
      __nanosleep(1000);
      continue;
    }

    // Ensure we observe the latest function table contents
    __threadfence();
    OpFn fn = nullptr;
    if (s_task.op >= 0 && s_task.op < GPUOS_MAX_OPS) {
      int phys = atomicAdd(&g_op_alias[s_task.op], 0);
      if (phys >= 0 && phys < GPUOS_MAX_OPS) {
        unsigned long long p = atomicAdd((unsigned long long*)&g_op_table[phys], 0ULL);
        fn = (OpFn)p;
      }
    }
    if (fn) {
      fn(s_task);
      __syncthreads();
      if (threadIdx.x == 0) {
        atomicAdd(&g_processed_count, 1ULL);
      }
    }
    __syncthreads();
  }
}
