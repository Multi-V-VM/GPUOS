#include "common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>

// Global jump table and counters
__device__ __managed__ OpFn g_op_table[GPUOS_MAX_OPS];
__device__ __managed__ int  g_op_alias[GPUOS_MAX_OPS];
__device__ __managed__ unsigned long long g_processed_count = 0ULL;
__device__ __managed__ unsigned long long g_heartbeat = 0ULL;
__device__ __managed__ int g_debug_level = 0;
__device__ __managed__ unsigned long long g_yield_every = 0ULL; // 0 means never yield

static constexpr int kBatchSlot = 10;

// Helpers for generic indexing (minimal, f32 only for builtin)
static __device__ inline int64_t linear_to_offset(const TensorRef& tr, int64_t idx) {
  // Convert linear idx into element offset using tr.sizes/tr.strides over out ndim
  int64_t off = 0;
  int nd = tr.ndim;
  for (int d = nd - 1; d >= 0; --d) {
    int64_t dim = tr.sizes[d] > 0 ? tr.sizes[d] : 1;
    int64_t i = idx % dim;
    idx /= dim;
    off += i * tr.strides[d];
  }
  return off;
}

// Built-in default operator: C = A + B (float32 generic, supports broadcast/strides)
extern "C" __device__ void op_add(const Task& t) {
  int64_t N = t.numel;
  for (int64_t li = threadIdx.x; li < N; li += blockDim.x) {
    int64_t oa = linear_to_offset(t.in0, li);
    int64_t ob = linear_to_offset(t.in1, li);
    int64_t oc = linear_to_offset(t.out0, li);
    const float* a = reinterpret_cast<const float*>(static_cast<const char*>(t.in0.data) + oa * sizeof(float));
    const float* b = reinterpret_cast<const float*>(static_cast<const char*>(t.in1.data) + ob * sizeof(float));
    float* c = reinterpret_cast<float*>(static_cast<char*>(t.out0.data) + oc * sizeof(float));
    *c = (*a) + (*b);
  }
}

extern "C" __device__ void op_mul(const Task& t) {
  int64_t N = t.numel;
  for (int64_t li = threadIdx.x; li < N; li += blockDim.x) {
    int64_t oa = linear_to_offset(t.in0, li);
    int64_t ob = linear_to_offset(t.in1, li);
    int64_t oc = linear_to_offset(t.out0, li);
    const float* a = reinterpret_cast<const float*>(static_cast<const char*>(t.in0.data) + oa * sizeof(float));
    const float* b = reinterpret_cast<const float*>(static_cast<const char*>(t.in1.data) + ob * sizeof(float));
    float* c = reinterpret_cast<float*>(static_cast<char*>(t.out0.data) + oc * sizeof(float));
    *c = (*a) * (*b);
  }
}

static __device__ inline float ld_as_float(const TensorRef& tr, int64_t off_elems) {
  const char* base = static_cast<const char*>(tr.data);
  switch (tr.dtype) {
    case kF32: return reinterpret_cast<const float*>(base)[off_elems];
    case kF16: return __half2float(reinterpret_cast<const __half*>(base)[off_elems]);
    case kBF16: return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(base)[off_elems]);
    default: return reinterpret_cast<const float*>(base)[off_elems];
  }
}

static __device__ inline void st_from_float(const TensorRef& tr, int64_t off_elems, float v) {
  char* base = static_cast<char*>(tr.data);
  switch (tr.dtype) {
    case kF32: reinterpret_cast<float*>(base)[off_elems] = v; break;
    case kF16: reinterpret_cast<__half*>(base)[off_elems] = __float2half_rn(v); break;
    case kBF16: reinterpret_cast<__nv_bfloat16*>(base)[off_elems] = __float2bfloat16(v); break;
    default: reinterpret_cast<float*>(base)[off_elems] = v; break;
  }
}

// Aggregated batch operator: executes a sequence of micro-ops in one kernel launch
extern "C" __device__ void op_batch(const Task& t) {
  const Task* req = reinterpret_cast<const Task*>(t.in0.data);
  if (req == nullptr) {
    return;
  }
  int m = static_cast<int>(t.numel);
  for (int k = 0; k < m; ++k) {
    const Task& u = req[k];
    if (g_debug_level > 1 && threadIdx.x == 0 && blockIdx.x == 0 && k < 4) {
      printf("[batch] sub k=%d op=%d numel=%lld in0=%p in1=%p out0=%p\n",
             k, u.op, (long long)u.numel, u.in0.data, u.in1.data, u.out0.data);
    }
    int64_t N = u.numel;
    for (int64_t li = threadIdx.x; li < N; li += blockDim.x) {
      int64_t oa = linear_to_offset(u.in0, li);
      int64_t ob = linear_to_offset(u.in1, li);
      int64_t oc = linear_to_offset(u.out0, li);
      float A = ld_as_float(u.in0, oa);
      float B = (u.in1.data) ? ld_as_float(u.in1, ob) : 0.0f;
      float R = A;
      switch (u.op) {
        case 0: R = A + B; break; // add
        case 1: R = A * B; break; // mul
        case 2: R = A - B; break; // sub
        case 3: R = (B != 0.f) ? (A / B) : 0.f; break; // div
        default: R = A; break;
      }
      st_from_float(u.out0, oc, R);
    }
    __syncthreads();
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
  if (idx == 1) {
    g_op_table[1] = op_mul;
  }
  if (idx == kBatchSlot) {
    g_op_table[kBatchSlot] = op_batch;
  }
}

// Persistent worker kernel: each thread acts as a consumer
extern "C" __global__ void persistent_worker(WorkQueue q) {
  if (q.capacity == 0) return;
  __shared__ Task s_task;
  __shared__ int s_has_work;
  while (atomicAdd(q.quit, 0) == 0) {
    if (threadIdx.x == 0) {
      s_has_work = 0;
      // Robust claim: CAS increments head only if work is available
      for (int attempt = 0; attempt < 1024; ++attempt) {
        int h = atomicAdd(q.head, 0);
        int t = atomicAdd(q.tail, 0);
        if (h >= t) break; // no work
        if (atomicCAS(q.head, h, h + 1) == h) {
          s_task = q.tasks[h % q.capacity];
          s_has_work = 1;
          if (g_debug_level > 0) {
            printf("[worker] picked idx=%d tail=%d op=%d numel=%lld in0=%p in1=%p out0=%p\n",
                   h, t, s_task.op, (long long)s_task.numel,
                   s_task.in0.data, s_task.in1.data, s_task.out0.data);
          }
          break;
        }
      }
    }
    __syncthreads();
    if (!s_has_work) {
      if (threadIdx.x == 0 && blockIdx.x == 0) {
        atomicAdd(&g_heartbeat, 1ULL);
      }
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
        if (g_debug_level > 1) {
          printf("[worker] completed op=%d, processed=%llu\n", s_task.op, (unsigned long long)g_processed_count);
        }
        // Optional yield policy: after every g_yield_every tasks, request global exit
        unsigned long long ye = atomicAdd(&g_yield_every, 0ULL);
        if (ye > 0ULL) {
          unsigned long long pc = atomicAdd(&g_processed_count, 0ULL);
          if ((pc % ye) == 0ULL) {
            // signal all blocks to exit by setting quit flag
            atomicExch(q.quit, 1);
          }
        }
      }
      __syncthreads();
      if (atomicAdd(q.quit, 0) != 0) {
        return; // exit this block immediately if quit was requested
      }
    }
  }
}

// Host-callable launchers to avoid <<< >>> in non-CUDA translation units
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t stream) {
  dim3 blk(128);
  dim3 grd((GPUOS_MAX_OPS + blk.x - 1) / blk.x);
  init_builtin_ops<<<grd, blk, 0, stream>>>();
  return cudaGetLastError();
}

extern "C" cudaError_t launch_persistent_worker(WorkQueue q, int blocks, int threads, cudaStream_t stream) {
  persistent_worker<<<blocks, threads, 0, stream>>>(q);
  return cudaGetLastError();
}

// Device-symbol helpers callable from host C++ TUs
extern "C" cudaError_t gpu_get_processed_count_async(unsigned long long* out, cudaStream_t s) {
  return cudaMemcpyFromSymbolAsync(out, g_processed_count, sizeof(*out), 0, cudaMemcpyDeviceToHost, s);
}

extern "C" cudaError_t gpu_set_op_table_async(int index, OpPtrInt fn, cudaStream_t s) {
  return cudaMemcpyToSymbolAsync(g_op_table, &fn, sizeof(fn), index * sizeof(OpFn), cudaMemcpyHostToDevice, s);
}

extern "C" cudaError_t gpu_set_alias_async(int logical_id, int physical_slot, cudaStream_t s) {
  return cudaMemcpyToSymbolAsync(g_op_alias, &physical_slot, sizeof(physical_slot), logical_id * sizeof(int), cudaMemcpyHostToDevice, s);
}

extern "C" cudaError_t gpu_get_heartbeat_async(unsigned long long* out, cudaStream_t s) {
  return cudaMemcpyFromSymbolAsync(out, g_heartbeat, sizeof(*out), 0, cudaMemcpyDeviceToHost, s);
}

extern "C" cudaError_t gpu_set_debug_level_async(int level, cudaStream_t s) {
  return cudaMemcpyToSymbolAsync(g_debug_level, &level, sizeof(level), 0, cudaMemcpyHostToDevice, s);
}

extern "C" cudaError_t gpu_set_yield_every_async(unsigned long long every, cudaStream_t s) {
  return cudaMemcpyToSymbolAsync(g_yield_every, &every, sizeof(every), 0, cudaMemcpyHostToDevice, s);
}
