#include "common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda/atomic>
#include <stdio.h>
#ifdef GPUOS_ENABLE_TMA
#include "tma_copy.cuh"
#endif

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

#ifdef GPUOS_ENABLE_TMA
// Slot 11 is opt-in: static shared scratch affects the whole resident kernel.
// The contract is float32 add. Irregular layouts use the existing add handler.
// One out-of-line accessor gives initialization, dispatch and shutdown the
// same CTA-local allocation even though operators keep the original Task ABI.
__device__ __noinline__ gpuos::TmaState<1024,2>& worker_tma_state() {
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ gpuos::TmaState<1024,2> state;
  return state;
}
struct TmaAddConsume {
  float* out;
  __device__ void operator()(const float* a, const float* b, int tile) const {
    for (int i = threadIdx.x; i < 1024; i += blockDim.x)
      out[tile * 1024 + i] = a[i] + b[i];
  }
};
extern "C" __device__ void op_tma_add(const Task& t) {
  bool eligible = t.numel >= 1024 && t.numel <= 0x7fffffff &&
    blockDim.x >= 32 && blockDim.x % 32 == 0;
  for (const TensorRef* r : {&t.in0, &t.in1, &t.out0})
    eligible = eligible && r->dtype == kF32 && r->ndim == 1 &&
      r->strides[0] == 1 && r->sizes[0] >= t.numel &&
      (reinterpret_cast<uintptr_t>(r->data) % 16 == 0);
  if (!eligible) { op_add(t); return; }
  const float* a = static_cast<const float*>(t.in0.data);
  const float* b = static_cast<const float*>(t.in1.data);
  float* out = static_cast<float*>(t.out0.data);
  int tiles = static_cast<int>(t.numel / 1024);
  gpuos::tma_pipeline<1024,2>(worker_tma_state(), tiles,
    gpuos::BulkPair<1024>{a,b}, TmaAddConsume{out});
  for (int64_t i = int64_t(tiles)*1024 + threadIdx.x; i < t.numel; i += blockDim.x)
    out[i] = a[i] + b[i];
}
#endif

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
#ifdef GPUOS_ENABLE_TMA
  if (idx == 11) g_op_table[11] = op_tma_add;
#endif
}

// Persistent worker kernel: each BLOCK acts as a consumer.
// Uses host-mapped memory (zero-copy) for reliable completion detection.
// Protocol:
//   1. Execute operator function
//   2. __threadfence_system() - ensure all writes are visible to CPU
//   3. atomicAdd_system() on sync->processed - CPU can poll this directly
extern "C" __global__ void persistent_worker(WorkQueue q) {
  if (q.capacity == 0) return;
#ifdef GPUOS_ENABLE_TMA
  gpuos::tma_init(worker_tma_state());
#endif
  __shared__ Task s_task;
  __shared__ int s_has_work;
  __shared__ int s_quit;
  __shared__ OpFn s_fn;

  // Signal kernel is ready (only block 0, thread 0)
  if (threadIdx.x == 0 && blockIdx.x == 0 && q.sync != nullptr) {
    __threadfence_system();
    atomicExch_system((int*)&q.sync->ready, 1);
  }
  __syncthreads();

  while (true) {
    // A per-thread quit test can split a block around __syncthreads().
    if (threadIdx.x == 0) s_quit = atomicAdd(q.quit, 0);
    __syncthreads();
    if (s_quit) {
#ifdef GPUOS_ENABLE_TMA
      gpuos::tma_destroy(worker_tma_state());
#endif
      return;
    }
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
        // Update heartbeat in both legacy counter and host-mapped sync state
        atomicAdd(&g_heartbeat, 1ULL);
        if (q.sync != nullptr) {
          // System-scope atomic ensures CPU sees update immediately
          atomicAdd_system((unsigned long long*)&q.sync->heartbeat, 1ULL);
        }
      }
      __nanosleep(1000);
      continue;
    }

    // Resolve once per block. A live alias change must not send different
    // lanes of one operator invocation to different functions/barriers.
    if (threadIdx.x == 0) {
      s_fn = nullptr;
      if (s_task.op >= 0 && s_task.op < GPUOS_MAX_OPS) {
        int phys = cuda::atomic_ref<int, cuda::thread_scope_device>(g_op_alias[s_task.op])
                     .load(cuda::memory_order_acquire);
        if (phys >= 0 && phys < GPUOS_MAX_OPS) {
          s_fn = cuda::atomic_ref<OpFn, cuda::thread_scope_device>(g_op_table[phys])
                     .load(cuda::memory_order_acquire);
        }
      }
    }
    __syncthreads();
    if (s_fn) {
      s_fn(s_task);
      // A fence only orders the calling thread's writes. Every output-writing
      // thread must fence before the leader signals system-visible completion.
      __threadfence_system();
      __syncthreads();
      if (threadIdx.x == 0) {
        // Update legacy counter (device-managed)
        atomicAdd(&g_processed_count, 1ULL);

        // Update host-mapped counter with system-scope atomic
        // The CPU can poll this directly without memory copies
        if (q.sync != nullptr) {
          atomicAdd_system((unsigned long long*)&q.sync->processed, 1ULL);
        }

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
    }
  }
}

// Host-callable launchers to avoid <<< >>> in non-CUDA translation units
__global__ void publish_op(int index, OpPtrInt fn);
__global__ void publish_alias(int logical, int physical);
extern "C" cudaError_t launch_init_builtin_ops(cudaStream_t stream) {
  // Resolve control kernels before a resident worker starts: lazy module
  // loading on first use may otherwise synchronize a never-ending context.
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, publish_op);
  if (status != cudaSuccess) return status;
  status = cudaFuncGetAttributes(&attributes, publish_alias);
  if (status != cudaSuccess) return status;
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

// Publication uses atomic stores, paired with the worker's acquire loads.
// These control kernels need spare residency; module reclamation still requires
// draining all tasks that can hold an old pointer.
__global__ void publish_op(int index, OpPtrInt fn) {
  cuda::atomic_ref<OpFn, cuda::thread_scope_device>(g_op_table[index])
      .store(reinterpret_cast<OpFn>(fn), cuda::memory_order_release);
}
__global__ void publish_alias(int logical, int physical) {
  cuda::atomic_ref<int, cuda::thread_scope_device>(g_op_alias[logical])
      .store(physical, cuda::memory_order_release);
}
extern "C" cudaError_t gpu_set_op_table_async(int index, OpPtrInt fn, cudaStream_t s) {
  if (index < 0 || index >= GPUOS_MAX_OPS) return cudaErrorInvalidValue;
  publish_op<<<1, 1, 0, s>>>(index, fn);
  return cudaGetLastError();
}

extern "C" cudaError_t gpu_set_alias_async(int logical_id, int physical_slot, cudaStream_t s) {
  if (logical_id < 0 || logical_id >= GPUOS_MAX_OPS ||
      physical_slot < 0 || physical_slot >= GPUOS_MAX_OPS) return cudaErrorInvalidValue;
  publish_alias<<<1, 1, 0, s>>>(logical_id, physical_slot);
  return cudaGetLastError();
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
