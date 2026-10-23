#pragma once
// CTA-local TMA input staging. This does not implement cluster communication.
#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>

namespace gpuos {
using TmaBarrier = cuda::barrier<cuda::thread_scope_block>;

// Lifetime is one worker, not one descriptor. The phase of each stage survives
// task boundaries, including tasks that consume an odd or zero number of tiles.
template<int Tile, int Stages> struct TmaState {
  alignas(128) float buffers[Stages][2][Tile];
  TmaBarrier ready[Stages];
  bool parity[Stages];
};
template<int Tile, int Stages>
__device__ void tma_init(TmaState<Tile,Stages>& state) {
  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) {
      init(&state.ready[s], 1);
      state.parity[s] = false;
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  }
  __syncthreads();
}
template<int Tile, int Stages>
__device__ void tma_destroy(TmaState<Tile,Stages>& state) {
  __syncthreads();
  if (threadIdx.x == 0)
    for (int s = 0; s < Stages; ++s) state.ready[s].~TmaBarrier();
  __syncthreads();
}

// All block threads participate in whole warps. An elected leader issues transfers.
// Each stage has one arrival and two tiles' worth of transaction bytes per phase.
// Consume must finish all reads before returning to the block-wide reuse fence.
template<int Tile, int Stages, class Issue, class Consume>
__device__ void tma_pipeline(TmaState<Tile,Stages>& state, int tiles,
                            Issue issue, Consume consume) {
  static_assert(Tile % 32 == 0 && (Stages == 1 || Stages == 2));
  auto& buffers = state.buffers;
  auto& ready = state.ready;
  bool parity[Stages];
  for (int s = 0; s < Stages; ++s) parity[s] = state.parity[s];
  const bool issuer = threadIdx.x < 32 && cuda::ptx::elect_sync(0xffffffff);
  if (issuer) {
    for (int s = 0; s < Stages && s < tiles; ++s) {
      issue(buffers[s][0], buffers[s][1], s, ready[s]);
      (void)cuda::device::barrier_arrive_tx(ready[s], 1, 2 * Tile * sizeof(float));
    }
  }
  for (int t = 0; t < tiles; ++t) {
    int s = t % Stages;
    ready[s].wait_parity(parity[s]);
    parity[s] = !parity[s];
    consume(buffers[s][0], buffers[s][1], t);
    // No reader may remain when the producer reuses this stage.
    __syncthreads();
    if (issuer && t + Stages < tiles) {
      issue(buffers[s][0], buffers[s][1], t + Stages, ready[s]);
      (void)cuda::device::barrier_arrive_tx(ready[s], 1, 2 * Tile * sizeof(float));
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int s = 0; s < Stages; ++s) state.parity[s] = parity[s];
  }
  __syncthreads();
}

template<int Tile>
struct BulkPair {
  const float *a, *b;
  __device__ void operator()(float* sa, float* sb, int t, TmaBarrier& bar) const {
    // Explicit CTA destination matches our local ownership contract. The
    // higher-level memcpy_async_tx wrapper uses shared::cluster instead.
    auto native = cuda::device::barrier_native_handle(bar);
    const uint32_t bytes = Tile * sizeof(float);
    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                           sa, a + t * Tile, bytes, native);
    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared, cuda::ptx::space_global,
                           sb, b + t * Tile, bytes, native);
  }
};

// Tensor maps describe two row-major 2D arrays of width 256 floats. Map box
// dimensions must be {256, Tile/256}, with no swizzle or interleave.
template<int Tile>
struct TensorPair {
  const CUtensorMap *a, *b;
  int row;
  __device__ void operator()(float* sa, float* sb, int t, TmaBarrier& bar) const {
    const int32_t xy[2] = {0, row + t * (Tile / 256)};
    auto native = cuda::device::barrier_native_handle(bar);
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared, cuda::ptx::space_global, sa, a, xy, native);
    cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared, cuda::ptx::space_global, sb, b, xy, native);
  }
};
} // namespace gpuos
