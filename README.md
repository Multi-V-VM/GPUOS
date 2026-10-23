# FuseOS: Fused Persistent Execution for Dynamic Inference

The [IPDPS manuscript](69e5c13c57662528991eb76c/FuseOS-IPDPS-2027.pdf) presents a system for **dynamism**: GPU request retirement, ready-tile assignment, operator fusion and storage lifetime reuse. The [current Qwen3-0.6B executor](experiments/fusion_persistent/README.md) runs complete decode in one bounded persistent launch after native prefill, without CUDA Graphs on the FuseOS path.

The new suite retains **1,380 complete requests, 120 shared-reservation decode trials, and 300 real-MPS groups**. Strong controls include fused/separate tile-grid+PDL and CLC+PDL, fixed workspace planning, cyclic claims, and actual four-process MPS with shared weights. Selected fused kernels pass memcheck and synccheck. Native numerical compatibility still fails in four of six cases; [paper.md](paper.md) states the exact performance and numerical boundaries.

The design offers a software pool for compatible trusted requests. It does not supply MIG isolation, continuous admission or general device allocation. The full ResNet-18 benchmark is retained as a boundary case where partial residual persistence loses.

See [paper sources and build instructions](69e5c13c57662528991eb76c/README.md). The manuscript uses ten body pages plus one references page. All original figures remain unchanged in a [separate comparison](69e5c13c57662528991eb76c/FuseOS-original-figures.pdf). Earlier [persistent-decode](experiments/persistent_decode/README.md), [launch-control](experiments/launch_controls/README.md), [selected-operator](experiments/e2e/README.md) and [online](benchmark_results/revision_20261003_extended/README.md) suites remain historical evidence with distinct protocols.

The legacy descriptor/function-table prototype below remains separate from the specialized dynamic inference executor.

`GPUOS_ENABLE_TMA=ON` adds an experimental bulk-TMA float32 add handler at slot 11.
It is off by default because shared scratch affects residency. Worker-lifetime
barrier state resolves the observed reuse warnings; all four sanitizer tools
pass on the bounded benchmark and actual handler. The replicated evaluation
contains 162,000 checked timings, including stronger float4 and subdivided-grid
controls. See the [raw results](benchmark_results/revision_20261003_followup/README.md).
The bounded tensor-map benchmark is a separate path; no
cross-SM TMA communication is implemented. See the [experiment instructions](experiments/README.md).

This remains a research prototype. The worker now takes a block-uniform operator
and shutdown snapshot, but the extension's ring reuse, dependency handling,
asynchronous tensor lifetimes, and live JIT module loading still need validation.
Control publication requires spare GPU residency; old modules must stay loaded
until their callers finish. The new correctness test covers alias updates and
termination, not the full PyTorch integration.

This example demonstrates a practical pattern for running a persistent kernel on NVIDIA GPUs while hot-swapping device-side operators at runtime using NVRTC JIT and a device function-pointer jump table.

Highlights:
- Persistent kernel with a global work queue (single producer on host, many consumers on device).
- Device jump table `g_op_table[]` of `__device__` function pointers.
- Host compiles new operators at runtime via NVRTC, loads them with the CUDA Driver API, fetches a device function pointer from the JIT module, and patches the jump table.

## Layout
- `src/common.h` — shared types: `Task`, `WorkQueue`, `OpFn` and extern globals.
- `src/persistent_kernel.cu` — persistent worker + built-in `op_add` + processed counter; `g_op_table` is `__managed__`.
- `src/host.cpp` — host program: sets up queue, launches workers, NVRTC-compiles `op_mul`, updates jump table, verifies results.
- `CMakeLists.txt` — builds with `-rdc=true` and links against `cudart`, `cuda_driver`, `nvrtc`.

## Requirements
- CUDA Toolkit 12.x (or newer) with NVRTC.
- GPU with device function pointer support (sm_50+; recommend sm_70+).

## Build
```
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Run
```
./persistent_jit
```
You should see the host JIT-inject `op[1] = op_mul` and the persistent kernel will call through the updated function pointer for tasks with `op=1`. The program waits for completion, signals the kernel to stop, and verifies `C = A * B` on a few elements.

### Tests
- Online in-place switch (overwrite slot 0): `./test_online_switch`
- Dual-slot alias switch with rollback (logical 0 -> slot 0/1): `./test_dual_slot_switch`

## PyTorch: Continuous Small Requests + Runtime JIT Aggregation

A minimal PyTorch extension demonstrates micro-batching many tiny ops and executing them as a single aggregated operator compiled at runtime with NVRTC.

What it does:
- Exposes functions to submit tiny add/mul requests on CUDA tensors without launching individual kernels.
- Accumulates pending requests on the host and, on `flush()`, JIT-compiles a batch operator (`op_batch`) via NVRTC (once) and enqueues a single Task that processes the entire batch on the persistent kernel.
- The batch operator iterates sub-requests and uses block-local threading to process each, maximizing GPU utilization without per-request launch overhead.

Build/run (example):
- Using dynamic build in-place with PyTorch tools:
  - `python examples/pytorch_batch_demo.py`
- Or pre-build the extension:
  - `cd pytorch_ext && python setup.py build_ext --inplace`
  - Then import `gpuos_ext` in Python.

API (gpuos_ext):
- `init(capacity=4096, threads_per_block=256)` — allocates queue, launches persistent kernel, installs builtins.
- `submit_add(a, b, out)` / `submit_mul(a, b, out)` — enqueue micro-requests to host-side pending buffer (expects float32 CUDA tensors).
- `flush(sync=False)` — JIT-install `op_batch` (once) and publish a single aggregated Task pointing to the batch of requests. With `sync=True`, waits for completion.
- `shutdown()` — signals quit and joins the persistent kernel.

Notes:
- Set `GPUOS_NVRTC_ARCH` (e.g., `compute_90`) to override NVRTC arch if needed.
- For simplicity, async `flush(sync=False)` does not reclaim the per-batch descriptor buffer immediately; use `sync=True` or add a small GC loop in production.

## PyTorch Scheduler Plugin (Open-Box Usage)

A lightweight TorchDispatch-based scheduler that transparently aggregates tiny pointwise ops (add/mul) into the persistent-kernel runtime. Use it as a context manager:

```
from pytorch_ext.scheduler import scheduler_context
import torch

with torch.no_grad():
  with scheduler_context(capacity=8192, threads_per_block=256, size_threshold=1<<15, auto_flush_ms=2.0):
    y = a + b   # small ops are queued and batched
    z = a * b
    # leaving the context flushes and waits
```

Demo:
- `python examples/pytorch_scheduler_demo.py` (builds the extension on the fly and runs a quick correctness check).
- `python examples/pytorch_scheduler_advanced_demo.py` (broadcast, non-contiguous views, mixed dtypes like fp16+fp32).
- `python examples/pytorch_reduce_demo.py` (single-dimension sum/mean over last dim, keepdim/nostack; non-contiguous input).

Behavior and constraints:
- Intercepts many unary/binary elementwise ops (relu/sigmoid/tanh/exp/log/sqrt/abs/sin/cos/gelu/hardsigmoid/hardswish/maximum/minimum/pow/leaky_relu/hardtanh/elu/softplus/clamp variants) in addition to add/mul/div/sub.
- Supports broadcasting and non-contiguous input strides; outputs are contiguous by default. Mixed dtypes (fp16/bf16/fp32) compute in fp32 and cast to output dtype.
- Falls back to regular PyTorch for unsupported ops, large tensors (beyond `size_threshold`), or when autograd is enabled (best used under `torch.no_grad()`).
- Ensures correctness by auto-flushing synchronously if a downstream op consumes a tensor produced by the scheduler before the batch is flushed.
- Background timer (`auto_flush_ms`) opportunistically flushes pending work to keep latency bounded.

Reductions (beta):
- Scheduler intercepts `aten::sum.dim_IntList` and `aten::mean.dim` for a single dimension equal to the last axis. Generates and caches a dedicated JIT reduce kernel (sum/mean), supports keepdim and non-contiguous inputs. More general multi-d reductions can be added similarly.

## Notes
- The queue is implemented with Unified Memory for simplicity. For production, prefer explicit device memory plus lightweight doorbells (atomics in mapped pinned memory) to avoid UM migration overhead.
- `g_op_table` is declared `__managed__`. Control kernels publish table and alias entries with release stores; the worker leader uses acquire loads and broadcasts one pointer to its block. Those control kernels are resolved before worker startup and require spare residency.
- The JIT module exports a bridge `__device__ void* op_mul_ptr = (void*)op_mul;` so the host can fetch the function pointer value via `cuModuleGetGlobal` + `cuMemcpyDtoH` and store it in `g_op_table[op_id]`.
- The sample keeps the CUmodule alive. In a real system, track modules per operator to unload/replace safely when no tasks are executing that operator.
- If you need to support multiple operator signatures, create multiple jump tables or a thin bytecode interpreter.

## Troubleshooting
- RDC: Both host build and NVRTC must enable relocatable device code (`-rdc=true` / `--relocatable-device-code=true`).
- Arch: Default NVRTC target is `--gpu-architecture=compute_90`. Override via env `GPUOS_NVRTC_ARCH` if needed.
- ABI: Keep `extern "C" __device__` and identical `Task` layout on host and in JIT sources.
- Pointer bridge: Always expose `__device__ void* op_x_ptr = (void*)op_x;` in JIT modules to fetch the function pointer value.
- Prefer PTX loading with the Driver API; `cudaLibraryLoadData` is also viable on CUDA 12+ if you use the runtime loader variants.
