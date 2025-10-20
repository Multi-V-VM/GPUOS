# Persistent Kernel + JIT-Injected Operators (CUDA)

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

## Notes
- The queue is implemented with Unified Memory for simplicity. For production, prefer explicit device memory plus lightweight doorbells (atomics in mapped pinned memory) to avoid UM migration overhead.
- `g_op_table` is declared `__managed__` to simplify host updates (we use `cudaMemcpyToSymbol` with an offset). Workers call `__threadfence()` before reading the table.
- The JIT module exports a bridge `__device__ void* op_mul_ptr = (void*)op_mul;` so the host can fetch the function pointer value via `cuModuleGetGlobal` + `cuMemcpyDtoH` and store it in `g_op_table[op_id]`.
- The sample keeps the CUmodule alive. In a real system, track modules per operator to unload/replace safely when no tasks are executing that operator.
- If you need to support multiple operator signatures, create multiple jump tables or a thin bytecode interpreter.

## Troubleshooting
- RDC: Both host build and NVRTC must enable relocatable device code (`-rdc=true` / `--relocatable-device-code=true`).
- Arch: Default NVRTC target is `--gpu-architecture=compute_90`. Override via env `GPUOS_NVRTC_ARCH` if needed.
- ABI: Keep `extern "C" __device__` and identical `Task` layout on host and in JIT sources.
- Pointer bridge: Always expose `__device__ void* op_x_ptr = (void*)op_x;` in JIT modules to fetch the function pointer value.
- Prefer PTX loading with the Driver API; `cudaLibraryLoadData` is also viable on CUDA 12+ if you use the runtime loader variants.
