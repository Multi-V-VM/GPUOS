"""
Micro-batch elementwise operation benchmarks
Compares GPUOS vs PyTorch baseline vs MPS vs MIG
"""
import torch
import os
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_utils import (
    BenchmarkTimer, BenchmarkResult, ResultsReporter,
    verify_correctness, get_gpu_info, MPSController
)

def load_gpuos_ext():
    try:
        import pytorch_ext.gpuos_ext as mod
        if hasattr(mod, 'abi_version') and mod.abi_version() >= 1:
            return mod
        else:
            print("GPUOS extension present but outdated; rebuilding...")
    except Exception:
        print("GPUOS extension not found; building...")

    # Build fresh extension in-place
    from torch.utils.cpp_extension import load
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    mod = load(
        name='gpuos_ext',
        sources=[
            os.path.join(root, 'pytorch_ext', 'gpuos_ext.cpp'),
            os.path.join(root, 'src', 'persistent_kernel.cu'),
        ],
        extra_cflags=['-O3', '-std=c++17'],
        extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                           '-gencode=arch=compute_121,code=sm_121'],
        extra_ldflags=['-lcuda', '-lnvrtc', '-lcudart'],
        with_cuda=True,
        verbose=False,
    )
    return mod

gpuos_ext = load_gpuos_ext()


def benchmark_pytorch_baseline(num_ops: int, tensor_size: int) -> tuple:
    """Baseline PyTorch: separate kernel launches"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

    def workload():
        for _ in range(num_ops):
            torch.add(a, b, out=out)

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    return mean_ms, std_ms, min_ms, max_ms, out.clone()


def benchmark_gpuos(num_ops: int, tensor_size: int) -> tuple:
    """GPUOS: persistent kernel with batching"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

    # Initialize GPUOS
    gpuos_ext.init(capacity=8192, threads_per_block=256)
    # Optional: configure device yield policy via env
    ye = os.getenv('GPUOS_YIELD_EVERY')
    if ye is not None:
        try:
            ye_val = int(ye)
            if hasattr(gpuos_ext, 'set_yield_every'):
                gpuos_ext.set_yield_every(ye_val)
        except Exception:
            pass

    debug_flush = os.getenv('GPUOS_DEBUG_FLUSH') not in (None, '0', 'false', 'False')

    def workload():
        for _ in range(num_ops):
            gpuos_ext.submit_add(a, b, out)
        if debug_flush:
            import time
            t0 = time.time()
            gpuos_ext.flush(sync=True)
            dt = time.time() - t0
            alive = getattr(gpuos_ext, 'worker_alive', lambda: None)()
            peek = getattr(gpuos_ext, 'peek_queue', lambda: {})()
            print(f"  flush returned in {dt:.3f}s; worker_alive={alive}; peek={peek}")
            return dt, dt, dt, dt, dt
        else:
            gpuos_ext.flush(sync=True)
        # Optional peek to observe queue state per iteration
        import os
        try:
            peek_every = int(os.getenv('GPUOS_PEEK_EVERY', '0'))
        except Exception:
            peek_every = 0
        if peek_every:
            # Use a counter on the function to print every N calls
            c = getattr(workload, '_cnt', 0) + 1
            setattr(workload, '_cnt', c)
            if c % peek_every == 0:
                try:
                    print('  peek:', gpuos_ext.peek_queue())
                except Exception as _:
                    pass

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    result_out = out.clone()
    gpuos_ext.shutdown()

    return mean_ms, std_ms, min_ms, max_ms, result_out


def benchmark_pytorch_compiled(num_ops: int, tensor_size: int) -> tuple:
    """PyTorch 2.x with torch.compile"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    @torch.compile
    def workload_compiled():
        out = a
        for _ in range(num_ops):
            out = out + b
        return out

    # Warmup for compilation
    _ = workload_compiled()
    torch.cuda.synchronize()

    def workload():
        return workload_compiled()

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    return mean_ms, std_ms, min_ms, max_ms, workload()


def run_microbatch_benchmark():
    """Run comprehensive micro-batch benchmark"""
    print("="*80)
    print("MICRO-BATCH ELEMENTWISE BENCHMARK")
    print("="*80)

    # Print GPU info
    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('gpu_name', 'Unknown')}")
    print(f"PyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")
    print()

    reporter = ResultsReporter()

    # Test configurations
    tensor_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]
    num_ops = 10000

    print(f"Running {num_ops} add operations for various tensor sizes...")
    print()

    for tensor_size in tensor_sizes:
        print(f"\n--- Tensor Size: {tensor_size} elements ({tensor_size * 4 / 1024:.1f} KB) ---")

        # PyTorch baseline
        print("  Running PyTorch baseline...")
        mean_ms, std_ms, min_ms, max_ms, reference = benchmark_pytorch_baseline(num_ops, tensor_size)
        result = BenchmarkResult(
            name=f"microbatch_add_{tensor_size}",
            config="PyTorch Baseline",
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms,
            throughput_ops_per_sec=num_ops / (mean_ms / 1000.0) if mean_ms > 0 else 0
        )
        reporter.add_result(result)
        print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

        # GPUOS
        print("  Running GPUOS...")
        try:
            mean_ms, std_ms, min_ms, max_ms, output = benchmark_gpuos(num_ops, tensor_size)

            # Verify correctness
            if not verify_correctness(reference, output):
                print("    WARNING: Correctness check failed!")

            result = BenchmarkResult(
                name=f"microbatch_add_{tensor_size}",
                config="GPUOS",
                mean_time_ms=mean_ms,
                std_time_ms=std_ms,
                min_time_ms=min_ms,
                max_time_ms=max_ms,
                throughput_ops_per_sec=num_ops / (mean_ms / 1000.0) if mean_ms > 0 else 0
            )
            reporter.add_result(result)
            print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

            # Calculate speedup
            baseline_time = next(r for r in reporter.results
                                if r.name == f"microbatch_add_{tensor_size}"
                                and r.config == "PyTorch Baseline").mean_time_ms
            speedup = baseline_time / mean_ms
            print(f"    Speedup: {speedup:.2f}×")

        except Exception as e:
            print(f"    ERROR: {e}")

        # PyTorch compiled (if available)
        if hasattr(torch, 'compile'):
            print("  Running PyTorch compiled...")
            try:
                mean_ms, std_ms, min_ms, max_ms, output = benchmark_pytorch_compiled(num_ops, tensor_size)

                result = BenchmarkResult(
                    name=f"microbatch_add_{tensor_size}",
                    config="PyTorch Compiled",
                    mean_time_ms=mean_ms,
                    std_time_ms=std_ms,
                    min_time_ms=min_ms,
                    max_time_ms=max_ms,
                    throughput_ops_per_sec=num_ops / (mean_ms / 1000.0) if mean_ms > 0 else 0
                )
                reporter.add_result(result)
                print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Save results
    reporter.save_json("microbatch_results.json")
    reporter.save_csv("microbatch_results.csv")
    reporter.print_summary()


if __name__ == "__main__":
    run_microbatch_benchmark()
