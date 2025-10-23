"""
Micro-batch elementwise operation benchmarks
Compares GPUOS vs PyTorch baseline vs MPS vs MIG
"""
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_utils import (
    BenchmarkTimer, BenchmarkResult, ResultsReporter,
    verify_correctness, get_gpu_info, MPSController
)

try:
    # Build extension on the fly if needed
    import torch.utils.cpp_extension
    import pytorch_ext.gpuos_ext as gpuos_ext
except ImportError:
    print("Building GPUOS extension...")
    import subprocess
    subprocess.run(['python', 'examples/pytorch_batch_demo.py'], check=True)
    import pytorch_ext.gpuos_ext as gpuos_ext


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

    def workload():
        for _ in range(num_ops):
            gpuos_ext.submit_add(a, b, out)
        gpuos_ext.flush(sync=True)

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
