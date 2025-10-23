"""
Mixed operation workload benchmarks
Tests realistic inference patterns with multiple operation types
"""
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_utils import (
    BenchmarkTimer, BenchmarkResult, ResultsReporter,
    verify_correctness, get_gpu_info
)

try:
    from pytorch_ext.scheduler import scheduler_context
    import pytorch_ext.gpuos_ext as gpuos_ext
    GPUOS_AVAILABLE = True
except ImportError:
    GPUOS_AVAILABLE = False
    print("WARNING: GPUOS not available, will only run PyTorch benchmarks")


def benchmark_pytorch_mixed(num_iterations: int, tensor_size: int) -> tuple:
    """PyTorch baseline: mixed operations"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    c = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    d = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    def workload():
        out = None
        for _ in range(num_iterations):
            x = a + b          # add
            y = x * c          # mul
            z = torch.relu(y)  # relu
            w = torch.sigmoid(z)  # sigmoid
            out = w / d        # div
        return out

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    return mean_ms, std_ms, min_ms, max_ms, workload()


def benchmark_pytorch_compiled_mixed(num_iterations: int, tensor_size: int) -> tuple:
    """PyTorch with torch.compile"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    c = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    d = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    @torch.compile
    def workload_compiled():
        out = None
        for _ in range(num_iterations):
            x = a + b
            y = x * c
            z = torch.relu(y)
            w = torch.sigmoid(z)
            out = w / d
        return out

    # Warmup for compilation
    _ = workload_compiled()
    torch.cuda.synchronize()

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload_compiled)

    return mean_ms, std_ms, min_ms, max_ms, workload_compiled()


def benchmark_gpuos_scheduler_mixed(num_iterations: int, tensor_size: int) -> tuple:
    """GPUOS with transparent scheduler"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    c = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    d = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    def workload():
        with torch.no_grad():
            with scheduler_context(capacity=8192, threads_per_block=256,
                                 size_threshold=tensor_size * 2, auto_flush_ms=None):
                out = None
                for _ in range(num_iterations):
                    x = a + b
                    y = x * c
                    z = torch.relu(y)
                    w = torch.sigmoid(z)
                    out = w / d
                # Context exit flushes automatically
        return out

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    return mean_ms, std_ms, min_ms, max_ms, workload()


def benchmark_gpuos_manual_mixed(num_iterations: int, tensor_size: int) -> tuple:
    """GPUOS with manual batching"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    c = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    d = torch.randn(tensor_size, device='cuda', dtype=torch.float32)

    # Pre-allocate outputs
    x = torch.empty_like(a)
    y = torch.empty_like(a)
    z = torch.empty_like(a)
    w = torch.empty_like(a)
    out = torch.empty_like(a)

    gpuos_ext.init(capacity=8192, threads_per_block=256)

    # Register operations
    add_slot = gpuos_ext.register_elementwise("add_manual", "(A + B)", 2)
    mul_slot = gpuos_ext.register_elementwise("mul_manual", "(A * B)", 2)
    relu_slot = gpuos_ext.register_elementwise("relu_manual", "(A > 0.f ? A : 0.f)", 1)
    sigmoid_slot = gpuos_ext.register_elementwise("sigmoid_manual", "1.f / (1.f + expf(-A))", 1)
    div_slot = gpuos_ext.register_elementwise("div_manual", "(A / B)", 2)

    def workload():
        for _ in range(num_iterations):
            gpuos_ext.submit_binary(add_slot, a, b, x)
            gpuos_ext.submit_binary(mul_slot, x, c, y)
            gpuos_ext.submit_unary(relu_slot, y, z)
            gpuos_ext.submit_unary(sigmoid_slot, z, w)
            gpuos_ext.submit_binary(div_slot, w, d, out)
        gpuos_ext.flush(sync=True)

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    result_out = out.clone()
    gpuos_ext.shutdown()

    return mean_ms, std_ms, min_ms, max_ms, result_out


def run_mixed_ops_benchmark():
    """Run comprehensive mixed operations benchmark"""
    print("="*80)
    print("MIXED OPERATIONS WORKLOAD BENCHMARK")
    print("="*80)

    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('gpu_name', 'Unknown')}")
    print(f"PyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")
    print()

    reporter = ResultsReporter()

    # Test configurations
    tensor_sizes = [1024, 2048, 4096, 8192]
    num_iterations = 1000

    for tensor_size in tensor_sizes:
        print(f"\n--- Tensor Size: {tensor_size} elements ---")
        print(f"Pattern: {num_iterations} iterations of (add → mul → relu → sigmoid → div)")

        # PyTorch baseline
        print("  Running PyTorch baseline...")
        mean_ms, std_ms, min_ms, max_ms, reference = benchmark_pytorch_mixed(num_iterations, tensor_size)
        result = BenchmarkResult(
            name=f"mixed_ops_{tensor_size}",
            config="PyTorch Baseline",
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms
        )
        reporter.add_result(result)
        print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

        # PyTorch compiled
        if hasattr(torch, 'compile'):
            print("  Running PyTorch compiled...")
            try:
                mean_ms, std_ms, min_ms, max_ms, output = benchmark_pytorch_compiled_mixed(num_iterations, tensor_size)
                result = BenchmarkResult(
                    name=f"mixed_ops_{tensor_size}",
                    config="PyTorch Compiled",
                    mean_time_ms=mean_ms,
                    std_time_ms=std_ms,
                    min_time_ms=min_ms,
                    max_time_ms=max_ms
                )
                reporter.add_result(result)
                print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

                baseline_time = next(r for r in reporter.results
                                    if r.name == f"mixed_ops_{tensor_size}"
                                    and r.config == "PyTorch Baseline").mean_time_ms
                speedup = baseline_time / mean_ms
                print(f"    Speedup: {speedup:.2f}×")
            except Exception as e:
                print(f"    ERROR: {e}")

        # GPUOS scheduler
        if GPUOS_AVAILABLE:
            print("  Running GPUOS (scheduler)...")
            try:
                mean_ms, std_ms, min_ms, max_ms, output = benchmark_gpuos_scheduler_mixed(num_iterations, tensor_size)

                # Verify correctness
                if not verify_correctness(reference, output, rtol=1e-3):
                    print("    WARNING: Correctness check failed!")

                result = BenchmarkResult(
                    name=f"mixed_ops_{tensor_size}",
                    config="GPUOS Scheduler",
                    mean_time_ms=mean_ms,
                    std_time_ms=std_ms,
                    min_time_ms=min_ms,
                    max_time_ms=max_ms
                )
                reporter.add_result(result)
                print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

                baseline_time = next(r for r in reporter.results
                                    if r.name == f"mixed_ops_{tensor_size}"
                                    and r.config == "PyTorch Baseline").mean_time_ms
                speedup = baseline_time / mean_ms
                print(f"    Speedup: {speedup:.2f}×")
            except Exception as e:
                print(f"    ERROR: {e}")

            # GPUOS manual
            print("  Running GPUOS (manual)...")
            try:
                mean_ms, std_ms, min_ms, max_ms, output = benchmark_gpuos_manual_mixed(num_iterations, tensor_size)

                # Verify correctness
                if not verify_correctness(reference, output, rtol=1e-3):
                    print("    WARNING: Correctness check failed!")

                result = BenchmarkResult(
                    name=f"mixed_ops_{tensor_size}",
                    config="GPUOS Manual",
                    mean_time_ms=mean_ms,
                    std_time_ms=std_ms,
                    min_time_ms=min_ms,
                    max_time_ms=max_ms
                )
                reporter.add_result(result)
                print(f"    Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

                baseline_time = next(r for r in reporter.results
                                    if r.name == f"mixed_ops_{tensor_size}"
                                    and r.config == "PyTorch Baseline").mean_time_ms
                speedup = baseline_time / mean_ms
                print(f"    Speedup: {speedup:.2f}×")
            except Exception as e:
                print(f"    ERROR: {e}")

    # Save results
    reporter.save_json("mixed_ops_results.json")
    reporter.save_csv("mixed_ops_results.csv")
    reporter.print_summary()


if __name__ == "__main__":
    run_mixed_ops_benchmark()
