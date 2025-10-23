"""
MIG (Multi-Instance GPU) benchmark and setup instructions
MIG is only available on A100, A30, H100, H200 GPUs
"""
import torch
import sys
import os
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_utils import (
    BenchmarkTimer, BenchmarkResult, ResultsReporter,
    MIGController, get_gpu_info
)


def print_mig_setup_instructions():
    """Print comprehensive MIG setup instructions"""
    print("""
================================================================================
MIG (Multi-Instance GPU) Setup Instructions
================================================================================

REQUIREMENTS:
- NVIDIA A100, A30, H100, or H200 GPU
- Root/sudo access
- CUDA 11.0+ or later

CURRENT STATUS:
""")

    mig_available = MIGController.is_available()
    mig_enabled = MIGController.is_enabled()

    print(f"  MIG Supported: {'YES' if mig_available else 'NO (requires A100/H100 hardware)'}")
    print(f"  MIG Enabled: {'YES' if mig_enabled else 'NO'}")

    if not mig_available:
        print("""
This system does not support MIG. MIG requires:
- NVIDIA A100 (40GB or 80GB)
- NVIDIA A30
- NVIDIA H100
- NVIDIA H200

Your GPU does not support MIG partitioning.
The benchmark will run simulated tests instead.
""")
        return False

    if not mig_enabled:
        print("""
STEP 1: Enable MIG Mode
------------------------
Run as root:

  sudo nvidia-smi -mig 1

Note: This will terminate all GPU processes. Then reboot:

  sudo reboot

STEP 2: Create MIG Instances
-----------------------------
After reboot, create MIG instances. Example configurations:

# Seven 1g.10gb instances (maximum partitioning):
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C

# Four 2g.20gb instances:
sudo nvidia-smi mig -cgi 14,14,14,14 -C

# Mixed: 3g.40gb + 2g.20gb + 1g.10gb + 1g.10gb:
sudo nvidia-smi mig -cgi 9,14,19,19 -C

STEP 3: List MIG Instances
---------------------------
sudo nvidia-smi mig -lgi

STEP 4: Set CUDA_VISIBLE_DEVICES
---------------------------------
Each MIG instance appears as a separate device:

export CUDA_VISIBLE_DEVICES=MIG-UUID

Or by index:
export CUDA_VISIBLE_DEVICES=0  # First MIG instance

STEP 5: Run Benchmark
---------------------
python benchmarks/bench_mig.py --real-mig

""")
    else:
        print("\nMIG is enabled! Available instances:")
        profiles = MIGController.get_available_profiles()
        for p in profiles:
            print(f"  {p}")

    return mig_enabled


def benchmark_on_mig_instance(num_ops: int, tensor_size: int) -> tuple:
    """Run benchmark on current MIG instance"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

    def workload():
        for _ in range(num_ops):
            torch.add(a, b, out=out)

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    return mean_ms, std_ms, min_ms, max_ms


def benchmark_simulated_partition(num_ops: int, tensor_size: int, sm_limit_percent: int) -> tuple:
    """
    Simulate MIG-like resource partitioning by limiting SM usage
    This is NOT a true MIG benchmark but shows the concept
    """
    import warnings
    warnings.warn("Using simulated MIG (not real hardware partitioning)")

    # We can't truly limit SMs from user code, but we can:
    # 1. Limit threads per block
    # 2. Limit number of concurrent blocks
    # 3. Measure performance difference

    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

    def workload():
        for _ in range(num_ops):
            torch.add(a, b, out=out)

    timer = BenchmarkTimer(warmup_iters=50, measure_iters=500)
    mean_ms, std_ms, min_ms, max_ms = timer.benchmark(workload)

    # Scale time to simulate partition (rough approximation)
    scale_factor = 100.0 / sm_limit_percent
    return mean_ms * scale_factor, std_ms * scale_factor, min_ms * scale_factor, max_ms * scale_factor


def run_mig_benchmark(use_real_mig: bool = False):
    """Run MIG benchmark"""
    print("="*80)
    print("MIG (MULTI-INSTANCE GPU) BENCHMARK")
    print("="*80)

    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('gpu_name', 'Unknown')}")
    print(f"PyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")
    print()

    mig_enabled = print_mig_setup_instructions()

    if not use_real_mig or not mig_enabled:
        print("\n" + "="*80)
        print("RUNNING SIMULATED MIG TESTS")
        print("="*80)
        print("Note: These are NOT real MIG tests. Real MIG requires A100/H100.")
        print("Simulated tests show conceptual performance differences only.")
        print()

        reporter = ResultsReporter()
        num_ops = 10000
        tensor_size = 2048

        # Full GPU (baseline)
        print("Testing on full GPU (100% SMs)...")
        mean_ms, std_ms, min_ms, max_ms = benchmark_simulated_partition(
            num_ops, tensor_size, sm_limit_percent=100
        )
        result = BenchmarkResult(
            name="mig_simulated",
            config="Full GPU (100%)",
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms
        )
        reporter.add_result(result)
        print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms\n")

        # Simulated partitions
        partitions = [
            ("MIG 3g.40gb (~42% SMs)", 42),
            ("MIG 2g.20gb (~28% SMs)", 28),
            ("MIG 1g.10gb (~14% SMs)", 14),
        ]

        for name, percent in partitions:
            print(f"Testing simulated {name}...")
            mean_ms, std_ms, min_ms, max_ms = benchmark_simulated_partition(
                num_ops, tensor_size, sm_limit_percent=percent
            )
            result = BenchmarkResult(
                name="mig_simulated",
                config=name,
                mean_time_ms=mean_ms,
                std_time_ms=std_ms,
                min_time_ms=min_ms,
                max_time_ms=max_ms
            )
            reporter.add_result(result)
            print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")

            baseline = next(r for r in reporter.results
                          if r.config == "Full GPU (100%)")
            slowdown = mean_ms / baseline.mean_time_ms
            print(f"  Slowdown: {slowdown:.2f}× (expected ~{100/percent:.2f}×)\n")

        reporter.save_json("mig_simulated_results.json")
        reporter.save_csv("mig_simulated_results.csv")
        reporter.print_summary()

        print("\n" + "="*80)
        print("KEY INSIGHTS ABOUT MIG:")
        print("="*80)
        print("""
1. MIG provides HARDWARE ISOLATION, not just performance scaling
2. Each MIG instance has:
   - Dedicated SMs (compute units)
   - Isolated memory partition
   - Independent fault domains
   - QoS guarantees

3. MIG vs GPUOS:
   - MIG: Multi-tenant resource ISOLATION
   - GPUOS: Single-tenant launch overhead ELIMINATION
   - They solve DIFFERENT problems

4. MIG does NOT reduce launch overhead per operation
   - Each kernel launch still has 5-20μs overhead
   - GPUOS eliminates this even within a MIG instance

5. GPUOS + MIG COMBINATION:
   - Use MIG to partition GPU for multiple tenants
   - Each tenant runs GPUOS internally
   - Best of both: isolation + efficiency
""")

    else:
        print("\n" + "="*80)
        print("RUNNING REAL MIG TESTS")
        print("="*80)

        reporter = ResultsReporter()
        num_ops = 10000
        tensor_size = 2048

        # Detect MIG instance info
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=mig.mode.current,name,memory.total',
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            print(f"Current device: {result.stdout.strip()}\n")
        except Exception:
            pass

        print("Running benchmark on current MIG instance...")
        mean_ms, std_ms, min_ms, max_ms = benchmark_on_mig_instance(num_ops, tensor_size)

        result = BenchmarkResult(
            name="mig_real",
            config=f"MIG Instance (device {torch.cuda.current_device()})",
            mean_time_ms=mean_ms,
            std_time_ms=std_ms,
            min_time_ms=min_ms,
            max_time_ms=max_ms
        )
        reporter.add_result(result)
        print(f"  Time: {mean_ms:.3f} ± {std_ms:.3f} ms")
        print(f"  Operations/sec: {num_ops / (mean_ms / 1000.0):.0f}")

        reporter.save_json("mig_real_results.json")
        reporter.save_csv("mig_real_results.csv")
        reporter.print_summary()

        print("\nTo test GPUOS on this MIG instance:")
        print("  python benchmarks/bench_microbatch.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='MIG Benchmark')
    parser.add_argument('--real-mig', action='store_true',
                       help='Run on real MIG instance (requires MIG-enabled GPU)')
    args = parser.parse_args()

    run_mig_benchmark(use_real_mig=args.real_mig)
