"""
Benchmark utilities for GPUOS, MPS, and MIG comparisons
"""
import time
import os
import torch
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
import subprocess
import os
import json
from dataclasses import dataclass, asdict


@dataclass
class BenchmarkResult:
    """Store benchmark results"""
    name: str
    config: str
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_per_sec: Optional[float] = None
    speedup_vs_baseline: Optional[float] = None
    gpu_utilization: Optional[float] = None
    power_watts: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class BenchmarkTimer:
    """High-precision GPU timing using CUDA events"""

    def __init__(self, warmup_iters: int = 100, measure_iters: int = 1000,
                 trim_percent: float = 0.05):
        # Allow environment overrides for faster debugging
        env_warm = os.getenv('GPUOS_BENCH_WARMUP')
        env_meas = os.getenv('GPUOS_BENCH_ITERS')
        if env_warm is not None:
            try:
                warmup_iters = int(env_warm)
            except Exception:
                pass
        if env_meas is not None:
            try:
                measure_iters = int(env_meas)
            except Exception:
                pass
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.trim_percent = trim_percent
        self.times = []

    def benchmark(self, func: Callable, *args, **kwargs) -> Tuple[float, float, float, float]:
        """
        Benchmark a function with CUDA events
        Returns: (mean_ms, std_ms, min_ms, max_ms)
        """
        # Warmup
        for _ in range(self.warmup_iters):
            func(*args, **kwargs)

        torch.cuda.synchronize()

        # Measurement
        times = []
        show_prog = os.getenv('GPUOS_BENCH_PROGRESS') is not None
        for i in range(self.measure_iters):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            func(*args, **kwargs)
            end_event.record()

            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event)  # milliseconds
            times.append(elapsed)
            if show_prog and (i % max(1, self.measure_iters // 10) == 0):
                print(f"  progress: {i}/{self.measure_iters}")

        # Trim outliers (skip when too few samples)
        times = np.array(times, dtype=float)
        if self.trim_percent > 0 and len(times) >= 20:
            lower = int(len(times) * self.trim_percent)
            upper = int(len(times) * (1 - self.trim_percent))
            lower = max(0, min(lower, len(times) - 1))
            upper = max(lower + 1, min(upper, len(times)))
            times = np.sort(times)[lower:upper]

        return (
            float(np.mean(times)),
            float(np.std(times)),
            float(np.min(times)),
            float(np.max(times))
        )


class GPUMonitor:
    """Monitor GPU utilization and power consumption"""

    @staticmethod
    def get_gpu_utilization() -> float:
        """Get current GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def get_power_usage() -> float:
        """Get current GPU power usage in watts"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=power.draw', '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    @staticmethod
    def monitor_during_benchmark(duration_sec: float, sample_rate_hz: float = 10) -> Dict[str, float]:
        """Monitor GPU metrics during benchmark execution"""
        import threading

        utilizations = []
        powers = []
        stop_flag = threading.Event()

        def monitor_loop():
            while not stop_flag.is_set():
                utilizations.append(GPUMonitor.get_gpu_utilization())
                powers.append(GPUMonitor.get_power_usage())
                time.sleep(1.0 / sample_rate_hz)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        time.sleep(duration_sec)
        stop_flag.set()
        thread.join(timeout=1.0)

        return {
            'avg_utilization': np.mean(utilizations) if utilizations else 0.0,
            'avg_power': np.mean(powers) if powers else 0.0,
            'max_power': np.max(powers) if powers else 0.0,
        }


class MPSController:
    """Control NVIDIA MPS (Multi-Process Service)"""

    @staticmethod
    def is_available() -> bool:
        """Check if MPS is available"""
        try:
            result = subprocess.run(['which', 'nvidia-cuda-mps-control'],
                                  capture_output=True, timeout=2)
            return result.returncode == 0
        except Exception:
            return False

    @staticmethod
    def start():
        """Start MPS server"""
        if not MPSController.is_available():
            print("WARNING: MPS not available on this system")
            return False

        try:
            # Stop any existing MPS instance
            MPSController.stop()

            # Ensure pipe and log directories are set and exist
            pipe_dir = os.environ.get('CUDA_MPS_PIPE_DIRECTORY') or f"/tmp/nvidia-mps-{os.getuid()}"
            log_dir = os.environ.get('CUDA_MPS_LOG_DIRECTORY') or f"/tmp/nvidia-mps-log-{os.getuid()}"
            os.makedirs(pipe_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            os.environ['CUDA_MPS_PIPE_DIRECTORY'] = pipe_dir
            os.environ['CUDA_MPS_LOG_DIRECTORY'] = log_dir

            # Start MPS daemon with environment
            subprocess.run(['nvidia-cuda-mps-control', '-d'],
                         check=True, timeout=10, env=os.environ.copy())
            time.sleep(1)  # Give it time to start
            print("MPS server started")
            return True
        except Exception as e:
            print(f"Failed to start MPS: {e}")
            return False

    @staticmethod
    def stop():
        """Stop MPS server"""
        try:
            subprocess.run(['echo', 'quit'],
                         stdout=subprocess.PIPE,
                         check=False, timeout=5)
            subprocess.run(['killall', '-9', 'nvidia-cuda-mps-control'],
                         check=False, timeout=5,
                         stderr=subprocess.DEVNULL)
            subprocess.run(['killall', '-9', 'nvidia-cuda-mps-server'],
                         check=False, timeout=5,
                         stderr=subprocess.DEVNULL)
            time.sleep(1)
            print("MPS server stopped")
        except Exception as e:
            print(f"Note: MPS stop had issues (may be ok): {e}")


class MIGController:
    """Control NVIDIA MIG (Multi-Instance GPU)"""

    @staticmethod
    def is_available() -> bool:
        """Check if MIG is supported on this GPU"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=mig.mode.current', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            output = result.stdout.strip()
            return output in ['Enabled', 'Disabled']
        except Exception:
            return False

    @staticmethod
    def is_enabled() -> bool:
        """Check if MIG is currently enabled"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=mig.mode.current', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=2
            )
            return result.stdout.strip() == 'Enabled'
        except Exception:
            return False

    @staticmethod
    def get_available_profiles() -> List[str]:
        """Get available MIG profiles"""
        if not MIGController.is_enabled():
            return []

        try:
            result = subprocess.run(
                ['nvidia-smi', 'mig', '-lgip'],
                capture_output=True, text=True, timeout=5
            )
            # Parse output for profile IDs
            profiles = []
            for line in result.stdout.split('\n'):
                if 'MIG' in line and 'GI' in line:
                    profiles.append(line.strip())
            return profiles
        except Exception:
            return []

    @staticmethod
    def enable():
        """Enable MIG mode (requires root and reboot)"""
        print("MIG can only be enabled by root user with:")
        print("  sudo nvidia-smi -mig 1")
        print("  sudo reboot")
        print("This benchmark will skip MIG tests if not available.")

    @staticmethod
    def create_instances(profile: str = "1g.5gb", count: int = 7):
        """Create MIG instances (requires root)"""
        print(f"To create MIG instances, run as root:")
        print(f"  sudo nvidia-smi mig -cgi {profile} -C")
        print("This benchmark will provide instructions for MIG testing.")


class ResultsReporter:
    """Generate benchmark reports"""

    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result"""
        self.results.append(result)

    def save_json(self, filename: str = "results.json"):
        """Save results to JSON"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        print(f"Results saved to {filepath}")

    def save_csv(self, filename: str = "results.csv"):
        """Save results to CSV"""
        filepath = os.path.join(self.output_dir, filename)

        if not self.results:
            return

        import csv
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())

        print(f"Results saved to {filepath}")

    def print_summary(self):
        """Print summary table to console"""
        if not self.results:
            print("No results to display")
            return

        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)

        # Group by benchmark name
        by_name = {}
        for r in self.results:
            if r.name not in by_name:
                by_name[r.name] = []
            by_name[r.name].append(r)

        for name, results in by_name.items():
            print(f"\n{name}:")
            print("-" * 80)
            print(f"{'Config':<25} {'Mean(ms)':<12} {'Std(ms)':<12} {'Speedup':<12}")
            print("-" * 80)

            # Find baseline for speedup calculation
            baseline = next((r for r in results if 'PyTorch' in r.config or 'Baseline' in r.config), None)

            for r in results:
                speedup_str = "-"
                if baseline and r != baseline and baseline.mean_time_ms > 0:
                    speedup = baseline.mean_time_ms / r.mean_time_ms
                    speedup_str = f"{speedup:.2f}×"

                print(f"{r.config:<25} {r.mean_time_ms:<12.3f} {r.std_time_ms:<12.3f} {speedup_str:<12}")

        print("\n" + "="*80)


def verify_correctness(reference: torch.Tensor, output: torch.Tensor,
                       rtol: float = 1e-4, atol: float = 1e-5) -> bool:
    """Verify numerical correctness against reference"""
    try:
        torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)
        return True
    except AssertionError as e:
        print(f"Correctness check failed: {e}")
        return False


def get_gpu_info() -> Dict[str, any]:
    """Get GPU information"""
    try:
        gpu_name = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()

        memory_total = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=2
        ).stdout.strip()

        cuda_version = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True, text=True, timeout=2
        ).stdout

        return {
            'gpu_name': gpu_name,
            'memory_total_mb': memory_total,
            'cuda_version': cuda_version,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
        }
    except Exception as e:
        return {'error': str(e)}
