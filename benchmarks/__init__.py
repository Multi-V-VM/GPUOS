"""
GPUOS Benchmark Suite

Comprehensive benchmarking framework for comparing GPUOS against:
- NVIDIA Multi-Process Service (MPS)
- NVIDIA Multi-Instance GPU (MIG)
- PyTorch baseline
- PyTorch torch.compile

Usage:
    python benchmarks/run_all_benchmarks.py --visualize

Individual benchmarks:
    python benchmarks/bench_microbatch.py
    python benchmarks/bench_mixed_ops.py
    python benchmarks/bench_mps.py
    python benchmarks/bench_mig.py
    python benchmarks/bench_attention.py

Visualization:
    python benchmarks/visualize_results.py
"""

__version__ = "1.0.0"
__all__ = [
    "benchmark_utils",
    "bench_microbatch",
    "bench_mixed_ops",
    "bench_mps",
    "bench_mig",
    "bench_attention",
    "visualize_results",
    "run_all_benchmarks"
]
