"""
Master script to run all GPUOS benchmarks
Runs comprehensive comparison against MPS and MIG
"""
import subprocess
import sys
import os
import argparse
import time

def run_command(cmd, description):
    """Run a command and report results"""
    print("\n" + "="*80)
    print(f"RUNNING: {description}")
    print("="*80)
    print(f"Command: {' '.join(cmd)}")
    print()

    start_time = time.time()

    try:
        result = subprocess.run(cmd, check=True)
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n✗ Failed after {elapsed:.1f}s: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Run all GPUOS benchmarks')
    parser.add_argument('--skip-microbatch', action='store_true',
                       help='Skip micro-batch benchmark')
    parser.add_argument('--skip-mixed', action='store_true',
                       help='Skip mixed operations benchmark')
    parser.add_argument('--skip-mps', action='store_true',
                       help='Skip MPS benchmark')
    parser.add_argument('--skip-mig', action='store_true',
                       help='Skip MIG benchmark')
    parser.add_argument('--skip-attention', action='store_true',
                       help='Skip attention benchmark')
    parser.add_argument('--real-mig', action='store_true',
                       help='Run real MIG tests (requires A100/H100)')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate visualizations after benchmarks')
    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════════════════════════╗
    ║                                                                       ║
    ║                    GPUOS BENCHMARK SUITE                              ║
    ║                                                                       ║
    ║  Comprehensive comparison of GPUOS vs MPS vs MIG                      ║
    ║                                                                       ║
    ╚═══════════════════════════════════════════════════════════════════════╝
    """)

    # Ensure we're in the right directory
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(os.path.dirname(benchmark_dir))  # Go to project root

    results = {}

    # 1. Micro-batch benchmark
    if not args.skip_microbatch:
        results['microbatch'] = run_command(
            [sys.executable, 'benchmarks/bench_microbatch.py'],
            "Micro-batch Elementwise Operations"
        )
    else:
        print("\nSkipping micro-batch benchmark")

    # 2. Mixed operations benchmark
    if not args.skip_mixed:
        results['mixed'] = run_command(
            [sys.executable, 'benchmarks/bench_mixed_ops.py'],
            "Mixed Operations Workload"
        )
    else:
        print("\nSkipping mixed operations benchmark")

    # 3. MPS benchmark
    if not args.skip_mps:
        results['mps'] = run_command(
            [sys.executable, 'benchmarks/bench_mps.py'],
            "MPS Multi-Process Benchmark"
        )
    else:
        print("\nSkipping MPS benchmark")

    # 4. MIG benchmark
    if not args.skip_mig:
        cmd = [sys.executable, 'benchmarks/bench_mig.py']
        if args.real_mig:
            cmd.append('--real-mig')
        results['mig'] = run_command(
            cmd,
            "MIG Multi-Instance GPU Benchmark" + (" (REAL)" if args.real_mig else " (SIMULATED)")
        )
    else:
        print("\nSkipping MIG benchmark")

    # 5. Attention benchmark
    if not args.skip_attention:
        results['attention'] = run_command(
            [sys.executable, 'benchmarks/bench_attention.py'],
            "Attention Mechanism Benchmark"
        )
    else:
        print("\nSkipping attention benchmark")

    # Summary
    print("\n\n" + "="*80)
    print("BENCHMARK SUITE COMPLETE")
    print("="*80)
    print("\nResults Summary:")
    for name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {name:<20}: {status}")

    total = len(results)
    passed = sum(1 for s in results.values() if s)
    print(f"\nTotal: {passed}/{total} benchmarks passed")

    # Visualize
    if args.visualize:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        run_command(
            [sys.executable, 'benchmarks/visualize_results.py'],
            "Result Visualization and Reporting"
        )

    print("\n" + "="*80)
    print("All benchmark results saved to: benchmark_results/")
    print("\nTo generate visualizations:")
    print("  python benchmarks/visualize_results.py")
    print("\nTo view results:")
    print("  cat benchmark_results/visualizations/SUMMARY.md")
    print("  # or open PNG files in benchmark_results/visualizations/")
    print("="*80)


if __name__ == "__main__":
    main()
