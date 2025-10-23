"""
MPS (Multi-Process Service) benchmarks
Tests multi-process GPU sharing with and without MPS
"""
import torch
import sys
import os
import time
import multiprocessing as mp
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_utils import (
    BenchmarkTimer, BenchmarkResult, ResultsReporter,
    MPSController, get_gpu_info
)


def worker_pytorch_task(rank: int, num_ops: int, tensor_size: int, barrier, result_queue):
    """Worker process running PyTorch operations"""
    try:
        # Each process uses the same GPU
        torch.cuda.set_device(0)

        a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
        b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
        out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

        # Wait for all processes to be ready
        barrier.wait()

        # Start timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Run workload
        for _ in range(num_ops):
            torch.add(a, b, out=out)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        result_queue.put({'rank': rank, 'time_ms': elapsed_ms, 'success': True})

    except Exception as e:
        result_queue.put({'rank': rank, 'error': str(e), 'success': False})


def worker_gpuos_task(rank: int, num_ops: int, tensor_size: int, barrier, result_queue):
    """Worker process running GPUOS operations"""
    try:
        import pytorch_ext.gpuos_ext as gpuos_ext

        torch.cuda.set_device(0)

        a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
        b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
        out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

        # Initialize GPUOS for this process
        gpuos_ext.init(capacity=4096, threads_per_block=256)

        # Wait for all processes to be ready
        barrier.wait()

        # Start timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        # Run workload
        for _ in range(num_ops):
            gpuos_ext.submit_add(a, b, out)
        gpuos_ext.flush(sync=True)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_ms = start_event.elapsed_time(end_event)

        gpuos_ext.shutdown()

        result_queue.put({'rank': rank, 'time_ms': elapsed_ms, 'success': True})

    except Exception as e:
        result_queue.put({'rank': rank, 'error': str(e), 'success': False})


def run_multiprocess_benchmark(num_processes: int, num_ops: int, tensor_size: int,
                               use_gpuos: bool = False) -> tuple:
    """Run multi-process benchmark"""

    # Create barrier and result queue
    barrier = mp.Barrier(num_processes)
    result_queue = mp.Queue()

    # Choose worker function
    worker_func = worker_gpuos_task if use_gpuos else worker_pytorch_task

    # Start all worker processes
    processes = []
    start_time = time.time()

    for rank in range(num_processes):
        p = mp.Process(target=worker_func, args=(rank, num_ops, tensor_size, barrier, result_queue))
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    total_time = time.time() - start_time

    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Check for errors
    errors = [r for r in results if not r.get('success', False)]
    if errors:
        print(f"  Errors occurred: {errors}")
        return None, None, None

    # Calculate statistics
    times = [r['time_ms'] for r in results if 'time_ms' in r]
    max_time = max(times) if times else 0
    avg_time = sum(times) / len(times) if times else 0

    return max_time, avg_time, total_time * 1000


def benchmark_sequential(num_processes: int, num_ops: int, tensor_size: int) -> float:
    """Baseline: run processes sequentially (no GPU sharing)"""
    a = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    b = torch.randn(tensor_size, device='cuda', dtype=torch.float32)
    out = torch.empty(tensor_size, device='cuda', dtype=torch.float32)

    total_time_ms = 0

    for _ in range(num_processes):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        for _ in range(num_ops):
            torch.add(a, b, out=out)

        end_event.record()
        torch.cuda.synchronize()

        total_time_ms += start_event.elapsed_time(end_event)

    return total_time_ms


def run_mps_benchmark():
    """Run comprehensive MPS benchmark"""
    print("="*80)
    print("MPS (MULTI-PROCESS SERVICE) BENCHMARK")
    print("="*80)

    gpu_info = get_gpu_info()
    print(f"\nGPU: {gpu_info.get('gpu_name', 'Unknown')}")
    print(f"PyTorch: {gpu_info.get('pytorch_version', 'Unknown')}")

    # Check MPS availability
    mps_available = MPSController.is_available()
    print(f"MPS Available: {mps_available}")
    print()

    reporter = ResultsReporter()

    # Test configuration
    num_processes_list = [2, 4, 8]
    num_ops = 1000
    tensor_size = 1024

    for num_procs in num_processes_list:
        print(f"\n--- {num_procs} Concurrent Processes ---")
        print(f"Each process: {num_ops} operations on {tensor_size} elements")

        # Sequential baseline (no sharing)
        print("  Running sequential baseline (no GPU sharing)...")
        sequential_time = benchmark_sequential(num_procs, num_ops, tensor_size)
        result = BenchmarkResult(
            name=f"multiprocess_{num_procs}",
            config="Sequential (No Sharing)",
            mean_time_ms=sequential_time,
            std_time_ms=0.0,
            min_time_ms=sequential_time,
            max_time_ms=sequential_time
        )
        reporter.add_result(result)
        print(f"    Total time: {sequential_time:.3f} ms")

        # Without MPS (concurrent but no MPS)
        print("  Running concurrent without MPS...")
        try:
            # Make sure MPS is stopped
            MPSController.stop()
            time.sleep(1)

            max_time, avg_time, wall_time = run_multiprocess_benchmark(
                num_procs, num_ops, tensor_size, use_gpuos=False
            )

            if max_time is not None:
                result = BenchmarkResult(
                    name=f"multiprocess_{num_procs}",
                    config="Concurrent (No MPS)",
                    mean_time_ms=wall_time,
                    std_time_ms=0.0,
                    min_time_ms=avg_time,
                    max_time_ms=max_time
                )
                reporter.add_result(result)
                print(f"    Wall time: {wall_time:.3f} ms")
                print(f"    Max per-process: {max_time:.3f} ms, Avg: {avg_time:.3f} ms")
                speedup = sequential_time / wall_time if wall_time > 0 else 0
                print(f"    Speedup vs sequential: {speedup:.2f}×")
        except Exception as e:
            print(f"    ERROR: {e}")

        # With MPS
        if mps_available:
            print("  Running concurrent WITH MPS...")
            try:
                # Start MPS
                if MPSController.start():
                    time.sleep(2)  # Give MPS time to initialize

                    max_time, avg_time, wall_time = run_multiprocess_benchmark(
                        num_procs, num_ops, tensor_size, use_gpuos=False
                    )

                    if max_time is not None:
                        result = BenchmarkResult(
                            name=f"multiprocess_{num_procs}",
                            config="MPS Enabled",
                            mean_time_ms=wall_time,
                            std_time_ms=0.0,
                            min_time_ms=avg_time,
                            max_time_ms=max_time
                        )
                        reporter.add_result(result)
                        print(f"    Wall time: {wall_time:.3f} ms")
                        print(f"    Max per-process: {max_time:.3f} ms, Avg: {avg_time:.3f} ms")
                        speedup = sequential_time / wall_time if wall_time > 0 else 0
                        print(f"    Speedup vs sequential: {speedup:.2f}×")

                        # Compare with non-MPS
                        no_mps_result = next((r for r in reporter.results
                                             if r.name == f"multiprocess_{num_procs}"
                                             and r.config == "Concurrent (No MPS)"), None)
                        if no_mps_result:
                            mps_speedup = no_mps_result.mean_time_ms / wall_time
                            print(f"    MPS speedup vs no-MPS: {mps_speedup:.2f}×")

                    MPSController.stop()
                    time.sleep(1)

            except Exception as e:
                print(f"    ERROR: {e}")
                MPSController.stop()
        else:
            print("  MPS not available on this system")

        # GPUOS multi-process (for comparison)
        print("  Running GPUOS multi-process...")
        try:
            MPSController.stop()  # Make sure MPS is off
            time.sleep(1)

            max_time, avg_time, wall_time = run_multiprocess_benchmark(
                num_procs, num_ops, tensor_size, use_gpuos=True
            )

            if max_time is not None:
                result = BenchmarkResult(
                    name=f"multiprocess_{num_procs}",
                    config="GPUOS Multi-Process",
                    mean_time_ms=wall_time,
                    std_time_ms=0.0,
                    min_time_ms=avg_time,
                    max_time_ms=max_time
                )
                reporter.add_result(result)
                print(f"    Wall time: {wall_time:.3f} ms")
                print(f"    Max per-process: {max_time:.3f} ms, Avg: {avg_time:.3f} ms")
                speedup = sequential_time / wall_time if wall_time > 0 else 0
                print(f"    Speedup vs sequential: {speedup:.2f}×")
        except Exception as e:
            print(f"    ERROR: {e}")

    # Save results
    reporter.save_json("mps_results.json")
    reporter.save_csv("mps_results.csv")
    reporter.print_summary()

    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("- MPS benefits: Concurrent kernel execution from multiple processes")
    print("- GPUOS benefits: Eliminates launch overhead within each process")
    print("- Best combination: MPS + GPUOS (each process uses GPUOS internally)")
    print("="*80)


if __name__ == "__main__":
    # Required for multiprocessing on some platforms
    mp.set_start_method('spawn', force=True)
    run_mps_benchmark()
