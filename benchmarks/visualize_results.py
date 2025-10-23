"""
Visualize benchmark results
Creates charts and tables from benchmark JSON/CSV files
"""
import json
import os
import sys
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_results(result_dir: str = "benchmark_results") -> Dict[str, List[Dict]]:
    """Load all JSON result files"""
    results = {}

    if not os.path.exists(result_dir):
        print(f"Result directory {result_dir} not found")
        return results

    for filename in os.listdir(result_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(result_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    name = filename.replace('.json', '')
                    results[name] = data
                    print(f"Loaded {len(data)} results from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    return results


def plot_microbatch_speedup(results: List[Dict], output_path: str):
    """Plot speedup vs tensor size for micro-batch benchmark"""
    # Group by tensor size
    by_size = {}
    for r in results:
        name = r['name']
        if 'microbatch_add' in name:
            size = int(name.split('_')[-1])
            if size not in by_size:
                by_size[size] = {}
            by_size[size][r['config']] = r['mean_time_ms']

    sizes = sorted(by_size.keys())

    # Calculate speedups relative to PyTorch baseline
    speedups = {}
    for size in sizes:
        baseline = by_size[size].get('PyTorch Baseline', 1.0)
        for config, time_ms in by_size[size].items():
            if config != 'PyTorch Baseline':
                if config not in speedups:
                    speedups[config] = []
                speedup = baseline / time_ms if time_ms > 0 else 0
                speedups[config].append(speedup)

    # Plot
    plt.figure(figsize=(10, 6))
    for config, values in speedups.items():
        plt.plot(sizes, values, marker='o', label=config, linewidth=2)

    plt.xlabel('Tensor Size (elements)', fontsize=12)
    plt.ylabel('Speedup vs PyTorch Baseline', fontsize=12)
    plt.title('Micro-Batch Speedup (10K operations)', fontsize=14, fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_mixed_ops_comparison(results: List[Dict], output_path: str):
    """Bar chart comparing mixed operations performance"""
    # Group by tensor size
    by_size = {}
    for r in results:
        name = r['name']
        if 'mixed_ops' in name:
            size = int(name.split('_')[-1])
            if size not in by_size:
                by_size[size] = {}
            by_size[size][r['config']] = r['mean_time_ms']

    if not by_size:
        print("No mixed ops results found")
        return

    # Use first tensor size for comparison
    size = sorted(by_size.keys())[0]
    data = by_size[size]

    configs = list(data.keys())
    times = [data[c] for c in configs]

    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(configs)), times, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title(f'Mixed Operations Performance (1K iterations, {size} elements)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(configs)), configs, rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, time) in enumerate(zip(bars, times)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{time:.1f}ms', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_mps_scaling(results: List[Dict], output_path: str):
    """Plot MPS multi-process scaling"""
    # Group by number of processes
    by_nproc = {}
    for r in results:
        name = r['name']
        if 'multiprocess' in name:
            nproc = int(name.split('_')[-1])
            if nproc not in by_nproc:
                by_nproc[nproc] = {}
            by_nproc[nproc][r['config']] = r['mean_time_ms']

    if not by_nproc:
        print("No MPS results found")
        return

    nprocs = sorted(by_nproc.keys())

    # Calculate speedups vs sequential
    speedups = {}
    for nproc in nprocs:
        baseline = by_nproc[nproc].get('Sequential (No Sharing)', 1.0)
        for config, time_ms in by_nproc[nproc].items():
            if 'Sequential' not in config:
                if config not in speedups:
                    speedups[config] = []
                speedup = baseline / time_ms if time_ms > 0 else 0
                speedups[config].append(speedup)

    # Plot
    plt.figure(figsize=(10, 6))
    x = np.arange(len(nprocs))
    width = 0.25

    configs = list(speedups.keys())
    for i, config in enumerate(configs):
        plt.bar(x + i*width, speedups[config], width, label=config)

    plt.xlabel('Number of Processes', fontsize=12)
    plt.ylabel('Speedup vs Sequential', fontsize=12)
    plt.title('Multi-Process GPU Sharing Performance', fontsize=14, fontweight='bold')
    plt.xticks(x + width, nprocs)
    plt.legend(fontsize=9)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_attention_throughput(results: List[Dict], output_path: str):
    """Plot attention tokens/sec"""
    # Group by config and seq_len
    by_config = {}
    for r in results:
        name = r['name']
        if 'attention' in name and r.get('throughput_ops_per_sec'):
            parts = name.split('_')
            seq_len = int(parts[1])
            config = r['config']
            if config not in by_config:
                by_config[config] = {'seq_lens': [], 'throughputs': []}
            by_config[config]['seq_lens'].append(seq_len)
            by_config[config]['throughputs'].append(r['throughput_ops_per_sec'])

    if not by_config:
        print("No attention results found")
        return

    # Sort and plot
    plt.figure(figsize=(10, 6))
    for config, data in by_config.items():
        # Sort by seq_len
        sorted_pairs = sorted(zip(data['seq_lens'], data['throughputs']))
        seq_lens, throughputs = zip(*sorted_pairs)
        plt.plot(seq_lens, throughputs, marker='o', label=config, linewidth=2)

    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Tokens Generated / Second', fontsize=12)
    plt.title('Token-by-Token Generation Throughput', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def generate_latex_table(results: Dict[str, List[Dict]], output_path: str):
    """Generate LaTeX table for paper"""
    with open(output_path, 'w') as f:
        f.write("% Benchmark Results LaTeX Tables\n\n")

        # Micro-batch table
        f.write("% Micro-batch results\n")
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Micro-batch Elementwise Performance (10K add operations)}\n")
        f.write("\\begin{tabular}{lrrrr}\n")
        f.write("\\hline\n")
        f.write("Tensor Size & PyTorch (ms) & GPUOS (ms) & Speedup & PyTorch Compiled (ms) \\\\\n")
        f.write("\\hline\n")

        # Extract micro-batch results
        if 'microbatch_results' in results:
            by_size = {}
            for r in results['microbatch_results']:
                name = r['name']
                if 'microbatch_add' in name:
                    size = int(name.split('_')[-1])
                    if size not in by_size:
                        by_size[size] = {}
                    by_size[size][r['config']] = r['mean_time_ms']

            for size in sorted(by_size.keys()):
                pytorch = by_size[size].get('PyTorch Baseline', 0)
                gpuos = by_size[size].get('GPUOS', 0)
                compiled = by_size[size].get('PyTorch Compiled', '-')
                speedup = pytorch / gpuos if gpuos > 0 else 0

                compiled_str = f"{compiled:.1f}" if isinstance(compiled, float) else compiled

                f.write(f"{size} & {pytorch:.1f} & {gpuos:.1f} & {speedup:.2f}× & {compiled_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

    print(f"Saved LaTeX table to {output_path}")


def generate_summary_report(results: Dict[str, List[Dict]], output_path: str):
    """Generate markdown summary report"""
    with open(output_path, 'w') as f:
        f.write("# GPUOS Benchmark Results Summary\n\n")
        f.write(f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for result_name, result_list in results.items():
            f.write(f"## {result_name.replace('_', ' ').title()}\n\n")

            # Group by benchmark name
            by_name = {}
            for r in result_list:
                name = r['name']
                if name not in by_name:
                    by_name[name] = []
                by_name[name].append(r)

            for name, entries in by_name.items():
                f.write(f"### {name}\n\n")
                f.write("| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |\n")
                f.write("|---------------|-----------|----------|----------|----------|\n")

                for entry in entries:
                    f.write(f"| {entry['config']:<25} | {entry['mean_time_ms']:>8.3f} | "
                           f"{entry['std_time_ms']:>7.3f} | {entry['min_time_ms']:>7.3f} | "
                           f"{entry['max_time_ms']:>7.3f} |\n")

                # Calculate speedups
                baseline = next((e for e in entries if 'Baseline' in e['config'] or 'PyTorch' in e['config']), None)
                if baseline:
                    f.write("\n**Speedups vs Baseline:**\n\n")
                    for entry in entries:
                        if entry != baseline:
                            speedup = baseline['mean_time_ms'] / entry['mean_time_ms'] if entry['mean_time_ms'] > 0 else 0
                            f.write(f"- {entry['config']}: **{speedup:.2f}×**\n")

                f.write("\n")

    print(f"Saved summary report to {output_path}")


def visualize_all_results():
    """Generate all visualizations"""
    print("="*80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    print()

    # Load results
    results = load_results()

    if not results:
        print("No results found. Run benchmarks first:")
        print("  python benchmarks/run_all_benchmarks.py")
        return

    # Create visualizations directory
    viz_dir = "benchmark_results/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    # Generate plots
    if 'microbatch_results' in results:
        print("Generating micro-batch speedup plot...")
        plot_microbatch_speedup(results['microbatch_results'],
                               os.path.join(viz_dir, 'microbatch_speedup.png'))

    if 'mixed_ops_results' in results:
        print("Generating mixed ops comparison...")
        plot_mixed_ops_comparison(results['mixed_ops_results'],
                                 os.path.join(viz_dir, 'mixed_ops_comparison.png'))

    if 'mps_results' in results:
        print("Generating MPS scaling plot...")
        plot_mps_scaling(results['mps_results'],
                        os.path.join(viz_dir, 'mps_scaling.png'))

    if 'attention_results' in results:
        print("Generating attention throughput plot...")
        plot_attention_throughput(results['attention_results'],
                                 os.path.join(viz_dir, 'attention_throughput.png'))

    # Generate tables and reports
    print("Generating LaTeX tables...")
    generate_latex_table(results, os.path.join(viz_dir, 'tables.tex'))

    print("Generating summary report...")
    generate_summary_report(results, os.path.join(viz_dir, 'SUMMARY.md'))

    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Results saved to: {viz_dir}/")
    print("="*80)


if __name__ == "__main__":
    visualize_all_results()
