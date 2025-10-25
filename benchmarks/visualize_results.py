"""
Visualize benchmark results
Creates charts and tables from benchmark JSON/CSV files
"""
import json
import os
import sys
import argparse
import csv
from typing import List, Dict, Callable, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import re


def _coerce_types(row: Dict) -> Dict:
    """Coerce known numeric fields to float if present"""
    numeric_keys = [
        'mean_time_ms', 'std_time_ms', 'min_time_ms', 'max_time_ms',
        'throughput_ops_per_sec', 'speedup_vs_baseline', 'gpu_utilization',
        'power_watts'
    ]
    out = dict(row)
    for k in numeric_keys:
        if k in out and out[k] not in (None, ""):
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                pass
    return out


def load_results(result_dir: str = "benchmark_results", extra_csv: Optional[List[str]] = None) -> Dict[str, List[Dict]]:
    """Load all JSON and CSV result files in a directory plus any extra CSVs.

    Returns a mapping of dataset base name (filename without extension) to list of entries.
    Each entry is a dict with keys like: name, config, mean_time_ms, ...
    """
    results: Dict[str, List[Dict]] = {}

    if result_dir and os.path.exists(result_dir):
        for filename in os.listdir(result_dir):
            filepath = os.path.join(result_dir, filename)
            base, ext = os.path.splitext(filename)
            try:
                if ext == '.json':
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        # Mark source for potential de-duplication
                        for d in data:
                            d['__source__'] = 'json'
                        results[base] = data
                        print(f"Loaded {len(data)} results from {filename}")
                elif ext == '.csv':
                    with open(filepath, 'r', newline='') as f:
                        reader = csv.DictReader(f)
                        rows = [_coerce_types({**row, '__source__': 'csv'}) for row in reader]
                        # If JSON of same base already loaded, keep JSON and store CSV under new key
                        key = base if base not in results else f"{base}__csv"
                        results[key] = rows
                        print(f"Loaded {len(rows)} results from {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    # Load any explicitly provided extra CSV files
    extra_csv = extra_csv or []
    for path in extra_csv:
        try:
            with open(path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                rows = [_coerce_types({**row, '__source__': 'csv'}) for row in reader]
                base = os.path.splitext(os.path.basename(path))[0]
                key = base if base not in results else f"{base}__csv_extra"
                results[key] = rows
                print(f"Loaded {len(rows)} results from {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    return results


def collect_results(results_map: Dict[str, List[Dict]], match: Callable[[Dict], bool]) -> List[Dict]:
    """Collect and de-duplicate results across all datasets matching predicate.

    Deduplicates by (name, config), preferring JSON over CSV when both present.
    """
    merged: Dict[tuple, Dict] = {}
    # First pass: JSON entries
    for entries in results_map.values():
        for r in entries:
            if not isinstance(r, dict):
                continue
            if not match(r):
                continue
            key = (r.get('name'), r.get('config'))
            src = r.get('__source__')
            if key not in merged or (src == 'json' and merged.get(key, {}).get('__source__') != 'json'):
                merged[key] = r
    # Second pass: fill in missing from CSV
    for entries in results_map.values():
        for r in entries:
            if not isinstance(r, dict):
                continue
            if not match(r):
                continue
            key = (r.get('name'), r.get('config'))
            if key not in merged:
                merged[key] = r
    return list(merged.values())


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


def plot_mig_simulated(results: List[Dict], output_path: str):
    """Bar chart for MIG simulated partition results"""
    # Filter and group by config
    entries = [r for r in results if isinstance(r.get('name'), str) and r['name'] == 'mig_simulated']
    if not entries:
        print("No MIG simulated results found")
        return

    # Map config -> time
    data = {e['config']: e['mean_time_ms'] for e in entries if 'config' in e and 'mean_time_ms' in e}
    if not data:
        print("MIG simulated entries missing required fields")
        return

    # Determine baseline and order by approximate % SMs if present
    baseline_key = next((k for k in data.keys() if 'Full GPU' in k or '100%' in k), None)
    def percent_from_config(cfg: str) -> float:
        m = re.search(r"(~?\s*(\d+))%", cfg)
        return float(m.group(2)) if m else (100.0 if cfg == baseline_key else 0.0)

    ordered = sorted(data.items(), key=lambda kv: -percent_from_config(kv[0]))

    configs = [k for k, _ in ordered]
    times = [v for _, v in ordered]

    # Compute slowdowns vs baseline
    baseline_time = data.get(baseline_key, times[0])
    slowdowns = [t / baseline_time if baseline_time else 0 for t in times]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(configs)), times, color='#1f77b4')
    plt.xlabel('Configuration', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('MIG Simulated Partition Performance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(configs)), configs, rotation=20, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    # Annotate slowdowns
    for bar, sd in zip(bars, slowdowns):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                 f"{sd:.2f}× slower" if sd >= 1.0 else f"{1/sd:.2f}× faster",
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")
    plt.close()


def plot_mig_real(results: List[Dict], output_path: str):
    """Bar chart for real MIG results (if multiple, compare by config)"""
    entries = [r for r in results if isinstance(r.get('name'), str) and r['name'] == 'mig_real']
    if not entries:
        print("No real MIG results found")
        return

    # Group by config
    data = {e.get('config', f"Run {i}"): e['mean_time_ms'] for i, e in enumerate(entries)}
    configs = list(data.keys())
    times = [data[c] for c in configs]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(range(len(configs)), times, color='#2ca02c')
    plt.xlabel('MIG Instance', fontsize=12)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.title('Real MIG Instance Performance', fontsize=14, fontweight='bold')
    plt.xticks(range(len(configs)), configs, rotation=15, ha='right')
    plt.grid(True, axis='y', alpha=0.3)

    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                 f"{t:.1f}ms", ha='center', va='bottom', fontsize=9)

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


def visualize_all_results(result_dir: str = "benchmark_results", extra_csv: Optional[List[str]] = None,
                         only: Optional[List[str]] = None, skip: Optional[List[str]] = None):
    """Generate all visualizations"""
    print("="*80)
    print("BENCHMARK RESULTS VISUALIZATION")
    print("="*80)
    print()

    # Load results
    results = load_results(result_dir=result_dir, extra_csv=extra_csv)

    if not results:
        print("No results found. Run benchmarks first:")
        print("  python benchmarks/run_all_benchmarks.py")
        return

    # Create visualizations directory
    viz_dir = "benchmark_results/visualizations"
    os.makedirs(viz_dir, exist_ok=True)

    only = set(only or [])
    skip = set(skip or [])

    def should_do(kind: str) -> bool:
        return (not only or kind in only) and (kind not in skip)

    # Generate plots from aggregated entries across all loaded files
    if should_do('microbatch'):
        microbatch_entries = collect_results(results, lambda r: isinstance(r.get('name'), str) and 'microbatch_add' in r['name'])
        if microbatch_entries:
            print("Generating micro-batch speedup plot...")
            plot_microbatch_speedup(microbatch_entries,
                                   os.path.join(viz_dir, 'microbatch_speedup.png'))
        else:
            print("No micro-batch results found in JSON/CSV")

    if should_do('mixed_ops'):
        mixed_entries = collect_results(results, lambda r: isinstance(r.get('name'), str) and 'mixed_ops' in r['name'])
        if mixed_entries:
            print("Generating mixed ops comparison...")
            plot_mixed_ops_comparison(mixed_entries,
                                     os.path.join(viz_dir, 'mixed_ops_comparison.png'))
        else:
            print("No mixed ops results found in JSON/CSV")

    if should_do('mps'):
        mps_entries = collect_results(results, lambda r: isinstance(r.get('name'), str) and 'multiprocess' in r['name'])
        if mps_entries:
            print("Generating MPS scaling plot...")
            plot_mps_scaling(mps_entries,
                            os.path.join(viz_dir, 'mps_scaling.png'))
        else:
            print("No MPS results found in JSON/CSV")

    if should_do('attention'):
        attention_entries = collect_results(results, lambda r: isinstance(r.get('name'), str) and 'attention_' in r['name'])
        if attention_entries:
            print("Generating attention throughput plot...")
            plot_attention_throughput(attention_entries,
                                     os.path.join(viz_dir, 'attention_throughput.png'))
        else:
            print("No attention results found in JSON/CSV")

    if should_do('mig'):
        mig_entries = collect_results(results, lambda r: isinstance(r.get('name'), str) and (r['name'] == 'mig_simulated' or r['name'] == 'mig_real'))
        if mig_entries:
            simulated = [e for e in mig_entries if e.get('name') == 'mig_simulated']
            real = [e for e in mig_entries if e.get('name') == 'mig_real']
            if simulated:
                print("Generating MIG simulated plot...")
                plot_mig_simulated(simulated, os.path.join(viz_dir, 'mig_simulated.png'))
            else:
                print("No MIG simulated results found")
            if real:
                print("Generating MIG real plot...")
                plot_mig_real(real, os.path.join(viz_dir, 'mig_real.png'))
            else:
                print("No real MIG results found")
        else:
            print("No MIG results found in JSON/CSV")

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
    parser = argparse.ArgumentParser(description='Visualize benchmark results')
    parser.add_argument('--result-dir', default='benchmark_results', help='Directory containing result JSON/CSV files')
    parser.add_argument('--extra-csv', nargs='*', default=[], help='Additional CSV files to include')
    parser.add_argument('--only', nargs='*', default=[], choices=['microbatch', 'mixed_ops', 'mps', 'attention', 'mig'], help='Only generate specific plots')
    parser.add_argument('--skip', nargs='*', default=[], choices=['microbatch', 'mixed_ops', 'mps', 'attention', 'mig'], help='Skip specific plots')

    args = parser.parse_args()
    visualize_all_results(result_dir=args.result_dir,
                          extra_csv=args.extra_csv,
                          only=args.only,
                          skip=args.skip)
