# GPUOS Benchmark Suite

Comprehensive benchmarking framework comparing GPUOS against NVIDIA MPS and MIG.

## Quick Start

Run all benchmarks:
```bash
python benchmarks/run_all_benchmarks.py --visualize
```

## Individual Benchmarks

### 1. Micro-batch Elementwise Operations
Tests performance of many small elementwise operations (add, mul, etc.)

```bash
python benchmarks/bench_microbatch.py
```

**What it measures:**
- 10,000 add operations on tensors of varying sizes (256 - 16K elements)
- Compares GPUOS vs PyTorch baseline vs torch.compile

**Expected results:**
- GPUOS: 10-15× speedup for small tensors (256-2048 elements)
- Speedup decreases as tensor size increases (computation dominates)

### 2. Mixed Operations Workload
Realistic inference pattern with multiple operation types

```bash
python benchmarks/bench_mixed_ops.py
```

**What it measures:**
- 1000 iterations of: add → mul → relu → sigmoid → div
- Tests both manual GPUOS batching and transparent scheduler

**Expected results:**
- GPUOS: 15-25× speedup over PyTorch
- Expression fusion provides additional 30-40% benefit

### 3. MPS Multi-Process Benchmark
Tests GPU sharing across multiple processes

```bash
python benchmarks/bench_mps.py
```

**What it measures:**
- 2, 4, and 8 concurrent processes running small operations
- Compares: Sequential, Concurrent (no MPS), MPS enabled, GPUOS multi-process

**Expected results:**
- MPS: 2-2.5× speedup for multi-process workloads
- GPUOS: Works orthogonally (each process benefits from GPUOS internally)
- Best: MPS + GPUOS combination

**Requirements:**
- MPS server (nvidia-cuda-mps-control)
- Available on Kepler+ GPUs

### 4. MIG Multi-Instance GPU Benchmark
Tests hardware partitioning (A100/H100 only)

```bash
# Simulated (any GPU):
python benchmarks/bench_mig.py

# Real MIG (requires A100/H100):
python benchmarks/bench_mig.py --real-mig
```

**What it measures:**
- Performance on different MIG partition sizes
- Shows MIG provides isolation, not launch overhead reduction

**Expected results:**
- MIG partitions have proportional performance (3g.40gb ≈ 42% of full GPU)
- Each MIG instance still benefits from GPUOS internally

**Requirements:**
- Real MIG tests: NVIDIA A100, A30, H100, or H200
- Root access for MIG configuration

### 5. Attention Pattern Benchmark
Token-by-token generation simulation

```bash
python benchmarks/bench_attention.py
```

**What it measures:**
- 100 tokens generated with attention over 128-512 sequence lengths
- Compares: Naive PyTorch, FlashAttention, GPUOS-scheduled

**Expected results:**
- GPUOS benefits for small batch sizes (batch=1)
- FlashAttention optimized for large batches
- Complementary: use both together

## Visualization

Generate plots and reports:
```bash
python benchmarks/visualize_results.py
```

**Outputs:**
- `benchmark_results/visualizations/*.png` - Performance charts
- `benchmark_results/visualizations/SUMMARY.md` - Markdown report
- `benchmark_results/visualizations/tables.tex` - LaTeX tables for paper

## Understanding the Results

### Key Metrics

1. **Mean Time (ms)**: Average execution time
2. **Speedup**: Ratio vs PyTorch baseline (higher is better)
3. **Throughput**: Operations/second (higher is better)

### When GPUOS Wins

✓ Many (>1000) small operations
✓ Tensor sizes < 16K elements
✓ Launch overhead dominates (>50% of time)
✓ Single-process workloads

### When MPS Wins

✓ Multiple processes competing for GPU
✓ GPU underutilized (<30% without sharing)
✓ Compute-bound workloads
✓ Each process has moderate-sized operations

### When MIG Wins

✓ Multi-tenant cloud environments
✓ Strict resource isolation required
✓ Different workloads need guaranteed resources
✓ Fault isolation critical

### Best Combination

**MPS + MIG + GPUOS:**
```
┌─────────────────────────────────┐
│         A100 GPU (80GB)         │
├──────────────┬──────────────────┤
│ MIG Instance │  MIG Instance    │
│   (20GB)     │    (60GB)        │
│              │                  │
│  MPS Server  │   Direct Access  │
│  ├─ Proc A   │   (Training)     │
│  │  (GPUOS) │                  │
│  └─ Proc B   │                  │
│     (GPUOS)  │                  │
└──────────────┴──────────────────┘
```

## System Requirements

### Minimum
- CUDA-capable GPU (compute capability 5.0+)
- CUDA Toolkit 12.0+
- PyTorch 2.0+
- Python 3.8+

### Recommended
- NVIDIA RTX 4090 or A100 (for full benchmarks)
- 16GB+ GPU memory
- Ubuntu 22.04 or later

### Optional
- MPS: Any Kepler+ GPU (2012+)
- MIG: A100, A30, H100, H200 only

## Interpreting Paper Results

The results in the paper were obtained on:
- GPU: NVIDIA RTX 4090 (128 SMs, 24GB)
- CUDA: 12.4
- PyTorch: 2.1.0

Your results may vary based on:
- GPU architecture (different launch overhead)
- CUDA/driver version
- System load and thermals

## Troubleshooting

### GPUOS extension not found
```bash
cd examples
python pytorch_batch_demo.py  # Builds extension
```

### MPS won't start
```bash
# Kill existing MPS
sudo killall nvidia-cuda-mps-control nvidia-cuda-mps-server

# Check permissions
ls -l /tmp/nvidia-mps

# Start manually
nvidia-cuda-mps-control -d
```

### Out of memory
Reduce test sizes in benchmark scripts:
- `num_ops = 10000` → `5000`
- `tensor_size = 16384` → `8192`

## Advanced Usage

### Custom Benchmark Configuration

Edit benchmark scripts to modify:
- Number of operations: `num_ops`
- Tensor sizes: `tensor_sizes` list
- Warmup iterations: `BenchmarkTimer(warmup_iters=...)`
- Measurement iterations: `BenchmarkTimer(measure_iters=...)`

### Running Subset of Tests

```bash
# Skip long-running tests
python benchmarks/run_all_benchmarks.py --skip-attention --skip-mps

# Only micro-batch
python benchmarks/bench_microbatch.py
```

### Exporting Results

All results saved as JSON and CSV in `benchmark_results/`:
- Easy import into spreadsheets
- Compatible with pandas/numpy
- Reproducible data for papers

## Citation

If you use these benchmarks in your research, please cite:

```bibtex
@article{gpuos2024,
  title={GPUOS: A Persistent GPU Kernel Runtime with Dynamic Operator Injection},
  author={[Authors]},
  journal={[Venue]},
  year={2024}
}
```

## Contributing

Found issues or want to add benchmarks?
- Open an issue on GitHub
- Submit a pull request
- Contact: [email]
