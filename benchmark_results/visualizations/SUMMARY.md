# GPUOS Benchmark Results Summary

Generated: 2025-10-25 03:56:14

## Attention Results

### attention_128_128

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| PyTorch Naive             |    4.615 |   0.048 |   4.562 |   4.818 |
| FlashAttention            |    7.516 |   0.034 |   7.438 |   7.606 |
| GPUOS Scheduled           |   21.606 |   0.284 |  21.116 |  22.424 |

**Speedups vs Baseline:**

- FlashAttention: **0.61×**
- GPUOS Scheduled: **0.21×**

### attention_256_128

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| PyTorch Naive             |    4.551 |   0.025 |   4.515 |   4.639 |
| FlashAttention            |    7.280 |   0.029 |   7.185 |   7.357 |
| GPUOS Scheduled           |   21.674 |   0.312 |  20.944 |  22.343 |

**Speedups vs Baseline:**

- FlashAttention: **0.63×**
- GPUOS Scheduled: **0.21×**

### attention_512_128

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| PyTorch Naive             |    4.542 |   0.025 |   4.460 |   4.599 |
| FlashAttention            |    7.339 |   0.037 |   7.271 |   7.507 |
| GPUOS Scheduled           |   21.733 |   0.971 |  20.964 |  26.929 |

**Speedups vs Baseline:**

- FlashAttention: **0.62×**
- GPUOS Scheduled: **0.21×**

## Simple Baseline

