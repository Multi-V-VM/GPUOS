# GPUOS Benchmark Results Summary

Generated: 2025-10-23 17:31:51

## Mig Simulated Results

### mig_simulated

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| Full GPU (100%)           |   40.506 |   3.454 |  34.478 |  44.791 |
| MIG 3g.40gb (~42% SMs)    |  106.147 |   4.482 | 101.857 | 122.377 |
| MIG 2g.20gb (~28% SMs)    |  172.259 |  26.816 | 146.376 | 233.843 |
| MIG 1g.10gb (~14% SMs)    |  337.781 |  29.552 | 292.260 | 383.069 |

## Mps Results

### multiprocess_2

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| Sequential (No Sharing)   |   20.444 |   0.000 |  20.444 |  20.444 |
| Concurrent (No MPS)       | 1930.988 |   0.000 |  16.588 |  16.779 |
| MPS Enabled               | 2015.376 |   0.000 |  16.544 |  16.652 |

### multiprocess_4

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| Sequential (No Sharing)   |   30.454 |   0.000 |  30.454 |  30.454 |
| Concurrent (No MPS)       | 2637.326 |   0.000 |  19.783 |  28.853 |
| MPS Enabled               | 2540.357 |   0.000 |  15.858 |  16.240 |

### multiprocess_8

| Configuration | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|---------------|-----------|----------|----------|----------|
| Sequential (No Sharing)   |   56.156 |   0.000 |  56.156 |  56.156 |
| Concurrent (No MPS)       | 3654.528 |   0.000 |  24.418 |  33.878 |
| MPS Enabled               | 3247.577 |   0.000 |  20.803 |  31.118 |

