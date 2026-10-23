# GPUOS experiments and historical benchmarks

For the revised paper, use the [controlled native CUDA experiment](../experiments/README.md). It compares identical operations across eager CUDA, serial and parallel CUDA Graphs, one-launch batching, and resident workers. It records raw samples and provenance and checks every output.

```bash
python experiments/run_dispatch.py --out benchmark_results/my_clean_run
python experiments/summarize_dispatch.py benchmark_results/my_clean_run
```

The existing `bench_*.py` scripts are historical prototypes. Their saved results do not substantiate the former headline speedups, attention gains, MPS/MIG coexistence, or energy claims. See [the paper evidence audit](../paper.md#why-the-previous-headline-results-were-removed).

Known issues that require repair before using those scripts for publication:

- Microbatch methods generate different random inputs and compare non-equivalent compiled and eager computations.
- CUDA event timing on the default stream is not a general end-to-end completion protocol for a worker on another stream.
- The extension's ring publication, asynchronous tensor lifetimes, stream ordering, and dependency handling need further validation.
- Simulated MIG scales a baseline time arithmetically; it does not run a hardware partition.
- Saved MPS records do not contain an MPS-enabled configuration and mix timing scopes.
- The saved attention summary and JSON/CSV contain different values; no run identity reconciles them.

Do not treat the historical charts as new measurements, automatically start or stop MPS services, or reconfigure MIG to reproduce the native dispatch experiment. The native experiment does not change GPU configuration.
