# Claim and source map

This map accompanies `main.tex`. Architectural requirements describe the
proposed system; only explicitly identified prototypes have implementation or
measurement claims. The original source is Nourai and Bastani,
[The GPU Needs an Operating System](https://arxiv.org/abs/2611.08720).

| Source idea | Manuscript location | Treatment |
|---|---|---|
| Decode roofline; expert-local batch; structural idle arithmetic | §2.1 | Derivation plus attributed OpenInfer measurements; distinct from our device |
| Stream priority, finite block lifetime, spatial partition limits | §§2.2–2.3, 4.5, 7.7 | Documented contracts and local context-switch diagnostic |
| Resource vector and workload-dependent prices | §3.3, §5.2 | Architecture, units, logical/physical accounting, allocation-only controls |
| Six-layer OS and slack admission | §§3.1, 5 | Full architecture; capability table distinguishes prototype coverage |
| Compiler-generated resumable slices and resource declarations | §4 | Strict FX/Triton subset; no arbitrary CUDA transformation claim |
| Resident model service and phase-boundary background roles | §4.4 | Attributed source design; distinguishes our frontier-level executor |
| Memory leases, cache, bandwidth, links, copies, power | §6 | Proposed governance with bounded implemented mechanisms identified |
| Quantile budgets, multi-rank tail and phase-switch overhead | §§5.1, 6.5, 8.3 | Statistical scope and distributed design, no multigpu measurement claim |
| Synthetic/real background and useful-work denominators | §§7.4–7.5 | Qualified local controls; custom GEMM solo rate is not GPU peak |
| Three hardware requests and supporting services | §§8.1–8.4 | Bounded pause, memory QoS, enforced partitions; proposed interfaces |
| Model/compiler/hardware co-design | §8.5 | Expert balance and attention regularity; no unmeasured speedup |

## Evidence anchors

- Retained Figures 1–2 are original design artwork: `img/gpuos_persistent_jit_tikz.pdf`
  and `osdi/figures/system-overview.pdf`, copied byte-for-byte to
  `gpuos/figures/01_system_architecture.pdf` and `02_execution_storage.pdf`.
  Their captions distinguish architecture from measured implementation.
- Qualified complete-request and search-replay numbers retain the data provenance
  documented by `../benchmark_results/revision_20261025_osdi/README.md` and
  `../paper/osdi/scripts/figures.py`; they now appear in text and tables.
- Slack comparison table, selection and worker diagnostics:
  `../benchmark_results/slack_20261110/RESULTS.md`, `verification.json`,
  `accepted/` records, and `../experiments/slack_scheduler/analyze.py`.
- Context-switch table and experiment:
  `../benchmark_results/process_compiler_20261110/accepted/preemption.json`;
  20 trials per mode with verified state and 170 distinct SM identifiers.
  Host latency includes submission and driver scheduling, not isolated save cost.
- Compiler correctness: `../benchmark_results/process_compiler_20261110/accepted/validation/`.
  Seven tests and three bounded sanitizer runs; not a full-model compiler evaluation.
- Process throughput rounds: retained under the same results directory, but
  paused before the planned repeated suite finished. No aggregate performance
  claim in this paper is derived from those partial rounds.

The main manuscript has two figures and five tables. `gpuos-layout-check.json`
records PDF layout, original figure hashes, and bibliography checks.
No new GPU benchmark is required to compile the manuscript.

## November 12 design additions

- §3.2 specifies logical prepare/grant/revoke/ack/retire operations and a
  generation-tagged lease. These are integration requirements, not a claim
  that the components expose one complete API.
- §4.3 derives a checkpoint feasibility interval under a serialized finite-slice
  model. Its costs are parameters, not new GPU measurements.
- §5.1 distinguishes joint-tail validation from adding component p99 values.
- §5.5 gives the admission procedure and a conditional safety argument;
  §5.6 distinguishes orderly checkpointing from crash-safe recovery.
- §9 cites primary publications for REEF, XSched, Salus, and Gandiva.
  They are related work, not measured baselines in this artifact.
- `DESIGN-CONTRACT.md` expands the interfaces, assumptions, failure cases, and
  tests needed for future integration. No new GPU experiment was run.
- `gpuos/build.py` checks the two original figure hashes, author list,
  bibliography, exact page count, and self-contained package compilation.
