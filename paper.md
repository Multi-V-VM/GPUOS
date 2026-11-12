# GPUOS: An Operating-System Architecture for Shared GPU Execution

The authoritative entry point is the **IPDPS manuscript
[main.tex](69e5c13c57662528991eb76c/main.tex)**, in the original IEEEtran format
with the existing author list. [Current PDF](69e5c13c57662528991eb76c/FuseOS-IPDPS-2027.pdf)
has **12 body pages plus one reference page**.
[Portable LaTeX package](69e5c13c57662528991eb76c/FuseOS-IPDPS-source.zip)
and [build instructions](69e5c13c57662528991eb76c/README.md) accompany it.

The text unifies process/task/lease semantics, compiler safe points,
scheduling, resource governance, admission, and hardware interfaces, with
attribution to [Nourai and Bastani](https://arxiv.org/abs/2611.08720).
It preserves the original GPUOS system-architecture and execution/storage
contract diagrams as Figures 1–2; measurements appear in text and tables. Existing measurements and proposed services remain distinct.
Implementation and additional GPU experiments remain paused.

The November 12 revision strengthens the system contract: versioned grants,
resource-return acknowledgements, conditional safety invariants, checkpoint
feasibility, joint-tail admission, and failure handling. The
[design companion](69e5c13c57662528991eb76c/DESIGN-CONTRACT.md) documents the
full protocol and future validation obligations. Four additional primary
related-work references clarify the contribution relative to existing GPU
sharing and preemption systems.

The previous update targeted `paper/osdi/main.tex` by mistake. That file is
retained as an alternate-format revision; the IPDPS `main.tex` above is now
the corrected target. Its earlier source and PDF were backed up locally before
replacement. [Claim provenance](69e5c13c57662528991eb76c/CLAIMS.md) documents
which evidence supports the architecture.

---

## Earlier implementation and evaluation notes

The previous [OSDI-format benefits and evaluation revision](paper/osdi/FuseOS-OSDI-evaluation.pdf) develops the **dynamism** motivation: changing live requests, ready work, and storage lifetimes inside a bounded inference request. The [LaTeX text](paper/osdi/evaluation.tex), [seven vector figures and reproduction instructions](paper/osdi/README.md), and [fresh numerical/timing evidence](benchmark_results/revision_20261025_osdi/README.md) are maintained together. This is an evaluation revision for the systems paper, not a completed OSDI submission.

## System objective and implemented design

### November 10 extension: slack-budgeted background execution

The [cooperative runtime](experiments/slack_scheduler/README.md) implements
checkpointed BF16 background GEMMs, stream-ordered foreground urgency,
per-worker issue pacing, and empirical admission under a 6% p99 budget,
inspired by [Nourai and Bastani](https://arxiv.org/abs/2611.08720).
It is a trusted single-process extension with manually written kernels;
sleeping CTAs keep their residency. Hardware preemption, memory QoS,
compiler-generated checkpoints and phase integration into the fused model
executor remain outside this implementation.

Three independent processes per workload retain **154,368 timed foreground
steps**, including calibration and adjacent standalone references. Selection
uses calibration samples only; all nine selected configurations pass held-out
validation. [Full results and raw records](benchmark_results/slack_20261110/RESULTS.md)
include failed configurations and source/binary hashes. The new
[LaTeX section](paper/osdi/slack-scheduling.tex) and generated table/figure are
included in the current OSDI PDF.

| Workload and background policy | p99 cost | Background TFLOP/s | Fraction of best custom solo |
|---|---:|---:|---:|
| Streaming GEMV proxy, 100-us gaps; yield q4/w64 | +0.62% | 0.741 | 6.507% |
| Same proxy; selected yield q64/w64 | +2.50% | 1.167 | 10.213% |
| Same proxy; no yield q4096/w64 | +23.68% | 4.359 | 38.252% |
| Proxy, no intentional gap; selected yield q64/w64 | +3.48% | 0.374 | 3.276% |
| Qwen3-0.6B replay; selected yield q64/w64 | −3.04% | 0.039 | 0.342% |
| Qwen3-0.6B replay; no yield q4096/w64 | −1.57% | 8.779 | 76.999% |

Values are medians of paired process ratios; ranges are in the full table.
`q` denotes K tiles per warp between checkpoints and `w` denotes CTAs, not
reserved SMs. The denominator is the fastest measured **custom GEMM** solo
configuration, not GPU peak. Negative costs are observations under unlocked
clocks, not claimed scheduler-induced speedups. Whole-step suspension protects
the streaming workload but leaves almost no useful background progress on
this Qwen workload; ordinary co-running is a stronger performance control there.

Qwen uses all 28 native layers plus the vocabulary head with cached graphs,
P128, up to four branches, and a fixed 32-step native search trace. Prefill,
capture, allocation, and search scoring are excluded. Final-replay logits
match the native reference bitwise. Bounded worker runs pass memcheck,
racecheck and synccheck, while an initial live-worker sanitizer attempt stalled;
these checks do not certify the full concurrent runtime. Separate flag-to-ack
p99 probes measure 10.69/21.85/738.68 us for q4/q64/q4096. This is not a
reproduction of OpenInfer's Kimi, FP8, green-context, MPS, or multi-GPU results.

### Existing dynamic inference executor

In output-driven search, model scores determine which branches survive, which histories are copied, and the next frontier's active width. A fixed model architecture therefore does not determine the runtime's execution plan. FuseOS combines bounded storage plans, fused operator handlers, and GPU work assignment to execute this changing workload.

The Qwen3-0.6B implementation keeps immutable ancestor KV and two private-history banks. Each frontier copies selected histories into the alternate bank, appends KV, and swaps banks. Physical capacity stays reserved while branch identities and active work bounds change. The persistent executor runs all 28 decoder layers and the vocabulary head in one cooperative launch **per frontier**. GPU counters assign ready tiles within a phase; barriers order dependent phases. Host branch scoring and metadata publication remain inside complete-request timing.

The FuseOS persistent path uses no CUDA graph. Matched tile-grid, PDL, and CLC executors reuse the fused handlers; cached graphs are retained as strong competing implementations. This separates the benefit of the fused implementation from the incremental value of its execution protocol.

## Real complete-request benefit

The current headline uses fresh measurements after repairing the inherited attention warp-ordering hazard. Three independent processes each run five timed requests per cell, with unchanged native numerical gates. The request freshly prefills a 128-token ancestor and performs output-dependent search with at most four live branches.

| Executor | 32 search steps (ms) | 64 search steps (ms) |
|---|---:|---:|
| Native shape-cached graphs | 143.293 | 274.924 |
| FuseOS persistent | **117.889** | **225.005** |
| Matched CLC+PDL, shape-cached | 113.184 | 215.208 |
| Matched CLC+PDL, fixed-topology graph | 121.330 | 231.966 |

FuseOS reduces median complete-request latency by **17.7% and 18.2%** relative to native shape-cached execution, or **1.215× and 1.222×** speedup. These are ratios of the median process latencies. All methods produce the same final candidate paths. Maximum recorded full-vocabulary NRMSE is **0.007762**, below the unchanged 0.01 gate; the cosine threshold remains 0.999.

![Complete-request latency](paper/osdi/figures/01_request.png)

The interval includes fresh prefill, ancestor-KV publication, all model layers and search frontiers, branch policy, private-history copies, final D2H/text decoding, and synchronization. Weights, tokenized inputs, executor buffers, and graph caches are resident. Stage medians are not added to construct this interval. Whiskers show the minimum and maximum of three process medians; they are not confidence intervals or request-tail percentiles.

The positive application result belongs to the complete fused dynamic executor. Cached CLC+PDL remains faster than persistence with the same handlers, so these results do not attribute the entire improvement to persistent scheduling. They also show why a graph baseline must be allowed to reuse previously encountered shapes.

## Where the benefit changes

A separate equal-work full-model replay holds native-generated parent/token choices constant while every backend still executes scoring. It starts from a ready prefilled ancestor and therefore has a different timing boundary from the complete requests above.

![Workload-dependent full-model benefit](paper/osdi/figures/02_workload_boundary.png)

The narrow case P128/B4/H32 favors the fused persistent implementation over native cached execution. P512/B16/H16 and P1024/B8/H32 do not retain that advantage, although persistence remains faster than the matched tile-grid and tile+PDL paths. All three replay cases pass the original numerical gate; maximum NRMSE is 0.008308. These configurations change several workload dimensions together and establish an observed applicability boundary, not a causal threshold for any one dimension.

## Allocation, fusion, and cross-SM transport

To distinguish runtime benefit from an imbalanced starting schedule, the [post-fix split-KV controls](benchmark_results/revision_20261025_cooperation/RESULTS.md) independently vary placement, task partitioning, phase fusion, and transport. ShareGPT supplies prompt lengths; Q/K/V are seeded BF16 tensors. These are attention-operator controls, not end-to-end ShareGPT serving.

![Placement, partitioning, and phase fusion](paper/osdi/figures/03_mechanisms.png)

Across nine prompt-length batches, replacing round-robin placement with LPT for identical 512-key jobs yields **1.234×** geometric-mean speedup, up to **1.925×**. Repartitioning an already sorted ordinary grid from four parts into 128-key jobs yields **1.843×**, up to **3.988×**. Both exploit input lengths known before dispatch. The latter also changes reduction partitioning.

With task geometry and compiled resources held constant, matched CLC phase fusion yields **0.994×** geometrically. Thus the large partitioning improvement cannot be presented as a persistent-fusion gain, and the separate ratios cannot be multiplied. This gives a concrete design requirement: a runtime must first expose enough well-sized ready work; removing a launch is valuable only if completion, resource, and duplicated-work costs permit it.

![Worker-budget sensitivity](paper/osdi/figures/05_worker_budget.png)

For B128/trace 1 with identical 512-key jobs, increasing the persistent budget from 170 to 680 CTAs improves staged LPT from 206.61 to 141.16 µs. Fused LPT takes 215.40 to 142.20 µs; the sorted grid stays around 137 µs. More workers help scheduling but do not make phase fusion profitable in this case.

![Matched transport and memory counters](paper/osdi/figures/04_transport.png)

Four-CTA TMA exchange yields **1.008×** geometric-mean speedup over matched global-memory exchange. A separate counter diagnostic observes lower L2 traffic but similar DRAM reads. It is a synchronized cluster-state exchange, not migration of a live CTA's registers/program counter and not a reproduction of Meld's asynchronous pipeline. Meld already includes input-aware scheduling; this experiment does not show that all of Meld's benefit is an allocation bug.

## Larger-model boundary

The [Qwen3.5-35B-A3B port](benchmark_results/revision_20261024_fuse35/RESULTS.md) runs custom routed-MoE handlers for microbatches 1–8 across all 40 routed layers and a decode path for 30 DeltaNet layers. Attention, prefill, shared experts, and dense projections remain native. The B16 ragged case has 26.47% custom-MoE token-layer coverage; the other five cases have 100% eligible coverage.

![Complete 35B requests](paper/osdi/figures/06_qwen35.png)

The default three-stage persistent path does not beat native eager across the six evaluated workloads. Native/persistent ratios of median latency range from 0.909× to 0.980×. All 9,792 recorded frontier comparisons have zero NRMSE against the native mixed-Q4 model, and all 108 free-path comparisons match. This is a measured performance boundary of a numerically compatible port.

![Selective fusion ablation](paper/osdi/figures/07_fusion_boundary.png)

Separating hidden quantization, down projection, and expert reduction while retaining persistent scheduling reduces the tree request from 721.11 to 682.92 ms and the ragged request from 554.12 to 547.39 ms. The former is close to native's 688.90 ms but does not establish a robust win across runs. The profiler records fewer launches but more custom-MoE GPU time with three stages. The two-case ablation supports choosing fusion boundaries by duplicated work and available parallelism, rather than minimizing kernel count alone.

## Contribution supported by this revision

FuseOS provides a concrete dynamic execution path that improves complete small-Qwen search requests under narrow-frontier workloads while preserving the measured native numerical contract and final paths. The experiments identify known-input work partitioning as a prerequisite and expose the limits of additional phase fusion and transport optimizations.

Persistent execution alone is not novel: [MPK](https://www.usenix.org/conference/osdi26/presentation/cheng) already supplies a persistent compiler/runtime, and [Meld](https://doi.org/10.1145/3830418.3843857) combines input-aware scheduling with cross-SM cooperation. A stronger distinct contribution requires demonstrating an advantage when ready work or storage liveness becomes known during execution, against competitors with the same prior information. The current evidence does not establish general allocation, continuous admission, or replacement of MIG's isolation. Historical unqualified MPS measurements are not used for the present application claim. Bounded model search is also separate from tool-using agent success or ShareGPT service throughput.

## Evidence and historical versions

The [current evidence record](benchmark_results/revision_20261025_osdi/README.md) contains 150 fresh complete requests and 315 equal-work replay timings, lifecycle checks, source/binary hashes, and sanitizer logs. The repaired ordered-attention path passes racecheck; selected full-model frame kernels pass memcheck and synccheck. These checks have explicit filters and do not certify all kernels in the framework.

The [previous Markdown manuscript](paper/osdi/data/paper_pre_osdi_20261025.md), [frozen IPDPS PDF](69e5c13c57662528991eb76c/FuseOS-IPDPS-2027.pdf), and [original-figure comparison](69e5c13c57662528991eb76c/FuseOS-original-figures.pdf) remain preserved. Earlier attention-based snapshots predate the warp-ordering repair and require requalification before reuse; the new figures use fresh accepted data or the separately qualified 35B path. Historical ResNet and other operator experiments remain linked from the original manuscript and repository README rather than being presented as newly rerun results.

AI-assisted tools were used for code edits, experiment orchestration, prose revision, and plotting. The current revision retains raw observations and reproducible transformations for author review.
