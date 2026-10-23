# FuseOS: Fused Persistent Execution for Dynamic Inference

This companion matches the [current IPDPS manuscript](69e5c13c57662528991eb76c/FuseOS-IPDPS-2027.pdf): **10 body pages + 1 reference page**, anonymous IEEE conference format. The contribution is a systems design for **dynamism**: changing live requests, ready work and storage lifetimes inside an admitted inference batch.

## Implemented design

After native prefill, a cooperative persistent kernel executes every Qwen3-0.6B decoder layer and remaining greedy token. The GPU retires requests at their generation limits or actual EOS, compacts stable request IDs, assigns ready tiles, and reuses scratch at last-consumer phase boundaries. FuseOS itself uses no CUDA Graph. Conditional graphs, PDL and CLC remain strong matched controls.

Paired gate/up projections and vertical SiLU fusion remove a phase and a materialized tensor. The transient global plan is **12 KiB per live request**, compared with 24 KiB for separate gate/up and SiLU. Phase-local shared storage is overlaid across nonoverlapping handlers. Weights, KV, logits and residual state are outside that scratch accounting. Physical reservations stay fixed; this is not general device malloc, KV-page reclamation or continuous admission.

For trusted compatible requests, the common ready-work pool provides an alternative to separate MPS clients or instance-based placement. It does **not** provide MIG memory/fault isolation. The RTX 5090 has no measured MIG configuration.

## Completed experiments

| Suite | Formal retained evidence | Timing boundary |
|---|---:|---|
| Fusion × persistent / tile+PDL / CLC+PDL and ablations | 138 cells, **1,380 requests**, 3 process rounds | Complete local request, plus decode breakdown |
| Identical-instruction shared reservation | **120 decode trials**, 3 process rounds | Complete decode; prefill outside timing |
| Four real MPS clients versus pooled execution | **300 four-request groups**, 3 rounds per case | Common post-prefill release through D2H/text |
| ResNet-18 boundary case | Retained earlier complete-network suite | Full model and local request |

Every primary cell has three warmups and ten timed requests. Figures report the median of three process medians and their range, not confidence intervals. Ratios are paired within a process. Frozen sources, binary hashes, input identities, numerical failures, raw records, profiler counters and checks are retained in [the new evidence directory](benchmark_results/revision_20261004_fusion).

## What the fusion experiments establish

| Complete decode (ms) | Uniform | Ragged | Actual EOS | Long |
|---|---:|---:|---:|---:|
| FuseOS fused persistent | 75.7 | 78.9 | 254.0 | 210.2 |
| Separate-MLP persistent | 96.4 | 101.6 | 327.4 | 263.5 |
| Fused tile-grid + PDL | 86.6 | 93.0 | 305.7 | 253.8 |
| Separate tile-grid + PDL | 101.0 | 111.9 | 376.7 | 318.8 |
| Fused CLC + PDL | 88.1 | 98.3 | 334.4 | 287.2 |
| Separate CLC + PDL | 86.1 | 92.6 | 303.6 | 252.6 |

Allowing the launch controls either MLP variant, their best within-process latency divided by FuseOS has median **1.139, 1.173, 1.194 and 1.198**. All matched controls produce bitwise-identical custom tokens and logits. Fusion can lose under CLC+PDL, so the evidence supports choosing fusion and execution together, not a universal fusion speedup.

The most discriminating controls are:

- Independent static shared arrays slow Uniform/Ragged to **106.9/116.0 ms**. The overlay implementation uses 21,856 rather than 36,704 shared bytes per block and changes compiler register allocation.
- Increasing only the launch-time reservation of the **same device instructions** to the larger footprint leaves Uniform/Ragged at **75.59/78.75 ms**, versus **75.60/78.75 ms** originally. SASS identity is verified. Extra shared capacity alone therefore does not explain the slower separate-array build.
- Actual active warps are about **16.7%** for all three profiled variants, and DRAM read traffic is similar. Generated execution and some L2 traffic differ; no complete single-counter explanation is claimed.
- A one-time fixed global region plan matches per-token rebuilding, with the same reuse and active work. Ragged region assignments drop from 62 to 2; latency is essentially unchanged.
- Cyclic claiming is slightly faster than atomic claiming on the tested dense tiles. Both use dynamic active sets and work bounds.

Selected fused decodes pass **memcheck and synccheck**. Additional tests pass for odd contexts, unequal retirement, fixed/rebuilt plans, exhausted arenas with untouched guards, and rejection of oversized cooperative grids. Profiler timings are not performance samples.

## Real MPS comparison

The four MPS processes share model weights through CUDA IPC, have private KV/scratch, and restore the same native prefill checkpoint as the pooled controls. All outputs/logits match bitwise. MPS server client lists are retained. A 25% active-thread cap is a soft provisioning control, not a MIG simulation.

| Post-prefill group makespan (ms) | Ragged | Actual EOS |
|---|---:|---:|
| FuseOS pooled persistent | 79.0 | 254.9 |
| Pooled CLC + PDL | 105.3 | 412.9 |
| Four MPS tile + PDL clients | 119.6 | 332.4 |
| Four MPS CLC + PDL clients | 115.0 | 353.5 |
| Four capped MPS CLC + PDL clients | 124.0 | 415.0 |

The paired ratios against MPS CLC on Ragged and MPS tile on EOS are **1.455× and 1.307×**. EOS pooled persistence varies from **254.4 to 336.8 ms** across process medians; one round is slower than MPS tile. Pooled CLC also varies substantially. The paper retains these ranges and makes no sustained-service or tail-latency guarantee. Startup, checkpoint restore and prefill are outside this sharing interval; it is distinct from the full-request benchmark.

## Native model compatibility and end-to-end performance

Native reference generation agrees with Hugging Face. All new BF16 paths use FP32 GEMM accumulation; framework reduced-precision BF16 reductions are disabled. Operator-local diagnostics resolved a substantial accumulation mismatch, but the full-model gate remains mixed.

| Qwen case | Native graph request | FuseOS request | Same-prefix NRMSE | Free greedy agreement | Gate |
|---|---:|---:|---:|---:|---|
| Single | 235.7 ms | 184.1 ms | 0.934% | 100% | Pass |
| Uniform | 131.8 ms | 92.1 ms | 1.073% | 84.4% | Fail |
| Ragged | 145.8 ms | 96.0 ms | 1.011% | 93.8% | Fail |
| Actual EOS | 455.2 ms | 275.3 ms | 1.181% | 58.2% | Fail |
| Long | 366.8 ms | 239.4 ms | 0.965% | 95.2% | Pass |
| Skew-8 | 315.2 ms | 186.4 ms | 1.191% | 73.8% | Fail |

The fixed gate is finite logits, NRMSE ≤ 1%, cosine ≥ 0.999, using the same reference prefixes. Failed cases do not support accuracy-equivalent native speedups. Long passes the logit gate but does not reproduce every greedy token. Actual EOS lengths differ between native (2/7/31/94) and custom (2/9/31/95), so their native timing workloads also differ. **Single is the strongest qualified native application comparison**.

ResNet-18 remains a complete-network boundary case: only eight residual add–ReLU operations are resident, with convolutions in the framework. At B32, native graph model time is **2.534 ms**, vector graph **2.536 ms**, and resident-32 **3.273 ms**. All numerical/top-1 checks pass. Partial persistence leaves coordination costs and does not establish a whole-network speedup. Batches replicate one processed photograph.

## Relationship to prior persistent systems

MPK already supports device-side retirement, continuous batching, paged KV management and shared-memory pages. FuseOS does not claim these general capabilities as new. Its narrower design uses implicit phase tasks, one compacted active map and last-consumer overlay, trading task queues and cross-phase pipelining for cooperative barriers. The paper compares mechanisms explicitly. A source compatibility probe of public MPK revision `60ae655` rejects CC 120 in its resource helper, and its linear dispatch excludes that architecture. This is not a full MPK build or timing result. No MPK or Meld latency is inferred from another GPU or model.

## What to refer to in Meld


The supplied [3830418.3843857.pdf](3830418.3843857.pdf) is *Taming Dynamism on GPUs: Cross-SM Kernel Fusion via SM Cooperation and Just-in-Time Reduction*, DOI [10.1145/3830418.3843857](https://doi.org/10.1145/3830418.3843857).

Useful connections are computation/scheduling separation (§4.1), skew-aware work distribution (§4.3), reusable execution wrappers and planning amortization (§4.4), and resource/synchronization contracts (§5). Meld itself uses a persistent GPU wrapper. FuseOS therefore does not claim “persistent kernels instead of Meld” as a novelty distinction. Its present design uses global-memory intermediates and GPU-wide phases; it does not implement Meld's topology-aware cooperation, remote shared-memory transport or just-in-time reduction. Just-in-time reduction is also different from NVRTC compilation.

## TMA, figures and reproducibility

TMA is a separate optional staging experiment and is compressed in the main paper. Its worker-local gains are not multiplied into Qwen or ResNet results. Original attention/MPS/MIG illustrations remain unchanged in the [original-figure comparison PDF](69e5c13c57662528991eb76c/FuseOS-original-figures.pdf); they are not new empirical evidence. The original four image hashes are checked by the paper build.

Use [the fusion experiment README](experiments/fusion_persistent/README.md) for commands and [paper build instructions](69e5c13c57662528991eb76c/README.md) for the template. Repeated AI-drafting footnotes are absent; required affected-section citations and the acknowledgment remain. The older paper and evidence are preserved under `build/pre_fusion` and the earlier result directories.
