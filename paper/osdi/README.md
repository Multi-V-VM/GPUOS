> The requested manuscript entry point is now corrected to
> [IPDPS main.tex](../../69e5c13c57662528991eb76c/main.tex).
> This directory retains the earlier alternate-format revision.

# GPUOS: current architecture manuscript

The current entry point is [main.tex](main.tex), producing
[GPUOS-arXiv.pdf](GPUOS-arXiv.pdf): **12 body pages plus one reference page**,
in the existing two-column USENIX format. It presents a unified GPU operating
system architecture and identifies the implemented component evidence.
Unimplemented services remain architectural proposals; no completed general
OS, arbitrary-kernel compiler, or new hardware preemption primitive is claimed.

Only the previous **Figure 1 and Figure 2** are included: `01_request.pdf`
and `02_workload_boundary.pdf`, now on pages 8 and 9. Other results use tables.
The earlier evaluation main file and PDF are preserved in
[archive/evaluation-20261110](archive/evaluation-20261110/).
Historical IPDPS sources are unchanged.

Build from the repository root with `bash paper/osdi/build.sh`. This compiles
retained evidence without launching experiments. `--refresh-evidence` also
regenerates numerical macros, tables, and artwork from existing records.
[Portable LaTeX sources](GPUOS-arXiv-source.zip) include only the required
manuscript files, two figures, references, style, and generated numerical inputs.
[Layout verification](layout-check.json) records the page and figure checks.

Text is split across `manuscript/design.tex`, `control.tex`, `evidence.tex`,
`retained-results.tex`, and `outlook.tex`. [Source/claim map](CLAIMS.md) links
original ideas, design extensions, and local evidence. The process sweep was
paused at the user's request; unfinished rounds are not aggregate evidence.

---

## Earlier evaluation revisions

**November 10 — slack scheduling extension:** the current
[evaluation PDF](FuseOS-OSDI-evaluation.pdf) includes
[implementation, protocol, and limitations](slack-scheduling.tex), the generated
[comparison table](slack-table.tex), and [tradeoff figure](figures/08_slack_tradeoff.pdf).
[Three-process raw data and derived results](../../benchmark_results/slack_20261110/RESULTS.md)
cover the streaming proxy and full native Qwen3-0.6B decode replay. The new section
reports the limited model benefit and the stronger non-yielding control as well
as positive proxy results. `experiments/slack_scheduler/analyze.py` audits source
and binary hashes and regenerates the new numbers, table and figure. The main
build script runs both old and new figure pipelines. Historical IPDPS sources
and PDFs remain unchanged. Earlier page counts below describe the October revision.

**October 28 — redrawn IPDPS Figure 2:** the
[OSDI-format full manuscript](GPUOS-OSDI-from-IPDPS.pdf) retains the IPDPS text
and replaces its system diagram with a two-panel execution/lifetime figure.
Figure 2 is on page 4; the document has 12 body pages and one reference page.
[Vector artwork and editable TikZ](system-figure-20261028/README-system.md)
and [complete Overleaf sources](GPUOS-Overleaf-20261028.zip) are included.
Select `ipdps.tex` as the main document on Overleaf. Other figures are unchanged;
the historical IPDPS measurements retain their original qualification scope.

The [two-page GPUOS KV-runtime note](GPUOS-KV-runtime.pdf) adds shared ancestor
page lifetimes, copy-on-write, bounded GPU/CPU paging, cached NVRTC page commands,
token-boundary priority, and warm weight restoration. Its two figures include
the sufficient-capacity allocation control and the faster cached-graph command
baseline. [Implementation](../../experiments/kv_runtime/README.md),
[three-process evidence](../../benchmark_results/revision_20261025_kv_runtime/RESULTS.md),
and [source/data archive](GPUOS-KV-runtime-artifact.zip) accompany it.
Build with `latexmk -pdf -jobname=GPUOS-KV-runtime kv-runtime-note.tex`.
This is a separate extension; earlier main-paper and recurrence-sharing results
keep their original measurement scopes.

The [two-page GPUOS recurrence KV-sharing supplement](GPUOS-KV-sharing.pdf)
adds the released BeNice 140M implementation, two vector figures, native
numerical qualification, and three-process comparisons of grid, CLC, and
atomic persistent execution. [Raw evidence](../../benchmark_results/revision_20261025_kvsharing/RESULTS.md)
distinguishes KV capacity savings from the mixed scheduling results.
[Source/data archive](GPUOS-KV-sharing-artifact.zip), with an adjacent SHA-256
file, includes the retained numerical pilots and reproducible adapter.
Build the separate note with `latexmk -pdf -jobname=GPUOS-KV-sharing kvsharing-note.tex`
from this directory. It supplements the main evaluation without changing its
earlier Qwen claims.

Open **[FuseOS-OSDI-evaluation.pdf](FuseOS-OSDI-evaluation.pdf)** for the current seven-page revision, or [../../paper.md](../../paper.md) for the repository narrative. This chapter develops the system's dynamism motivation and measured benefit; it is not a completed OSDI submission. Historical IPDPS sources, PDFs, and original figures remain unchanged. The [revision bundle](FuseOS-OSDI-revision-20261025.zip) includes this paper, all plotted raw inputs, plotting/validation scripts, and the private corrected source and binaries; its SHA-256 is in the adjacent `.zip.sha256` file.

The current positive application result is **17.7% / 18.2% lower complete-request latency** for Qwen3-0.6B at P128/B4/H32 and H64, respectively, against native shape-cached execution. The same fused handlers with cached CLC+PDL are faster still. Placement/partitioning results and the 35B boundary are reported separately.

## Figures and exact sources

All figures are available as vector PDF, editable-text SVG, and 240-dpi PNG under [figures](figures/). The numbers below are manuscript figure numbers; filename prefixes remain stable.

| Figure | Vector file | Evidence and plotted statistic |
|---|---|---|
| 1. Complete requests | [01_request.pdf](figures/01_request.pdf) | `revision_20261025_osdi/request_round{1,2,3}.json`; median of process medians and min–max range; five requests/process |
| 2. Workload boundary | [02_workload_boundary.pdf](figures/02_workload_boundary.pdf) | `revision_20261025_osdi/replay_round{1,2,3}.json`; median and range of round-indexed native/executor ratios |
| 3. Mechanism controls | [03_mechanisms.pdf](figures/03_mechanisms.pdf) | `revision_20261025_cooperation/full_r{0,1,2}_w340.jsonl`; matched placement, partitioning, and phase-fusion ratios |
| 4. Worker budget | [05_worker_budget.pdf](figures/05_worker_budget.pdf) | cooperation `worker_sensitivity.csv` and corresponding raw sensitivity runs; B128/trace 1, 512-key jobs |
| 5. Transport | [04_transport.pdf](figures/04_transport.pdf) | cooperation full runs and `profiler_summary.csv`; nine trace ratios and their geometric mean; separate profiler diagnostic |
| 6. Larger model | [06_qwen35.pdf](figures/06_qwen35.pdf) | `revision_20261024_fuse35/accepted/*_r{1,2,3}.json` and audited summary; median and range of paired ratios |
| 7. Fusion boundary | [07_fusion_boundary.pdf](figures/07_fusion_boundary.pdf) | 35B accepted native/three-stage runs and five-stage ablation; complete-request process medians |

Paths in this table are relative to `benchmark_results/`. The [source manifest](data/source_manifest.json) records every input read by the figure pipeline with SHA-256. The [figure manifest](data/figure_manifest.json) records physical dimensions and output hashes. [claims.json](data/claims.json), [request_summary.csv](data/request_summary.csv), [replay_summary.csv](data/replay_summary.csv), [mechanisms.csv](data/mechanisms.csv), and [transport.csv](data/transport.csv) expose the plotted/quoted values. `numbers.tex` is generated from the same records as the figures.

## Rebuild without running the GPU

From the repository root, using the existing Python environment (NumPy, Matplotlib, Pillow) and TeX Live/latexmk:

```bash
bash paper/osdi/build.sh
```

Or, with an equivalent Python environment:

```bash
python paper/osdi/scripts/figures.py
cd paper/osdi
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=FuseOS-OSDI-evaluation main.tex
```

The plotting pipeline checks numerical qualification, request-path agreement, sample counts, the accepted cooperation binary hash, and original 35B records before plotting. It excludes quarantined/pre-fix cooperation records and unqualified pilot runs. It never launches a GPU benchmark.

## Reproduce the repaired Qwen3-0.6B measurements

See the [fresh evidence README](../../benchmark_results/revision_20261025_osdi/README.md). [requalify.py](scripts/requalify.py) builds a private source copy under `build/osdi_requalification/workspace`, inserts one warp barrier before the attention denominator is overwritten, and rebuilds both full-model variants plus ordered attention. It leaves previous source snapshots and binaries intact. It refuses to overwrite an existing private workspace; individual benchmark runners refuse to overwrite result files. For another run, use a fresh checkout/output directory or explicitly archive the existing workspace/results first.

```bash
build/numerics-venv/bin/python paper/osdi/scripts/requalify.py
bash paper/osdi/scripts/validate.sh
bash paper/osdi/build.sh
```

GPU reproduction needs the existing local Qwen3-0.6B cache at revision `c1899de289a04d12100db370d81485cdf75e47ca`, CUDA 13.1, the repository's CUDA compatibility-header overlay, PyTorch/Transformers dependencies, and an available compatible GPU. `validate.sh` also expects the local Compute Sanitizer 13.4.92 installation. These scripts describe this artifact's tested environment; model weights, toolkit, Python environment, and profiler installations are not redistributed in the revision bundle.

Qualification uses finite logits, NRMSE ≤ 0.01, and cosine ≥ 0.999 at every tested frontier, with final paths checked independently. Racecheck targets rebuilt ordered attention; memcheck/synccheck target selected full-model frame kernels. See the logs for filter scope; these are not an exhaustive certification of the entire runtime.

## Publication layout

The revision uses the unmodified official [USENIX template](https://www.usenix.org/conferences/author-resources/paper-templates), with sources and hashes in [template_provenance.json](template_provenance.json). The [OSDI 2027 author requirements](https://www.usenix.org/conference/osdi27/requirements-authors) and [call for papers](https://www.usenix.org/conference/osdi27/call-for-papers) were checked on 2026-10-25.

The layout is US letter, two columns, 10-point Times body on 12-point leading, a 7×9-inch text block, and a 0.33-inch gutter. Submission page numbers are enabled. This seven-page chapter is within the stated 12-page body limit, but does not replace a complete introduction/design/related-work manuscript.

Plots are authored at 7-inch double-column or 3.335-inch single-column width. Minimum authored label size is 9.2 points, our readability choice rather than an asserted conference minimum. Figures have embedded fonts, white backgrounds, zero-origin absolute bars, explicit ratio baselines, and redundant hatches/markers/line styles for grayscale. Process ranges are labeled as ranges, never confidence intervals. Different timing scopes and ablation populations are explicit in captions. [layout_qa.json](layout_qa.json) records the local checks; no conference upload checker was run.

## Evidence boundaries and history

Complete requests include fresh prefill and final decoding, while equal-work replay starts from a prefilled ancestor. Weights, tokenized inputs, executor buffers, and graph caches are warm. Host policy remains timed; EOS is excluded by the bounded-search benchmark. This is output-driven model search, not a tool-agent success test. ShareGPT supplies attention-control lengths, not an end-to-end serving trace. 35B uses a native mixed-Q4 reference and has explicitly reported custom-kernel coverage.

The earlier [paper.md snapshot](data/paper_pre_osdi_20261025.md) is preserved verbatim. Its relative links refer to the original repository-root location. Historical attention-based data predates the warp fix and must not be pooled with the fresh 0.6B records. Older MPS/MIG or ResNet observations are not newly qualified here. The manuscript retains one drafting disclosure and the MPK/Meld comparison.
