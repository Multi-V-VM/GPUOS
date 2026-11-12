# GPUOS manuscript — IPDPS main.tex

[main.tex](main.tex) is the authoritative manuscript entry point.
[Current PDF](FuseOS-IPDPS-2027.pdf) · [Portable source package](FuseOS-IPDPS-source.zip)

The current manuscript is **GPUOS: An Operating-System Architecture for Shared
GPU Execution**, in the existing IEEEtran conference format: **12 body pages
plus one reference page**. The original six-author list is preserved.
This is an architecture paper with explicitly scoped component evidence;
proposed services are not described as completed implementations.

The two retained figures are the original **GPUOS system architecture** and
**GPUOS execution and storage contract** diagrams, copied byte-for-byte into
`gpuos/figures/01_system_architecture.pdf` and `02_execution_storage.pdf`.
They remain Figures 1 and 2; the current page positions are recorded in
`gpuos-layout-check.json`. Qualified request/replay
measurements are presented in text and tables, not replacement figures.
The manuscript text and generated numerical inputs are under `gpuos/`.
Additional scheduling and failure-handling semantics are in
`gpuos/manuscript/control.tex`. [Claim provenance](CLAIMS.md) distinguishes
source ideas, architecture, and local measurements.

The November 12 design revision adds a five-operation lease API, generation
and resource-return semantics, three safety invariants with a conditional
argument, checkpoint feasibility constraints, statistical admission limits,
and explicit comparisons with REEF, XSched, Salus, and Gandiva.
[Detailed design companion](DESIGN-CONTRACT.md) expands the contracts,
crash-recovery requirements, and future integration validation cases.
It does not report those future cases as completed experiments.

Build and validate from this directory:

```bash
python3 gpuos/build.py --package --verify-package
```

The source package is self-contained; select `main.tex` in Overleaf. It needs
no GPU, model download, Python environment, or experiment rerun.
`gpuos-layout-check.json` records pagination, figure hashes, and reference checks.
Twelve pages is the requested manuscript length, not a claim about conference
submission eligibility.

The previous main file and PDF were preserved in
`archive/pre-gpuos-20261111/` before editing. Other historical source files
are unchanged and are not included by the current main file.
The earlier `paper/osdi/main.tex` revision remains an alternate-format artifact;
it is not the target entry point for this correction.
