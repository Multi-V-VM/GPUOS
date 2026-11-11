#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == "--refresh-evidence" ]]; then
  build/numerics-venv/bin/python paper/osdi/scripts/figures.py
  build/numerics-venv/bin/python experiments/slack_scheduler/analyze.py > build/slack_scheduler/paper-analysis.log
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--refresh-evidence]" >&2
  exit 2
fi
cd paper/osdi
latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=GPUOS-arXiv main.tex
