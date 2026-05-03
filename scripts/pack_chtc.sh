#!/usr/bin/env bash
set -euo pipefail
# From repo root: ./scripts/pack_chtc.sh
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "${HERE}"

STAMP="$(date +%Y%m%d%H%M)"
CODE_TAR="${HERE}/pack_code_${STAMP}.tar.gz"

tar -czvf "${CODE_TAR}" \
  --exclude=".git" \
  --exclude="venv" \
  --exclude=".venv" \
  --exclude="*.tar.gz" \
  --exclude="runs" \
  --exclude="results" \
  --exclude="__pycache__" \
  --exclude="external_ORCA" \
  orca_rank run_experiment.py third_party configs requirements*.txt README.md htcondor scripts

echo "Wrote ${CODE_TAR}"
cp -f "${CODE_TAR}" "${HERE}/code.tar.gz"
echo "Updated ${HERE}/code.tar.gz (Condor transfer_input_files)"
