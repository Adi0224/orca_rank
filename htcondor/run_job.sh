#!/usr/bin/env bash
# HTCondor executable: unpack venv + flattened code tree, run one GPU job.
# Pack `code.tar.gz` from repo root so `run_experiment.py` lives next to `orca_rank/`.
#
# CHTC pools often expose **Tesla P100** (Pascal sm_60). Torch bf16 autocast is **disabled**
# automatically when unsupported; GPU smoke verifies OTDD + one short `orca_otdd` train.
#
# JOB_MODE:
#   full      — production-style run (METHOD, EMBED_EP, etc.)
#   gpu_smoke — minimal `orca_otdd` + skips eval; run once after packaging to validate the pool
set -euo pipefail
WORK="${_CONDOR_SCRATCH_DIR:-$(pwd)}"
cd "${WORK}"

# Append human-readable disk/GPU snapshot (and mirror to stdout). Written under ${OUT}/
# so it is included in orca_rank_gpu_results.tar.gz — use to tune request_disk / pool choice.
log_condor_resources() {
  local label="$1"
  local log_file="$2"
  mkdir -p "$(dirname "$log_file")"
  {
    echo "=== condor_resource_usage: ${label} ==="
    echo "date: $(date -Is 2>/dev/null || date)"
    echo "WORK=${WORK}"
    echo "--- scratch total ---"
    du -sh . 2>/dev/null || true
    echo "--- large dirs (du -sh) ---"
    du -sh venv hf_cache runs 2>/dev/null | sort -h || true
    echo "--- transfer_input_files ---"
    ls -lh venv.tar.gz code.tar.gz 2>/dev/null || true
    echo "--- GPU ---"
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader
    else
      echo "nvidia-smi not in PATH"
    fi
    echo "--- end ${label} ---"
    echo ""
  } | tee -a "$log_file"
}

if [[ ! -f venv.tar.gz || ! -f code.tar.gz ]]; then
  echo "Missing transfer_input_files: venv.tar.gz and/or code.tar.gz" >&2
  exit 9
fi

tar -xzf venv.tar.gz
tar -xzf code.tar.gz
export PATH="${PWD}/venv/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${PWD}/hf_cache}"
mkdir -p "${HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

JOB_MODE="${JOB_MODE:-full}"
if [[ "${JOB_MODE}" == "gpu_smoke" ]]; then
  SKIP_EVAL="${SKIP_EVAL:-1}"
fi

EXTRA_OPTS=()
if [[ "${SKIP_EVAL:-0}" == "1" ]]; then
  EXTRA_OPTS+=(--skip_eval)
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  EXTRA_OPTS+=(--dry_run)
fi

if [[ "${JOB_MODE}" == "gpu_smoke" ]]; then
  echo "==========================================="
  echo "JOB_MODE=gpu_smoke (CHTC / Pascal-safe path)"
  echo "==========================================="
  python - <<'PROBE'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    print("cuda.is_bf16_supported", fn() if fn else None)
PROBE
  if [[ "${VERIFY_OTDD:-1}" != "0" ]]; then
    python scripts/verify_otdd.py
  fi
  METHOD="${METHOD:-orca_otdd}"
  EMBED_EP="${EMBED_EP:-1}"
  JOB_TAG="${JOB_TAG:-gpu_smoke_chtc}"
  OUT="${OUT:-runs/${JOB_TAG}}"
  OUTPUT_TARBALL="${OUTPUT_TARBALL:-orca_rank_gpu_results.tar.gz}"
  mkdir -p "${OUT}"

  set +e
  PYTHONPATH="${PWD}:${PYTHONPATH:-}" python run_experiment.py \
    --method "${METHOD}" \
    --lora_rank "${LORA_RANK:-8}" \
    --seed "${SEED:-0}" \
    --embedder_epochs "${EMBED_EP}" \
    --hard_train_samples 96 \
    --easy_pool_samples 96 \
    --max_proxy_source_embeddings 256 \
    --otdd_maxsamples_per_class 12 \
    --stage_b_epochs 1 \
    --grad_accum 1 \
    --limit_train_batches 12 \
    --max_stage_b_steps 6 \
    --output_dir "${OUT}" \
    "${EXTRA_OPTS[@]}"
  _rc=${?}
  set -euo pipefail

  {
    echo "run_experiment exit_code=${_rc}"
    date -Is 2>/dev/null || date
  } >>"${OUT}/condor_job_status.txt"

  log_condor_resources "pre_pack_smoke" "${OUT}/condor_resource_usage.txt"

  echo "Smoke finished (python exit ${_rc}) — packing ${OUT}"
  tar -czf "${OUTPUT_TARBALL}" "${OUT}"
  ls -lh "${OUTPUT_TARBALL}"

  exit "${_rc}"
fi

METHOD="${METHOD:-lora_only}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-0}"
EMBED_EP="${EMBED_EP:-60}"
JOB_TAG="${JOB_TAG:-${METHOD}_r${LORA_RANK}_s${SEED}}"
OUT="${OUT:-runs/${JOB_TAG}}"
OUTPUT_TARBALL="${OUTPUT_TARBALL:-orca_rank_gpu_results.tar.gz}"
mkdir -p "${OUT}"

set +e
python run_experiment.py \
  --method "${METHOD}" \
  --lora_rank "${LORA_RANK}" \
  --seed "${SEED}" \
  --embedder_epochs "${EMBED_EP}" \
  --output_dir "${OUT}" \
  "${EXTRA_OPTS[@]}"
_rc=${?}
set -euo pipefail

echo "run_experiment exit_code=${_rc}" >>"${OUT}/condor_job_status.txt"

log_condor_resources "pre_pack_full" "${OUT}/condor_resource_usage.txt"

tar -czf "${OUTPUT_TARBALL}" "${OUT}"
exit "${_rc}"
