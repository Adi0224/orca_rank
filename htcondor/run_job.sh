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

  echo "Smoke OK — output dir: ${OUT}"
  OUTPUT_TARBALL="${OUTPUT_TARBALL:-orca_rank_gpu_results.tar.gz}"
  tar -czf "${OUTPUT_TARBALL}" "${OUT}" || true
  exit 0
fi

METHOD="${METHOD:-lora_only}"
LORA_RANK="${LORA_RANK:-8}"
SEED="${SEED:-0}"
EMBED_EP="${EMBED_EP:-30}"
JOB_TAG="${JOB_TAG:-${METHOD}_r${LORA_RANK}_s${SEED}}"
OUT="${OUT:-runs/${JOB_TAG}}"
OUTPUT_TARBALL="${OUTPUT_TARBALL:-orca_rank_gpu_results.tar.gz}"

python run_experiment.py \
  --method "${METHOD}" \
  --lora_rank "${LORA_RANK}" \
  --seed "${SEED}" \
  --embedder_epochs "${EMBED_EP}" \
  --output_dir "${OUT}" \
  "${EXTRA_OPTS[@]}"

tar -czf "${OUTPUT_TARBALL}" "${OUT}" || true
