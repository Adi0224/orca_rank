# ORCA × LoRA (GSM8K pilot)

Pythia (GPT-NeoX) + optional **input adapter** trained with vendored **[OTDD](https://github.com/sjunhongshen/ORCA)** (`third_party/orca_otdd`), then **LoRA** fine-tuning on a budget GSM8K split (“easy proxy” rows vs harder train rows, same modality).

## Local setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-chtc-torch-example.txt  # swap for your CUDA
pip install -r requirements.txt
```

## Smoke tests

```bash
python run_experiment.py --dry_run --method lora_only --output_dir /tmp/o1   # no PyTorch required
python run_experiment.py --dry_run --method orca_otdd --output_dir /tmp/o2   # needs torch + geomloss/POT stack
```

Tiny train (downloads models/datasets):

```bash
python run_experiment.py --method lora_only --output_dir runs/smoke \
  --limit_train_batches 4 --skip_eval
```

**CPU / Mac micro-test** (first run downloads Pythia + GSM8K; use `--grad_accum 1` so a few batches still trigger optimizer steps):

```bash
python3 -m venv .venv && .venv/bin/pip install torch transformers accelerate peft datasets safetensors sentencepiece protobuf tqdm pyyaml numpy
.venv/bin/python run_experiment.py --method lora_only --output_dir runs/mac_cpu_smoke \
  --hard_train_samples 16 --val_samples 4 --easy_pool_samples 16 \
  --stage_b_epochs 1 --limit_train_batches 4 --skip_eval --lora_rank 4 --grad_accum 1
```

## CHTC tarballs

1. On **Linux x86_64**, build **`venv/`** matching GPU workers’ CUDA, then **`tar -czf venv.tar.gz venv`**.
2. From repo root: **`./scripts/pack_chtc.sh`** → `pack_code_*.tar.gz`; **`cp`** that tarball to **`code.tar.gz`** in the folder from which you submit (alongside **`venv.tar.gz`**).
3. **`mkdir -p htcondor_logs`**.
4. **First-time pool check (P100-friendly):** submit **`htcondor/submit_gpu_smoke.sub`**, which sets **`JOB_MODE=gpu_smoke`**. That runs a CUDA capability probe, **`scripts/verify_otdd.py`**, and a short **`orca_otdd`** train. Use the same **`venv.tar.gz` + `code.tar.gz`** layout as below.
5. Full experiment: set env (passed through **`getenv = True`**) then submit, e.g.  
   **`export METHOD=orca_otdd LORA_RANK=8 SEED=0`**  
   **`condor_submit htcondor/submit_example.sub`**

**P100 (Pascal sm_60):** PyTorch usually reports **no bf16 CUDA autocast** on this hardware. The launcher **turns off bf16** and trains in **fp32 on the GPU** when needed (see **`config.json`** / **`bf16_autocast_disabled_reason`**). Use a CUDA build that still ships **sm_60** SASS (many **cu118** wheels do). Details: **`htcondor/README_CHTC.txt`**.

Edit **`htcondor/submit_example.sub`** for your pool’s **`request_gpus`** / **`requirements`**.

Returned tarball: **`orca_rank_gpu_results.tar.gz`** (`runs/<tag>/metrics.json`, etc., inside).

Artifacts under output dir: **`metrics.json`**, **`config.json`**, **`lora_adapter/`**, optional **`adapter.pt`**.


## Aggregation

```bash
python scripts/aggregate_metrics.py --runs-root runs --out metrics_grid.csv
```

## License

Upstream OTDD code is MIT (`third_party/orca_otdd/LICENSE-OTDD`). This scaffold is yours to license.
