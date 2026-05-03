Submit one Condor GPU job that runs the bundled **gpu_smoke** pipeline (CUDA probe +
`scripts/verify_otdd.py` + minimal `orca_otdd`).

Build inputs (adjust paths):

  ./scripts/pack_chtc.sh
  ln -sf pack_code_*.tar.gz code.tar.gz
  tar -czf venv.tar.gz venv/

Environment is passed via HTCondor; set JOB_MODE:

  JOB_MODE = gpu_smoke

Pool notes (UW–Madison CHTC and similar):

- **Tesla P100** uses Pascal **sm_60**. PyTorch **bf16 CUDA autocast is off** automatically;
  runs use fp32 on GPU unless you disable that logic (not recommended on P100).
- Use a Torch wheel matching the execute host CUDA (often **cu118** still ships Pascal SASS).
- First run needs Hub/download time; HF cache defaults to ${_CONDOR_SCRATCH_DIR}/hf_cache inside run_job.sh.

See htcondor/submit_example.sub for full / single-queue template; duplicate it and append:

  environment            = "JOB_MODE=gpu_smoke"
  +JobBatchName           = orca_rank_gpu_smoke

Or rely on getenv = True from your laptop:

  export JOB_MODE=gpu_smoke
  condor_submit htcondor/submit_gpu_smoke.sub

