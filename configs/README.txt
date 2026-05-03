Optional presets (YAML/JSON loaders not wired yet). Use run_experiment.py CLI flags instead.
Suggested Condor sweep (env vars + getenv line in submit file):
METHOD=orca_otdd LORA_RANK=4 SEED=0
METHOD=orca_otdd LORA_RANK=8 SEED=0
METHOD=lora_only LORA_RANK=4 SEED=0
