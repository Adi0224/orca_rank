from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ExperimentConfig:
    """Single Condor/SGE job boundary (one seed × rank × method)."""

    seed: int = 0
    method: str = "lora_only"  # lora_only | orca_otdd | ce_warm_f
    lora_r: int = 8
    lora_alpha: int | None = None

    model_name: str = "EleutherAI/pythia-70m"
    max_length: int = 256

    embedder_epochs: int = 60
    otdd_inner_exact: bool = False  # True = inner_ot_method exact (slower); else gaussian_approx
    otdd_maxsamples_per_class: int = 64
    max_proxy_source_embeddings: int = 2048
    alignment_lr: float = 1e-4
    easy_answer_lt: int = 80  # pool "easy" GSM8K train rows with |gold| < this (same modality trick)

    hard_train_samples: int = 512
    val_samples: int = 128
    easy_pool_samples: int = 768
    data_seed: int = 42

    stage_b_epochs: int = 2
    stage_b_lr: float = 1e-4
    stage_b_microbatch: int = 2
    stage_b_gradient_accumulation: int = 8
    max_stage_b_steps: int | None = 600

    bf16: bool = True
    gradient_clip: float = 1.0

    output_dir: str = "runs/default"

    adapter_bottleneck: int = 128

    dry_run: bool = False

    limit_train_batches: int | None = None
    skip_eval: bool = False

    def resolved_lora_alpha(self) -> int:
        if self.lora_alpha is not None:
            return self.lora_alpha
        return 2 * self.lora_r

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


def label_mod10(answer_text: str) -> int:
    """Discrete label aligned with GSM8K answer #### final number."""
    tail = answer_text.split("####")[-1].strip()
    digs = "".join(c for c in tail if (c.isdigit() or c == "-"))
    if not digs.replace("-", ""):
        return 0
    try:
        v = int(abs(int(digs.replace("-", "", 1) if digs.count("-") <= 1 else digs)))
    except ValueError:
        return 0
    return int(v % 10)
