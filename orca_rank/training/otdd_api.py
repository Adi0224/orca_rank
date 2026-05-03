"""Thin wrapper matching ORCA `embedder.otdd` (+ explicit per-target labels)."""

from __future__ import annotations

import torch
from otdd.pytorch.distance import DatasetDistance


def otdd_scalar(
    feats: torch.Tensor,
    ys: torch.Tensor,
    src_train_dataset: torch.utils.data.TensorDataset,
    exact: bool,
    maxsamples: int,
) -> torch.Tensor:
    """Differentiable OTDD between frozen source embeddings and pooled target feats."""

    feats = feats.to(dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(feats, ys.long())
    ms = max(1, min(maxsamples, len(src_train_dataset), len(dataset)))

    dist = DatasetDistance(
        src_train_dataset,
        dataset,
        inner_ot_method="exact" if exact else "gaussian_approx",
        debiased_loss=True,
        inner_ot_debiased=True,
        p=2,
        inner_ot_p=2,
        entreg=0.1,
        ignore_target_labels=False,
        device=str(feats.device),
        load_prev_dyy1=None,
    )
    d = dist.distance(maxsamples=ms)
    return d if torch.is_tensor(d) else torch.as_tensor(float(d), device=feats.device, dtype=torch.float32)
