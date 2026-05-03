"""Thin wrapper matching ORCA `embedder.otdd` (+ explicit per-target labels)."""

from __future__ import annotations

import torch
from otdd.pytorch.distance import DatasetDistance


def _stabilize_feats_for_geomloss(feats: torch.Tensor) -> torch.Tensor:
    """GeomLoss Sinkhorn builds an epsilon ladder from cloud `diameter`; near-collapsed
    batches (often a few pooled LM vectors) yield invalid schedules (ValueError: arange).

    Small iid jitter breaks exact degeneracy; scale uses row std when available, else
    an absolute floor so zero-variance pooled vectors still perturb.
    """
    if feats.numel() == 0:
        return feats
    flat = feats.detach().flatten(1)
    sd = float(flat.std(dim=1, unbiased=False).mean().cpu())
    noise_scale = max(sd * 1e-5, 1e-4)
    return feats + noise_scale * torch.randn_like(feats)


def otdd_scalar(
    feats: torch.Tensor,
    ys: torch.Tensor,
    src_train_dataset: torch.utils.data.TensorDataset,
    exact: bool,
    maxsamples: int,
) -> torch.Tensor:
    """Differentiable OTDD between frozen source embeddings and pooled target feats."""

    feats = feats.to(dtype=torch.float32)
    feats = _stabilize_feats_for_geomloss(feats)
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
