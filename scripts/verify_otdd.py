#!/usr/bin/env python3
"""Verify vendored OTDD (DatasetDistance) imports and one backward pass on CPU."""
from __future__ import annotations

import sys

import torch
from torch.utils.data import TensorDataset


def main() -> int:
    import orca_rank  # noqa: F401 — prepend third_party/orca_otdd

    from otdd.pytorch.distance import DatasetDistance

    torch.manual_seed(0)
    device = "cpu"
    d_h = 64
    # Balanced labels so Gaussian per-class moments are well-defined under maxsamples caps.
    n_per_digit = 8
    n_classes = 10
    n_src = n_per_digit * n_classes
    n_tgt = n_per_digit * n_classes

    feats_s = torch.randn(n_src, d_h, dtype=torch.float32)
    lab_s = torch.arange(n_classes, dtype=torch.long).repeat_interleave(n_per_digit)

    feats_t = torch.randn(n_tgt, d_h, dtype=torch.float32, requires_grad=True)
    lab_t = torch.arange(n_classes, dtype=torch.long).repeat_interleave(n_per_digit)

    src = TensorDataset(feats_s, lab_s)
    tgt = TensorDataset(feats_t, lab_t)

    dist_obj = DatasetDistance(
        src,
        tgt,
        inner_ot_method="gaussian_approx",
        debiased_loss=True,
        inner_ot_debiased=True,
        p=2,
        inner_ot_p=2,
        entreg=0.1,
        ignore_target_labels=False,
        device=device,
        verbose=0,
        load_prev_dyy1=None,
    )
    loss = dist_obj.distance(maxsamples=min(n_src, n_tgt, 320))
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss, dtype=torch.float32, device=feats_t.device)
    loss = loss.float()

    loss.backward()
    if feats_t.grad is None or not torch.isfinite(feats_t.grad).all():
        print("OTDD smoke FAILED: bad grad on target features", file=sys.stderr)
        return 2

    print("OTDD smoke OK: loss=", float(loss.detach()), "grad_norm=", float(feats_t.grad.norm()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
