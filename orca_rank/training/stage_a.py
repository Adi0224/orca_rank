"""Stage A: freeze LM, train adapter f with class-wise OTDD (ORCA-style outer loop)."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import TensorDataset

from orca_rank.models.pythia_lm import PythiaFrontendCausalLM
from orca_rank.training.otdd_api import otdd_scalar


@torch.no_grad()
def pooled_no_adapter(
    embedder: torch.nn.Embedding,
    input_ids: torch.Tensor,
    attn: torch.Tensor,
) -> torch.Tensor:
    e = embedder(input_ids).float()
    m = attn.unsqueeze(-1).to(dtype=e.dtype)
    return (e * m).sum(1) / m.sum(1).clamp(min=1e-6)


def build_source_tensors(
    model: PythiaFrontendCausalLM,
    easy_ds_mapped,
    device: torch.device,
    max_rows: int,
    pad_id: int,
    microbatch: int = 8,
) -> TensorDataset:
    emb = model.get_input_embeddings()
    chunks_f: list[torch.Tensor] = []
    chunks_y: list[torch.Tensor] = []
    n = len(easy_ds_mapped)
    for start in range(0, n, microbatch):
        end = min(start + microbatch, n)
        rows = [easy_ds_mapped[j] for j in range(start, end)]
        ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(r["input_ids_alignment"], dtype=torch.long) for r in rows],
            batch_first=True,
            padding_value=pad_id,
        )
        attn = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(r["attention_mask_alignment"], dtype=torch.long) for r in rows],
            batch_first=True,
            padding_value=0,
        ).to(device)
        ids = ids.to(device)
        pooled = pooled_no_adapter(emb, ids, attn)
        chunks_f.append(pooled.detach().cpu())
        chunks_y.append(torch.tensor([r["label10"] for r in rows], dtype=torch.long))

    feats = torch.cat(chunks_f, 0)
    ys = torch.cat(chunks_y, 0)
    if len(feats) > max_rows:
        pick = torch.randperm(len(feats))[:max_rows]
        feats = feats[pick]
        ys = ys[pick]
    feats = feats.to(device=device, dtype=torch.float32)
    ys = ys.long().to(device)
    return TensorDataset(feats, ys)


def _class_indices(mapped_ds) -> dict[int, list[int]]:
    buckets: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(mapped_ds)):
        c = int(mapped_ds[idx]["label10"])
        buckets[c].append(idx)
    return buckets


def train_stage_a_alignment(
    model: PythiaFrontendCausalLM,
    cfg,
    align_hard_mapped,
    source_ds: TensorDataset,
    device: torch.device,
    pad_token_id: int,
) -> dict[str, Any]:
    for p in model.lm.parameters():
        p.requires_grad = False

    if model.adapter is None:
        raise RuntimeError("Stage A requires adapter (method orca_otdd).")

    for p in model.adapter.parameters():
        p.requires_grad = True

    adapter_params = list(model.adapter.parameters())
    opt = torch.optim.AdamW(adapter_params, lr=cfg.alignment_lr, weight_decay=0.01)

    buckets = _class_indices(align_hard_mapped)
    classes_present = sorted(buckets.keys())

    counts = torch.zeros(10)
    for c, idxs in buckets.items():
        counts[c] = len(idxs)
    counts_sum = counts.sum().clamp(min=1.0)

    losses_out: list[float] = []
    t0 = time.perf_counter()

    for ep in range(cfg.embedder_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        rng = np.random.RandomState(cfg.seed + ep)
        ci_order = rng.permutation(classes_present).tolist()
        eps_terms: list[torch.Tensor] = []

        for ci in ci_order:
            idxs = buckets[ci][: cfg.otdd_maxsamples_per_class]
            idcs = list(idxs)
            rng.shuffle(idcs)

            rows = [align_hard_mapped[j] for j in idcs]
            if len(rows) < 2:
                continue

            ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r["input_ids_alignment"], dtype=torch.long) for r in rows],
                batch_first=True,
                padding_value=pad_token_id,
            ).to(device)
            attn = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(r["attention_mask_alignment"], dtype=torch.long) for r in rows],
                batch_first=True,
                padding_value=0,
            ).to(device)

            feats = model.pooled_masked_mean(ids, attn)

            wt = (
                counts[int(ci)].to(dtype=torch.float32, device=device)
                / counts_sum.to(dtype=torch.float32, device=device).clamp(min=1e-8)
            )
            ys_tgt = torch.full((feats.shape[0],), int(ci), device=device, dtype=torch.long)

            ms = max(1, min(len(source_ds), cfg.otdd_maxsamples_per_class * 20, 8192))

            d = otdd_scalar(
                feats,
                ys_tgt,
                source_ds,
                exact=cfg.otdd_inner_exact,
                maxsamples=ms,
            )
            eps_terms.append(wt * d.float())

        if not eps_terms:
            losses_out.append(0.0)
            continue

        loss = torch.stack(eps_terms).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter_params, cfg.gradient_clip)
        opt.step()
        losses_out.append(float(loss.detach().cpu().item()))

    wall = float(time.perf_counter() - t0)
    return {
        "embedder_epochs": cfg.embedder_epochs,
        "alignment_loss_curve": losses_out,
        "wall_seconds_stage_a": wall,
    }
