from __future__ import annotations

import dataclasses
from typing import Any

import torch


@dataclasses.dataclass
class Batch:
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor


def collate_lm_batch(
    features: list[dict[str, Any]],
    pad_token_id: int,
    max_length: int | None = None,
) -> Batch:
    max_len = max(len(f["input_ids"]) for f in features)
    if max_length is not None:
        max_len = min(max_len, max_length)
    bsz = len(features)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attn = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    for i, f in enumerate(features):
        ids = f["input_ids"][:max_len]
        la = f["labels_lm"][:max_len]
        am = f["attention_mask"][:max_len]
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = torch.tensor(am, dtype=torch.long)
        labels[i, :L] = torch.tensor(la, dtype=torch.long)
    return Batch(input_ids=input_ids, attention_mask=attn, labels=labels)
