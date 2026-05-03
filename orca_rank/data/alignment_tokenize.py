"""Tokenize prompts-only rows for embedding alignment."""

from __future__ import annotations

from datasets import Dataset
from transformers import AutoTokenizer


def tokenize_alignment_prompt_batch(
    dataset: Dataset,
    tokenizer_name: str,
    max_length: int,
) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def proc(batch):
        input_ids_batch = []
        attn_batch = []
        lbl10 = []
        qs = batch["question"]
        ans_texts = batch["answer"]
        for q, ans in zip(qs, ans_texts):
            prompt = "Question:\n" + q.strip() + "\n\nAnswer:\n"
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                padding=False,
                add_special_tokens=True,
            )
            input_ids_batch.append(enc["input_ids"])
            attn_batch.append(enc["attention_mask"])

            tail = ans.split("####")[-1].strip()
            try:
                v = abs(int(round(float(tail.replace(",", "").split()[0]))))
            except (ValueError, IndexError):
                v = 0
            lbl10.append(int(v % 10))

        return {
            "input_ids_alignment": input_ids_batch,
            "attention_mask_alignment": attn_batch,
            "label10": lbl10,
        }

    return dataset.map(proc, batched=True, batch_size=32, desc="alignment_tokenize")


def alignment_collate(
    samples: list[dict],
    pad_token_id: int,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad variable-length prompt tokenizations."""
    import torch as T

    max_len = max(len(s["input_ids_alignment"]) for s in samples)
    max_len = min(max_len, max_length)
    bsz = len(samples)
    ids = T.full((bsz, max_len), pad_token_id, dtype=T.long)
    attn = T.zeros((bsz, max_len), dtype=T.long)
    ys = []
    for i, s in enumerate(samples):
        row_ids = s["input_ids_alignment"][:max_len]
        row_attn = s["attention_mask_alignment"][:max_len]
        L = len(row_ids)
        ids[i, :L] = T.tensor(row_ids, dtype=T.long)
        attn[i, :L] = T.tensor(row_attn, dtype=T.long)
        ys.append(int(s["label10"]))
    lbl = T.tensor(ys, dtype=T.long)
    return ids, attn, lbl


class AlignmentDatasetWrapper:
    """Random access by HF dataset row index."""

    def __init__(self, mapped_ds: Dataset):
        self.ds = mapped_ds

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        return self.ds[idx]
