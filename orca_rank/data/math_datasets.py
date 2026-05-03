"""GSM8K splits (easy proxy vs harder train) sharing same modality + tokenizer."""

from __future__ import annotations

from typing import Any

import numpy as np
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def gsm8k_split_tail(answer_full: str) -> str:
    return answer_full.split("####")[-1].strip()


def parse_gold_numeric(answer_full: str) -> int | None:
    tail = gsm8k_split_tail(answer_full)
    if not tail:
        return None
    try:
        v = float(tail.replace(",", ""))
        return int(round(v))
    except ValueError:
        return None


def format_prompt_completion(question: str, answer_full: str) -> tuple[str, str]:
    prompt = "Question:\n" + question.strip() + "\n\nAnswer:\n"
    completion = answer_full.strip()
    return prompt, completion


def label_mod10_from_answer(answer_full: str) -> int:
    from orca_rank.config import label_mod10

    return label_mod10(answer_full)


def build_index_splits(
    easy_answer_lt: int,
    data_seed: int,
    hard_train_n: int,
    val_n: int,
    easy_pool_n: int,
) -> dict[str, list[int]]:
    """Return disjoint index lists over GSM8K train by numeric magnitude proxy."""
    rng = np.random.RandomState(data_seed)
    ds_train = load_dataset("gsm8k", "main", split="train")
    n_all = len(ds_train)

    def row_info(i: int):
        ans = ds_train[i]["answer"]
        g_abs = parse_gold_numeric(ans)
        return abs(g_abs) if g_abs is not None else None

    idx_perm = rng.permutation(n_all).tolist()
    easy: list[int] = []
    for i in idx_perm:
        gv = row_info(i)
        if gv is not None and gv < easy_answer_lt:
            easy.append(i)
        if len(easy) >= easy_pool_n:
            break
    easy_set = set(easy[:easy_pool_n])

    hard_val_pool: list[int] = []
    for i in idx_perm:
        if i in easy_set:
            continue
        hard_val_pool.append(i)

    train_hard = hard_val_pool[:hard_train_n]
    val_idxs = hard_val_pool[hard_train_n : hard_train_n + val_n]

    easy_pool = easy[:easy_pool_n]

    return {"easy_proxy": easy_pool, "hard_train": train_hard, "hard_val": val_idxs}


def build_hf_splits(cfg) -> tuple[Dataset, Dataset, Dataset, dict[str, Any]]:
    splits = build_index_splits(
        easy_answer_lt=cfg.easy_answer_lt,
        data_seed=cfg.data_seed,
        hard_train_n=cfg.hard_train_samples,
        val_n=cfg.val_samples,
        easy_pool_n=cfg.easy_pool_samples,
    )

    ds_train_full = load_dataset("gsm8k", "main", split="train")

    def subset_rows(indices: list[int]) -> Dataset:
        rows = [ds_train_full[int(i)] for i in indices]
        return Dataset.from_list(rows)

    easy_ds = subset_rows(splits["easy_proxy"])
    hard_train_ds = subset_rows(splits["hard_train"])
    hard_val_ds = subset_rows(splits["hard_val"])
    splits["sizes"] = {
        "easy": len(easy_ds),
        "hard_train": len(hard_train_ds),
        "hard_val": len(hard_val_ds),
    }
    return easy_ds, hard_train_ds, hard_val_ds, splits


def tokenize_split(dataset: Dataset, tokenizer_name: str, max_length: int) -> Dataset:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def proc(batch):
        input_ids_batch = []
        attn_batch = []
        labels_batch = []
        prompt_lens = []
        lbl10 = []
        golds = []
        for q, ans in zip(batch["question"], batch["answer"]):
            prompt, completion = format_prompt_completion(q, ans)
            full_text = prompt + completion
            enc_full = tokenizer(
                full_text,
                max_length=max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            enc_prompt = tokenizer(
                prompt,
                max_length=max_length,
                truncation=True,
                padding=False,
                add_special_tokens=True,
                return_tensors=None,
            )
            inp = enc_full["input_ids"]
            attn = enc_full["attention_mask"]
            lbl = list(inp)
            plen = len(enc_prompt["input_ids"])
            for j in range(min(plen, len(lbl))):
                lbl[j] = -100
            input_ids_batch.append(inp)
            attn_batch.append(attn)
            labels_batch.append(lbl)
            prompt_lens.append(plen)
            lbl10.append(label_mod10_from_answer(ans))
            golds.append(ans)
        return {
            "input_ids": input_ids_batch,
            "attention_mask": attn_batch,
            "labels_lm": labels_batch,
            "prompt_len": prompt_lens,
            "label10": lbl10,
            "gold_answer_text": golds,
        }

    return dataset.map(
        proc,
        batched=True,
        batch_size=32,
        remove_columns=list(dataset.column_names),
        desc="tokenize_gsm8k",
    )
