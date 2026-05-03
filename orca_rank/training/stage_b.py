"""LoRA adaptation on GSM8K hard train (masked CE on answer span)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from orca_rank.data.collate import Batch, collate_lm_batch
from orca_rank.eval.gsm8k_em import (
    extract_from_generation,
    gold_from_answer_field,
    gsm8k_exact_batch,
)
from orca_rank.training.gpu_utils import autocast_for_device


class _HFDatasetView(Dataset):
    def __init__(self, hf_ds):
        self.hf_ds = hf_ds

    def __len__(self) -> int:
        return len(self.hf_ds)

    def __getitem__(self, idx: int):
        row = self.hf_ds[int(idx)]
        return {
            "input_ids": list(row["input_ids"]),
            "attention_mask": list(row["attention_mask"]),
            "labels_lm": list(row["labels_lm"]),
        }


def attach_lora_once(lm_module, r: int, alpha: int, dropout: float = 0.05):
    if hasattr(lm_module, "peft_config"):
        return lm_module  # already wrapped
    targ = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=int(alpha),
        lora_dropout=dropout,
        bias="none",
        target_modules=targ,
        inference_mode=False,
    )
    return get_peft_model(lm_module, cfg)


def train_stage_b(
    model_wrapper,
    train_hf_ds,
    val_raw_hf_ds,
    cfg,
    device: torch.device,
    tokenizer: AutoTokenizer,
    output_dir: Path,
) -> dict[str, Any]:
    model_wrapper.lm = attach_lora_once(
        model_wrapper.lm,
        r=cfg.lora_r,
        alpha=cfg.resolved_lora_alpha(),
    )

    lora_trainable = [p for p in model_wrapper.lm.parameters() if p.requires_grad]
    groups = [{"params": lora_trainable, "lr": cfg.stage_b_lr}]
    if model_wrapper.adapter is not None:
        ada_p = list(model_adapter_params(model_wrapper))
        if ada_p:
            groups.append({"params": ada_p, "lr": cfg.stage_b_lr * 0.25})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    def collate_fn(rows):
        return collate_lm_batch(
            rows,
            pad_token_id=int(tokenizer.pad_token_id),
            max_length=cfg.max_length,
        )

    pytorch_ds = _HFDatasetView(train_hf_ds)
    loader = DataLoader(
        pytorch_ds,
        shuffle=True,
        batch_size=cfg.stage_b_microbatch,
        collate_fn=collate_fn,
        drop_last=False,
    )

    global_step = 0
    t0 = time.perf_counter()
    epoch_tail_losses: list[float] = []

    for ep in range(cfg.stage_b_epochs):
        model_wrapper.train()
        accum = cfg.stage_b_gradient_accumulation
        optimizer.zero_grad(set_to_none=True)
        running_sum = 0.0
        micro = 0
        accumulated = 0

        for i, batch in enumerate(loader):
            if cfg.limit_train_batches is not None and micro >= cfg.limit_train_batches:
                break
            if not isinstance(batch, Batch):
                raise TypeError(batch)
            inp = batch.input_ids.to(device)
            attn = batch.attention_mask.to(device)
            labels = batch.labels.to(device)

            with autocast_for_device(device, cfg.bf16):
                out = model_wrapper(input_ids=inp, attention_mask=attn, labels=labels)
            loss = out.loss / float(accum)

            loss.backward()

            running_sum += float(out.loss.detach().cpu().item())
            micro += 1
            accumulated += 1

            if accumulated % accum == 0:
                pooled = list(p for p in model_wrapper.lm.parameters() if p.grad is not None)
                ada = list(model_wrapper.adapter.parameters()) if model_wrapper.adapter is not None else []
                pooled += [p for p in ada if p.grad is not None]
                clip_params = pooled
                if clip_params:
                    torch.nn.utils.clip_grad_norm_(clip_params, cfg.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if cfg.max_stage_b_steps and global_step >= int(cfg.max_stage_b_steps):
                break

        epoch_tail_losses.append(running_sum / max(micro, 1))

        if cfg.max_stage_b_steps and global_step >= int(cfg.max_stage_b_steps):
            break

    wall_b = float(time.perf_counter() - t0)

    em = float("nan")
    if not cfg.skip_eval:
        pred_dump = (output_dir / "val_predictions.json") if cfg.dump_val_predictions else None
        em, _, _ = evaluate_gsm8k(
            model_wrapper,
            tokenizer,
            val_raw_hf_ds,
            device,
            cfg.max_length,
            use_bf16_autocast=cfg.bf16,
            dump_predictions_json=pred_dump,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    em_out: float | None
    if cfg.skip_eval:
        em_out = None
    else:
        em_out = float(em)

    metrics = {
        "wall_seconds_stage_b": wall_b,
        "epochs_ran_stage_b": len(epoch_tail_losses),
        "global_steps_stage_b": global_step,
        "train_loss_epochs_mean_tail": epoch_tail_losses[-1:] if epoch_tail_losses else [],
        "val_exact_match_mean": em_out,
    }
    if cfg.dump_val_predictions and not cfg.skip_eval:
        metrics["val_predictions_filename"] = "val_predictions.json"
    (output_dir / "metrics.partial.json").write_text(json.dumps(metrics, indent=2))

    adapter_dir = output_dir / "lora_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model_wrapper.lm.save_pretrained(adapter_dir)

    tokenizer.save_pretrained(output_dir)
    if model_wrapper.adapter:
        torch.save(model_wrapper.adapter.state_dict(), output_dir / "adapter.pt")

    return metrics


def model_adapter_params(model_wrapper) -> list[torch.Tensor]:
    if model_wrapper.adapter is None:
        return []
    return [p for p in model_wrapper.adapter.parameters() if p.requires_grad]


def evaluate_gsm8k(
    model_wrapper,
    tokenizer,
    raw_val_hf,
    device,
    max_prompt_len,
    *,
    use_bf16_autocast: bool,
    dump_predictions_json: Path | None = None,
):
    model_wrapper.eval()
    preds: list[str] = []
    golds: list[str] = []
    questions: list[str] = []
    pad_id = tokenizer.pad_token_id

    ac = autocast_for_device(device, use_bf16_autocast)

    with torch.no_grad():
        for i in range(len(raw_val_hf)):
            row = raw_val_hf[i]
            q = row["question"].strip()
            questions.append(q)
            gold_a = row["answer"]
            prompt = "Question:\n" + q + "\n\nAnswer:\n"
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors="pt",
                add_special_tokens=True,
            )
            input_ids = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with ac:
                out_ids = model_wrapper.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=192,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen = out_ids[:, input_ids.shape[1] :]
            decoded = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            preds.append(decoded)
            golds.append(gold_a)

    em, ok, total = gsm8k_exact_batch(preds, golds)
    if dump_predictions_json is not None:
        rows: list[dict[str, Any]] = []
        for i in range(total):
            pp = preds[i]
            ga = golds[i]
            parsed_p = extract_from_generation(pp)
            parsed_g = gold_from_answer_field(ga)
            rows.append(
                {
                    "idx": i,
                    "question": questions[i],
                    "prediction_raw": pp,
                    "gold_answer_field": ga,
                    "parsed_pred_numeric": parsed_p,
                    "parsed_gold_numeric": parsed_g,
                    # Same rule as gsm8k_exact_batch: both parsed nonempty and equal.
                    "numeric_match": bool(parsed_p == parsed_g and parsed_g != ""),
                }
            )
        dump_predictions_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"val_exact_match_mean": float(em), "ok": ok, "total": total, "items": rows}
        dump_predictions_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return float(em), int(ok), int(total)
