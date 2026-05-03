#!/usr/bin/env python3
"""
Single-job launcher (one Condor GPU task): GSM8K align-then-LoRA pilot.

`import orca_rank` prepends vendored OTDD to sys.path.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--method", choices=("lora_only", "orca_otdd", "ce_warm_f"), default="lora_only")
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--embedder_epochs", type=int, default=60)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--skip_eval", action="store_true")
    p.add_argument("--limit_train_batches", type=int, default=None)
    p.add_argument("--stage_b_epochs", type=int, default=None, help="Override ExperimentConfig.stage_b_epochs")
    p.add_argument("--hard_train_samples", type=int, default=None)
    p.add_argument("--val_samples", type=int, default=None)
    p.add_argument("--easy_pool_samples", type=int, default=None)
    p.add_argument(
        "--grad_accum",
        type=int,
        default=None,
        help="Override stage_b_gradient_accumulation (use 1 for tiny CPU tests)",
    )
    p.add_argument(
        "--max_proxy_source_embeddings",
        type=int,
        default=None,
        help="Cap easy-proxy vectors for frozen OTDD source TensorDataset",
    )
    p.add_argument(
        "--otdd_maxsamples_per_class",
        type=int,
        default=None,
        help="Hard-target rows sampled per digit class during Stage A OTDD",
    )
    p.add_argument(
        "--otdd_inner_exact",
        action="store_true",
        help="Use slower exact inner OT; default gaussian_approx",
    )
    p.add_argument(
        "--max_stage_b_steps",
        type=int,
        default=None,
        help="Early-stop Stage B after this many optimizer steps (None = use config)",
    )
    p.add_argument("--no_bf16", action="store_true", help="Disable bf16 autocast (fp32 on GPU)")
    p.add_argument(
        "--dump_val_predictions",
        action="store_true",
        help="Write GSM8K val generations to output_dir/val_predictions.json (after Stage B)",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu", "mps"),
        default="auto",
        help="Accelerator: auto picks cuda>mps>cpu. "
        "orca_otdd auto-downgrades from mps→cpu (POT coupling is not MPS-safe).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None):
    argv = argv if argv is not None else sys.argv[1:]
    args_ns = parse_args(argv)

    from orca_rank.config import ExperimentConfig

    cfg = ExperimentConfig(
        seed=args_ns.seed,
        method=args_ns.method,
        lora_r=args_ns.lora_rank,
        output_dir=args_ns.output_dir,
        embedder_epochs=args_ns.embedder_epochs,
        dry_run=args_ns.dry_run,
        skip_eval=args_ns.skip_eval,
        limit_train_batches=args_ns.limit_train_batches,
        dump_val_predictions=args_ns.dump_val_predictions,
    )
    ov: dict = {}
    if args_ns.stage_b_epochs is not None:
        ov["stage_b_epochs"] = args_ns.stage_b_epochs
    if args_ns.hard_train_samples is not None:
        ov["hard_train_samples"] = args_ns.hard_train_samples
    if args_ns.val_samples is not None:
        ov["val_samples"] = args_ns.val_samples
    if args_ns.easy_pool_samples is not None:
        ov["easy_pool_samples"] = args_ns.easy_pool_samples
    if args_ns.grad_accum is not None:
        ov["stage_b_gradient_accumulation"] = args_ns.grad_accum
    if args_ns.max_proxy_source_embeddings is not None:
        ov["max_proxy_source_embeddings"] = args_ns.max_proxy_source_embeddings
    if args_ns.otdd_maxsamples_per_class is not None:
        ov["otdd_maxsamples_per_class"] = args_ns.otdd_maxsamples_per_class
    if args_ns.otdd_inner_exact:
        ov["otdd_inner_exact"] = True
    if args_ns.max_stage_b_steps is not None:
        ov["max_stage_b_steps"] = args_ns.max_stage_b_steps
    if args_ns.no_bf16:
        ov["bf16"] = False
    if ov:
        cfg = dataclasses.replace(cfg, **ov)

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    import orca_rank  # noqa: F401 — third_party OTDD path

    if cfg.dry_run:
        (Path(cfg.output_dir) / "config.json").write_text(json.dumps(cfg.to_jsonable(), indent=2, default=str))
        if cfg.method == "ce_warm_f":
            raise NotImplementedError("ce_warm_f not implemented.")
        if cfg.method == "orca_otdd":
            import torch

            from orca_rank.training.gpu_utils import mps_is_available, pick_device

            if args_ns.device == "auto":
                _chosen = pick_device()
            else:
                _chosen = torch.device(args_ns.device)
            if _chosen.type == "mps":
                print(
                    "dry_run: method=orca_otdd would avoid MPS and use cpu (see --device / OT coupling)."
                )

            torch.cuda.is_available()

            from otdd.pytorch.distance import DatasetDistance

            print(
                "dry_run:",
                DatasetDistance,
                "CUDA?",
                torch.cuda.is_available(),
                "MPS?",
                mps_is_available(),
                "cli_device:",
                args_ns.device,
                "auto_pick_device:",
                pick_device(),
            )
        else:
            print("dry_run (lora_only): config OK — full train needs torch/transformers/peft")

        print("dry_run:", json.dumps(cfg.to_jsonable(), indent=2, default=str))
        return

    if cfg.method == "ce_warm_f":
        raise NotImplementedError("ce_warm_f not implemented.")

    import torch

    from orca_rank.training.gpu_utils import cuda_supports_bf16_autocast, mps_is_available, pick_device

    if args_ns.device == "auto":
        device = pick_device()
    else:
        device = torch.device(args_ns.device)

    orca_otdd_cpu_fallback_reason: str | None = None
    if cfg.method == "orca_otdd" and device.type == "mps":
        orca_otdd_cpu_fallback_reason = (
            "method=orca_otdd: POT/GeomLoss coupling is not reliably differentiable on Apple MPS; "
            "using CPU for the whole run. Use NVIDIA+CUDA or --device cuda for GPU acceleration."
        )
        print(f"orca_rank: {orca_otdd_cpu_fallback_reason}", file=sys.stderr)
        device = torch.device("cpu")

    bf16_trimmed_for_gpu = False
    bf16_trim_reason: str | None = None
    if cfg.bf16 and device.type == "cuda" and not cuda_supports_bf16_autocast():
        cfg = dataclasses.replace(cfg, bf16=False)
        bf16_trimmed_for_gpu = True
        bf16_trim_reason = (
            "GPU does not support bf16 CUDA autocast (e.g. CHTC Tesla P100, Pascal sm_60); "
            "using fp32 compute."
        )
    elif cfg.bf16 and device.type == "mps":
        cfg = dataclasses.replace(cfg, bf16=False)
        bf16_trimmed_for_gpu = True
        bf16_trim_reason = "MPS: using fp32 (bf16 autocast not enabled for portability)."

    meta = cfg.to_jsonable()
    if bf16_trimmed_for_gpu and bf16_trim_reason:
        meta["bf16_autocast_disabled_reason"] = bf16_trim_reason
    meta.update(
        {
            "torch_version": torch.__version__,
            "selected_device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_capability": list(torch.cuda.get_device_capability())
            if torch.cuda.is_available()
            else None,
            "cuda_bf16_autocast_supported": cuda_supports_bf16_autocast()
            if torch.cuda.is_available()
            else False,
            "mps_available": mps_is_available(),
            "mps_used": device.type == "mps",
            "cli_device_request": args_ns.device,
            "orca_otdd_cpu_fallback_reason": orca_otdd_cpu_fallback_reason,
        }
    )
    (Path(cfg.output_dir) / "config.json").write_text(json.dumps(meta, indent=2, default=str))

    from orca_rank.data.alignment_tokenize import tokenize_alignment_prompt_batch
    from orca_rank.data.math_datasets import build_hf_splits, tokenize_split
    from orca_rank.models.pythia_lm import PythiaFrontendCausalLM
    from orca_rank.training.stage_b import train_stage_b
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.manual_seed(cfg.seed)

    easy_raw, hard_train_raw, hard_val_raw, split_info = build_hf_splits(cfg)

    meta["data_splits"] = split_info

    tok_name = cfg.model_name
    hard_train_tok = tokenize_split(hard_train_raw, tok_name, cfg.max_length)

    align_easy = tokenize_alignment_prompt_batch(easy_raw, tok_name, cfg.max_length)
    align_hard = tokenize_alignment_prompt_batch(hard_train_raw, tok_name, cfg.max_length)

    use_adapter = cfg.method == "orca_otdd"

    frontend = PythiaFrontendCausalLM(
        cfg.model_name,
        use_adapter=use_adapter,
        adapter_bottleneck=cfg.adapter_bottleneck,
    )
    frontend.to(device)
    frontend.train()

    stage_a_metrics: dict = {}
    if cfg.method == "orca_otdd":
        from orca_rank.training.stage_a import build_source_tensors, train_stage_a_alignment

        src_td = build_source_tensors(
            frontend,
            align_easy,
            device,
            cfg.max_proxy_source_embeddings,
            int(tokenizer.pad_token_id),
        )
        stage_a_metrics = train_stage_a_alignment(
            frontend,
            cfg,
            align_hard,
            src_td,
            device,
            pad_token_id=int(tokenizer.pad_token_id),
        )
        frontend.train()

    stage_b_metrics = train_stage_b(
        frontend,
        hard_train_tok,
        hard_val_raw,
        cfg,
        device,
        tokenizer,
        Path(cfg.output_dir),
    )

    all_m = {"config": meta, **stage_a_metrics, **stage_b_metrics}
    (Path(cfg.output_dir) / "metrics.json").write_text(json.dumps(all_m, indent=2, default=str))


if __name__ == "__main__":
    main()
