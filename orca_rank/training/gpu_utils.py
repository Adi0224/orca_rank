"""GPU capability helpers (Condor pools often use Tesla P100 = Pascal sm_60)."""

from __future__ import annotations

import torch


def cuda_supports_bf16_autocast() -> bool:
    """True iff PyTorch can use CUDA autocast with bfloat16 on the current device.

    Ampere (sm_80+) typically returns True; **Pascal P100 (sm_60) returns False** — use fp32 there.
    """
    if not torch.cuda.is_available():
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        return bool(fn())
    major, _minor = torch.cuda.get_device_capability()
    return major >= 8


def autocast_maybe_bf16_cuda():
    """Training/eval autocast: bf16 only when supported; otherwise disabled → fp32 on GPU."""
    if not torch.cuda.is_available():
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=False)
    use = cuda_supports_bf16_autocast()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use)
