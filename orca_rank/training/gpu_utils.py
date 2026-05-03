"""GPU / Metal capability helpers (CUDA, Apple MPS, CPU)."""

from __future__ import annotations

import torch


def mps_is_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and bool(mps.is_available())


def pick_device() -> torch.device:
    """Prefer CUDA, then Apple Metal (MPS), else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def bf16_autocast_enabled(device: torch.device, bf16_requested: bool) -> bool:
    """bf16 autocast only on CUDA when hardware supports it; MPS/CPU train in fp32."""
    if not bf16_requested:
        return False
    if device.type == "cuda":
        return cuda_supports_bf16_autocast()
    return False


def autocast_for_device(device: torch.device, bf16_requested: bool):
    """Training/eval autocast: bf16 on capable CUDA; otherwise disabled (fp32)."""
    use = bf16_autocast_enabled(device, bf16_requested)
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use)


def autocast_maybe_bf16_cuda(device: torch.device | None = None, *, bf16: bool = True):
    """Backward-compatible alias; prefer :func:`autocast_for_device`."""
    dev = device if device is not None else pick_device()
    return autocast_for_device(dev, bf16)
