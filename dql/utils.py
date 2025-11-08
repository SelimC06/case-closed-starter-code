"""Misc helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def linear_epsilon(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return eps_end
    frac = min(1.0, step / decay_steps)
    return eps_start + (eps_end - eps_start) * frac


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)


def to_tensor(array, device: torch.device) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device)
    return torch.as_tensor(array, device=device)
