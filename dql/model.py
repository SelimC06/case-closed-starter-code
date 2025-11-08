"""Neural network modules."""

from __future__ import annotations

from typing import List

import torch
from torch import nn


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, crop_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.crop_size = crop_size
        self.out_channels = 32

    @property
    def output_dim(self) -> int:
        return self.out_channels * self.crop_size * self.crop_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.flatten(self.net(x))


class DQNNetwork(nn.Module):
    def __init__(self, in_channels: int, crop_size: int, scalar_dim: int, hidden_sizes: List[int], num_actions: int = 4):
        super().__init__()
        self.encoder = ConvEncoder(in_channels, crop_size)
        layers: List[nn.Module] = []
        prev_dim = self.encoder.output_dim + scalar_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))
        self.head = nn.Sequential(*layers)

    def forward(self, crop: torch.Tensor, scalars: torch.Tensor | None) -> torch.Tensor:
        if crop.dim() == 3:
            crop = crop.unsqueeze(0)
        if scalars is not None and scalars.dim() == 1:
            scalars = scalars.unsqueeze(0)
        encoded = self.encoder(crop)
        if scalars is not None and scalars.shape[1] > 0:
            x = torch.cat([encoded, scalars], dim=-1)
        else:
            x = encoded
        return self.head(x)
