"""Experience replay buffer."""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: Deque[Dict[str, np.ndarray]] = deque(maxlen=capacity)

    def add(
        self,
        obs_crop: np.ndarray,
        obs_scalars: np.ndarray,
        legal_mask: np.ndarray,
        action: int,
        reward: float,
        next_crop: np.ndarray,
        next_scalars: np.ndarray,
        next_legal: np.ndarray,
        done: bool,
    ) -> None:
        self.storage.append(
            {
                "obs_crop": obs_crop.astype(np.float32, copy=True),
                "obs_scalars": obs_scalars.astype(np.float32, copy=True),
                "legal_mask": legal_mask.astype(bool, copy=True),
                "action": int(action),
                "reward": float(reward),
                "next_crop": next_crop.astype(np.float32, copy=True),
                "next_scalars": next_scalars.astype(np.float32, copy=True),
                "next_legal": next_legal.astype(bool, copy=True),
                "done": bool(done),
            }
        )

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if batch_size > len(self.storage):
            raise ValueError("Not enough samples in replay buffer")
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False)
        keys = self.storage[0].keys()
        batch = {k: [] for k in keys}
        for idx in indices:
            transition = self.storage[idx]
            for key, value in transition.items():
                batch[key].append(value)
        return {k: np.stack(v) if isinstance(v[0], np.ndarray) else np.asarray(v) for k, v in batch.items()}
