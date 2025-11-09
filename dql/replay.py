"""Prioritized experience replay buffer."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.storage: List[Dict[str, np.ndarray]] = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.next_idx = 0

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
        data = {
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

        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data

        max_prio = self.priorities.max() if self.storage else 1.0
        if max_prio == 0:
            max_prio = 1.0
        self.priorities[self.next_idx] = max_prio

        self.next_idx = (self.next_idx + 1) % self.capacity

    def __len__(self) -> int:
        return len(self.storage)

    def sample(self, batch_size: int, beta: float) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        if batch_size > len(self.storage):
            raise ValueError("Not enough samples in replay buffer")
        priorities = self.priorities[: len(self.storage)]
        if priorities.sum() == 0:
            priorities = np.ones_like(priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.storage), size=batch_size, replace=False, p=probs)
        weights = (len(self.storage) * probs[indices]) ** (-beta)
        weights /= weights.max()

        keys = self.storage[0].keys()
        batch = {k: [] for k in keys}
        for idx in indices:
            transition = self.storage[idx]
            for key, value in transition.items():
                batch[key].append(value)
        stacked = {k: np.stack(v) if isinstance(v[0], np.ndarray) else np.asarray(v) for k, v in batch.items()}
        return stacked, indices, weights.astype(np.float32), probs[indices]

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        td_errors = np.abs(td_errors) + 1e-3
        for idx, err in zip(indices, td_errors):
            self.priorities[idx] = float(err)
