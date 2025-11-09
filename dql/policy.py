"""Shared policy loader for inference/eval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from case_closed_game import Direction, Game

from .config import Config, load_config
from .game_utils import ALL_DIRECTIONS
from .model import DQNNetwork
from .observation import ObservationBuilder
from .stance import StanceTracker


@dataclass
class PolicyOutput:
    action_idx: int
    q_values: np.ndarray
    legal_mask: np.ndarray
    observation_scalars: np.ndarray


class DQNPolicy:
    def __init__(self, config: Config, checkpoint_path: str):
        self.config = config
        self.device = torch.device("cpu")
        self.checkpoint_path = checkpoint_path
        self.builder = ObservationBuilder(config.observation, config.stance, config.actions)
        self.stance_trackers = {1: StanceTracker(config.stance), 2: StanceTracker(config.stance)}
        self.model, self.meta = self._load_model(checkpoint_path)
        self.rng = np.random.default_rng(config.training.seed)

    def _load_model(self, path: str) -> Tuple[DQNNetwork, dict]:
        payload = torch.load(path, map_location="cpu")
        hidden_sizes = payload.get("hidden_sizes") or self.config.dqn.hidden_sizes
        num_actions = payload.get("num_actions", len(ALL_DIRECTIONS))
        model = DQNNetwork(
            payload["crop_channels"],
            payload.get("crop_size", self.config.observation.crop_size),
            payload["scalar_dim"],
            hidden_sizes,
            num_actions=num_actions,
        )
        state_dict = payload.get("state_dict")
        if state_dict:
            model.load_state_dict(state_dict)
        else:
            print("[policy] Warning: checkpoint missing state_dict, using random weights")
        model.eval()
        return model, payload

    def reset(self) -> None:
        for tracker in self.stance_trackers.values():
            tracker.reset()

    def _get_agents(self, game: Game, player_number: int):
        if player_number == 1:
            return game.agent1, game.agent2
        return game.agent2, game.agent1

    def predict(self, game: Game, player_number: int = 1) -> Tuple[Direction, PolicyOutput]:
        me, opp = self._get_agents(game, player_number)
        stance_ctx = self.stance_trackers[player_number].update(game, me, opp)
        obs = self.builder.build(game, me, opp, stance_ctx)
        if obs.crop.shape[0] != self.meta["crop_channels"] or obs.crop.shape[1] != self.meta["crop_size"]:
            raise ValueError("Observation crop shape does not match checkpoint metadata")
        if obs.scalars.shape[0] != self.meta["scalar_dim"]:
            raise ValueError("Scalar feature mismatch with checkpoint")
        crop = torch.as_tensor(obs.crop, dtype=torch.float32, device=self.device).unsqueeze(0)
        scalars = torch.as_tensor(obs.scalars, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(crop, scalars).cpu().numpy()[0]
        mask = obs.legal_actions.astype(bool)
        if self.config.inference.mask_invalid:
            q_values = q_values.copy()
            q_values[~mask] = -1e9
        best = q_values.max()
        tie_eps = self.config.exploration.tie_eps
        if tie_eps > 0:
            candidates = np.where(np.abs(q_values - best) <= tie_eps)[0]
        else:
            candidates = np.where(q_values == best)[0]
        action_idx = int(self.rng.choice(candidates)) if len(candidates) else int(np.argmax(q_values))
        direction = ALL_DIRECTIONS[action_idx]
        return direction, PolicyOutput(action_idx, q_values, mask, obs.scalars)


def load_policy(checkpoint_path: str, config_path: Optional[str] = None) -> DQNPolicy:
    config = load_config(config_path)
    return DQNPolicy(config, checkpoint_path)
