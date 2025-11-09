"""Observation builder for Case Closed DQN agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from case_closed_game import Agent, Game

from .config import ActionConfig, ObservationConfig, StanceConfig
from .game_utils import boost_legal_mask, legal_action_mask, local_degree, torus_delta


@dataclass
class Observation:
    crop: np.ndarray
    scalars: np.ndarray
    legal_actions: np.ndarray
    stance_ctx: Dict[str, float]


class ObservationBuilder:
    def __init__(self, obs_cfg: ObservationConfig, stance_cfg: StanceConfig, action_cfg: ActionConfig):
        self.cfg = obs_cfg
        self.stance_cfg = stance_cfg
        self.action_cfg = action_cfg
        self._scalar_dim_cache = None

    def _global_layers(self, game: Game, me: Agent, opp: Agent) -> np.ndarray:
        H, W = game.board.height, game.board.width
        if self.cfg.channels_mode == "split2":
            layers = np.zeros((2, H, W), dtype=np.float32)
            for (x, y) in me.trail:
                layers[0, y % H, x % W] = 1.0
            for (x, y) in opp.trail:
                layers[1, y % H, x % W] = 1.0
            return layers
        grid = np.asarray(game.board.grid, dtype=np.float32)
        return grid[None, :, :]

    def _crop(self, layers: np.ndarray, center: Tuple[int, int], width: int, height: int) -> np.ndarray:
        size = self.cfg.crop_size
        cx, cy = center
        half = size // 2
        if self.cfg.centering_mode == "odd-center":
            start_x = cx - half
            start_y = cy - half
        else:
            start_x = cx - (half - 1)
            start_y = cy - (half - 1)
        crop = np.zeros((layers.shape[0], size, size), dtype=np.float32)
        for iy in range(size):
            for ix in range(size):
                gx = (start_x + ix) % width
                gy = (start_y + iy) % height
                crop[:, iy, ix] = layers[:, gy, gx]
        return crop

    def _scalar_vector(
        self,
        game: Game,
        me: Agent,
        opp: Agent,
        turn_fraction: float,
        stance_ctx: Dict[str, float],
    ) -> np.ndarray:
        scalars = []
        if self.cfg.include_degree:
            scalars.append(local_degree(game.board, *me.trail[-1]))
        if self.cfg.include_opponent_degree:
            scalars.append(local_degree(game.board, *opp.trail[-1]))
        if self.cfg.include_relative_delta:
            dx, dy = torus_delta(
                me.trail[-1][0],
                me.trail[-1][1],
                opp.trail[-1][0],
                opp.trail[-1][1],
                game.board.width,
                game.board.height,
            )
            scalars.append(dx / max(1, game.board.width // 2))
            scalars.append(dy / max(1, game.board.height // 2))
        if self.cfg.include_turn_fraction:
            scalars.append(turn_fraction)
        if self.cfg.include_boosts:
            scalars.append(me.boosts_remaining / 3.0)
            scalars.append(opp.boosts_remaining / 3.0)
        if self.stance_cfg.use_stance_features:
            scalars.extend(
                [
                    stance_ctx.get("agg_score", 0.0),
                    stance_ctx.get("distance_delta", 0.0),
                    stance_ctx.get("proximity", 0.0),
                    stance_ctx.get("heading_alignment", 0.0),
                ]
            )
        return np.asarray(scalars, dtype=np.float32)

    def build(self, game: Game, me: Agent, opp: Agent, stance_ctx: Dict[str, float]) -> Observation:
        H, W = game.board.height, game.board.width
        turn_fraction = min(1.0, game.turns / 200.0)
        layers = self._global_layers(game, me, opp)
        crop = self._crop(layers, me.trail[-1], W, H)
        scalars = self._scalar_vector(game, me, opp, turn_fraction, stance_ctx)
        mask = legal_action_mask(game.board, me)
        if self.action_cfg.use_boost:
            boost_mask = boost_legal_mask(game.board, me)
            legal = np.concatenate([mask, boost_mask])
        else:
            legal = mask
        return Observation(crop=crop, scalars=scalars, legal_actions=legal, stance_ctx=stance_ctx)

    @property
    def scalar_dim(self) -> int:
        if self._scalar_dim_cache is None:
            dummy = Game()
            obs = self.build(dummy, dummy.agent1, dummy.agent2, {})
            self._scalar_dim_cache = int(obs.scalars.shape[0])
        return self._scalar_dim_cache
