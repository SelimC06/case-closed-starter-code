"""Stance detector and aggression tracker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from case_closed_game import Agent, Game

from .config import StanceConfig
from .game_utils import DIR_VECS, flood_fill_area, torus_delta


STANCE_PACIFIST = "PACIFIST"
STANCE_AGGRESSIVE = "AGGRESSIVE"


class StanceTracker:
    def __init__(self, cfg: StanceConfig):
        self.cfg = cfg
        self.agg_score = 0.0
        self.prev_distance = None
        self.prev_area = None
        self.ctx: Dict[str, float] = {}

    def reset(self) -> None:
        self.agg_score = 0.0
        self.prev_distance = None
        self.prev_area = None
        self.ctx = {}

    def update(self, game: Game, me: Agent, opp: Agent) -> Dict[str, float]:
        dx, dy = torus_delta(
            me.trail[-1][0],
            me.trail[-1][1],
            opp.trail[-1][0],
            opp.trail[-1][1],
            game.board.width,
            game.board.height,
        )
        distance = abs(dx) + abs(dy)
        distance_delta = 0.0 if self.prev_distance is None else self.prev_distance - distance
        self.prev_distance = distance

        area = flood_fill_area(game.board, me.trail[-1], cap=self.cfg.area_cap)
        area_delta = 0.0 if self.prev_area is None else area - self.prev_area
        self.prev_area = area

        proximity = 1.0 if distance <= self.cfg.proximity_radius else 0.0

        heading_alignment = 0.0
        if opp.direction is not None:
            vx, vy = DIR_VECS[opp.direction]
            heading_alignment = -float(vx * dx + vy * dy)
        heading_alignment = max(-1.0, min(1.0, heading_alignment))

        signal = (
            0.3 * (1.0 if distance_delta < 0 else 0.0)
            + 0.2 * (1.0 if area_delta < 0 else 0.0)
            + 0.3 * proximity
            + self.cfg.heading_alignment_weight * max(0.0, heading_alignment)
        )
        self.agg_score = (1 - self.cfg.aggression_alpha) * self.agg_score + self.cfg.aggression_alpha * signal

        stance = STANCE_AGGRESSIVE if self.agg_score >= self.cfg.hysteresis_high else STANCE_PACIFIST
        if self.agg_score <= self.cfg.hysteresis_low:
            stance = STANCE_PACIFIST

        self.ctx = {
            "stance": stance,
            "agg_score": float(self.agg_score),
            "distance_delta": float(distance_delta),
            "proximity": float(proximity),
            "heading_alignment": float(heading_alignment),
            "area_delta": float(area_delta),
        }
        return self.ctx

    def get_ctx(self) -> Dict[str, float]:
        return dict(self.ctx)
