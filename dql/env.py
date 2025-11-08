"""Gym-like environment built on top of case_closed_game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from case_closed_game import Direction, Game, GameResult

from .config import Config
from .game_utils import ALL_DIRECTIONS, legal_action_mask, local_degree
from .observation import Observation, ObservationBuilder
from .opponents import OpponentPool, PolicyFn
from .stance import StanceTracker


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any]


class CaseClosedEnv:
    def __init__(self, config: Config, seed: Optional[int] = None):
        self.config = config
        self.game = Game()
        self.rng = random.Random(seed or config.training.seed)
        self.observer = ObservationBuilder(config.observation, config.stance)
        self.stance_tracker = StanceTracker(config.stance)
        self.opponents = OpponentPool(config.opponents.names, config.opponents.weights)
        self.current_opponent_name: Optional[str] = None
        self.current_policy: Optional[PolicyFn] = None

    def reset(self, opponent_name: Optional[str] = None) -> Tuple[Observation, Dict[str, Any]]:
        self.game.reset()
        self.stance_tracker.reset()
        if opponent_name:
            self.current_opponent_name = opponent_name
            self.current_policy = self.opponents.get_policy(opponent_name)
        else:
            name, policy = self.opponents.sample(self.rng)
            self.current_opponent_name = name
            self.current_policy = policy
        stance_ctx = self.stance_tracker.update(self.game, self.game.agent1, self.game.agent2)
        obs = self.observer.build(self.game, self.game.agent1, self.game.agent2, stance_ctx)
        return obs, {"opponent": self.current_opponent_name, "stance_ctx": stance_ctx}

    def _opponent_move(self) -> Direction:
        if self.current_policy is None:
            raise RuntimeError("Opponent policy not initialized")
        return self.current_policy(self.game, 2)

    def _base_reward(self, result: Optional[GameResult]) -> float:
        if result is None:
            return 0.0
        if result == GameResult.AGENT1_WIN:
            return 1.0
        if result == GameResult.AGENT2_WIN:
            return -1.0
        return 0.0

    def _shaping(self, done: bool) -> Tuple[float, Dict[str, float]]:
        cfg = self.config.rewards
        shaping_terms: Dict[str, float] = {}
        if not done and cfg.living_bonus:
            shaping_terms["living"] = float(np.clip(cfg.living_bonus, -cfg.reward_clip, cfg.reward_clip))
        if cfg.degree_penalty:
            deg = local_degree(self.game.board, *self.game.agent1.trail[-1])
            if deg <= cfg.degree_threshold:
                penalty = -abs(cfg.degree_penalty)
                shaping_terms["degree"] = float(np.clip(penalty, -cfg.reward_clip, cfg.reward_clip))
        total = float(np.clip(sum(shaping_terms.values()), -cfg.reward_clip, cfg.reward_clip)) if shaping_terms else 0.0
        return total, shaping_terms

    def step(self, action_idx: int) -> StepResult:
        me = self.game.agent1
        dirs = list(ALL_DIRECTIONS)
        mask = legal_action_mask(self.game.board, me)
        if not mask[action_idx]:
            legal_indices = np.where(mask)[0]
            if len(legal_indices):
                action_idx = int(self.rng.choice(list(legal_indices)))
        my_action = dirs[action_idx]
        opp_action = self._opponent_move()
        result = self.game.step(my_action, opp_action)

        done = result is not None
        if self.game.turns >= 200:
            done = True
            if result is None:
                result = GameResult.DRAW

        base_reward = self._base_reward(result)
        shaping, shaping_terms = self._shaping(done)
        reward = float(base_reward + shaping)

        stance_ctx = self.stance_tracker.update(self.game, self.game.agent1, self.game.agent2)
        obs = self.observer.build(self.game, self.game.agent1, self.game.agent2, stance_ctx)

        info = {
            "result": result.name if result else None,
            "opponent": self.current_opponent_name,
            "turn": self.game.turns,
            "shaping": shaping_terms,
            "reward_base": base_reward,
            "reward_shaping": shaping,
            "legal_mask": obs.legal_actions,
            "stance_ctx": stance_ctx,
        }
        if done:
            info["death_cause"] = self._death_cause(result)
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _death_cause(self, result: Optional[GameResult]) -> str:
        if result == GameResult.AGENT1_WIN:
            return "opponent_crash"
        if result == GameResult.AGENT2_WIN:
            return "self_crash"
        if result == GameResult.DRAW:
            if not self.game.agent1.alive and not self.game.agent2.alive:
                return "double_crash"
            return "max_turns"
        return "running"
