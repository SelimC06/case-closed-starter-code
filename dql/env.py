"""Gym-like environment built on top of case_closed_game."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from case_closed_game import Agent, Direction, Game, GameBoard, GameResult

from .config import Config
from .game_utils import ALL_DIRECTIONS, boost_legal_mask, flood_fill_area, legal_action_mask, local_degree
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
        self.observer = ObservationBuilder(config.observation, config.stance, config.actions)
        self.stance_tracker = StanceTracker(config.stance)
        self.opponents = OpponentPool(config.opponents.names, config.opponents.weights)
        self.current_opponent_name: Optional[str] = None
        self.current_policy: Optional[PolicyFn] = None
        self._action_dim = len(ALL_DIRECTIONS) * (2 if config.actions.use_boost else 1)
        self.prev_area: Optional[int] = None

    def reset(self, opponent_name: Optional[str] = None) -> Tuple[Observation, Dict[str, Any]]:
        self.game.reset()
        if self.config.training.randomize_starts:
            self._randomize_start_positions()
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
        self.prev_area = flood_fill_area(
            self.game.board, self.game.agent1.trail[-1], cap=self.config.rewards.area_cap
        )
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

    def _shaping(self, done: bool, area_delta: float) -> Tuple[float, Dict[str, float]]:
        cfg = self.config.rewards
        shaping_terms: Dict[str, float] = {}
        if not done:
            living_bonus = cfg.living_bonus_max
            if self.game.turns >= cfg.living_bonus_decay_turn:
                decay_den = max(1, 200 - cfg.living_bonus_decay_turn)
                decay_frac = min(1.0, (self.game.turns - cfg.living_bonus_decay_turn) / decay_den)
                living_bonus = cfg.living_bonus_max - (cfg.living_bonus_max - cfg.living_bonus_min) * decay_frac
            living_bonus = float(np.clip(living_bonus, cfg.living_bonus_min, cfg.living_bonus_max))
            if living_bonus:
                shaping_terms["living"] = living_bonus
        if cfg.degree_penalty:
            deg = local_degree(self.game.board, *self.game.agent1.trail[-1])
            if deg <= cfg.degree_threshold:
                penalty = -abs(cfg.degree_penalty)
                shaping_terms["degree"] = float(np.clip(penalty, -cfg.reward_clip, cfg.reward_clip))
        if cfg.area_bonus_scale:
            board_area = self.game.board.width * self.game.board.height
            delta_norm = area_delta / max(1, board_area)
            delta_clipped = float(np.clip(delta_norm, -cfg.area_delta_clip, cfg.area_delta_clip))
            if delta_clipped:
                area_reward = cfg.area_bonus_scale * delta_clipped
                shaping_terms["area"] = area_reward
        total = float(np.clip(sum(shaping_terms.values()), -cfg.reward_clip, cfg.reward_clip)) if shaping_terms else 0.0
        return total, shaping_terms

    def step(self, action_idx: int) -> StepResult:
        me = self.game.agent1
        dirs = list(ALL_DIRECTIONS)
        mask = legal_action_mask(self.game.board, me)
        if self.config.actions.use_boost:
            boost_mask = boost_legal_mask(self.game.board, me)
            full_mask = np.concatenate([mask, boost_mask])
        else:
            full_mask = mask
        if not full_mask[action_idx]:
            legal_indices = np.where(full_mask)[0]
            if len(legal_indices):
                action_idx = int(self.rng.choice(list(legal_indices)))
        dir_index = action_idx % len(dirs)
        use_boost = self.config.actions.use_boost and action_idx >= len(dirs)
        my_action = dirs[dir_index]
        opp_action = self._opponent_move()
        result = self.game.step(my_action, opp_action, boost1=use_boost)

        done = result is not None
        if self.game.turns >= 200:
            done = True
            if result is None:
                result = GameResult.DRAW

        base_reward = self._base_reward(result)
        area_now = flood_fill_area(
            self.game.board, self.game.agent1.trail[-1], cap=self.config.rewards.area_cap
        )
        area_delta = 0.0
        if self.prev_area is not None:
            area_delta = area_now - self.prev_area
        self.prev_area = area_now
        shaping, shaping_terms = self._shaping(done, area_delta)
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

    def _randomize_start_positions(self) -> None:
        board = GameBoard()
        directions = list(Direction)
        occupied = set()

        def sample_start():
            while True:
                x = self.rng.randrange(board.width)
                y = self.rng.randrange(board.height)
                start = (x, y)
                direction = self.rng.choice(directions)
                dx, dy = direction.value
                second = ((x + dx) % board.width, (y + dy) % board.height)
                if start in occupied or second in occupied:
                    continue
                occupied.add(start)
                occupied.add(second)
                return start, direction

        start1, dir1 = sample_start()
        start2, dir2 = sample_start()

        self.game.board = board
        self.game.agent1 = Agent(agent_id=1, start_pos=start1, start_dir=dir1, board=board)
        self.game.agent2 = Agent(agent_id=2, start_pos=start2, start_dir=dir2, board=board)
        self.game.turns = 0
