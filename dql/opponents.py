"""Rule-based opponents for training/evaluation."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from case_closed_game import Direction, Game

from .game_utils import DIR_VECS, OPPOSITE, legal_action_mask


PolicyFn = Callable[[Game, int], Direction]


def _forward_or_turn(game: Game, player: int, prefer_left: bool = True) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    current = agent.direction or Direction.RIGHT
    dirs = list(DIR_VECS.keys())
    if mask[dirs.index(current)]:
        return current
    ordered = dirs if prefer_left else list(reversed(dirs))
    for d in ordered:
        if mask[dirs.index(d)]:
            return d
    return current


def wall_hugger(game: Game, player: int) -> Direction:
    return _forward_or_turn(game, player, prefer_left=True)


def cutter(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    if agent.direction and mask[dirs.index(agent.direction)]:
        return agent.direction
    perp = {
        Direction.UP: [Direction.LEFT, Direction.RIGHT],
        Direction.DOWN: [Direction.RIGHT, Direction.LEFT],
        Direction.LEFT: [Direction.DOWN, Direction.UP],
        Direction.RIGHT: [Direction.UP, Direction.DOWN],
    }
    if agent.direction:
        for d in perp[agent.direction]:
            if mask[dirs.index(d)]:
                return d
    for d in dirs:
        if mask[dirs.index(d)]:
            return d
    return agent.direction or Direction.RIGHT


def random_inertia(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    if agent.direction and mask[dirs.index(agent.direction)] and random.random() < 0.7:
        return agent.direction
    legal = [d for d in dirs if mask[dirs.index(d)]]
    return random.choice(legal) if legal else agent.direction or Direction.RIGHT


def straight_biased(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    if agent.direction and mask[dirs.index(agent.direction)]:
        return agent.direction
    cx, cy = game.board.width // 2, game.board.height // 2
    hx, hy = agent.trail[-1]
    if hx < cx and mask[dirs.index(Direction.RIGHT)]:
        return Direction.RIGHT
    if hx > cx and mask[dirs.index(Direction.LEFT)]:
        return Direction.LEFT
    if hy < cy and mask[dirs.index(Direction.DOWN)]:
        return Direction.DOWN
    if hy > cy and mask[dirs.index(Direction.UP)]:
        return Direction.UP
    for d in dirs:
        if mask[dirs.index(d)]:
            return d
    return agent.direction or Direction.RIGHT


POLICIES: Dict[str, PolicyFn] = {
    "wall_hugger": wall_hugger,
    "cutter": cutter,
    "random_inertia": random_inertia,
    "straight_biased": straight_biased,
}


class OpponentPool:
    def __init__(self, names: List[str], weights: List[float]):
        if len(names) != len(weights):
            raise ValueError("Opponent names and weights must align")
        self.names = names
        self.weights = weights

    def sample(self, rng: random.Random) -> Tuple[str, PolicyFn]:
        choice = rng.choices(self.names, weights=self.weights, k=1)[0]
        return choice, POLICIES[choice]

    def get_policy(self, name: str) -> PolicyFn:
        if name not in POLICIES:
            raise KeyError(f"Unknown opponent: {name}")
        return POLICIES[name]
