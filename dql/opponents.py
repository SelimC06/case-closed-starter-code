"""Rule-based opponents for training/evaluation."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from case_closed_game import Direction, Game

from .game_utils import DIR_VECS, OPPOSITE, flood_fill_area, legal_action_mask, torus_delta, wrap_pos


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


def spiral_runner(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    preferred = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
    current_index = preferred.index(agent.direction) if agent.direction in preferred else 0
    for offset in range(len(preferred)):
        idx = (current_index + offset) % len(preferred)
        direction = preferred[idx]
        if mask[dirs.index(direction)]:
            return direction
    for direction in dirs:
        if mask[dirs.index(direction)]:
            return direction
    return agent.direction or Direction.RIGHT


def aggressive_chaser(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    opp = game.agent2 if player == 1 else game.agent1
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    hx, hy = agent.trail[-1]
    ox, oy = opp.trail[-1]
    width, height = game.board.width, game.board.height

    best_dir = None
    best_dist = float("inf")
    for direction in dirs:
        if not mask[dirs.index(direction)]:
            continue
        dx, dy = DIR_VECS[direction]
        nx, ny = wrap_pos(hx + dx, hy + dy, width, height)
        tx, ty = torus_delta(nx, ny, ox, oy, width, height)
        dist = abs(tx) + abs(ty)
        if dist < best_dist:
            best_dist = dist
            best_dir = direction
    if best_dir is not None:
        return best_dir
    return agent.direction or Direction.RIGHT


def random_boost_imitator(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    if agent.direction and mask[dirs.index(agent.direction)] and random.random() < 0.6:
        return agent.direction
    forward = agent.direction or random.choice(dirs)
    clockwise = {
        Direction.UP: Direction.RIGHT,
        Direction.RIGHT: Direction.DOWN,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
    }
    options = [forward, clockwise[forward], clockwise[clockwise[forward]], clockwise[clockwise[clockwise[forward]]]]
    for d in options:
        if mask[dirs.index(d)]:
            return d
    return agent.direction or Direction.RIGHT


def safe_pocket_seeker(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    best = None
    best_area = -1
    for direction in dirs:
        if not mask[dirs.index(direction)]:
            continue
        dx, dy = DIR_VECS[direction]
        nx, ny = wrap_pos(agent.trail[-1][0] + dx, agent.trail[-1][1] + dy, game.board.width, game.board.height)
        area = flood_fill_area(game.board, (nx, ny), cap=400)
        if area > best_area:
            best_area = area
            best = direction
    if best is not None:
        return best
    return agent.direction or Direction.RIGHT


def mirror_player(game: Game, player: int) -> Direction:
    agent = game.agent1 if player == 1 else game.agent2
    opponent = game.agent2 if player == 1 else game.agent1
    dirs = list(DIR_VECS.keys())
    mask = legal_action_mask(game.board, agent, include_reverse=False)
    if opponent.direction and mask[dirs.index(opponent.direction)] and random.random() < 0.8:
        return opponent.direction
    if agent.direction and mask[dirs.index(agent.direction)]:
        return agent.direction
    legal = [d for d in dirs if mask[dirs.index(d)]]
    return random.choice(legal) if legal else agent.direction or Direction.RIGHT


POLICIES: Dict[str, PolicyFn] = {
    "wall_hugger": wall_hugger,
    "cutter": cutter,
    "random_inertia": random_inertia,
    "straight_biased": straight_biased,
    "spiral_runner": spiral_runner,
    "aggressive_chaser": aggressive_chaser,
    "random_boost": random_boost_imitator,
    "safe_pocket": safe_pocket_seeker,
    "mirror_player": mirror_player,
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
