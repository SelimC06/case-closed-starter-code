"""Utility helpers shared across the DQN stack."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from case_closed_game import AGENT, Direction, GameBoard


DIR_VECS: Dict[Direction, Tuple[int, int]] = {
    Direction.UP: (0, -1),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
    Direction.RIGHT: (1, 0),
}

OPPOSITE: Dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

ALL_DIRECTIONS = tuple(DIR_VECS.keys())


def wrap_pos(x: int, y: int, width: int, height: int) -> Tuple[int, int]:
    return x % width, y % height


def is_blocked(board: GameBoard, x: int, y: int) -> bool:
    return board.grid[y % board.height][x % board.width] == AGENT


def legal_action_mask(board: GameBoard, agent, include_reverse: bool = False):
    mask = np.ones(len(ALL_DIRECTIONS), dtype=bool)
    head_x, head_y = agent.trail[-1]
    for idx, direction in enumerate(ALL_DIRECTIONS):
        if not include_reverse and agent.direction and direction == OPPOSITE[agent.direction]:
            mask[idx] = False
            continue
        dx, dy = DIR_VECS[direction]
        nx, ny = wrap_pos(head_x + dx, head_y + dy, board.width, board.height)
        if is_blocked(board, nx, ny):
            mask[idx] = False
    return mask


def boost_legal_mask(board: GameBoard, agent) -> np.ndarray:
    mask = legal_action_mask(board, agent)
    if agent.boosts_remaining <= 0:
        return np.zeros_like(mask, dtype=bool)
    head_x, head_y = agent.trail[-1]
    boost_mask = mask.copy()
    for idx, direction in enumerate(ALL_DIRECTIONS):
        if not boost_mask[idx]:
            continue
        dx, dy = DIR_VECS[direction]
        nx, ny = wrap_pos(head_x + dx, head_y + dy, board.width, board.height)
        nx2, ny2 = wrap_pos(nx + dx, ny + dy, board.width, board.height)
        if is_blocked(board, nx2, ny2):
            boost_mask[idx] = False
    return boost_mask


def local_degree(board: GameBoard, x: int, y: int) -> int:
    deg = 0
    for dx, dy in DIR_VECS.values():
        nx, ny = wrap_pos(x + dx, y + dy, board.width, board.height)
        if not is_blocked(board, nx, ny):
            deg += 1
    return deg


def torus_delta(ax: int, ay: int, bx: int, by: int, width: int, height: int) -> Tuple[int, int]:
    dx = bx - ax
    dy = by - ay
    if abs(dx) > width // 2:
        dx -= np.sign(dx) * width
    if abs(dy) > height // 2:
        dy -= np.sign(dy) * height
    return dx, dy


def flood_fill_area(board: GameBoard, start: Tuple[int, int], cap: int = 9999) -> int:
    x0, y0 = wrap_pos(start[0], start[1], board.width, board.height)
    if is_blocked(board, x0, y0):
        return 0
    seen = {(x0, y0)}
    stack = [(x0, y0)]
    area = 0
    while stack and area < cap:
        x, y = stack.pop()
        area += 1
        for dx, dy in DIR_VECS.values():
            nx, ny = wrap_pos(x + dx, y + dy, board.width, board.height)
            if (nx, ny) not in seen and not is_blocked(board, nx, ny):
                seen.add((nx, ny))
                stack.append((nx, ny))
    return area
