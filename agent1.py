"""
Case Closed agent server (Flask) compatible with the Judge Engine.

This file is the entry point for your agent. It exposes the required
HTTP endpoints and contains a simple, safe baseline move policy so the
agent runs out of the box. You can later replace the decision logic with
your own (e.g., by importing from dqn_agent.py).
"""

import os
import random
from flask import Flask, request, jsonify

app = Flask(__name__)

# Identity for latency check endpoint
PARTICIPANT = os.getenv("PARTICIPANT", "Participant")
AGENT_NAME = os.getenv("AGENT_NAME", "MyAgent")

# Simple in-memory state store updated by /send-state
game_state = {
    "board": None,
    "agent1_trail": [],
    "agent2_trail": [],
    "agent1_length": 0,
    "agent2_length": 0,
    "agent1_alive": True,
    "agent2_alive": True,
    "agent1_boosts": 3,
    "agent2_boosts": 3,
    "turn_count": 0,
    "player_number": 1,
}


@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge pushes the current game state here as JSON."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400

    game_state.update(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge requests the agent's move for the current tick.

    Return: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    """
    player_number = request.args.get("player_number", default=1, type=int)
    turn_count = request.args.get("turn_count", default=game_state.get("turn_count", 0), type=int)

    if player_number == 1:
        my_trail = game_state.get("agent1_trail", [])
        my_boosts = game_state.get("agent1_boosts", 3)
        other_trail = game_state.get("agent2_trail", [])
    else:
        my_trail = game_state.get("agent2_trail", [])
        my_boosts = game_state.get("agent2_boosts", 3)
        other_trail = game_state.get("agent1_trail", [])

    board = game_state.get("board") or [[0 for _ in range(20)] for _ in range(18)]
    move = decide_safe_move(board, my_trail, other_trail, turn_count, my_boosts)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies that the match finished and provides final state."""
    data = request.get_json(silent=True) or {}
    result = data.get("result", "UNKNOWN")
    print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


def decide_safe_move(board, my_trail, other_trail, turn_count, my_boosts):
    """Heuristic policy that tries to avoid immediate collisions.

    - Prefer continuing straight unless it's unsafe
    - Avoid moving into any cell occupied by a trail
    - Use boost mid-game when available (optional)
    """
    height = len(board)
    width = len(board[0]) if height > 0 else 20

    # Determine current head and direction
    if not my_trail:
        # If we don't have state yet, default to RIGHT
        return "RIGHT"

    head = my_trail[-1]
    cur_dir = infer_direction(my_trail)

    # Candidate directions, avoid opposite of current
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    if cur_dir in opposite and opposite[cur_dir] in directions:
        directions.remove(opposite[cur_dir])

    # Rank with current direction first
    if cur_dir in directions:
        ordered = [cur_dir] + [d for d in directions if d != cur_dir]
    else:
        ordered = directions

    # Build occupied set from board and trails for safety
    occupied = set()
    for y in range(height):
        for x in range(width):
            if board[y][x] != 0:
                occupied.add((x, y))
    for p in my_trail:
        occupied.add(tuple(p))
    for p in other_trail:
        occupied.add(tuple(p))

    # Choose first safe move; fallback to first available
    for d in ordered:
        nx, ny = next_pos(head, d, width, height)
        if (nx, ny) not in occupied:
            use_boost = my_boosts > 0 and 30 <= turn_count <= 80
            return f"{d}:BOOST" if use_boost else d

    # If no safe move, pick current or any direction to avoid forfeit
    use_boost = False
    fallback = cur_dir if cur_dir in ["UP", "DOWN", "LEFT", "RIGHT"] else random.choice(["UP", "DOWN", "LEFT", "RIGHT"]) 
    return f"{fallback}:BOOST" if use_boost else fallback


def infer_direction(trail):
    if len(trail) < 2:
        return "RIGHT"
    prev = trail[-2]
    head = trail[-1]
    dx = head[0] - prev[0]
    dy = head[1] - prev[1]
    # Normalize for torus wrapping
    if abs(dx) > 1:
        dx = -1 if dx > 0 else 1
    if abs(dy) > 1:
        dy = -1 if dy > 0 else 1
    if dx == 1:
        return "RIGHT"
    if dx == -1:
        return "LEFT"
    if dy == 1:
        return "DOWN"
    if dy == -1:
        return "UP"
    return "RIGHT"


def next_pos(head, direction, width, height):
    x, y = head
    if direction == "UP":
        y -= 1
    elif direction == "DOWN":
        y += 1
    elif direction == "LEFT":
        x -= 1
    elif direction == "RIGHT":
        x += 1
    # Torus wrap
    x %= width
    y %= height
    return x, y


if __name__ == "__main__":
    # Default to port 5008 to pair with sample_agent (5009)
    port = int(os.environ.get("PORT", "5009"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
