"""
Agent1 Flask server that behaves like agent.py but uses a DQN model
for action selection if available. Falls back to the same safe
heuristic if the model or torch is unavailable.

Default port: 5009 (pairs with agent.py on 5008 for local judging).
"""

import os
import random
from typing import Tuple, List

from flask import Flask, request, jsonify

# Optional ML imports (fallback if not available)
try:
    import numpy as np
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

app = Flask(__name__)

# Identity for latency check endpoint
PARTICIPANT = os.getenv("PARTICIPANT", "Participant")
AGENT_NAME = os.getenv("AGENT_NAME", "Agent1DQN")

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
    "player_number": 2,  # default identity for this server
}


# --------------------
# DQN model definition
# --------------------
class DQNNet(nn.Module):
    def __init__(self, in_channels=5, num_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Flatten()
        )
        conv_out = 64 * 18 * 20  # for 18x20 input
        self.fc = nn.Sequential(
            nn.Linear(conv_out + 8, 512),  # +8 for auxiliary features
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, img, aux):
        x = self.conv(img)
        x = torch.cat([x, aux], dim=1)
        return self.fc(x)


MODEL = None
DEVICE = "cpu"

def try_load_model():
    global MODEL
    if not TORCH_AVAILABLE:
        return
    model_path = os.getenv("MODEL_PATH", "von_model.pth")
    if os.path.isfile(model_path):
        m = DQNNet()
        state = torch.load(model_path, map_location=DEVICE)
        m.load_state_dict(state)
        m.eval()
        MODEL = m


def encode_state(board, my_trail, other_trail) -> Tuple['np.ndarray', 'np.ndarray']:
    """Build (img, aux) suitable for DQNNet input.

    img: (5, H, W)
      - 0: my trail
      - 1: other trail
      - 2: my head
      - 3: other head
      - 4: occupancy from board (non-zero)
    aux: (8,) simple features: [turn/500, my_len/200, other_len/200, my_boost/3, other_boost/3, H/20, W/20, bias]
    """
    H = len(board)
    W = len(board[0]) if H > 0 else 20

    import numpy as _np  # local alias even if np missing at import time
    img = _np.zeros((5, H, W), dtype=_np.float32)

    # Board occupancy
    for y in range(H):
        for x in range(W):
            if board[y][x] != 0:
                img[4, y, x] = 1.0

    # Trails and heads
    if my_trail:
        for (x, y) in my_trail:
            img[0, y % H, x % W] = 1.0
        hx, hy = my_trail[-1]
        img[2, hy % H, hx % W] = 1.0
    if other_trail:
        for (x, y) in other_trail:
            img[1, y % H, x % W] = 1.0
        ox, oy = other_trail[-1]
        img[3, oy % H, ox % W] = 1.0

    my_len = len(my_trail) if my_trail else 0
    other_len = len(other_trail) if other_trail else 0
    turn = game_state.get("turn_count", 0)
    my_boost = game_state.get("agent1_boosts" if game_state.get("player_number", 2) == 1 else "agent2_boosts", 3)
    other_boost = game_state.get("agent2_boosts" if game_state.get("player_number", 2) == 1 else "agent1_boosts", 3)

    aux = _np.array([
        float(turn) / 500.0,
        float(my_len) / 200.0,
        float(other_len) / 200.0,
        float(my_boost) / 3.0,
        float(other_boost) / 3.0,
        float(H) / 20.0,
        float(W) / 20.0,
        1.0,  # bias
    ], dtype=_np.float32)

    return img, aux


def infer_direction(trail: List[Tuple[int, int]]):
    if len(trail) < 2:
        return "RIGHT"
    prev = trail[-2]
    head = trail[-1]
    dx = head[0] - prev[0]
    dy = head[1] - prev[1]
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
    x %= width
    y %= height
    return x, y


def decide_safe_move(board, my_trail, other_trail, turn_count, my_boosts):
    H = len(board)
    W = len(board[0]) if H > 0 else 20
    if not my_trail:
        return "RIGHT"
    head = my_trail[-1]
    cur_dir = infer_direction(my_trail)
    directions = ["UP", "DOWN", "LEFT", "RIGHT"]
    opposite = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    if cur_dir in opposite and opposite[cur_dir] in directions:
        directions.remove(opposite[cur_dir])
    ordered = [cur_dir] + [d for d in directions if d != cur_dir] if cur_dir in directions else directions
    occupied = set()
    for y in range(H):
        for x in range(W):
            if board[y][x] != 0:
                occupied.add((x, y))
    for p in my_trail:
        occupied.add(tuple(p))
    for p in other_trail:
        occupied.add(tuple(p))
    for d in ordered:
        nx, ny = next_pos(head, d, W, H)
        if (nx, ny) not in occupied:
            use_boost = my_boosts > 0 and 30 <= turn_count <= 80
            return f"{d}:BOOST" if use_boost else d
    fallback = cur_dir if cur_dir in ["UP", "DOWN", "LEFT", "RIGHT"] else random.choice(["UP", "DOWN", "LEFT", "RIGHT"]) 
    return fallback


def dqn_decide_move(board, my_trail, other_trail, turn_count, my_boosts):
    if not TORCH_AVAILABLE or MODEL is None:
        return decide_safe_move(board, my_trail, other_trail, turn_count, my_boosts)
    img_np, aux_np = encode_state(board, my_trail, other_trail)
    img_t = torch.from_numpy(img_np).unsqueeze(0)  # (1,C,H,W)
    aux_t = torch.from_numpy(aux_np).unsqueeze(0)  # (1,8)
    with torch.no_grad():
        q = MODEL(img_t, aux_t)
        action = int(q.argmax(dim=1).item())
    idx_to_dir = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    chosen = idx_to_dir.get(action, "RIGHT")
    use_boost = my_boosts > 0 and 30 <= turn_count <= 80
    return f"{chosen}:BOOST" if use_boost else chosen


@app.route("/", methods=["GET"])
def info():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def receive_state():
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    game_state.update(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    player_number = request.args.get("player_number", default=game_state.get("player_number", 2), type=int)
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
    move = dqn_decide_move(board, my_trail, other_trail, turn_count, my_boosts)
    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    data = request.get_json(silent=True) or {}
    result = data.get("result", "UNKNOWN")
    print(f"\nGame Over! Result: {result}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    try_load_model()
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} ({PARTICIPANT}) on port {port}... TORCH_AVAILABLE={TORCH_AVAILABLE}, MODEL={'yes' if MODEL is not None else 'no'}")
    app.run(host="0.0.0.0", port=port, debug=False)
