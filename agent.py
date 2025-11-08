import os
import uuid
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

MODEL = None
N_ACTIONS = 8

try:
    import torch
    import torch.nn.functional as F
    from model_def import PolicyNet, INPUT_SIZE, N_ACTIONS as MODEL_ACTIONS

    if MODEL_ACTIONS:
        N_ACTIONS = MODEL_ACTIONS
except Exception:
    torch = None
    F = None
    PolicyNet = None
    INPUT_SIZE = None

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"

def load_model():
    global MODEL
    if torch is None or PolicyNet is None:
        print("[MODEL] Torch/model_def not available; running heuristic-only.")
        MODEL = None
        return
    
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, "model.pth")
        if os.path.exists(model_path):
            m = PolicyNet()
            state_dict = torch.load(model_path, map_location="cpu")
            m.load_state_dict(state_dict)
            m.eval()
            MODEL = m
            print("[MODEL] Loaded model.pth successfully.")
        else:
            print("[MODEL] model.pth not found; running heuristic-only.")
            MODEL = None
    except Exception as e:
        print(f"[MODEL] Failed to load model.pth: {e}")
        MODEL = None

load_model()

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

DIR_ORDER = ["UP", "RIGHT", "DOWN", "LEFT"]
DIR_VECS = {
    "UP": (0, -1),
    "RIGHT": (1, 0),
    "DOWN": (0, 1),
    "LEFT": (-1, 0),
}

def _get_board(state):
    board = state.get("board")
    if not board:
        return None, 0, 0
    h = len(board)
    w = len(board[0]) if h > 0 else 0
    return board, w, h

def _build_blocked(board):
    blocked = set()
    if not board:
        return blocked
    h = len(board)
    w = len(board[0]) if h > 0 else 0
    for y in range(h):
        for x in range(w):
            if board[y][x] != 0:
                blocked.add((x, y))
    return blocked

def _get_my_trail_and_boosts(state, player_number):
    if player_number == 1:
        trail = state.get("agent1_trail", [])
        boosts = int(state.get("agent1_boosts", 0))
    else:
        trail = state.get("agent2_trail", [])
        boosts = int(state.get("agent2_boosts", 0))
    return trail, boosts

def state_to_tensor_for_model(state, player_number):
    if torch is None or INPUT_SIZE is None:
        return None
    
    board, w, h = _get_board(state)
    if board is None or w == 0 or h == 0:
        return None
    
    flat = []
    for y in range(h):
        for x in range(w):
            v = board[y][x]
            flat.append(1.0 if v != 0 else 0.0)

    if player_number == 1:
        my_prefix, opp_prefix = "agent1", "agent2"
    else:
        my_prefix, opp_prefix = "agent2", "agent1"

    extras = [
        float(state.get("turn_count", 0)) / 400.0,
        float(state.get(f"{my_prefix}_boosts", 0)) / 5.0,
        float(state.get(f"{opp_prefix}_boosts", 0)) / 5.0,
        float(state.get(f"{my_prefix}_alive", 1)),
        float(state.get(f"{opp_prefix}_alive", 1)),
        float(state.get(f"{my_prefix}_length", 1)) / 400.0,
        float(state.get(f"{opp_prefix}_length", 1)) / 400.0,
        float(player_number - 1),
    ]

    vec = flat + extras
    if len(vec) != INPUT_SIZE:
        print(f"[MODEL] Feature size mismatch: got {len(vec)}, expected {INPUT_SIZE}")
        return None
    
    return torch.tensor(vec, dtype=torch.float32).unsqueeze(0)

def get_legal_actions_from_state(state, player_number):
    board, w, h = _get_board(state)
    if board is None:
        return list(range(N_ACTIONS))
    
    my_trail, my_boosts = _get_my_trail_and_boosts(state, player_number)
    if not my_trail:
        return list(range(N_ACTIONS))
    
    head_x, head_y = my_trail[-1]
    blocked = _build_blocked(board)

    legal = []
    dirs = [(0, -1), (1, 0), (0, 1), (-1, 0)]

    for a in range(N_ACTIONS):
        use_boost = a >= 4
        dir_idx = a % 4
        dx, dy = dirs[dir_idx]

        if use_boost and my_boosts <= 0:
            continue

        nx1 = (head_x + dx) % w
        ny1 = (head_y + dy) % h
        if (nx1, ny1) in blocked:
            continue

        if use_boost:
            nx2 = (nx1 + dx) % w
            ny2 = (ny1 + dy) % h
            if (nx2, ny2) in blocked:
                continue

        legal.append(a)
    
    return legal or list(range(N_ACTIONS))

def action_index_to_command(a, state, player_number):
    board, w, h = _get_board(state)
    if board is None:
        return None
    
    my_trail, my_boosts = _get_my_trail_and_boosts(state, player_number)
    if not my_trail:
        return None
    
    use_boost = a >= 4
    dir_idx = a % 4
    dir_name = DIR_ORDER[dir_idx]
    dx, dy = DIR_VECS[dir_name]

    head_x, head_y = my_trail[-1]
    blocked = _build_blocked(board)

    nx1 = (head_x + dx) % w
    ny1 = (head_y + dy) % h
    if (nx1, ny1) in blocked:
        return None
    
    if use_boost:
        if my_boosts <= 0:
            return None
        nx2 = (nx1 + dx) % w
        ny2 = (ny1 + dy) % h
        if (nx2, ny2) in blocked:
            return None
        return f"{dir_name}:BOOST"
    
    return dir_name

def choose_move_learned(state, player_number):
    if MODEL is None or torch is None or F is None:
        return None
    
    x = state_to_tensor_for_model(state, player_number)
    if x is None:
        return None
    
    with torch.no_grad():
        logits = MODEL(x)
        probs = F.softmax(logits, dim=-1).squeeze(0)

    legal = get_legal_actions_from_state(state, player_number)
    mask = torch.zeros_like(probs)
    mask[legal] = 1.0
    masked = probs * mask

    if masked.sum().item() > 0:
        masked = masked / masked.sum()
    else:
        masked = probs / probs.sum()

    for a in torch.argsort(masked, descending=True).tolist():
        move = action_index_to_command(a, state, player_number)
        if move:
            return move
        
    return None

def choose_move_heuristic(state, player_number):
    board, w, h = _get_board(state)
    if board is None or w == 0 or h == 0:
        return "RIGHT"
    
    my_trail, _ = _get_my_trail_and_boosts(state, player_number)
    if not my_trail:
        return "RIGHT"
    
    head_x, head_y = my_trail[-1]
    blocked = _build_blocked(board)

    for name in DIR_ORDER:
        dx, dy = DIR_VECS[name]
        nx = (head_x + dx) % w
        ny = (head_y + dy) % h
        if (nx, ny) not in blocked:
            return name
        
    return "RIGHT"

@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = int(request.args.get("player_number", default=1, type=int))

    with game_lock:
        state = dict(LAST_POSTED_STATE)

    move = choose_move_learned(state, player_number)

    if move is None or not isinstance(move, str) or not move:
        move = choose_move_heuristic(state, player_number)

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
        print(f"[END] Result: {data.get('result')}")
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    print(f"Starting {AGENT_NAME} on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
