# --- Heuristic fallback for Case Closed (torus aware) ---

from case_closed_game import Direction

DIRS = {
    Direction.UP:    (0, -1),
    Direction.DOWN:  (0,  1),
    Direction.LEFT:  (-1, 0),
    Direction.RIGHT: (1,  0),
}
ALL_DIRS = [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]
OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}

def _wrap(x, y, W, H):
    return (x % W, y % H)

def _is_open(board, x, y):
    return board[y][x] == 0

def _neighbors_open(board, x, y, W, H):
    out = []
    for d,(dx,dy) in DIRS.items():
        nx, ny = _wrap(x+dx, y+dy, W, H)
        if _is_open(board, nx, ny):
            out.append((d, nx, ny))
    return out

def _degree(board, x, y, W, H):
    # how many exits from this cell
    deg = 0
    for (dx,dy) in DIRS.values():
        nx, ny = _wrap(x+dx, y+dy, W, H)
        if _is_open(board, nx, ny):
            deg += 1
    return deg

def _flood_area(board, x0, y0, W, H, cap=9999):
    # conservative reachability count from (x0,y0)
    if not _is_open(board, x0, y0):
        return 0
    seen = {(x0,y0)}
    stack = [(x0,y0)]
    size = 0
    while stack and size < cap:
        x,y = stack.pop()
        size += 1
        for (dx,dy) in DIRS.values():
            nx, ny = _wrap(x+dx, y+dy, W, H)
            if (nx,ny) not in seen and _is_open(board, nx, ny):
                seen.add((nx,ny))
                stack.append((nx,ny))
    return size

def _head_on_risk(my_next, opp_head, opp_dir, W, H):
    if opp_dir is None:
        return False
    dx, dy = DIRS[opp_dir]
    ox, oy = opp_head
    opp_next = _wrap(ox+dx, oy+dy, W, H)
    return my_next == opp_next  # both try to enter the same tile

def _current_dir_from_trail(trail, W, H):
    if len(trail) < 2:
        return None
    (x2,y2), (x1,y1) = trail[-2], trail[-1]
    # account for wrap: choose delta with |d|<=1 by wrapping back
    dx = x1 - x2
    dy = y1 - y2
    if abs(dx) > 1:
        dx = -1 if dx > 0 else 1
    if abs(dy) > 1:
        dy = -1 if dy > 0 else 1
    for d,(mx,my) in DIRS.items():
        if (mx,my) == (dx,dy):
            return d
    return None

def choose_heuristic_move(game, me_player_number=1, allow_boost=False):
    """
    Returns either 'UP'/'DOWN'/'LEFT'/'RIGHT' (and optionally ':BOOST').
    Uses only board + trails in `game` (no mutation).
    """
    board = game.board.grid
    H, W = game.board.height, game.board.width

    me   = game.agent1 if me_player_number == 1 else game.agent2
    opp  = game.agent2 if me_player_number == 1 else game.agent1

    (mx, my) = me.trail[-1]
    (ox, oy) = opp.trail[-1]
    my_dir   = me.direction           # Direction enum (may be None at start)
    opp_dir  = opp.direction

    # 1) Legal candidate moves (disallow 180° turn)
    cands = []
    for d,(dx,dy) in DIRS.items():
        if my_dir and d == OPPOSITE[my_dir]:
            continue
        nx, ny = _wrap(mx+dx, my+dy, W, H)
        if _is_open(board, nx, ny):
            cands.append((d, nx, ny))

    if not cands:
        # no legal moves; keep going forward if possible, else pick anything
        if my_dir:
            return my_dir.name
        return Direction.RIGHT.name

    # 2) Score each candidate with a simple, effective heuristic
    # Weights (tune lightly if you want)
    W_DEG      = 1.0    # prefer high local degree
    W_AREA     = 0.05   # prefer larger reachable area
    W_HEAD_ON  = -5.0   # avoid head-on conflicts
    W_EDGEHUG  = 0.0    # not used; torus removes edges in practice

    best = None
    best_score = -1e18

    for d, nx, ny in cands:
        deg  = _degree(board, nx, ny, W, H)
        area = _flood_area(board, nx, ny, W, H, cap=400)  # cap for speed
        head_on = _head_on_risk((nx,ny), (ox,oy), opp_dir, W, H)

        score = 0.0
        score += W_DEG * deg
        score += W_AREA * area
        if head_on:
            score += W_HEAD_ON

        # bias to continue straight if tie
        if my_dir and d == my_dir:
            score += 0.1

        if score > best_score:
            best_score = score
            best = d

    # 3) Optional conservative boost: only if two steps are safe and degree stays decent
    move = best.name  # "UP"/"DOWN"/"LEFT"/"RIGHT"
    if allow_boost and me.boosts_remaining > 0:
        dx, dy = DIRS[best]
        nx1, ny1 = _wrap(mx+dx, my+dy, W, H)
        nx2, ny2 = _wrap(nx1+dx, ny1+dy, W, H)
        safe1 = _is_open(board, nx1, ny1)
        safe2 = _is_open(board, nx2, ny2)
        deg2  = _degree(board, nx2, ny2, W, H)
        if safe1 and safe2 and deg2 >= 2 and not _head_on_risk((nx1,ny1), (ox,oy), opp_dir, W, H):
            move = f"{move}:BOOST"

    return move
