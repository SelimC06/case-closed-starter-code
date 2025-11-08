import torch
import torch.nn.functional as F
import torch.optim as optim

from train_env import CaseClosedEnv
from model_def import PolicyNet, INPUT_SIZE, N_ACTIONS

device = torch.device("cpu")

def obs_to_tenor(obs):
    board = obs["board"]
    h = len(board)
    w = len(board[0])

    flat = []
    for y in range(h):
        for x in range(w):
            v = board[y][x]
            flat.append(1.0 if v != 0 else 0.0)

    extras = [
        float(obs.get("turn_count", 0)) / 400.0,
        float(obs.get("agent1_boosts", 0)) / 5.0,
        float(obs.get("agent2_boosts", 0)) / 5.0,
        float(obs.get("agent1_alive", 1)),
        float(obs.get("agent2_alive", 1)),
        float(obs.get("agent1_length", 1)) / 400.0,
        float(obs.get("agent2_length", 1)) / 400.0,
        float(obs.get("player_number", 1)) - 1.0,
    ]

    vec = flat + extras
    if len(vec) != INPUT_SIZE:
        raise ValueError(f"Expected {INPUT_SIZE} features, got {len(vec)}")
    
    return torch.tensor(vec, dtype=torch.float32, device=device)

def get_legal_actions(env, obs):
    board = obs["board"]
    h = len(board)
    w = len(board[0])

    player_num = obs["player_number"]
    if player_num == 1:
        my_trail = obs["agent1_trail"]
        my_boosts = obs["agent1_boosts"]
    else:
        my_trail = obs["agent2_trail"]
        my_boosts = obs["agent2_boosts"]

    if not my_trail:
        return list(range(N_ACTIONS))
    
    head_x, head_y = my_trail[-1]

    blocked = set()
    for y in range(h):
        for x in range(w):
            if board[y][x] != 0:
                blocked.add((x, y))
    
    legal = []
    dirs = [(-0, -1), (1, 0), (0, 1), (-1, 0)]

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

    if not legal:
        return list(range(N_ACTIONS))
    return legal

def select_action(policy, env, obs):
    x = obs_to_tenor(obs).unsqueeze(0)
    logits = policy(x)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    
    legal = get_legal_actions(env, obs)

    mask = torch.zeros_like(probs)
    mask[legal] = 1.0
    masked_probs = probs * mask

    if masked_probs.sum().item() <= 0:
        masked_probs = probs

    masked_probs = masked_probs / masked_probs.sum()

    dist = torch.distributions.Categorical(masked_probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    
    return int(action.item()), log_prob

def train_policy(episodes=500):
    policy = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    gamma = 0.99

    for ep in range(episodes):
        env = CaseClosedEnv(as_player=1)
        obs = env.reset()

        log_probs = []
        rewards = []
        done = False

        while not done:
            action, log_p = select_action(policy, env, obs)
            obs, r, done, info = env.step(action)
            log_probs.append(log_p)
            rewards.append(r)

        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.append(G)
        
        returns.reverse()

        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        if returns.std() > 1e-6:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        for log_p, Gt in zip(log_probs, returns):
            loss -= log_p * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (ep + 1) % 50 == 0:
            print(
                f"Episode {ep+1}/{episodes}: "
                f"total_reward={sum(rewards):.3f}, "
                f"loss={loss.item():.4f}, "
                f"result={info['result']}"
            )

    torch.save(policy.state_dict(), "model.pth")
    print("Saved model.pth")
    return policy

if __name__ == "__main__":
    train_policy(episodes=500)