"""Training entrypoint for python -m dql.train."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn

from .config import Config, load_config
from .env import CaseClosedEnv
from .model import DQNNetwork
from .observation import Observation
from .replay import ReplayBuffer
from .utils import ensure_dir, linear_epsilon, soft_update, to_tensor


def select_action(model: DQNNetwork, obs: Observation, epsilon: float, rng: np.random.Generator, device: torch.device) -> int:
    legal_indices = np.where(obs.legal_actions)[0]
    if len(legal_indices) == 0:
        legal_indices = np.arange(len(obs.legal_actions))
    if rng.random() < epsilon:
        return int(rng.choice(legal_indices))
    crop_t = to_tensor(obs.crop, device=device).unsqueeze(0)
    scalars_t = to_tensor(obs.scalars, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = model(crop_t, scalars_t).cpu().numpy()[0]
    masked = np.full_like(q_values, -1e9)
    masked[obs.legal_actions] = q_values[obs.legal_actions]
    max_indices = np.argwhere(masked == masked.max()).flatten()
    return int(np.random.choice(max_indices))


def compute_td_loss(
    batch: Dict[str, np.ndarray],
    model: DQNNetwork,
    target_model: DQNNetwork,
    config: Config,
    device: torch.device,
) -> torch.Tensor:
    obs_crop = to_tensor(batch["obs_crop"], device=device)
    obs_scalars = to_tensor(batch["obs_scalars"], device=device)
    next_crop = to_tensor(batch["next_crop"], device=device)
    next_scalars = to_tensor(batch["next_scalars"], device=device)
    actions = to_tensor(batch["action"], device=device).long()
    rewards = to_tensor(batch["reward"], device=device)
    dones = to_tensor(batch["done"], device=device).float()
    legal_mask = torch.as_tensor(batch["next_legal"], device=device)

    q_values = model(obs_crop, obs_scalars)
    q_selected = q_values.gather(1, actions.view(-1, 1)).squeeze(1)

    with torch.no_grad():
        if config.dqn.double_dqn:
            next_q_online = model(next_crop, next_scalars)
            next_actions = torch.argmax(next_q_online.masked_fill(~legal_mask, -1e9), dim=1)
            target_q = target_model(next_crop, next_scalars)
            next_q = target_q.gather(1, next_actions.view(-1, 1)).squeeze(1)
        else:
            next_q = target_model(next_crop, next_scalars)
            next_q = torch.max(next_q.masked_fill(~legal_mask, -1e9), dim=1).values
        target = rewards + (1.0 - dones) * config.dqn.gamma * next_q
    loss = nn.SmoothL1Loss()(q_selected, target)
    return loss


def save_checkpoint(model: DQNNetwork, config: Config, obs: Observation, path: str) -> None:
    ensure_dir(Path(path).parent.as_posix())
    payload = {
        "state_dict": model.state_dict(),
        "config": config.to_dict(),
        "crop_channels": int(obs.crop.shape[0]),
        "scalar_dim": int(obs.scalars.shape[0]),
        "crop_size": int(obs.crop.shape[1]),
        "hidden_sizes": config.dqn.hidden_sizes,
    }
    torch.save(payload, path)


def train(config: Config, episodes: int, checkpoint_path: str | None = None) -> None:
    env = CaseClosedEnv(config, seed=config.training.seed)
    obs, info = env.reset()
    device = torch.device("cpu")
    model = DQNNetwork(
        obs.crop.shape[0],
        config.observation.crop_size,
        obs.scalars.shape[0],
        config.dqn.hidden_sizes,
    ).to(device)
    target_model = DQNNetwork(
        obs.crop.shape[0],
        config.observation.crop_size,
        obs.scalars.shape[0],
        config.dqn.hidden_sizes,
    ).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.dqn.lr)
    replay = ReplayBuffer(config.dqn.replay_size)
    rng = np.random.default_rng(config.training.seed)
    telemetry_path = Path(config.paths.telemetry_csv)
    ensure_dir(telemetry_path.parent.as_posix())
    header = [
        "episode",
        "opponent",
        "result",
        "turns",
        "epsilon",
        "reward",
        "shaping",
    ]
    if not telemetry_path.exists():
        with telemetry_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

    opponent_stats = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "turns": []})

    global_step = 0
    for episode in range(1, episodes + 1):
        episode_reward = 0.0
        done = False
        turns = 0
        opponent = info.get("opponent")
        while not done:
            epsilon = linear_epsilon(
                global_step,
                config.exploration.eps_start,
                config.exploration.eps_end,
                config.exploration.eps_decay_steps,
            )
            action = select_action(model, obs, epsilon, rng, device)
            result = env.step(action)
            replay.add(
                obs.crop,
                obs.scalars,
                obs.legal_actions,
                action,
                result.reward,
                result.observation.crop,
                result.observation.scalars,
                result.observation.legal_actions,
                result.done,
            )
            obs = result.observation
            info = result.info
            episode_reward += result.reward
            global_step += 1
            turns = info["turn"]
            done = result.done

            if len(replay) >= config.dqn.min_replay_size:
                batch = replay.sample(config.dqn.batch_size)
                loss = compute_td_loss(batch, model, target_model, config, device)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                soft_update(target_model, model, config.dqn.target_tau)

        result_str = info.get("result") or "RUNNING"
        if result_str == "AGENT1_WIN":
            opponent_stats[opponent]["wins"] += 1
        elif result_str == "AGENT2_WIN":
            opponent_stats[opponent]["losses"] += 1
        elif result_str == "DRAW":
            opponent_stats[opponent]["draws"] += 1
        opponent_stats[opponent]["turns"].append(turns)

        row = {
            "episode": episode,
            "opponent": opponent,
            "result": result_str,
            "turns": turns,
            "epsilon": round(epsilon, 4),
            "reward": round(episode_reward, 4),
            "shaping": json.dumps(info.get("shaping", {})),
        }
        with telemetry_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writerow(row)
        obs, info = env.reset()
        if episode % config.training.log_interval == 0:
            print(f"Episode {episode}/{episodes} reward={episode_reward:.2f} result={row['result']} epsilon={row['epsilon']}")

    stats_path = Path(config.paths.log_dir) / "opponent_stats.json"
    ensure_dir(stats_path.parent.as_posix())
    stats_payload = {
        name: {**vals, "avg_turns": float(np.mean(vals["turns"])) if vals["turns"] else 0.0}
        for name, vals in opponent_stats.items()
    }
    stats_path.write_text(json.dumps(stats_payload, indent=2), encoding="utf-8")
    save_checkpoint(model, config, obs, checkpoint_path or config.paths.checkpoint_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Case Closed DQN agent")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config overrides")
    parser.add_argument("--episodes", type=int, default=None, help="Number of training episodes")
    parser.add_argument("--checkpoint", type=str, default=None, help="Where to save the checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    episodes = args.episodes or config.training.max_episodes
    train(config, episodes, checkpoint_path=args.checkpoint)


if __name__ == "__main__":
    main()
