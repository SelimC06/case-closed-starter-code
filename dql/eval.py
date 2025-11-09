"""Evaluation helper: python -m dql.eval --matches 500."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

import numpy as np

from .config import load_config
from .env import CaseClosedEnv
from .policy import DQNPolicy
from .utils import ensure_dir


def run_matches(env: CaseClosedEnv, policy: DQNPolicy, opponent: str, matches: int) -> Dict[str, float]:
    wins = losses = draws = 0
    total_turns: List[int] = []
    for _ in range(matches):
        policy.reset()
        obs, info = env.reset(opponent_name=opponent)
        done = False
        final_info = info
        while not done:
            action_idx, _ = policy.predict(env.game, player_number=1)
            result = env.step(action_idx)
            done = result.done
            final_info = result.info
        if final_info["result"] == "AGENT1_WIN":
            wins += 1
        elif final_info["result"] == "AGENT2_WIN":
            losses += 1
        else:
            draws += 1
        total_turns.append(final_info.get("turn", 0))
    total = max(1, matches)
    return {
        "opponent": opponent,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / total,
        "avg_turns": float(np.mean(total_turns)) if total_turns else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the trained Case Closed agent")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="model.pth")
    parser.add_argument("--matches", type=int, default=100)
    args = parser.parse_args()

    config = load_config(args.config)
    policy = DQNPolicy(config, args.checkpoint)
    env = CaseClosedEnv(config)

    rows = []
    for opponent in config.opponents.names:
        stats = run_matches(env, policy, opponent, args.matches)
        rows.append(stats)

    print("Opponent | Wins | Losses | Draws | Win% | AvgTurns")
    for row in rows:
        print(
            f"{row['opponent']:<10} {row['wins']:>4} {row['losses']:>7} {row['draws']:>6} {row['win_rate']*100:6.2f}% {row['avg_turns']:8.1f}"
        )

    csv_path = Path(config.paths.log_dir) / "evaluation.csv"
    ensure_dir(csv_path.parent.as_posix())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved evaluation summary to {csv_path}")


if __name__ == "__main__":
    main()
