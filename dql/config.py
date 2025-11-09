"""Configuration objects and helpers for the Case Closed DQN agent."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ObservationConfig:
    crop_size: int = 13
    centering_mode: str = "odd-center"  # or "even-center-UL"
    channels_mode: str = "split2"  # or "occupancy1"
    include_degree: bool = True
    include_opponent_degree: bool = True
    include_relative_delta: bool = True
    include_turn_fraction: bool = True
    include_boosts: bool = True


@dataclass
class StanceConfig:
    use_stance_features: bool = False
    aggression_alpha: float = 0.2
    hysteresis_high: float = 0.65
    hysteresis_low: float = 0.35
    proximity_radius: int = 4
    area_cap: int = 60
    heading_alignment_weight: float = 0.4


@dataclass
class RewardConfig:
    living_bonus_max: float = 0.015
    living_bonus_min: float = 0.005
    living_bonus_decay_turn: int = 120
    degree_penalty: float = 0.01
    degree_threshold: int = 1
    reward_clip: float = 0.02
    area_bonus_scale: float = 0.005
    area_delta_clip: float = 0.1
    area_cap: int = 400
    use_stance_rewards: bool = False


@dataclass
class ActionConfig:
    use_boost: bool = True


@dataclass
class DQNConfig:
    replay_size: int = 100_000
    batch_size: int = 256
    gamma: float = 0.99
    lr: float = 1e-3
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 128])
    target_tau: float = 0.001
    hard_target_interval: int = 1_000
    double_dqn: bool = True
    min_replay_size: int = 20_000
    prioritized_alpha: float = 0.6
    prioritized_beta_start: float = 0.4
    prioritized_beta_steps: int = 200_000
    target_ensembles: int = 2


@dataclass
class ExplorationConfig:
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 100_000
    tie_eps: float = 0.0


@dataclass
class OpponentConfig:
    names: List[str] = field(default_factory=lambda: [
        "wall_hugger",
        "cutter",
        "random_inertia",
        "straight_biased",
        "spiral_runner",
        "aggressive_chaser",
        "random_boost",
        "safe_pocket",
        "mirror_player",
    ])
    weights: List[float] = field(default_factory=lambda: [1.5, 1.5, 1.0, 1.2, 1.0, 1.3, 0.8, 1.1, 1.1])


@dataclass
class InferenceConfig:
    time_budget_ms: float = 40.0
    heuristic_on_timeout: bool = True
    mask_invalid: bool = True


@dataclass
class PathConfig:
    checkpoint_path: str = "model.pth"
    log_dir: str = "logs"
    telemetry_csv: str = "logs/training_metrics.csv"


@dataclass
class TrainingConfig:
    max_episodes: int = 500
    eval_interval: int = 50
    log_interval: int = 10
    seed: int = 7
    randomize_starts: bool = True


@dataclass
class Config:
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    stance: StanceConfig = field(default_factory=StanceConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)
    actions: ActionConfig = field(default_factory=ActionConfig)
    dqn: DQNConfig = field(default_factory=DQNConfig)
    exploration: ExplorationConfig = field(default_factory=ExplorationConfig)
    opponents: OpponentConfig = field(default_factory=OpponentConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _update_dataclass(instance, updates: Dict[str, Any]):
    for key, value in updates.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config key: {key}")
        attr = getattr(instance, key)
        if hasattr(attr, "__dataclass_fields__") and isinstance(value, dict):
            _update_dataclass(attr, value)
        else:
            setattr(instance, key, value)


def _load_dict_from_path(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object")
    return data


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    config = Config()
    updates: Dict[str, Any] = {}
    if path is not None:
        updates.update(_load_dict_from_path(Path(path)))
    if overrides:
        updates.update(overrides)
    if updates:
        _update_dataclass(config, updates)
    return config


def save_config(config: Config, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)
