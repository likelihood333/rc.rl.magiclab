"""PPO configuration independent of the installed RSL-RL API version."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import math


@dataclass
class PolicySettings:
    actor_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 0.8
    noise_std_type: str = "scalar"


@dataclass
class PPOSettings:
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    schedule: str = "adaptive"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0


@dataclass
class RunnerSettings:
    experiment_name: str = "z1_23dof_standup"
    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 200
    seed: int = 42
    empirical_normalization: bool = False
    clip_actions: float | None = None
    policy: PolicySettings = field(default_factory=PolicySettings)
    algorithm: PPOSettings = field(default_factory=PPOSettings)

    def to_runner_dict(self) -> dict:
        result = asdict(self)
        result["policy"]["class_name"] = "ActorCritic"
        result["algorithm"]["class_name"] = "PPO"
        result["logger"] = "tensorboard"
        return result


TASK_ACTIONS = {
    "Magiclab-Z1-12dof-Velocity": 12,
    "Magiclab-Z1-23dof-Standup-Ground": 23,
    "Magiclab-Z1-23dof-Standup-Prone": 23,
    "Magiclab-Z1-23dof-Standup-Platform": 23,
    "Magiclab-Z1-23dof-Standup-Slope": 23,
    "Magiclab-Z1-23dof-Standup-Wall": 23,
}


def load_settings(task: str, config_file: str | None = None) -> RunnerSettings:
    if task not in TASK_ACTIONS:
        raise ValueError(f"Unsupported vendored PPO task: {task}")
    cfg = RunnerSettings()
    if TASK_ACTIONS[task] == 12:
        cfg.experiment_name = "z1_12dof_velocity"
        cfg.max_iterations = 50000
        cfg.save_interval = 100
        cfg.policy.init_noise_std = 1.0
    if config_file is not None:
        with Path(config_file).expanduser().open(encoding="utf-8") as stream:
            overrides = json.load(stream)
        if not isinstance(overrides, dict):
            raise ValueError("Training config must be a JSON object.")
        for section, values in overrides.items():
            if section not in {"runner", "policy", "algorithm"} or not isinstance(values, dict):
                raise ValueError(f"Invalid config section: {section}")
            target = cfg if section == "runner" else getattr(cfg, section)
            for key, value in values.items():
                if key not in target.__dataclass_fields__ or key in {"policy", "algorithm"}:
                    raise ValueError(f"Unknown config key: {section}.{key}")
                setattr(target, key, value)
    if cfg.num_steps_per_env <= 0 or cfg.max_iterations <= 0 or cfg.save_interval <= 0:
        raise ValueError("Rollout length, iterations and save interval must be positive.")
    if cfg.algorithm.num_mini_batches <= 0 or cfg.algorithm.num_learning_epochs <= 0:
        raise ValueError("PPO batch count and learning epochs must be positive.")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if not cfg.experiment_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in cfg.experiment_name):
        raise ValueError("experiment_name must be a non-empty directory label.")
    for dims in (cfg.policy.actor_hidden_dims, cfg.policy.critic_hidden_dims):
        if not isinstance(dims, list) or not dims or any(not isinstance(dim, int) or dim <= 0 for dim in dims):
            raise ValueError("Hidden dimensions must be a non-empty list of positive integers.")
    if cfg.policy.activation not in {"elu", "selu", "relu", "crelu", "lrelu", "tanh", "sigmoid", "identity"}:
        raise ValueError("Unsupported policy activation.")
    if cfg.policy.noise_std_type not in {"scalar", "log"}:
        raise ValueError("noise_std_type must be scalar or log.")
    for name, value in (
        ("init_noise_std", cfg.policy.init_noise_std),
        ("learning_rate", cfg.algorithm.learning_rate),
        ("clip_param", cfg.algorithm.clip_param),
        ("max_grad_norm", cfg.algorithm.max_grad_norm),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if cfg.algorithm.schedule not in {"fixed", "adaptive"}:
        raise ValueError("PPO schedule must be fixed or adaptive.")
    if not 0 <= cfg.algorithm.gamma <= 1 or not 0 <= cfg.algorithm.lam <= 1:
        raise ValueError("gamma and lam must lie between zero and one.")
    return cfg
