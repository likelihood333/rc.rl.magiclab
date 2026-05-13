"""Train Z1 tasks using the pinned PPO implementation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import re

from isaaclab.app import AppLauncher

from backend import activate_backend


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--config", help="Local PPO JSON overrides.")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--max_iterations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", type=Path, help="Explicit checkpoint path from a previous vendored PPO run.")
    parser.add_argument("--weights_only", action="store_true", help="Load model weights and start a fresh optimizer.")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--log_dir", type=Path, default=Path("logs/vendor_ppo"))
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.weights_only and args.resume is None:
        parser.error("--weights_only requires --resume")
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if not args.resume.is_file():
            parser.error(f"Checkpoint not found: {args.resume}")
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num_envs must be positive")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max_iterations must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run_name must contain only letters, digits, underscores or hyphens")
    return args


def train(args) -> None:
    # Simulator initialization must precede task/config imports.
    import gymnasium as gym
    import numpy as np
    import torch

    activate_backend()
    from magiclab_rsl.runners import OnPolicyRunner
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
    import magiclab_rl_lab.tasks  # noqa: F401
    from magiclab_rl_lab.algorithms.vendor_ppo.config import TASK_ACTIONS, load_settings
    from magiclab_rl_lab.algorithms.vendor_ppo.vec_env import MagiclabVecEnv
    from magiclab_rl_lab.algorithms.vendor_ppo.metadata import policy_contract, save_run_metadata, validate_resume
    from isaaclab.utils.io import dump_yaml

    cfg = load_settings(args.task, args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.max_iterations is not None:
        cfg.max_iterations = args.max_iterations
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    env_cfg.seed = cfg.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    device = args.device or env_cfg.sim.device
    env_cfg.sim.device = device

    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    if args.run_name:
        run_name += f"_{args.run_name}"
    log_dir = args.log_dir.expanduser().resolve() / cfg.experiment_name / run_name
    log_dir.mkdir(parents=True, exist_ok=False)
    params_dir = log_dir / "params"
    params_dir.mkdir()
    print(f"[INFO] Vendored PPO task={args.task}, device={device}, log_dir={log_dir}")

    env = gym.make(args.task, cfg=env_cfg)
    try:
        wrapped = MagiclabVecEnv(env, clip_actions=cfg.clip_actions)
        if wrapped.num_actions != TASK_ACTIONS[args.task]:
            raise ValueError(f"Unexpected action count for {args.task}: {wrapped.num_actions}")
        rollout_size = wrapped.num_envs * cfg.num_steps_per_env
        if rollout_size < cfg.algorithm.num_mini_batches:
            raise ValueError("Rollout must contain at least one sample per PPO mini-batch.")
        print(f"[INFO] Rollout samples: {rollout_size}; mini-batches: {cfg.algorithm.num_mini_batches}")
        runner_cfg = cfg.to_runner_dict()
        # The upstream runner pops class names, so keep a separate config copy for logging.
        with (params_dir / "runner.json").open("w", encoding="utf-8") as stream:
            json.dump(runner_cfg, stream, indent=2)
        dump_yaml(str(params_dir / "env.yaml"), env_cfg)
        contract = policy_contract(wrapped, args.task, cfg)
        if args.resume is not None:
            validate_resume(args.resume, contract)
        save_run_metadata(
            params_dir / "run.json", contract=contract,
            repo_root=Path(__file__).resolve().parents[3], seed=cfg.seed,
            resume=args.resume, weights_only=args.weights_only,
        )
        runner = OnPolicyRunner(wrapped, cfg.to_runner_dict(), log_dir=str(log_dir), device=device)
        runner.add_git_repo_to_log(__file__)
        if args.resume is not None:
            print(f"[INFO] Loading checkpoint: {args.resume}; weights_only={args.weights_only}")
            runner.load(str(args.resume), load_optimizer=not args.weights_only)
            if args.weights_only:
                runner.current_learning_iteration = 0
        runner.learn(num_learning_iterations=cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    launcher = AppLauncher(args)
    try:
        train(args)
    finally:
        launcher.app.close()


if __name__ == "__main__":
    main()
