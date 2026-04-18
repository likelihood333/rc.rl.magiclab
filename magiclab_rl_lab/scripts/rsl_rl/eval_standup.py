"""Evaluate stand-up policy with HoST-aligned metrics."""

import argparse
from importlib.metadata import version
import os
import time

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate stand-up policy.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of eval environments.")
parser.add_argument("--task", type=str, required=True, help="Task name.")
parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path.")
parser.add_argument("--num_episodes", type=int, default=10, help="Episodes per environment.")
parser.add_argument("--head_success_height", type=float, default=0.95, help="Head relative-height success threshold.")
parser.add_argument("--head_fail_height", type=float, default=0.5, help="Head relative-height fail threshold after success.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import magiclab_rl_lab.tasks  # noqa: F401
from magiclab_rl_lab.utils.parser_cfg import parse_env_cfg


def _get_body_ids(env):
    robot = env.unwrapped.scene["robot"]
    head_ids, _ = robot.find_bodies("head_link")
    feet_ids, _ = robot.find_bodies(".*ankle_roll_link")
    if len(head_ids) == 0 or len(feet_ids) == 0:
        raise RuntimeError("Failed to resolve head/feet body ids for stand-up metrics.")
    return head_ids, feet_ids


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not getattr(args_cli, "disable_fabric", False),
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    head_ids, feet_ids = _get_body_ids(env)
    num_envs = env.num_envs

    success = torch.zeros(num_envs, dtype=torch.bool, device=env.unwrapped.device)
    reached = torch.zeros_like(success)
    fell_after_reach = torch.zeros_like(success)
    energy = torch.zeros(num_envs, dtype=torch.float, device=env.unwrapped.device)
    smooth = torch.zeros(num_envs, dtype=torch.float, device=env.unwrapped.device)
    feet_move = torch.zeros(num_envs, dtype=torch.float, device=env.unwrapped.device)
    last_feet_pos = None
    episode_counter = torch.zeros(num_envs, dtype=torch.long, device=env.unwrapped.device)

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            robot = env.unwrapped.scene["robot"]
            body_pos = robot.data.body_pos_w
            head_h = body_pos[:, head_ids, 2].mean(dim=1)
            feet_h = body_pos[:, feet_ids, 2].mean(dim=1)
            rel_h = head_h - feet_h

            reached |= rel_h > args_cli.head_success_height
            fell_after_reach |= reached & (rel_h < args_cli.head_fail_height)
            success = reached & ~fell_after_reach

            energy += torch.sum(torch.abs(robot.data.joint_vel) * torch.abs(robot.data.applied_torque), dim=1)
            smooth += torch.sum(torch.square(env.unwrapped.action_manager.action - env.unwrapped.action_manager.prev_action), dim=1)

            feet_pos = body_pos[:, feet_ids, :2].reshape(num_envs, -1)
            if last_feet_pos is not None:
                feet_move += torch.norm(feet_pos - last_feet_pos, dim=1)
            last_feet_pos = feet_pos

            done_ids = dones.nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                episode_counter[done_ids] += 1
                reached[done_ids] = False
                fell_after_reach[done_ids] = False

            if torch.all(episode_counter >= args_cli.num_episodes):
                break

        time.sleep(0.0)

    denom = torch.clamp(episode_counter.float(), min=1.0)
    print("========== Standup Eval ==========")
    print(f"task: {args_cli.task}")
    print(f"checkpoint: {resume_path}")
    print(f"success_rate(%): {(success.float().mean() * 100).item():.2f}")
    print(f"avg_energy: {(energy / denom).mean().item():.4f}")
    print(f"avg_smoothness: {(smooth / denom).mean().item():.4f}")
    print(f"avg_feet_movement: {(feet_move / denom).mean().item():.4f}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

