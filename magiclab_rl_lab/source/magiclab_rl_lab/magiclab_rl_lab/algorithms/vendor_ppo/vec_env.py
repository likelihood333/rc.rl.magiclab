"""Adapt Isaac Lab manager-based environments to the pinned VecEnv API."""

from __future__ import annotations

import torch

from magiclab_rsl.env import VecEnv


class MagiclabVecEnv(VecEnv):
    def __init__(self, env, clip_actions: float | None = None):
        self.env = env
        self.num_envs = env.unwrapped.num_envs
        self.num_actions = env.unwrapped.action_manager.total_action_dim
        self.device = env.unwrapped.device
        self.cfg = env.unwrapped.cfg
        self.max_episode_length = env.unwrapped.max_episode_length
        if clip_actions is not None and clip_actions <= 0:
            raise ValueError("clip_actions must be positive or None.")
        self.clip_actions = clip_actions
        self._observations = None
        self.reset()

    @property
    def unwrapped(self):
        return self.env.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        self.unwrapped.episode_length_buf = value

    def _pack(self, observations, extras: dict) -> tuple[torch.Tensor, dict]:
        if "policy" not in observations:
            raise ValueError("Environment must provide a policy observation group.")
        policy = observations["policy"]
        if policy.ndim != 2 or policy.shape[0] != self.num_envs:
            raise ValueError(f"Policy observations must be [num_envs, dim], got {policy.shape}.")
        groups = dict(extras.get("observations", {}))
        groups.update({name: value for name, value in observations.items() if name != "policy"})
        critic = groups.get("critic", policy)
        if critic.ndim != 2 or critic.shape[0] != self.num_envs:
            raise ValueError(f"Critic observations must be [num_envs, dim], got {critic.shape}.")
        infos = dict(extras)
        infos["observations"] = groups
        return policy, infos

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        # Returning cached output avoids advancing the observation manager history.
        return self._pack(self._observations, {})

    def reset(self) -> tuple[torch.Tensor, dict]:
        observations, extras = self.env.reset()
        self._observations = observations
        return self._pack(observations, extras)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Unexpected action shape: {actions.shape}.")
        actions = actions.to(self.device)
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        observations, rewards, terminated, truncated, extras = self.env.step(actions)
        self._observations = observations
        policy, infos = self._pack(observations, extras)
        if not self.cfg.is_finite_horizon:
            infos["time_outs"] = truncated & ~terminated
        else:
            infos.pop("time_outs", None)
        dones = (terminated | truncated).to(dtype=torch.long)
        return policy, rewards, dones, infos

    def close(self) -> None:
        self.env.close()
