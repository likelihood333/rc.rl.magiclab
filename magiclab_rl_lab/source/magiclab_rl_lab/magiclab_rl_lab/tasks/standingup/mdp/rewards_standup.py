from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def upright_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward upright torso orientation.

    projected_gravity_b[:, 2] approaches -1 when torso is upright.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 1.0)


def head_height_reward(
    env: ManagerBasedRLEnv,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["head_link"]),
    feet_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*ankle_roll_link"]),
) -> torch.Tensor:
    """Reward head height relative to feet mean height."""
    asset: Articulation = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].mean(dim=1)
    feet_height = asset.data.body_pos_w[:, feet_cfg.body_ids, 2].mean(dim=1)
    rel = head_height - feet_height
    # 0.95m is a practical stand-up target borrowed from HoST-style setups.
    return torch.clamp(rel / 0.95, 0.0, 1.2)


def joint_power_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.joint_vel) * torch.abs(asset.data.applied_torque), dim=-1)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize fast changes in action.

    This uses previous action cache provided by ActionManager.
    """
    current = env.action_manager.action
    previous = env.action_manager.prev_action
    return torch.sum(torch.square(current - previous), dim=-1)


def head_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.25,
    head_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["head_link"]),
) -> torch.Tensor:
    """Termination helper: returns True when head is too close to ground."""
    asset: Articulation = env.scene[head_cfg.name]
    head_height = asset.data.body_pos_w[:, head_cfg.body_ids, 2].mean(dim=1)
    return head_height < minimum_height

