from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_root_pose_fallen(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    roll_range: tuple[float, float],
    pitch_range: tuple[float, float],
    yaw_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset robot to fallen orientation ranges (HoST-style initialization)."""
    asset: Articulation = env.scene[asset_cfg.name]
    ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if ids.numel() == 0:
        return

    roll = torch.empty_like(ids, dtype=torch.float).uniform_(roll_range[0], roll_range[1])
    pitch = torch.empty_like(ids, dtype=torch.float).uniform_(pitch_range[0], pitch_range[1])
    yaw = torch.empty_like(ids, dtype=torch.float).uniform_(yaw_range[0], yaw_range[1])
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    root_state = asset.data.default_root_state[ids].clone()
    root_state[:, 3:7] = quat
    asset.write_root_state_to_sim(root_state, env_ids=ids)

