"""HoST observation assembly in policy joint order."""

from __future__ import annotations

import numpy as np


def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Body IMU quaternion (w,x,y,z) -> unit vector in body frame."""
    qw, qx, qy, qz = quaternion_wxyz.astype(np.float64)
    return np.array(
        [
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ],
        dtype=np.float32,
    )


def build_policy_obs(
    snap: dict[str, np.ndarray],
    last_action: np.ndarray,
    cfg,
    leg_sdk_to_policy: np.ndarray,
    arm_control_indices: np.ndarray,
) -> np.ndarray:
    qj = np.concatenate(
        [snap["leg_pos"][leg_sdk_to_policy], snap["waist_pos"][:1],
         snap["arm_pos"][arm_control_indices]], dtype=np.float32,
    )
    dqj = np.concatenate(
        [snap["leg_vel"][leg_sdk_to_policy], snap["waist_vel"][:1],
         snap["arm_vel"][arm_control_indices]], dtype=np.float32,
    )
    if len(qj) != cfg.num_actions or len(dqj) != cfg.num_actions:
        raise ValueError(f"State joint dimensions must equal {cfg.num_actions}.")
    obs = np.concatenate(
        [
            snap["imu_ang_vel"] * cfg.ang_vel_scale,
            get_gravity_orientation(snap["imu_quat_wxyz"]) * cfg.gravity_scale,
            qj * cfg.dof_pos_scale,
            dqj * cfg.dof_vel_scale,
            last_action,
            np.array([cfg.action_rescale_obs], dtype=np.float32),
        ],
        dtype=np.float32,
    )
    if len(obs) != cfg.num_obs:
        raise ValueError(f"Obs dim mismatch, got {len(obs)} expected {cfg.num_obs}.")
    return obs
