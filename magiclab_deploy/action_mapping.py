"""Convert HoST actions into joint targets without creating SDK commands."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointTargets:
    leg_sdk: np.ndarray
    waist: np.ndarray
    arm: np.ndarray


def relative_joint_targets(
    action: np.ndarray,
    snap: dict[str, np.ndarray],
    *,
    action_scale: float,
    max_joint_delta: float,
    leg_sdk_to_policy: np.ndarray,
    leg_policy_to_sdk: np.ndarray,
    arm_control_indices: np.ndarray,
) -> JointTargets:
    """Clip policy deltas, add in policy order, then restore SDK leg order."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape != (23,):
        raise ValueError(f"Expected 23 actions, got shape={action.shape}.")
    if not np.isfinite(action).all():
        raise ValueError("Policy action contains non-finite values.")
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be finite and positive.")
    if not np.isfinite(max_joint_delta):
        raise ValueError("max_joint_delta must be finite.")

    delta = np.clip(action, -1.0, 1.0) * action_scale
    if max_joint_delta > 0:
        delta = np.clip(delta, -max_joint_delta, max_joint_delta)

    # Both operands must use policy order before converting the target back.
    q_leg_policy = snap["leg_pos"][leg_sdk_to_policy]
    target_leg_policy = q_leg_policy + delta[:12]
    target_waist = snap["waist_pos"][:1] + delta[12:13]
    target_arm = snap["arm_pos"][arm_control_indices] + delta[13:23]
    targets = JointTargets(
        leg_sdk=target_leg_policy[leg_policy_to_sdk],
        waist=target_waist,
        arm=target_arm,
    )
    for name, target, size in (
        ("leg", targets.leg_sdk, 12),
        ("waist", targets.waist, 1),
        ("arm", targets.arm, 10),
    ):
        if target.shape != (size,) or not np.isfinite(target).all():
            raise ValueError(f"Invalid {name} target: shape={target.shape}.")
    return targets
