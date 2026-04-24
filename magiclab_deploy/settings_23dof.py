"""Configuration parsing for the 23DOF deployment entry point."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from config_z1 import Config


@dataclass(frozen=True)
class Z1Deploy23dofSettings:

    policy_path: str
    local_ip: str
    device: str
    policy_format: str
    actor_activation: str
    num_actions: int
    num_obs: int
    his_obs: int
    infer_hz: int
    cmd_hz: int
    state_timeout_s: float
    initial_stream_warmup_s: float
    state_warmup_publish: bool
    state_warmup_interval_s: float
    gait_settle_s: float
    lowlevel_retry: int
    lowlevel_retry_interval_s: float
    force_highlevel_reset_before_lowlevel: bool
    dry_run: bool
    prepare_gait: bool
    arm_control_indices: str
    leg_sdk_to_policy: str
    leg_policy_to_sdk: str
    leg_operation_mode: int
    waist_operation_mode: int
    arm_operation_mode: int
    arm_hold_operation_mode: int
    max_joint_delta: float
    ang_vel_scale: float
    gravity_scale: float
    dof_pos_scale: float
    dof_vel_scale: float
    action_scale: float
    action_rescale_obs: float
    kps: tuple[float, ...]
    kds: tuple[float, ...]
    safe_stop_duration_s: float
    safe_stop_kp_scale: float
    safe_stop_kd_scale: float
    log_command_values: bool
    log_command_detail: bool
    log_joint_command_each_publish: bool
    release_on_exit: bool
    release_duration_s: float


def _indices_field_to_csv(value: Any, name: str) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(x)) for x in value)
    raise ValueError(f"{name} must be a comma-separated string or a list of integers.")


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off", ""):
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def load_23dof_settings(config_path: str) -> Z1Deploy23dofSettings:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"23DOF config not found: {config_path}")

    loader = Config(config_path)
    d: dict[str, Any] = loader.config

    policy_path = loader.policy_path
    if not policy_path or not isinstance(policy_path, str):
        raise ValueError("YAML must set policy_path.")

    kps = d.get("kps")
    kds = d.get("kds")
    if not isinstance(kps, (list, tuple)) or not isinstance(kds, (list, tuple)):
        raise ValueError("YAML kps/kds must be lists of length num_actions.")
    kps_t = tuple(float(x) for x in kps)
    kds_t = tuple(float(x) for x in kds)

    num_actions = int(d["num_actions"])
    if len(kps_t) != num_actions or len(kds_t) != num_actions:
        raise ValueError("kps/kds length must equal num_actions.")

    cfg = Z1Deploy23dofSettings(
        policy_path=policy_path,
        local_ip=str(loader.local_ip),
        device=str(d["device"]),
        policy_format=str(loader.policy_format),
        actor_activation=str(loader.actor_activation),
        num_actions=num_actions,
        num_obs=int(d["num_obs"]),
        his_obs=int(d["his_obs"]),
        infer_hz=int(d["infer_hz"]),
        cmd_hz=int(d["cmd_hz"]),
        state_timeout_s=float(d["state_timeout_s"]),
        initial_stream_warmup_s=float(d["initial_stream_warmup_s"]),
        state_warmup_publish=_as_bool(d.get("state_warmup_publish"), True),
        state_warmup_interval_s=float(d["state_warmup_interval_s"]),
        gait_settle_s=float(d["gait_settle_s"]),
        lowlevel_retry=int(d["lowlevel_retry"]),
        lowlevel_retry_interval_s=float(d["lowlevel_retry_interval_s"]),
        force_highlevel_reset_before_lowlevel=_as_bool(
            d.get("force_highlevel_reset_before_lowlevel"), True
        ),
        dry_run=_as_bool(d.get("dry_run"), False),
        prepare_gait=_as_bool(d.get("prepare_gait"), True),
        arm_control_indices=_indices_field_to_csv(
            d["arm_control_indices"], "arm_control_indices"
        ),
        leg_sdk_to_policy=_indices_field_to_csv(
            d["leg_sdk_to_policy"], "leg_sdk_to_policy"
        ),
        leg_policy_to_sdk=_indices_field_to_csv(
            d["leg_policy_to_sdk"], "leg_policy_to_sdk"
        ),
        leg_operation_mode=int(d["leg_operation_mode"]),
        waist_operation_mode=int(d["waist_operation_mode"]),
        arm_operation_mode=int(d["arm_operation_mode"]),
        arm_hold_operation_mode=int(d["arm_hold_operation_mode"]),
        max_joint_delta=float(d["max_joint_delta"]),
        ang_vel_scale=float(d["ang_vel_scale"]),
        gravity_scale=float(d["gravity_scale"]),
        dof_pos_scale=float(d["dof_pos_scale"]),
        dof_vel_scale=float(d["dof_vel_scale"]),
        action_scale=float(d["action_scale"]),
        action_rescale_obs=float(d["action_rescale_obs"]),
        kps=kps_t,
        kds=kds_t,
        safe_stop_duration_s=float(d["safe_stop_duration_s"]),
        safe_stop_kp_scale=float(d["safe_stop_kp_scale"]),
        safe_stop_kd_scale=float(d["safe_stop_kd_scale"]),
        log_command_values=_as_bool(d.get("log_command_values"), False),
        log_command_detail=_as_bool(d.get("log_command_detail"), False),
        log_joint_command_each_publish=_as_bool(
            d.get("log_joint_command_each_publish"), False
        ),
        release_on_exit=_as_bool(d.get("release_on_exit"), True),
        release_duration_s=float(d["release_duration_s"]),
    )

    if cfg.infer_hz <= 0 or cfg.cmd_hz <= 0:
        raise ValueError("infer_hz and cmd_hz must be positive.")
    if cfg.gait_settle_s < 0:
        raise ValueError("gait_settle_s must be non-negative.")
    if cfg.lowlevel_retry <= 0:
        raise ValueError("lowlevel_retry must be positive.")
    if cfg.lowlevel_retry_interval_s < 0:
        raise ValueError("lowlevel_retry_interval_s must be non-negative.")
    if cfg.state_warmup_interval_s <= 0:
        raise ValueError("state_warmup_interval_s must be positive.")
    if cfg.initial_stream_warmup_s < 0:
        raise ValueError("initial_stream_warmup_s must be non-negative.")
    if cfg.safe_stop_duration_s < 0:
        raise ValueError("safe_stop_duration_s must be non-negative.")
    if cfg.safe_stop_kp_scale < 0 or cfg.safe_stop_kd_scale < 0:
        raise ValueError("safe_stop scales must be non-negative.")
    if cfg.release_duration_s < 0:
        raise ValueError("release_duration_s must be non-negative.")
    if cfg.device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'.")
    if cfg.policy_format != "checkpoint":
        raise ValueError(
            "23DOF deploy only loads PyTorch checkpoints (training torch.save with "
            "model_state_dict). Set policy_format: checkpoint in YAML."
        )
    return cfg
