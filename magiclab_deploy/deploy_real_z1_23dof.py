#!/usr/bin/env python3
"""
Deploy HoST Z1 23DOF policy to real robot.

Key points:
- Model output is normalized action (not absolute joint angle).
- Target joint command uses: target = current_pos + clip(action) * action_scale.
- Before switching to low-level control, robot gait must be RECOVERY_STAND or PASSIVE.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

import magicbot_z1_python as magicbot
from config_z1 import Config
from deploy_z1_base import (
    RobotStateBuffer,
    build_zero_torque_command,
    connect_robot,
    get_gravity_orientation,
    load_policy,
    make_joint_command,
    switch_to_low_level,
)


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


def default_23dof_config_path() -> str:
    return os.path.join(os.path.dirname(__file__), "configs", "z1_23dof.yaml")


def resolved_23dof_config_file(cli_config: str | None) -> str:
    """
    与 12DOF 一致：`--config` 为 `configs/` 下文件名（如 z1_23dof.yaml）或绝对路径；
    未传参时使用环境变量 Z1_23DOF_CONFIG，再否则为脚本目录下默认 YAML。
    """
    if cli_config is not None:
        if os.path.isabs(cli_config):
            return cli_config
        return f"configs/{cli_config}"
    return os.environ.get("Z1_23DOF_CONFIG", default_23dof_config_path())


class MagicBotController:
    def __init__(self, cfg: Z1Deploy23dofSettings):
        self.cfg = cfg
        self.running = True
        self.robot: Optional[magicbot.MagicRobot] = None
        self.controller = None
        # 异步回调写、主线程 snapshot 读；关节角用 posH 与 HoST 训练一致（见 deploy_z1_base.RobotStateBuffer）
        self.state = RobotStateBuffer()
        self._published_any_real_command = False
        self._safe_stop_sent = False

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if cfg.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
        self.device = torch.device(cfg.device)
        logging.info("Loading policy on device=%s ...", self.device)
        self.policy = load_policy(
            policy_path=cfg.policy_path,
            device=self.device,
            obs_dim=cfg.num_obs * cfg.his_obs,
            action_dim=cfg.num_actions,
            policy_format=cfg.policy_format,
            actor_activation=cfg.actor_activation,
        )
        self.history_obs = torch.zeros(
            1, cfg.num_obs * cfg.his_obs, dtype=torch.float32, device=self.device
        )
        self.last_action = np.zeros(cfg.num_actions, dtype=np.float32)
        self.cached_action = np.zeros(cfg.num_actions, dtype=np.float32)
        self.infer_period = 1.0 / cfg.infer_hz
        self.cmd_period = 1.0 / cfg.cmd_hz
        self.publish_window_total = 0
        self.publish_window_ok = {"leg": 0, "waist": 0, "arm": 0}
        self.publish_seq = 0
        self.prev_diag_leg_pos: Optional[np.ndarray] = None
        self.prev_diag_waist_pos: Optional[np.ndarray] = None
        self.prev_diag_arm_ctrl_pos: Optional[np.ndarray] = None

        self.arm_control_indices = np.array(
            [int(x) for x in cfg.arm_control_indices.split(",") if x.strip() != ""],
            dtype=np.int32,
        )
        self.arm_control_index_set = set(self.arm_control_indices.tolist())
        if len(self.arm_control_indices) != 10:
            raise ValueError(
                "Expected 10 arm indices for 23d policy. "
                f"Got {len(self.arm_control_indices)}."
            )

        self.kps = np.array(cfg.kps, dtype=np.float32)
        self.kds = np.array(cfg.kds, dtype=np.float32)
        if len(self.kps) != cfg.num_actions or len(self.kds) != cfg.num_actions:
            raise ValueError("kps/kds length must equal num_actions.")
        self.last_cmd_log: Optional[dict[str, np.ndarray]] = None

        self.leg_sdk_to_policy = np.array(
            [int(x) for x in cfg.leg_sdk_to_policy.split(",") if x.strip() != ""],
            dtype=np.int32,
        )
        self.leg_policy_to_sdk = np.array(
            [int(x) for x in cfg.leg_policy_to_sdk.split(",") if x.strip() != ""],
            dtype=np.int32,
        )
        if len(self.leg_sdk_to_policy) != 12 or len(self.leg_policy_to_sdk) != 12:
            raise ValueError("leg_sdk_to_policy/leg_policy_to_sdk must contain 12 indices.")
        logging.info(
            "Leg remap sdk->policy=%s policy->sdk=%s",
            self.leg_sdk_to_policy.tolist(),
            self.leg_policy_to_sdk.tolist(),
        )

    def _signal_handler(self, signum, _frame) -> None:
        logging.info("Received signal %s, stopping...", signum)
        self.running = False

    def _log_joint_command(self, tag: str, cmd: magicbot.JointCommand) -> None:
        modes = np.array([int(j.operation_mode) for j in cmd.joints], dtype=np.int32)
        pos = np.array([float(j.pos) for j in cmd.joints], dtype=np.float32)
        vel = np.array([float(j.vel) for j in cmd.joints], dtype=np.float32)
        toq = np.array([float(j.toq) for j in cmd.joints], dtype=np.float32)
        kp = np.array([float(j.kp) for j in cmd.joints], dtype=np.float32)
        kd = np.array([float(j.kd) for j in cmd.joints], dtype=np.float32)
        logging.info(
            "joint_cmd[%s] mode=%s pos=%s vel=%s toq=%s kp=%s kd=%s",
            tag,
            np.array2string(modes, max_line_width=200),
            np.array2string(pos, precision=4, max_line_width=200),
            np.array2string(vel, precision=4, max_line_width=200),
            np.array2string(toq, precision=4, max_line_width=200),
            np.array2string(kp, precision=4, max_line_width=200),
            np.array2string(kd, precision=4, max_line_width=200),
        )

    def _wait_state_ready(self, timeout_s: float) -> None:
        start = time.time()
        next_log = start
        next_warmup = start + 0.5
        while self.running and (time.time() - start) < timeout_s:
            if self.state.is_ready(arm_dim_min=self.arm_control_indices.max() + 1):
                return
            now = time.time()
            if self.cfg.state_warmup_publish and now >= next_warmup:
                self._publish_prepare_commands_once()
                next_warmup = now + self.cfg.state_warmup_interval_s
            if now >= next_log:
                s = self.state.stream_status()
                logging.info(
                    "Waiting callbacks... leg=%s(%d) waist=%s(%d) arm=%s(%d) imu=%s",
                    s["leg"],
                    s["leg_dim"],
                    s["waist"],
                    s["waist_dim"],
                    s["arm"],
                    s["arm_dim"],
                    s["imu"],
                )
                next_log = now + 1.0
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting for robot state callbacks.")

    def _publish_prepare_commands_once(self) -> None:
        """Publish mode=200 zero commands to warm up low-level stream."""
        if self.controller is None:
            return
        try:
            self.controller.publish_leg_command(
                build_zero_torque_command(int(magicbot.LEG_JOINT_NUM))
            )
            self.controller.publish_waist_command(
                build_zero_torque_command(int(magicbot.WAIST_JOINT_NUM))
            )
            self.controller.publish_arm_command(
                build_zero_torque_command(int(magicbot.ARM_JOINT_NUM))
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning("warmup publish error: %s", exc)

    def _set_required_gait_before_lowlevel(self) -> None:
        """Low-level switch requires RECOVERY_STAND or PASSIVE."""
        high = self.robot.get_high_level_motion_controller()
        if not high.initialize():
            raise RuntimeError("Failed to initialize high-level controller.")

        logging.info("Preparing gait for low-level switch...")
        status = high.set_gait(magicbot.GaitMode.GAIT_RECOVERY_STAND, 10000)
        if status.code != magicbot.ErrorCode.OK:
            logging.warning(
                "set GAIT_RECOVERY_STAND failed: %s, fallback to GAIT_PASSIVE",
                status.message,
            )
            status = high.set_gait(magicbot.GaitMode.GAIT_PASSIVE, 10000)
            if status.code != magicbot.ErrorCode.OK:
                high.shutdown()
                raise RuntimeError(
                    "Failed to set required gait before low-level, "
                    f"code={status.code}, msg={status.message}"
                )
        logging.info(
            "Gait command accepted, wait %.2fs before low-level switch.",
            self.cfg.gait_settle_s,
        )
        time.sleep(self.cfg.gait_settle_s)
        high.shutdown()

    def _reset_to_highlevel(self) -> None:
        """Try to reset motion control level to HighLevel."""
        try:
            status = self.robot.set_motion_control_level(magicbot.ControllerLevel.HighLevel)
            if status.code == magicbot.ErrorCode.OK:
                logging.info("Reset control level to HighLevel.")
            else:
                logging.warning("Reset to HighLevel failed: %s", status.message)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Reset to HighLevel exception: %s", exc)

    def _switch_to_lowlevel_with_retry(self) -> None:
        last_msg = ""
        for i in range(1, self.cfg.lowlevel_retry + 1):
            if self.cfg.force_highlevel_reset_before_lowlevel:
                self._reset_to_highlevel()
                time.sleep(self.cfg.lowlevel_retry_interval_s)
            if self.cfg.prepare_gait:
                self._set_required_gait_before_lowlevel()
            try:
                switch_to_low_level(self.robot)
            except RuntimeError as e:
                last_msg = str(e)
                logging.warning(
                    "Low-level switch failed on attempt %d/%d: %s",
                    i,
                    self.cfg.lowlevel_retry,
                    last_msg,
                )
                if i < self.cfg.lowlevel_retry:
                    time.sleep(self.cfg.lowlevel_retry_interval_s)
                continue
            logging.info("Switched to low-level control on attempt %d.", i)
            return
        raise RuntimeError(
            "Failed to switch to low-level control, "
            f"msg={last_msg}, retries={self.cfg.lowlevel_retry}"
        )

    def _assert_no_robot_faults(self, stage: str) -> None:
        """Abort deploy if robot reports any active faults."""
        monitor = self.robot.get_state_monitor()
        status, state = monitor.get_current_state()
        if status.code != magicbot.ErrorCode.OK:
            raise RuntimeError(
                f"[{stage}] Failed to get robot state, code={status.code}, msg={status.message}"
            )
        faults = list(state.faults)
        if not faults:
            logging.info("[%s] Robot faults check passed: no active faults.", stage)
            return

        fault_lines = []
        for fault in faults:
            code = getattr(fault, "error_code", "unknown")
            msg = getattr(fault, "error_message", "")
            fault_lines.append(f"{code}:{msg}")
        raise RuntimeError(
            f"[{stage}] Active robot faults detected, aborting deploy: {' | '.join(fault_lines)}"
        )

    def initialize_robot(self) -> None:
        """连接机器人（与 12DOF `initialize_robot` 同名；后续含故障检查、切低层重试等 23DOF 流程）。"""
        logging.info("SDK robot model: %s", magicbot.get_robot_model())
        self.robot = connect_robot(self.cfg.local_ip)
        logging.info("Successfully connected to robot")
        self._assert_no_robot_faults("post-connect")

        self._switch_to_lowlevel_with_retry()
        self._assert_no_robot_faults("post-lowlevel-switch")

        self.controller = self.robot.get_low_level_motion_controller()
        if hasattr(self.controller, "initialize"):
            try:
                ok = self.controller.initialize()
                logging.info("Low-level controller initialize() returned: %s", ok)
            except Exception as exc:  # noqa: BLE001
                logging.warning("Low-level controller initialize() exception: %s", exc)
        self.controller.subscribe_leg_state(self.state.update_leg)
        self.controller.subscribe_waist_state(self.state.update_waist)
        self.controller.subscribe_arm_state(self.state.update_arm)
        self.controller.subscribe_body_imu(self.state.update_imu)
        logging.info("Subscribed leg/waist/arm/imu state callbacks.")

        # Warm-up：部分机型订阅后需先有下行（零力矩）流量，状态回调才稳定；避免 _wait_state_ready 空等。
        if self.cfg.initial_stream_warmup_s > 0:
            logging.info(
                "Initial stream warmup for %.2fs before readiness wait.",
                self.cfg.initial_stream_warmup_s,
            )
            end_t = time.time() + self.cfg.initial_stream_warmup_s
            while self.running and time.time() < end_t:
                self._publish_prepare_commands_once()
                time.sleep(self.cfg.state_warmup_interval_s)

        self._wait_state_ready(self.cfg.state_timeout_s)
        logging.info("Robot state ready, start control loop.")

    def _build_policy_obs(self, snap: dict[str, np.ndarray]) -> np.ndarray:
        # HoST Z1 23d order: [leg12, waist1, arm10]
        # Remap SDK leg order to policy leg order.
        q_leg = snap["leg_pos"][self.leg_sdk_to_policy]
        dq_leg = snap["leg_vel"][self.leg_sdk_to_policy]
        q_waist = snap["waist_pos"][:1]
        dq_waist = snap["waist_vel"][:1]
        q_arm = snap["arm_pos"][self.arm_control_indices]
        dq_arm = snap["arm_vel"][self.arm_control_indices]

        qj = np.concatenate([q_leg, q_waist, q_arm], dtype=np.float32)
        dqj = np.concatenate([dq_leg, dq_waist, dq_arm], dtype=np.float32)
        if len(qj) != self.cfg.num_actions:
            raise ValueError(
                f"State joint dim mismatch, got {len(qj)} expected {self.cfg.num_actions}."
            )

        omega = snap["imu_ang_vel"] * self.cfg.ang_vel_scale
        gravity = get_gravity_orientation(snap["imu_quat_wxyz"]) * self.cfg.gravity_scale
        obs = np.concatenate(
            [
                omega.astype(np.float32),
                gravity.astype(np.float32),
                qj * self.cfg.dof_pos_scale,
                dqj * self.cfg.dof_vel_scale,
                self.last_action.astype(np.float32),
                np.array([self.cfg.action_rescale_obs], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        if len(obs) != self.cfg.num_obs:
            raise ValueError(
                f"Obs dim mismatch, got {len(obs)} expected {self.cfg.num_obs}."
            )
        return obs

    def _publish_action(self, action: np.ndarray, snap: dict[str, np.ndarray]) -> None:
        q_leg = snap["leg_pos"]
        q_waist = snap["waist_pos"][:1]
        q_arm_full = snap["arm_pos"]
        q_arm_ctrl = q_arm_full[self.arm_control_indices]

        scaled = np.clip(action, -1.0, 1.0) * self.cfg.action_scale
        delta_leg_raw = scaled[0:12]
        delta_waist_raw = scaled[12:13]
        delta_arm_raw = scaled[13:23]
        target_leg = q_leg + delta_leg_raw
        target_waist = q_waist + delta_waist_raw
        target_arm_ctrl = q_arm_ctrl + delta_arm_raw

        if self.cfg.max_joint_delta > 0.0:
            target_leg = np.clip(
                target_leg, q_leg - self.cfg.max_joint_delta, q_leg + self.cfg.max_joint_delta
            )
            target_waist = np.clip(
                target_waist,
                q_waist - self.cfg.max_joint_delta,
                q_waist + self.cfg.max_joint_delta,
            )
            target_arm_ctrl = np.clip(
                target_arm_ctrl,
                q_arm_ctrl - self.cfg.max_joint_delta,
                q_arm_ctrl + self.cfg.max_joint_delta,
            )

        # Convert policy leg order back to SDK leg order for publishing.
        target_leg_sdk = target_leg[self.leg_policy_to_sdk]
        leg_kp_sdk = self.kps[0:12][self.leg_policy_to_sdk]
        leg_kd_sdk = self.kds[0:12][self.leg_policy_to_sdk]
        leg_delta_sdk = target_leg_sdk - q_leg
        waist_delta = target_waist - q_waist
        arm_delta_ctrl = target_arm_ctrl - q_arm_ctrl
        leg_cmd = make_joint_command(
            target_leg_sdk, leg_kp_sdk, leg_kd_sdk, mode=self.cfg.leg_operation_mode
        )
        waist_cmd = make_joint_command(
            target_waist, self.kps[12:13], self.kds[12:13], mode=self.cfg.waist_operation_mode
        )

        arm_target_full = q_arm_full.copy()
        arm_target_full[self.arm_control_indices] = target_arm_ctrl
        arm_kp = np.zeros_like(arm_target_full, dtype=np.float32)
        arm_kd = np.zeros_like(arm_target_full, dtype=np.float32)
        arm_kp[self.arm_control_indices] = self.kps[13:23]
        arm_kd[self.arm_control_indices] = self.kds[13:23]

        arm_cmd = magicbot.JointCommand()
        for i in range(len(arm_target_full)):
            j = magicbot.SingleJointCommand()
            if i in self.arm_control_index_set:
                j.operation_mode = int(self.cfg.arm_operation_mode)
            else:
                j.operation_mode = int(self.cfg.arm_hold_operation_mode)
            j.pos = float(arm_target_full[i])
            j.vel = 0.0
            j.toq = 0.0
            j.kp = float(arm_kp[i])
            j.kd = float(arm_kd[i])
            arm_cmd.joints.append(j)

        if self.cfg.log_command_values:
            # clip ratio: percentage of joints touching max_joint_delta bound
            # after action_scale. Helps judge whether control is limit-saturated.
            eps = 1e-6
            if self.cfg.max_joint_delta > 0.0:
                leg_clip = np.abs(leg_delta_sdk) >= (self.cfg.max_joint_delta - eps)
                waist_clip = np.abs(waist_delta) >= (self.cfg.max_joint_delta - eps)
                arm_clip = np.abs(arm_delta_ctrl) >= (self.cfg.max_joint_delta - eps)
                clip_ratio_leg = float(np.mean(leg_clip))
                clip_ratio_waist = float(np.mean(waist_clip))
                clip_ratio_arm = float(np.mean(arm_clip))
                clip_ratio_total = float(
                    np.mean(np.concatenate([leg_clip, waist_clip, arm_clip]))
                )
            else:
                clip_ratio_leg = 0.0
                clip_ratio_waist = 0.0
                clip_ratio_arm = 0.0
                clip_ratio_total = 0.0
            self.last_cmd_log = {
                "target_leg_sdk": target_leg_sdk.copy(),
                "delta_leg_sdk": leg_delta_sdk.copy(),
                "target_waist": target_waist.copy(),
                "delta_waist": waist_delta.copy(),
                "target_arm_ctrl": target_arm_ctrl.copy(),
                "delta_arm_ctrl": arm_delta_ctrl.copy(),
                "clip_ratio_leg": clip_ratio_leg,
                "clip_ratio_waist": clip_ratio_waist,
                "clip_ratio_arm": clip_ratio_arm,
                "clip_ratio_total": clip_ratio_total,
            }

        if self.cfg.log_joint_command_each_publish:
            self.publish_seq += 1
            logging.info("joint_cmd_seq=%d", self.publish_seq)
            self._log_joint_command("leg", leg_cmd)
            self._log_joint_command("waist", waist_cmd)
            self._log_joint_command("arm", arm_cmd)

        if not self.cfg.dry_run:
            s_leg = self.controller.publish_leg_command(leg_cmd)
            s_waist = self.controller.publish_waist_command(waist_cmd)
            s_arm = self.controller.publish_arm_command(arm_cmd)
            self._published_any_real_command = True
            self.publish_window_total += 1
            if s_leg.code == magicbot.ErrorCode.OK:
                self.publish_window_ok["leg"] += 1
            if s_waist.code == magicbot.ErrorCode.OK:
                self.publish_window_ok["waist"] += 1
            if s_arm.code == magicbot.ErrorCode.OK:
                self.publish_window_ok["arm"] += 1
            if (
                s_leg.code != magicbot.ErrorCode.OK
                or s_waist.code != magicbot.ErrorCode.OK
                or s_arm.code != magicbot.ErrorCode.OK
            ):
                logging.warning(
                    "Publish status: leg=%s waist=%s arm=%s",
                    s_leg.message,
                    s_waist.message,
                    s_arm.message,
                )

    def run(self) -> None:
        infer_count = 0
        cmd_count = 0
        stat_start = time.perf_counter()
        next_infer_time = time.perf_counter()
        next_cmd_time = time.perf_counter() + self.cmd_period
        while self.running:
            now = time.perf_counter()

            if now >= next_infer_time:
                snap = self.state.snapshot()
                obs = self._build_policy_obs(snap)
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )
                self.history_obs = torch.cat(
                    [self.history_obs[:, self.cfg.num_obs :], obs_t], dim=-1
                )

                with torch.inference_mode():
                    action = self.policy(self.history_obs).detach().cpu().numpy().reshape(-1)
                if len(action) != self.cfg.num_actions:
                    raise ValueError(
                        f"Action dim mismatch got={len(action)} expected={self.cfg.num_actions}"
                    )
                self.cached_action = action.astype(np.float32)
                self.last_action = self.cached_action.copy()
                infer_count += 1
                while next_infer_time <= now:
                    next_infer_time += self.infer_period

            now = time.perf_counter()
            if now >= next_cmd_time:
                snap_cmd = self.state.snapshot()
                self._publish_action(self.cached_action, snap_cmd)
                cmd_count += 1
                while next_cmd_time <= now:
                    next_cmd_time += self.cmd_period

            now = time.perf_counter()
            elapsed = now - stat_start
            if elapsed >= 1.0:
                action_abs = np.abs(self.cached_action)
                action_sat_ratio = float(np.mean(action_abs >= 0.98))
                logging.info(
                    (
                        "running... infer=%.1fHz cmd=%.1fHz target(infer/cmd)=%.1f/%.1f "
                        "dry_run=%s action[min,max,sat]=[%.3f,%.3f,%.1f%%]"
                    ),
                    infer_count / elapsed,
                    cmd_count / elapsed,
                    self.cfg.infer_hz,
                    self.cfg.cmd_hz,
                    self.cfg.dry_run,
                    float(np.min(self.cached_action)),
                    float(np.max(self.cached_action)),
                    action_sat_ratio * 100.0,
                )
                if self.cfg.log_command_values and self.last_cmd_log is not None:
                    logging.info(
                        (
                            "delta_clip(edge): total=%.1f%% leg=%.1f%% waist=%.1f%% "
                            "arm=%.1f%% limit=%.3f"
                        ),
                        self.last_cmd_log["clip_ratio_total"] * 100.0,
                        self.last_cmd_log["clip_ratio_leg"] * 100.0,
                        self.last_cmd_log["clip_ratio_waist"] * 100.0,
                        self.last_cmd_log["clip_ratio_arm"] * 100.0,
                        self.cfg.max_joint_delta,
                    )
                    if self.publish_window_total > 0:
                        logging.info(
                            (
                                "publish_ok(window): leg=%d/%d waist=%d/%d arm=%d/%d"
                            ),
                            self.publish_window_ok["leg"],
                            self.publish_window_total,
                            self.publish_window_ok["waist"],
                            self.publish_window_total,
                            self.publish_window_ok["arm"],
                            self.publish_window_total,
                        )

                    # Compute tracking error from latest state snapshot:
                    # tracking_err = cmd_target - current_q
                    snap_now = self.state.snapshot()
                    err_leg = self.last_cmd_log["target_leg_sdk"] - snap_now["leg_pos"]
                    err_waist = self.last_cmd_log["target_waist"] - snap_now["waist_pos"][:1]
                    err_arm = self.last_cmd_log["target_arm_ctrl"] - snap_now["arm_pos"][
                        self.arm_control_indices
                    ]
                    logging.info(
                        (
                            "tracking_err(abs mean/max): "
                            "leg=%.4f/%.4f waist=%.4f/%.4f arm=%.4f/%.4f"
                        ),
                        float(np.mean(np.abs(err_leg))),
                        float(np.max(np.abs(err_leg))),
                        float(np.mean(np.abs(err_waist))),
                        float(np.max(np.abs(err_waist))),
                        float(np.mean(np.abs(err_arm))),
                        float(np.max(np.abs(err_arm))),
                    )
                    q_leg_now = snap_now["leg_pos"]
                    q_waist_now = snap_now["waist_pos"][:1]
                    q_arm_ctrl_now = snap_now["arm_pos"][self.arm_control_indices]
                    if (
                        self.prev_diag_leg_pos is not None
                        and self.prev_diag_waist_pos is not None
                        and self.prev_diag_arm_ctrl_pos is not None
                    ):
                        move_leg = q_leg_now - self.prev_diag_leg_pos
                        move_waist = q_waist_now - self.prev_diag_waist_pos
                        move_arm = q_arm_ctrl_now - self.prev_diag_arm_ctrl_pos
                        logging.info(
                            (
                                "state_move_1s(abs mean/max): "
                                "leg=%.4f/%.4f waist=%.4f/%.4f arm=%.4f/%.4f"
                            ),
                            float(np.mean(np.abs(move_leg))),
                            float(np.max(np.abs(move_leg))),
                            float(np.mean(np.abs(move_waist))),
                            float(np.max(np.abs(move_waist))),
                            float(np.mean(np.abs(move_arm))),
                            float(np.max(np.abs(move_arm))),
                        )

                        cmd_leg = self.last_cmd_log["delta_leg_sdk"]
                        cmd_waist = self.last_cmd_log["delta_waist"]
                        cmd_arm = self.last_cmd_log["delta_arm_ctrl"]
                        cmd_has_effort = (
                            float(np.max(np.abs(cmd_leg))) > 0.02
                            or float(np.max(np.abs(cmd_waist))) > 0.02
                            or float(np.max(np.abs(cmd_arm))) > 0.02
                        )
                        state_almost_static = (
                            float(np.max(np.abs(move_leg))) < 0.002
                            and float(np.max(np.abs(move_waist))) < 0.002
                            and float(np.max(np.abs(move_arm))) < 0.002
                        )
                        if cmd_has_effort and state_almost_static:
                            logging.warning(
                                "possible_no_execution: cmd_delta present but state_move_1s is tiny."
                            )
                    self.prev_diag_leg_pos = q_leg_now.copy()
                    self.prev_diag_waist_pos = q_waist_now.copy()
                    self.prev_diag_arm_ctrl_pos = q_arm_ctrl_now.copy()
                    if self.cfg.log_command_detail:
                        logging.info(
                            "cmd_target leg_sdk=%s waist=%s arm_ctrl=%s",
                            np.array2string(
                                self.last_cmd_log["target_leg_sdk"],
                                precision=3,
                                max_line_width=200,
                            ),
                            np.array2string(self.last_cmd_log["target_waist"], precision=3),
                            np.array2string(
                                self.last_cmd_log["target_arm_ctrl"],
                                precision=3,
                                max_line_width=200,
                            ),
                        )
                        logging.info(
                            "cmd_delta  leg_sdk=%s waist=%s arm_ctrl=%s",
                            np.array2string(
                                self.last_cmd_log["delta_leg_sdk"],
                                precision=4,
                                max_line_width=200,
                            ),
                            np.array2string(self.last_cmd_log["delta_waist"], precision=4),
                            np.array2string(
                                self.last_cmd_log["delta_arm_ctrl"],
                                precision=4,
                                max_line_width=200,
                            ),
                        )
                infer_count = 0
                cmd_count = 0
                stat_start = now
                self.publish_window_total = 0
                self.publish_window_ok = {"leg": 0, "waist": 0, "arm": 0}

            wakeup_time = min(next_infer_time, next_cmd_time)
            sleep_s = wakeup_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def _send_safe_stop_sequence(self) -> None:
        if self.cfg.dry_run or self.controller is None or self._safe_stop_sent:
            return
        if not self._published_any_real_command:
            logging.info("Skip safe-stop sequence: no real control command was published.")
            return

        try:
            if not self.state.is_ready(arm_dim_min=self.arm_control_indices.max() + 1):
                logging.warning("Skip safe-stop sequence: robot state not ready.")
                return
            snap = self.state.snapshot()
            leg_target = snap["leg_pos"]
            waist_target = snap["waist_pos"][:1]
            arm_target_full = snap["arm_pos"]

            leg_kp = self.kps[0:12] * self.cfg.safe_stop_kp_scale
            leg_kd = self.kds[0:12] * self.cfg.safe_stop_kd_scale
            waist_kp = self.kps[12:13] * self.cfg.safe_stop_kp_scale
            waist_kd = self.kds[12:13] * self.cfg.safe_stop_kd_scale
            arm_kp = np.zeros_like(arm_target_full, dtype=np.float32)
            arm_kd = np.zeros_like(arm_target_full, dtype=np.float32)
            arm_kp[self.arm_control_indices] = self.kps[13:23] * self.cfg.safe_stop_kp_scale
            arm_kd[self.arm_control_indices] = self.kds[13:23] * self.cfg.safe_stop_kd_scale

            leg_cmd = make_joint_command(leg_target, leg_kp, leg_kd, self.cfg.leg_operation_mode)
            waist_cmd = make_joint_command(
                waist_target, waist_kp, waist_kd, self.cfg.waist_operation_mode
            )

            arm_cmd = magicbot.JointCommand()
            for i in range(len(arm_target_full)):
                j = magicbot.SingleJointCommand()
                j.operation_mode = (
                    int(self.cfg.arm_operation_mode)
                    if i in self.arm_control_index_set
                    else int(self.cfg.arm_hold_operation_mode)
                )
                j.pos = float(arm_target_full[i])
                j.vel = 0.0
                j.toq = 0.0
                j.kp = float(arm_kp[i])
                j.kd = float(arm_kd[i])
                arm_cmd.joints.append(j)

            loop_hz = max(1, self.cfg.cmd_hz)
            repeat = max(1, int(self.cfg.safe_stop_duration_s * loop_hz))
            for _ in range(repeat):
                self.controller.publish_leg_command(leg_cmd)
                self.controller.publish_waist_command(waist_cmd)
                self.controller.publish_arm_command(arm_cmd)
                time.sleep(1.0 / loop_hz)
            self._safe_stop_sent = True
            logging.info("Safe-stop sequence sent.")
        except Exception as exc:  # noqa: BLE001
            logging.warning("safe-stop sequence error: %s", exc)

    def _send_zero_torque_release(self) -> None:
        """Release all joints by publishing mode=200 zero commands."""
        if self.cfg.dry_run or self.controller is None:
            return
        try:
            leg_n = int(magicbot.LEG_JOINT_NUM)
            waist_n = int(magicbot.WAIST_JOINT_NUM)
            arm_n = int(magicbot.ARM_JOINT_NUM)
            head_n = int(magicbot.HEAD_JOINT_NUM) if hasattr(magicbot, "HEAD_JOINT_NUM") else 0

            leg_cmd = build_zero_torque_command(leg_n)
            waist_cmd = build_zero_torque_command(waist_n)
            arm_cmd = build_zero_torque_command(arm_n)
            head_cmd = build_zero_torque_command(head_n) if head_n > 0 else None

            loop_hz = max(1, self.cfg.cmd_hz)
            repeat = max(1, int(self.cfg.release_duration_s * loop_hz))
            for _ in range(repeat):
                self.controller.publish_leg_command(leg_cmd)
                self.controller.publish_waist_command(waist_cmd)
                self.controller.publish_arm_command(arm_cmd)
                if head_cmd is not None and hasattr(self.controller, "publish_head_command"):
                    self.controller.publish_head_command(head_cmd)
                time.sleep(1.0 / loop_hz)

            logging.info("Zero-torque release sequence sent for %.2fs.", self.cfg.release_duration_s)
        except Exception as exc:  # noqa: BLE001
            logging.warning("zero-torque release error: %s", exc)

    def cleanup(self) -> None:
        logging.info("Cleaning up...")
        if self.cfg.release_on_exit:
            self._send_zero_torque_release()
        else:
            self._send_safe_stop_sequence()
        try:
            if self.controller is not None:
                self.controller.shutdown()
        except Exception as exc:  # noqa: BLE001
            logging.warning("controller shutdown error: %s", exc)

        try:
            if self.robot is not None:
                self.robot.disconnect()
                self.robot.shutdown()
        except Exception as exc:  # noqa: BLE001
            logging.warning("robot cleanup error: %s", exc)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Deploy HoST Z1 23DOF to MagicBot Z1.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML: file name under configs/ (e.g. z1_23dof.yaml) or absolute path; "
        "if omitted, use env Z1_23DOF_CONFIG or default beside this script.",
    )
    args = parser.parse_args()
    config_path = resolved_23dof_config_file(args.config)
    logging.info("Loading 23DOF config from %s", config_path)
    try:
        settings = load_23dof_settings(config_path)
    except Exception as exc:  # noqa: BLE001
        logging.error("Invalid config: %s", exc)
        return 1

    controller: Optional[MagicBotController] = None
    try:
        controller = MagicBotController(settings)
        controller.initialize_robot()
        controller.run()
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user (Ctrl+C).")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.error("Deployment failed: %s", exc)
        return 1
    finally:
        if controller is not None:
            controller.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

