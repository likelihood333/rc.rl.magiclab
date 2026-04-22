from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import threading
from typing import Any, Optional

import numpy as np
import torch

import magicbot_z1_python as magicbot


def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Body IMU quaternion (w,x,y,z) -> unit vector in body frame"""
    qw, qx, qy, qz = quaternion_wxyz.astype(np.float64)
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


def update_hist(hist_buf: torch.Tensor, cur: torch.Tensor, dim: int) -> torch.Tensor:
    cur = cur.to(hist_buf.device)
    return torch.cat([hist_buf[:, dim:], cur], dim=-1)


def get_activation_module(act_name: str) -> torch.nn.Module:
    act_name = act_name.lower()
    if act_name == "elu":
        return torch.nn.ELU()
    if act_name == "selu":
        return torch.nn.SELU()
    if act_name == "relu":
        return torch.nn.ReLU()
    if act_name == "lrelu":
        return torch.nn.LeakyReLU()
    if act_name == "tanh":
        return torch.nn.Tanh()
    if act_name == "sigmoid":
        return torch.nn.Sigmoid()
    raise ValueError(f"Unsupported actor activation: {act_name}")


def build_actor_from_state_dict(
    actor_state_dict: dict[str, torch.Tensor], activation: str = "elu"
) -> torch.nn.Module:
    linear_indices = sorted(
        int(k.split(".")[0]) for k in actor_state_dict.keys() if k.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError("No linear layers found in actor_state_dict.")

    modules: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for i, idx in enumerate(linear_indices):
        out_dim, in_dim = actor_state_dict[f"{idx}.weight"].shape
        modules[str(idx)] = torch.nn.Linear(in_dim, out_dim)
        modules[str(idx + 1)] = (
            get_activation_module(activation)
            if i < len(linear_indices) - 1
            else torch.nn.Tanh()
        )
    actor = torch.nn.Sequential(modules)
    actor.load_state_dict(actor_state_dict, strict=False)
    return actor


def load_policy(
    policy_path: str,
    device: torch.device,
    obs_dim: int,
    action_dim: int,
    policy_format: str = "auto",
    actor_activation: str = "elu",
) -> torch.nn.Module:
    errors: list[str] = []

    if policy_format in ("auto", "jit"):
        try:
            policy = torch.jit.load(policy_path, map_location=device)
            policy.eval()
            out = policy(torch.zeros(1, obs_dim, dtype=torch.float32, device=device))
            if out.shape[-1] != action_dim:
                raise ValueError(f"JIT output={out.shape[-1]} expected={action_dim}")
            return policy
        except Exception as exc:  # noqa: BLE001
            errors.append(f"jit={exc}")
            if policy_format == "jit":
                raise

    if policy_format in ("auto", "checkpoint"):
        try:
            checkpoint = torch.load(policy_path, map_location=device)
            state_dict = (
                checkpoint["model_state_dict"]
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
                else checkpoint
            )
            actor_state_dict = {
                k[len("actor.") :]: v
                for k, v in state_dict.items()
                if k.startswith("actor.")
            }
            if not actor_state_dict:
                raise ValueError("No actor.* keys found in checkpoint.")

            in_dim = actor_state_dict["0.weight"].shape[1]
            out_idx = max(
                int(k.split(".")[0])
                for k in actor_state_dict.keys()
                if k.endswith(".weight")
            )
            out_dim = actor_state_dict[f"{out_idx}.weight"].shape[0]

            if in_dim != obs_dim:
                raise ValueError(f"checkpoint input={in_dim} expected={obs_dim}")
            if out_dim != action_dim:
                raise ValueError(f"checkpoint output={out_dim} expected={action_dim}")

            actor = build_actor_from_state_dict(
                actor_state_dict, activation=actor_activation
            )
            actor.to(device).eval()
            return actor
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ckpt={exc}")
            if policy_format == "checkpoint":
                raise

    raise RuntimeError(
        f"Failed to load policy format={policy_format}, details={' | '.join(errors)}"
    )


def make_joint_command(
    target: np.ndarray, kp: np.ndarray, kd: np.ndarray, mode: int
) -> magicbot.JointCommand:
    cmd = magicbot.JointCommand()
    for i in range(len(target)):
        j = magicbot.SingleJointCommand()
        j.operation_mode = int(mode)
        j.pos = float(target[i])
        j.vel = 0.0
        j.toq = 0.0
        j.kp = float(kp[i])
        j.kd = float(kd[i])
        cmd.joints.append(j)
    return cmd


def build_zero_torque_command(num_joints: int) -> magicbot.JointCommand:
    cmd = magicbot.JointCommand()
    for _ in range(num_joints):
        j = magicbot.SingleJointCommand()
        j.operation_mode = 200
        j.pos = 0.0
        j.vel = 0.0
        j.toq = 0.0
        j.kp = 0.0
        j.kd = 0.0
        cmd.joints.append(j)
    return cmd


def publish_all_zero_torque(controller) -> None:
    leg_cmd = build_zero_torque_command(int(magicbot.LEG_JOINT_NUM))
    arm_cmd = build_zero_torque_command(int(magicbot.ARM_JOINT_NUM))
    waist_cmd = build_zero_torque_command(int(magicbot.WAIST_JOINT_NUM))
    head_cmd = build_zero_torque_command(int(magicbot.HEAD_JOINT_NUM))
    controller.publish_leg_command(leg_cmd)
    controller.publish_arm_command(arm_cmd)
    controller.publish_waist_command(waist_cmd)
    controller.publish_head_command(head_cmd)


def connect_robot(local_ip: str) -> magicbot.MagicRobot:
    """Create MagicRobot, initialize(local_ip), connect() -> fails with RuntimeError"""
    robot = magicbot.MagicRobot()
    if not robot.initialize(str(local_ip)):
        raise RuntimeError("Failed to initialize robot SDK.")
    status = robot.connect()
    if status.code != magicbot.ErrorCode.OK:
        raise RuntimeError(
            f"Failed to connect robot, code={status.code}, msg={status.message}"
        )
    return robot


def switch_to_low_level(robot: magicbot.MagicRobot) -> None:
    """Single LowLevel switch -> fails with RuntimeError"""
    status = robot.set_motion_control_level(magicbot.ControllerLevel.LowLevel)
    if status.code != magicbot.ErrorCode.OK:
        raise RuntimeError(
            f"Failed to switch to low-level controller: {status.message}"
        )


@dataclass
class RobotStateBuffer:
    """
    SDK 在后台线程调 subscribe_*_state / subscribe_body_imu，把最新状态写进这里；
    主线程 snapshot() 读一份拷贝做观测与下发，避免与回调竞态。

    posH：SDK 关节状态里与 HoST/策略训练一致的关节角定义（与 posL 可能差固定偏置或左右定义）；
    23DOF 部署与仿真对齐时用 posH。
    """

    lock: threading.Lock = field(default_factory=threading.Lock)
    leg_pos: Optional[np.ndarray] = None
    leg_vel: Optional[np.ndarray] = None
    waist_pos: Optional[np.ndarray] = None
    waist_vel: Optional[np.ndarray] = None
    arm_pos: Optional[np.ndarray] = None
    arm_vel: Optional[np.ndarray] = None
    imu_quat_wxyz: Optional[np.ndarray] = None
    imu_ang_vel: Optional[np.ndarray] = None

    def update_leg(self, joint_state) -> None:
        with self.lock:
            self.leg_pos = np.array(
                [float(j.posH) for j in joint_state.joints], dtype=np.float32
            )
            self.leg_vel = np.array(
                [float(j.vel) for j in joint_state.joints], dtype=np.float32
            )

    def update_waist(self, joint_state) -> None:
        with self.lock:
            self.waist_pos = np.array(
                [float(j.posH) for j in joint_state.joints], dtype=np.float32
            )
            self.waist_vel = np.array(
                [float(j.vel) for j in joint_state.joints], dtype=np.float32
            )

    def update_arm(self, joint_state) -> None:
        with self.lock:
            self.arm_pos = np.array(
                [float(j.posH) for j in joint_state.joints], dtype=np.float32
            )
            self.arm_vel = np.array(
                [float(j.vel) for j in joint_state.joints], dtype=np.float32
            )

    def update_imu(self, imu_data) -> None:
        with self.lock:
            self.imu_quat_wxyz = np.array(
                [
                    float(imu_data.orientation[0]),
                    float(imu_data.orientation[1]),
                    float(imu_data.orientation[2]),
                    float(imu_data.orientation[3]),
                ],
                dtype=np.float32,
            )
            self.imu_ang_vel = np.array(
                [
                    float(imu_data.angular_velocity[0]),
                    float(imu_data.angular_velocity[1]),
                    float(imu_data.angular_velocity[2]),
                ],
                dtype=np.float32,
            )

    def is_ready(self, arm_dim_min: int, waist_dim_min: int = 1) -> bool:
        with self.lock:
            return (
                self.leg_pos is not None
                and self.leg_vel is not None
                and self.waist_pos is not None
                and self.waist_vel is not None
                and self.arm_pos is not None
                and self.arm_vel is not None
                and self.imu_quat_wxyz is not None
                and self.imu_ang_vel is not None
                and len(self.arm_pos) >= arm_dim_min
                and len(self.waist_pos) >= waist_dim_min
            )

    def stream_status(self) -> dict[str, object]:
        with self.lock:
            return {
                "leg": self.leg_pos is not None and self.leg_vel is not None,
                "waist": self.waist_pos is not None and self.waist_vel is not None,
                "arm": self.arm_pos is not None and self.arm_vel is not None,
                "imu": self.imu_quat_wxyz is not None and self.imu_ang_vel is not None,
                "leg_dim": 0 if self.leg_pos is None else int(len(self.leg_pos)),
                "waist_dim": 0 if self.waist_pos is None else int(len(self.waist_pos)),
                "arm_dim": 0 if self.arm_pos is None else int(len(self.arm_pos)),
            }

    def snapshot(self) -> dict[str, np.ndarray]:
        with self.lock:
            return {
                "leg_pos": self.leg_pos.copy(),
                "leg_vel": self.leg_vel.copy(),
                "waist_pos": self.waist_pos.copy(),
                "waist_vel": self.waist_vel.copy(),
                "arm_pos": self.arm_pos.copy(),
                "arm_vel": self.arm_vel.copy(),
                "imu_quat_wxyz": self.imu_quat_wxyz.copy(),
                "imu_ang_vel": self.imu_ang_vel.copy(),
            }
