#!/usr/bin/env python3
# ===================== Metadata =====================
# Author: Mingzhu Li
# Last Modified: 2026-04-21 01:42:13 CST
# ====================================================
"""
Deploy host RL policy to MagicBot Z1 real robot via magicbot-z1_sdk.

Target environment:
- Ubuntu 22.04
- Python 3.10
- magicbot_z1_python SDK binding available in PYTHONPATH

Policy:
- /home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

import magicbot_z1_python as magicbot


def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Project gravity vector to body frame from quaternion (w, x, y, z)."""
    qw, qx, qy, qz = quaternion_wxyz
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2.0 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2.0 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return gravity_orientation


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
        raise ValueError("No linear layers found in actor state_dict.")

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


@dataclass
class RobotStateBuffer:
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


class HostPolicyDeployer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = True
        self.robot: Optional[magicbot.MagicRobot] = None
        self.controller = None
        self.state = RobotStateBuffer()

        self.device = torch.device(args.device)
        self.policy = load_policy(
            policy_path=args.model_path,
            device=self.device,
            obs_dim=args.num_obs * args.his_obs,
            action_dim=args.num_actions,
            policy_format=args.policy_format,
            actor_activation=args.actor_activation,
        )
        self.history_obs = torch.zeros(
            1, args.num_obs * args.his_obs, dtype=torch.float32, device=self.device
        )
        self.last_action = np.zeros(args.num_actions, dtype=np.float32)
        self.cached_action = np.zeros(args.num_actions, dtype=np.float32)
        self.infer_period = 1.0 / args.infer_hz
        self.cmd_period = 1.0 / args.cmd_hz

        self.arm_control_indices = np.array(
            [int(x) for x in args.arm_control_indices.split(",") if x.strip() != ""],
            dtype=np.int32,
        )
        self.arm_control_index_set = set(self.arm_control_indices.tolist())
        if len(self.arm_control_indices) != 10:
            raise ValueError(
                "Expected 10 controlled arm joints for 23d policy. "
                f"Got {len(self.arm_control_indices)} from --arm_control_indices."
            )

        self.kps = np.array(args.kps, dtype=np.float32)
        self.kds = np.array(args.kds, dtype=np.float32)
        if len(self.kps) != args.num_actions or len(self.kds) != args.num_actions:
            raise ValueError("kps/kds length must equal num_actions.")

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, _frame) -> None:
        logging.info("Received signal %s, stopping...", signum)
        self.running = False

    def _wait_state_ready(self, timeout_s: float) -> None:
        start = time.time()
        while self.running and (time.time() - start) < timeout_s:
            if self.state.is_ready(arm_dim_min=self.arm_control_indices.max() + 1):
                return
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting for robot state callbacks.")

    def setup_robot(self) -> None:
        logging.info("SDK robot model: %s", magicbot.get_robot_model())
        self.robot = magicbot.MagicRobot()

        if not self.robot.initialize(self.args.local_ip):
            raise RuntimeError("Failed to initialize robot SDK.")

        status = self.robot.connect()
        if status.code != magicbot.ErrorCode.OK:
            raise RuntimeError(
                f"Failed to connect robot, code={status.code}, msg={status.message}"
            )

        if self.args.set_lowlevel_gait:
            high_controller = self.robot.get_high_level_motion_controller()
            if not high_controller.initialize():
                raise RuntimeError("Failed to initialize high-level controller.")
            status = high_controller.set_gait(magicbot.GaitMode.GAIT_LOWLEVL_SDK, 10000)
            if status.code != magicbot.ErrorCode.OK:
                raise RuntimeError(
                    f"Failed to set GAIT_LOWLEVL_SDK, code={status.code}, msg={status.message}"
                )
            high_controller.shutdown()
            logging.info("Set gait to GAIT_LOWLEVL_SDK.")

        status = self.robot.set_motion_control_level(magicbot.ControllerLevel.LowLevel)
        if status.code != magicbot.ErrorCode.OK:
            raise RuntimeError(
                f"Failed to switch to low-level control, code={status.code}, msg={status.message}"
            )

        self.controller = self.robot.get_low_level_motion_controller()
        self.controller.subscribe_leg_state(self.state.update_leg)
        self.controller.subscribe_waist_state(self.state.update_waist)
        self.controller.subscribe_arm_state(self.state.update_arm)
        self.controller.subscribe_body_imu(self.state.update_imu)
        logging.info("Subscribed leg/waist/arm/imu state callbacks.")

        self._wait_state_ready(self.args.state_timeout_s)
        logging.info("Robot state ready, start control loop.")

    def _build_policy_obs(self, snap: dict[str, np.ndarray]) -> np.ndarray:
        # host_23d policy order: [leg12, waist1, arm10]
        q_leg = snap["leg_pos"]
        dq_leg = snap["leg_vel"]
        q_waist = snap["waist_pos"][:1]
        dq_waist = snap["waist_vel"][:1]
        q_arm = snap["arm_pos"][self.arm_control_indices]
        dq_arm = snap["arm_vel"][self.arm_control_indices]

        qj = np.concatenate([q_leg, q_waist, q_arm], dtype=np.float32)
        dqj = np.concatenate([dq_leg, dq_waist, dq_arm], dtype=np.float32)
        if len(qj) != self.args.num_actions:
            raise ValueError(
                f"State joint dim mismatch, got {len(qj)} expected {self.args.num_actions}."
            )

        omega = snap["imu_ang_vel"] * self.args.ang_vel_scale
        gravity = (
            get_gravity_orientation(snap["imu_quat_wxyz"]) * self.args.gravity_scale
        )

        obs = np.concatenate(
            [
                omega.astype(np.float32),
                gravity.astype(np.float32),
                qj * self.args.dof_pos_scale,
                dqj * self.args.dof_vel_scale,
                self.last_action.astype(np.float32),
                np.array([self.args.action_rescale_obs], dtype=np.float32),
            ],
            dtype=np.float32,
        )
        if len(obs) != self.args.num_obs:
            raise ValueError(
                f"Obs dim mismatch, got {len(obs)} expected {self.args.num_obs}."
            )
        return obs

    def _publish_action(self, action: np.ndarray, snap: dict[str, np.ndarray]) -> None:
        # relative_current_pos: target = current + action * scale
        q_leg = snap["leg_pos"]
        q_waist = snap["waist_pos"][:1]
        q_arm_full = snap["arm_pos"]
        q_arm_ctrl = q_arm_full[self.arm_control_indices]

        scaled = np.clip(action, -1.0, 1.0) * self.args.action_scale
        target_leg = q_leg + scaled[0:12]
        target_waist = q_waist + scaled[12:13]
        target_arm_ctrl = q_arm_ctrl + scaled[13:23]

        if self.args.max_joint_delta > 0.0:
            target_leg = np.clip(
                target_leg,
                q_leg - self.args.max_joint_delta,
                q_leg + self.args.max_joint_delta,
            )
            target_waist = np.clip(
                target_waist,
                q_waist - self.args.max_joint_delta,
                q_waist + self.args.max_joint_delta,
            )
            target_arm_ctrl = np.clip(
                target_arm_ctrl,
                q_arm_ctrl - self.args.max_joint_delta,
                q_arm_ctrl + self.args.max_joint_delta,
            )

        leg_cmd = make_joint_command(
            target_leg, self.kps[0:12], self.kds[0:12], mode=self.args.leg_operation_mode
        )
        waist_cmd = make_joint_command(
            target_waist,
            self.kps[12:13],
            self.kds[12:13],
            mode=self.args.waist_operation_mode,
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
                j.operation_mode = int(self.args.arm_operation_mode)
            else:
                # Keep non-policy joints in prepare mode by default.
                j.operation_mode = int(self.args.arm_hold_operation_mode)
            j.pos = float(arm_target_full[i])
            j.vel = 0.0
            j.toq = 0.0
            j.kp = float(arm_kp[i])
            j.kd = float(arm_kd[i])
            arm_cmd.joints.append(j)

        if not self.args.dry_run:
            s_leg = self.controller.publish_leg_command(leg_cmd)
            s_waist = self.controller.publish_waist_command(waist_cmd)
            s_arm = self.controller.publish_arm_command(arm_cmd)
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

            # Inference loop runs at infer_hz.
            if now >= next_infer_time:
                snap = self.state.snapshot()
                obs = self._build_policy_obs(snap)
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(
                    device=self.device, dtype=torch.float32
                )
                self.history_obs = torch.cat(
                    [self.history_obs[:, self.args.num_obs :], obs_t], dim=-1
                )

                with torch.inference_mode():
                    action = (
                        self.policy(self.history_obs).detach().cpu().numpy().reshape(-1)
                    )
                if len(action) != self.args.num_actions:
                    raise ValueError(
                        f"Action dim mismatch got={len(action)} expected={self.args.num_actions}"
                    )

                self.cached_action = action.astype(np.float32)
                self.last_action = self.cached_action.copy()
                infer_count += 1
                while next_infer_time <= now:
                    next_infer_time += self.infer_period

            # Command publishing loop runs at cmd_hz.
            now = time.perf_counter()
            if now >= next_cmd_time:
                snap_cmd = self.state.snapshot()
                self._publish_action(self.cached_action, snap_cmd)
                cmd_count += 1
                while next_cmd_time <= now:
                    next_cmd_time += self.cmd_period

            # Periodic status log, including separated infer/cmd rates.
            now = time.perf_counter()
            elapsed = now - stat_start
            if elapsed >= 1.0:
                logging.info(
                    "running... infer=%.1fHz cmd=%.1fHz target(infer/cmd)=%.1f/%.1f dry_run=%s action[0:3]=%s",
                    infer_count / elapsed,
                    cmd_count / elapsed,
                    self.args.infer_hz,
                    self.args.cmd_hz,
                    self.args.dry_run,
                    np.array2string(self.cached_action[:3], precision=3),
                )
                infer_count = 0
                cmd_count = 0
                stat_start = now

            wakeup_time = min(next_infer_time, next_cmd_time)
            sleep_s = wakeup_time - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)

    def cleanup(self) -> None:
        logging.info("Cleaning up...")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy host_gym_model_12000.pt to real MagicBot Z1."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ubuntu/rc.rl.magiclab/sim2real/model/host_gym_model_12000.pt",
    )
    parser.add_argument("--local_ip", type=str, default="192.168.54.119") # 大脑算力板 ip
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--policy_format", type=str, default="auto", choices=["auto", "jit", "checkpoint"])
    parser.add_argument("--actor_activation", type=str, default="elu")

    parser.add_argument("--num_actions", type=int, default=23)
    parser.add_argument("--num_obs", type=int, default=76)
    parser.add_argument("--his_obs", type=int, default=6)

    parser.add_argument(
        "--control_hz",
        type=int,
        default=None,
        help="Deprecated compatibility option: if set, overrides infer_hz/cmd_hz.",
    )
    parser.add_argument("--infer_hz", type=int, default=50)
    parser.add_argument("--cmd_hz", type=int, default=100)
    parser.add_argument("--state_timeout_s", type=float, default=8.0)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument(
        "--set_lowlevel_gait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Call high-level set_gait(GAIT_LOWLEVL_SDK) before low-level control.",
    )

    parser.add_argument(
        "--arm_control_indices",
        type=str,
        default="0,1,2,3,4,7,8,9,10,11",
        help="10 controlled arm joints in SDK arm-state order.",
    )

    parser.add_argument("--leg_operation_mode", type=int, default=4)
    parser.add_argument("--waist_operation_mode", type=int, default=4)
    parser.add_argument("--arm_operation_mode", type=int, default=4)
    parser.add_argument("--arm_hold_operation_mode", type=int, default=200)
    parser.add_argument("--max_joint_delta", type=float, default=0.25)

    parser.add_argument("--ang_vel_scale", type=float, default=0.25)
    parser.add_argument("--gravity_scale", type=float, default=1.0)
    parser.add_argument("--dof_pos_scale", type=float, default=1.0)
    parser.add_argument("--dof_vel_scale", type=float, default=0.05)
    parser.add_argument("--action_scale", type=float, default=0.4)
    parser.add_argument("--action_rescale_obs", type=float, default=0.5)

    parser.add_argument(
        "--kps",
        type=float,
        nargs="+",
        default=[
            150.0, 150.0, 150.0, 200.0, 40.0, 40.0,
            150.0, 150.0, 150.0, 200.0, 40.0, 40.0,
            100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
        ],
    )
    parser.add_argument(
        "--kds",
        type=float,
        nargs="+",
        default=[
            4.0, 4.0, 4.0, 6.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 6.0, 2.0, 2.0,
            4.0,
            4.0, 4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0, 4.0,
        ],
    )
    args = parser.parse_args()

    if args.control_hz is not None:
        args.infer_hz = args.control_hz
        args.cmd_hz = args.control_hz

    if args.infer_hz <= 0 or args.cmd_hz <= 0:
        raise ValueError("infer_hz and cmd_hz must be positive.")

    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    deployer = HostPolicyDeployer(args)
    try:
        deployer.setup_robot()
        deployer.run()
        return 0
    except Exception as exc:  # noqa: BLE001
        logging.error("Deployment failed: %s", exc)
        return 1
    finally:
        deployer.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
