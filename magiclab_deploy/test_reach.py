#!/usr/bin/env python3
"""
目的：检查电机是否能够跟随目标指令（最小实现）
方式：读取当前关节角 -> 施加固定增量 -> 持续下发目标并打印误差
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import magicbot_z1_python as magicbot


@dataclass
class StateBuffer:
    lock: threading.Lock = field(default_factory=threading.Lock)
    leg_pos: Optional[np.ndarray] = None
    waist_pos: Optional[np.ndarray] = None
    arm_pos: Optional[np.ndarray] = None

    def update_leg(self, joint_state) -> None:
        with self.lock:
            self.leg_pos = np.array([float(j.posH) for j in joint_state.joints], dtype=np.float32)

    def update_waist(self, joint_state) -> None:
        with self.lock:
            self.waist_pos = np.array([float(j.posH) for j in joint_state.joints], dtype=np.float32)

    def update_arm(self, joint_state) -> None:
        with self.lock:
            self.arm_pos = np.array([float(j.posH) for j in joint_state.joints], dtype=np.float32)

    def ready(self) -> bool:
        with self.lock:
            return self.leg_pos is not None and self.waist_pos is not None and self.arm_pos is not None

    def snapshot(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.lock:
            return self.leg_pos.copy(), self.waist_pos.copy(), self.arm_pos.copy()


def make_joint_command(
    target: np.ndarray,
    kp: np.ndarray,
    kd: np.ndarray,
    mode: int,
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


class ReachTester:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.running = True
        self.state = StateBuffer()
        self.robot: Optional[magicbot.MagicRobot] = None
        self.controller = None
        signal.signal(signal.SIGINT, self._sig_stop)
        signal.signal(signal.SIGTERM, self._sig_stop)

    def _sig_stop(self, signum, _frame) -> None:
        logging.info("Received signal %s, stopping...", signum)
        self.running = False

    def setup(self) -> None:
        self.robot = magicbot.MagicRobot()
        if not self.robot.initialize(self.args.local_ip):
            raise RuntimeError("robot.initialize() failed")
        st = self.robot.connect()
        if st.code != magicbot.ErrorCode.OK:
            raise RuntimeError(f"robot.connect failed: {st.message}")

        high = self.robot.get_high_level_motion_controller()
        if not high.initialize():
            raise RuntimeError("high level controller initialize failed")
        st = high.set_gait(magicbot.GaitMode.GAIT_RECOVERY_STAND, 10000)
        if st.code != magicbot.ErrorCode.OK:
            st = high.set_gait(magicbot.GaitMode.GAIT_PASSIVE, 10000)
            if st.code != magicbot.ErrorCode.OK:
                high.shutdown()
                raise RuntimeError(f"set_gait failed: {st.message}")
        time.sleep(self.args.gait_settle_s)
        high.shutdown()

        st = self.robot.set_motion_control_level(magicbot.ControllerLevel.LowLevel)
        if st.code != magicbot.ErrorCode.OK:
            raise RuntimeError(f"switch low-level failed: {st.message}")

        self.controller = self.robot.get_low_level_motion_controller()
        if hasattr(self.controller, "initialize"):
            try:
                self.controller.initialize()
            except Exception as exc:  # noqa: BLE001
                logging.warning("low-level initialize exception: %s", exc)
        self.controller.subscribe_leg_state(self.state.update_leg)
        self.controller.subscribe_waist_state(self.state.update_waist)
        self.controller.subscribe_arm_state(self.state.update_arm)

        start = time.time()
        while self.running and time.time() - start < self.args.state_timeout_s:
            if self.state.ready():
                logging.info("State callback ready.")
                return
            time.sleep(0.01)
        raise TimeoutError("Timeout waiting leg/waist/arm callbacks")

    def _release_all(self) -> None:
        if self.controller is None:
            return
        try:
            leg_n = int(magicbot.LEG_JOINT_NUM)
            waist_n = int(magicbot.WAIST_JOINT_NUM)
            arm_n = int(magicbot.ARM_JOINT_NUM)

            def _zero_cmd(n: int) -> magicbot.JointCommand:
                cmd = magicbot.JointCommand()
                for _ in range(n):
                    j = magicbot.SingleJointCommand()
                    j.operation_mode = 200
                    j.pos = 0.0
                    j.vel = 0.0
                    j.toq = 0.0
                    j.kp = 0.0
                    j.kd = 0.0
                    cmd.joints.append(j)
                return cmd

            leg_cmd = _zero_cmd(leg_n)
            waist_cmd = _zero_cmd(waist_n)
            arm_cmd = _zero_cmd(arm_n)
            repeat = max(1, int(self.args.release_s * self.args.cmd_hz))
            for _ in range(repeat):
                self.controller.publish_leg_command(leg_cmd)
                self.controller.publish_waist_command(waist_cmd)
                self.controller.publish_arm_command(arm_cmd)
                time.sleep(1.0 / self.args.cmd_hz)
            logging.info("Released all joints with mode=200 for %.2fs.", self.args.release_s)
        except Exception as exc:  # noqa: BLE001
            logging.warning("release failed: %s", exc)

    def run(self) -> None:
        leg_q0, waist_q0, arm_q0 = self.state.snapshot()
        target_leg = leg_q0.copy()
        target_waist = waist_q0.copy()
        target_arm = arm_q0.copy()

        if self.args.group == "leg" and not self.args.all_in_group:
            if self.args.joint_index < 0 or self.args.joint_index >= len(target_leg):
                raise ValueError(f"leg joint_index out of range: {self.args.joint_index}")
        if self.args.group == "waist" and not self.args.all_in_group:
            if self.args.joint_index < 0 or self.args.joint_index >= len(target_waist):
                raise ValueError(f"waist joint_index out of range: {self.args.joint_index}")
        if self.args.group == "arm" and not self.args.all_in_group:
            if self.args.joint_index < 0 or self.args.joint_index >= len(target_arm):
                raise ValueError(f"arm joint_index out of range: {self.args.joint_index}")

        if self.args.group == "leg":
            if self.args.all_in_group:
                target_leg += self.args.delta
            else:
                target_leg[self.args.joint_index] += self.args.delta
        elif self.args.group == "waist":
            target_waist[self.args.joint_index] += self.args.delta
        elif self.args.group == "arm":
            if self.args.all_in_group:
                target_arm += self.args.delta
            else:
                target_arm[self.args.joint_index] += self.args.delta
        else:
            raise ValueError(f"Unsupported group: {self.args.group}")

        logging.info(
            "Test target built: group=%s joint_index=%d all_in_group=%s delta=%.4f",
            self.args.group,
            self.args.joint_index,
            self.args.all_in_group,
            self.args.delta,
        )

        kp_leg = np.full_like(target_leg, self.args.kp_leg, dtype=np.float32)
        kd_leg = np.full_like(target_leg, self.args.kd_leg, dtype=np.float32)
        kp_waist = np.full_like(target_waist, self.args.kp_waist, dtype=np.float32)
        kd_waist = np.full_like(target_waist, self.args.kd_waist, dtype=np.float32)
        kp_arm = np.full_like(target_arm, self.args.kp_arm, dtype=np.float32)
        kd_arm = np.full_like(target_arm, self.args.kd_arm, dtype=np.float32)

        hold_end = time.time() + self.args.hold_s
        next_log = time.time()
        period = 1.0 / self.args.cmd_hz

        while self.running and time.time() < hold_end:
            leg_q, waist_q, arm_q = self.state.snapshot()
            leg_err = target_leg - leg_q
            waist_err = target_waist - waist_q
            arm_err = target_arm - arm_q

            # Only drive selected group. Other groups are released to avoid coupling confusion.
            if self.args.group == "leg":
                leg_cmd = make_joint_command(target_leg, kp_leg, kd_leg, self.args.mode_leg)
                waist_cmd = make_joint_command(
                    waist_q,
                    np.zeros_like(waist_q, dtype=np.float32),
                    np.zeros_like(waist_q, dtype=np.float32),
                    200,
                )
                arm_cmd = make_joint_command(
                    arm_q,
                    np.zeros_like(arm_q, dtype=np.float32),
                    np.zeros_like(arm_q, dtype=np.float32),
                    200,
                )
            elif self.args.group == "waist":
                leg_cmd = make_joint_command(
                    leg_q,
                    np.zeros_like(leg_q, dtype=np.float32),
                    np.zeros_like(leg_q, dtype=np.float32),
                    200,
                )
                waist_cmd = make_joint_command(target_waist, kp_waist, kd_waist, self.args.mode_waist)
                arm_cmd = make_joint_command(
                    arm_q,
                    np.zeros_like(arm_q, dtype=np.float32),
                    np.zeros_like(arm_q, dtype=np.float32),
                    200,
                )
            else:  # arm
                leg_cmd = make_joint_command(
                    leg_q,
                    np.zeros_like(leg_q, dtype=np.float32),
                    np.zeros_like(leg_q, dtype=np.float32),
                    200,
                )
                waist_cmd = make_joint_command(
                    waist_q,
                    np.zeros_like(waist_q, dtype=np.float32),
                    np.zeros_like(waist_q, dtype=np.float32),
                    200,
                )
                arm_cmd = make_joint_command(target_arm, kp_arm, kd_arm, self.args.mode_arm)

            self.controller.publish_leg_command(leg_cmd)
            self.controller.publish_waist_command(waist_cmd)
            self.controller.publish_arm_command(arm_cmd)

            now = time.time()
            if now >= next_log:
                logging.info(
                    "tracking_err(abs mean/max): leg=%.4f/%.4f waist=%.4f/%.4f arm=%.4f/%.4f",
                    float(np.mean(np.abs(leg_err))),
                    float(np.max(np.abs(leg_err))),
                    float(np.mean(np.abs(waist_err))),
                    float(np.max(np.abs(waist_err))),
                    float(np.mean(np.abs(arm_err))),
                    float(np.max(np.abs(arm_err))),
                )
                next_log = now + self.args.log_interval_s

            time.sleep(period)

    def cleanup(self) -> None:
        self._release_all()
        try:
            if self.controller is not None and hasattr(self.controller, "shutdown"):
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
    p = argparse.ArgumentParser(description="Minimal reach/follow test for Z1 joints.")
    p.add_argument("--local_ip", type=str, default="192.168.54.119")
    p.add_argument("--group", type=str, default="leg", choices=["leg", "waist", "arm"])
    p.add_argument("--joint_index", type=int, default=0, help="Index inside the selected group.")
    p.add_argument("--all_in_group", action="store_true", help="Apply delta to all joints in group.")
    p.add_argument("--delta", type=float, default=0.10, help="Target increment in rad.")
    p.add_argument("--hold_s", type=float, default=6.0, help="Command hold duration.")
    p.add_argument("--cmd_hz", type=int, default=50)
    p.add_argument("--state_timeout_s", type=float, default=15.0)
    p.add_argument("--gait_settle_s", type=float, default=2.0)
    p.add_argument("--log_interval_s", type=float, default=0.5)
    p.add_argument("--release_s", type=float, default=0.3)
    p.add_argument("--mode_leg", type=int, default=3)
    p.add_argument("--mode_waist", type=int, default=3)
    p.add_argument("--mode_arm", type=int, default=200)
    p.add_argument("--kp_leg", type=float, default=80.0)
    p.add_argument("--kd_leg", type=float, default=2.5)
    p.add_argument("--kp_waist", type=float, default=60.0)
    p.add_argument("--kd_waist", type=float, default=2.0)
    p.add_argument("--kp_arm", type=float, default=60.0)
    p.add_argument("--kd_arm", type=float, default=2.0)
    args = p.parse_args()
    if args.cmd_hz <= 0:
        raise ValueError("cmd_hz must be positive")
    if args.hold_s <= 0:
        raise ValueError("hold_s must be positive")
    if args.log_interval_s <= 0:
        raise ValueError("log_interval_s must be positive")
    return args


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    tester: Optional[ReachTester] = None
    try:
        tester = ReachTester(args)
        tester.setup()
        tester.run()
        return 0
    except KeyboardInterrupt:
        logging.warning("Interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.error("test_reach failed: %s", exc)
        return 1
    finally:
        if tester is not None:
            tester.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

