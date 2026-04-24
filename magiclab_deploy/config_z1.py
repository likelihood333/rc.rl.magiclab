from pathlib import Path
from typing import Any
import numpy as np
import yaml


class Config:
    def __init__(self, file_path: str) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict):
            raise ValueError(f"Config file must be a mapping: {file_path}")

        self.config: dict[str, Any] = config

        self.policy_path = self.config.get("policy_path")
        if isinstance(self.policy_path, str) and self.policy_path:
            policy_root = Path(file_path).resolve().parent / self.config.get("policy_root", "..")
            self.policy_path = str((policy_root / Path(self.policy_path).expanduser()).resolve())
        # self.odom_path = self.config.get("odom_path")
        self.local_ip = self.config.get("local_ip", "192.168.54.111")
        self.policy_format = self.config.get("policy_format", "auto")
        self.actor_activation = self.config.get("actor_activation", "elu")

        # 12DOF timing/control
        self.simulation_duration = self.config.get("simulation_duration")
        self.simulation_dt = self.config.get("simulation_dt")
        self.control_dt = self.simulation_dt
        self.control_decimation = self.config.get("control_decimation")

        # 12DOF leg/arm/waist/head parameters (optional for 23DOF configs)
        self.leg_mode = self._to_array("leg_mode", np.int32)
        self.leg_kp = self._to_array("kps", np.float32)
        self.leg_kd = self._to_array("kds", np.float32)
        self.rec_leg_kp = self._to_array("rec_leg_kp", np.float32)
        self.rec_leg_kd = self._to_array("rec_leg_kd", np.float32)
        self.default_leg_angles = self._to_array("default_angles", np.float32)

        self.arm_mode = self._to_array("upper_arm_mode", np.int32)
        self.arm_kp = self._to_array("upper_arm_kp", np.float32)
        self.arm_kd = self._to_array("upper_arm_kd", np.float32)
        self.default_arm_angles = self._to_array("default_arm_angles", np.float32)

        self.waist_mode = self._to_array("upper_waist_mode", np.int32)
        self.waist_kp = self._to_array("upper_waist_kp", np.float32)
        self.waist_kd = self._to_array("upper_waist_kd", np.float32)
        self.default_waist_angles = self._to_array("default_waist_angles", np.float32)

        self.head_mode = self._to_array("upper_head_mode", np.int32)
        self.head_kp = self._to_array("upper_head_kp", np.float32)
        self.head_kd = self._to_array("upper_head_kd", np.float32)
        self.default_head_angles = self._to_array("default_head_angles", np.float32)

        # Observation/action scaling
        self.lin_vel_scale = self.config.get("lin_vel_scale")
        self.ang_vel_scale = self.config.get("ang_vel_scale")
        self.rpy_scale = self.config.get("rpy_scale")
        self.dof_pos_scale = self.config.get("dof_pos_scale")
        self.dof_vel_scale = self.config.get("dof_vel_scale")
        self.action_scale = self.config.get("action_scale")
        self.cmd_scale = self._to_array("cmd_scale", np.float32)

        self.num_actions = self.config.get("num_actions")
        self.num_obs = self.config.get("num_obs")
        self.his_obs = self.config.get("his_obs")

        self.cmd_init = self._to_array("cmd_init", np.float32)
        if self.cmd_init is None:
            self.cmd_init = np.zeros(3, dtype=np.float32)
        self.controller_scale = self.config.get("controller_scale")

    def _to_array(self, key: str, dtype: np.dtype) -> np.ndarray | None:
        value = self.config.get(key)
        if value is None:
            return None
        return np.array(value, dtype=dtype)
