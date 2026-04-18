import os
import time
from collections import OrderedDict

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

np.set_printoptions(precision=3, suppress=True)


def get_gravity_orientation(quaternion):
    qw, qx, qy, qz = quaternion
    gravity_orientation = np.zeros(3, dtype=np.float32)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)
    return gravity_orientation


def pd_control(target_q, q, kp, dq, kd):
    return (target_q - q) * kp - dq * kd


def apply_velocity_push(d, magnitude, direction, random_yaw=True):
    direction = np.array(direction, dtype=np.float32)
    if np.linalg.norm(direction) < 1e-8:
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if random_yaw:
        yaw = np.random.uniform(-np.pi, np.pi)
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        direction[:2] = rot @ direction[:2]
    direction = direction / np.linalg.norm(direction)
    # MuJoCo free joint qvel[:3] is root linear velocity.
    d.qvel[0:3] += magnitude * direction


def reset_to_init_state(m, d, qpos_addr, init_joint_angles, init_target_angles, init_base_pos, init_base_quat_xyzw):
    d.qpos[qpos_addr] = init_joint_angles
    if init_base_pos is not None:
        d.qpos[0:3] = np.array(init_base_pos, dtype=np.float32)
    if init_base_quat_xyzw is not None:
        quat_xyzw = np.array(init_base_quat_xyzw, dtype=np.float32)
        q_norm = np.linalg.norm(quat_xyzw)
        if q_norm > 1e-8:
            quat_xyzw = quat_xyzw / q_norm
            d.qpos[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    d.qvel[:] = 0.0
    # Refresh derived states after direct qpos/qvel assignment.
    mujoco.mj_forward(m, d)
    return init_target_angles.copy()


def update_hist(hist_buf, cur, dim):
    cur = cur.to(hist_buf.device)
    return torch.cat([hist_buf[:, dim:], cur], dim=-1)


def get_activation_module(act_name: str):
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


def build_actor_from_state_dict(actor_state_dict, activation="elu"):
    linear_indices = sorted(
        int(k.split(".")[0]) for k in actor_state_dict.keys() if k.endswith(".weight")
    )
    if not linear_indices:
        raise ValueError("No linear layers found in actor state_dict.")

    modules = OrderedDict()
    for i, idx in enumerate(linear_indices):
        out_dim, in_dim = actor_state_dict[f"{idx}.weight"].shape
        modules[str(idx)] = torch.nn.Linear(in_dim, out_dim)
        modules[str(idx + 1)] = (
            get_activation_module(activation) if i < len(linear_indices) - 1 else torch.nn.Tanh()
        )
    actor = torch.nn.Sequential(modules)
    actor.load_state_dict(actor_state_dict, strict=False)
    return actor


def load_policy(policy_path, device, obs_dim, action_dim, policy_format="auto", actor_activation="elu"):
    errors = []

    if policy_format in ("auto", "jit"):
        try:
            policy = torch.jit.load(policy_path, map_location=device)
            policy.eval()
            test_out = policy(torch.zeros(1, obs_dim, dtype=torch.float32, device=device))
            if test_out.shape[-1] != action_dim:
                raise ValueError(f"JIT output={test_out.shape[-1]}, expected={action_dim}")
            return policy
        except Exception as e:
            errors.append(f"jit={e}")
            if policy_format == "jit":
                raise

    if policy_format in ("auto", "checkpoint"):
        try:
            checkpoint = torch.load(policy_path, map_location=device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            actor_state_dict = {
                k[len("actor."):]: v for k, v in state_dict.items() if k.startswith("actor.")
            }
            if not actor_state_dict:
                raise ValueError("No actor.* keys found in checkpoint.")
            in_dim = actor_state_dict["0.weight"].shape[1]
            out_idx = max(int(k.split(".")[0]) for k in actor_state_dict.keys() if k.endswith(".weight"))
            out_dim = actor_state_dict[f"{out_idx}.weight"].shape[0]
            if in_dim != obs_dim:
                raise ValueError(f"checkpoint input={in_dim}, expected={obs_dim}")
            if out_dim != action_dim:
                raise ValueError(f"checkpoint output={out_dim}, expected={action_dim}")
            actor = build_actor_from_state_dict(actor_state_dict, activation=actor_activation)
            actor.to(device).eval()
            return actor
        except Exception as e:
            errors.append(f"ckpt={e}")
            if policy_format == "checkpoint":
                raise

    raise RuntimeError(f"Failed to load policy, format={policy_format}, details={' | '.join(errors)}")


def build_control_mapping(model, controlled_actuator_indices):
    """Build qpos/qvel address map for selected actuators.

    NOTE: 23d policy often needs to skip one actuator in full xml (e.g. head joint).
    Use controlled_actuator_indices in yaml to map action output -> actuator index.
    """
    nu = model.nu
    if controlled_actuator_indices is None:
        controlled_actuator_indices = list(range(nu))
    controlled_actuator_indices = np.array(controlled_actuator_indices, dtype=np.int32)
    if controlled_actuator_indices.min() < 0 or controlled_actuator_indices.max() >= nu:
        raise ValueError(f"controlled_actuator_indices out of range [0, {nu - 1}]")

    joint_ids = []
    qpos_addr = []
    qvel_addr = []
    for act_id in controlled_actuator_indices:
        joint_id = model.actuator_trnid[act_id, 0]
        joint_ids.append(joint_id)
        qpos_addr.append(model.jnt_qposadr[joint_id])
        qvel_addr.append(model.jnt_dofadr[joint_id])
    return (
        controlled_actuator_indices,
        np.array(joint_ids, dtype=np.int32),
        np.array(qpos_addr, dtype=np.int32),
        np.array(qvel_addr, dtype=np.int32),
    )


def resolve_path(path_value, config_file):
    if os.path.isabs(path_value):
        if os.path.exists(path_value):
            return os.path.normpath(path_value)
        raise FileNotFoundError(f"Absolute path not found: {path_value}")

    config_dir = os.path.abspath(os.path.dirname(config_file))
    candidates = [
        os.path.abspath(path_value),  # from current working directory
        os.path.join(config_dir, path_value),  # from config directory
        os.path.join(config_dir, "..", path_value),  # from deploy_mujoco/
        os.path.join(config_dir, "..", "..", path_value),  # from project root
    ]
    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        f"Path not found: '{path_value}'. Tried: " + " | ".join(os.path.normpath(p) for p in candidates)
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        "--config",
        dest="config_file",
        default="deploy_mujoco/configs/z1_23dof_host.yaml",
        type=str,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without mujoco viewer (for servers without DISPLAY).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="Only used in --headless mode. 0 means run forever.",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = cfg["policy_path"]
    xml_path = cfg["xml_path"]
    policy_format = cfg.get("policy_format", "auto")
    actor_activation = cfg.get("actor_activation", "elu")
    obs_mode = cfg.get("obs_mode", "host_ground_v1")
    control_mode = cfg.get("control_mode", "relative_current_pos")
    controlled_actuator_indices = cfg.get("controlled_actuator_indices", None)

    init_mode = cfg.get("init_mode", None)
    init_presets = cfg.get("init_presets", {})
    selected_init = {}
    if init_mode is not None:
        if init_mode not in init_presets:
            raise ValueError(f"init_mode='{init_mode}' not found in init_presets.")
        selected_init = init_presets[init_mode] or {}

    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    kps = np.array(cfg["kps"], dtype=np.float32)
    kds = np.array(cfg["kds"], dtype=np.float32)
    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    init_joint_angles = np.array(
        selected_init.get("init_joint_angles", cfg.get("init_joint_angles", default_angles)),
        dtype=np.float32,
    )
    init_target_angles = np.array(
        selected_init.get("init_target_angles", cfg.get("init_target_angles", init_joint_angles)),
        dtype=np.float32,
    )
    init_base_pos = selected_init.get("init_base_pos", cfg.get("init_base_pos", None))
    init_base_quat_xyzw = selected_init.get("init_base_quat_xyzw", cfg.get("init_base_quat_xyzw", None))
    unactuated_policy_steps = int(
        selected_init.get("unactuated_policy_steps", cfg.get("unactuated_policy_steps", 0))
    )
    ang_vel_scale = float(cfg["ang_vel_scale"])
    dof_pos_scale = float(cfg["dof_pos_scale"])
    dof_vel_scale = float(cfg["dof_vel_scale"])
    action_scale = float(cfg["action_scale"])
    torque_limits = np.array(cfg.get("torque_limits", []), dtype=np.float32)
    gravity_scale = float(cfg.get("gravity_scale", 1.0))
    action_rescale_obs = float(cfg.get("action_rescale_obs", 1.0))
    num_actions = int(cfg["num_actions"])
    num_obs = int(cfg["num_obs"])
    his_obs = int(cfg["his_obs"])
    enable_periodic_push = bool(cfg.get("enable_periodic_push", False))
    push_interval_s = float(cfg.get("push_interval_s", 10.0))
    push_start_after_s = float(cfg.get("push_start_after_s", 2.0))
    push_velocity_magnitude = float(cfg.get("push_velocity_magnitude", 1.8))
    push_direction = np.array(cfg.get("push_direction", [1.0, 0.0, 0.0]), dtype=np.float32)
    push_random_yaw = bool(cfg.get("push_random_yaw", True))
    push_time_basis = cfg.get("push_time_basis", "wall_time")
    enable_periodic_reset = bool(cfg.get("enable_periodic_reset", False))
    reset_interval_s = float(cfg.get("reset_interval_s", 20.0))
    reset_start_after_s = float(cfg.get("reset_start_after_s", 12.0))
    reset_time_basis = cfg.get("reset_time_basis", "wall_time")
    root_lin_vel_clip = float(cfg.get("root_lin_vel_clip", 3.0))
    root_ang_vel_clip = float(cfg.get("root_ang_vel_clip", 10.0))
    enable_safety_reset = bool(cfg.get("enable_safety_reset", True))
    safety_max_base_height = float(cfg.get("safety_max_base_height", 1.8))
    safety_max_base_xy = float(cfg.get("safety_max_base_xy", 3.0))

    if obs_mode != "host_ground_v1":
        raise ValueError("This new 23dof file only supports obs_mode=host_ground_v1.")
    if control_mode not in ("relative_current_pos", "default_angles"):
        raise ValueError("control_mode must be relative_current_pos or default_angles.")

    xml_path = resolve_path(xml_path, args.config_file)
    policy_path = resolve_path(policy_path, args.config_file)

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    act_idx, joint_ids, qpos_addr, qvel_addr = build_control_mapping(m, controlled_actuator_indices)
    if len(act_idx) != num_actions:
        raise ValueError(f"controlled actuators={len(act_idx)} != num_actions={num_actions}")
    joint_ranges = m.jnt_range[joint_ids]
    target_lower = joint_ranges[:, 0].astype(np.float32)
    target_upper = joint_ranges[:, 1].astype(np.float32)
    if (
        len(default_angles) != num_actions
        or len(init_joint_angles) != num_actions
        or len(init_target_angles) != num_actions
        or len(kps) != num_actions
        or len(kds) != num_actions
    ):
        raise ValueError("default/init angles and kps/kds length must equal num_actions.")
    if torque_limits.size == 0:
        raise ValueError(
            "Missing torque_limits in config. host_z1 training clips torques by effort limits; "
            "without clipping MuJoCo can become unstable."
        )
    if len(torque_limits) != num_actions:
        raise ValueError("torque_limits length must equal num_actions.")

    # Set initial posture for controlled joints only.
    d.qpos[qpos_addr] = init_joint_angles
    if init_base_pos is not None:
        if len(init_base_pos) != 3:
            raise ValueError("init_base_pos must have length 3.")
        d.qpos[0:3] = np.array(init_base_pos, dtype=np.float32)
    if init_base_quat_xyzw is not None:
        if len(init_base_quat_xyzw) != 4:
            raise ValueError("init_base_quat_xyzw must have length 4.")
        quat_xyzw = np.array(init_base_quat_xyzw, dtype=np.float32)
        q_norm = np.linalg.norm(quat_xyzw)
        if q_norm < 1e-8:
            raise ValueError("init_base_quat_xyzw norm is too small.")
        quat_xyzw = quat_xyzw / q_norm
        # MuJoCo uses wxyz order for free-joint quaternion.
        d.qpos[3:7] = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)

    expected_one_step_obs = 3 + 3 + num_actions + num_actions + num_actions + 1
    if num_obs != expected_one_step_obs:
        raise ValueError(f"num_obs={num_obs}, expected={expected_one_step_obs} for host_ground_v1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, model_nu={m.nu}, controlled={len(act_idx)}")
    if init_mode is not None:
        print(f"Using init_mode: {init_mode}")
    if enable_periodic_push:
        print(
            f"Periodic push enabled ({push_time_basis}): interval={push_interval_s}s, "
            f"start_after={push_start_after_s}s, magnitude={push_velocity_magnitude}"
        )
    if enable_periodic_reset:
        print(
            f"Periodic reset enabled ({reset_time_basis}): interval={reset_interval_s}s, "
            f"start_after={reset_start_after_s}s"
        )
    if enable_safety_reset:
        print(
            f"Safety reset enabled: max_base_height={safety_max_base_height}, "
            f"max_base_xy={safety_max_base_xy}, v_clip=({root_lin_vel_clip}, {root_ang_vel_clip})"
        )

    policy = load_policy(
        policy_path=policy_path,
        device=device,
        obs_dim=num_obs * his_obs,
        action_dim=num_actions,
        policy_format=policy_format,
        actor_activation=actor_activation,
    )
    print(f"Loaded policy: {policy_path}")

    history_obs = torch.zeros(1, num_obs * his_obs, dtype=torch.float32, device=device)
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = init_target_angles.copy()

    counter = 0
    policy_counter = 0
    now_wall = time.time()
    next_push_wall_time = now_wall + push_start_after_s
    next_push_sim_time = push_start_after_s
    next_reset_wall_time = now_wall + reset_start_after_s
    next_reset_sim_time = reset_start_after_s

    def step_once(
        action,
        target_dof_pos,
        history_obs,
        counter,
        policy_counter,
        next_push_wall_time,
        next_push_sim_time,
        next_reset_wall_time,
        next_reset_sim_time,
    ):
        step_start = time.time()

        qj = d.qpos[qpos_addr]
        dqj = d.qvel[qvel_addr]
        tau = pd_control(target_dof_pos, qj, kps, dqj, kds)
        tau = np.clip(tau, -torque_limits, torque_limits)
        d.ctrl[:] = 0.0
        d.ctrl[act_idx] = tau
        mujoco.mj_step(m, d)
        # Safety: cap root velocities to avoid runaway energy blow-up.
        d.qvel[0:3] = np.clip(d.qvel[0:3], -root_lin_vel_clip, root_lin_vel_clip)
        d.qvel[3:6] = np.clip(d.qvel[3:6], -root_ang_vel_clip, root_ang_vel_clip)

        do_push = False
        now = time.time()
        if enable_periodic_push:
            if push_time_basis == "wall_time" and now >= next_push_wall_time:
                do_push = True
                next_push_wall_time += push_interval_s
            elif push_time_basis == "sim_time" and d.time >= next_push_sim_time:
                do_push = True
                next_push_sim_time += push_interval_s
        if do_push:
            apply_velocity_push(
                d=d,
                magnitude=push_velocity_magnitude,
                direction=push_direction,
                random_yaw=push_random_yaw,
            )
            print(f"[push] sim_t={d.time:.3f}s")

        counter = counter + 1
        if counter % control_decimation == 0:
            step_start = time.time()
            policy_counter = policy_counter + 1
            qj = d.qpos[qpos_addr]
            dqj = d.qvel[qvel_addr]
            quat = d.qpos[3:7]
            omega = d.qvel[3:6]
            gravity_orientation = get_gravity_orientation(quat)

            current_obs = np.concatenate(
                [
                    omega * ang_vel_scale,
                    gravity_orientation * gravity_scale,
                    qj * dof_pos_scale,
                    dqj * dof_vel_scale,
                    action,
                    np.array([action_rescale_obs], dtype=np.float32),
                ]
            ).astype(np.float32)
            if policy_counter <= unactuated_policy_steps:
                current_obs *= 0.0
            current_obs_t = torch.from_numpy(current_obs).unsqueeze(0).to(device=device, dtype=torch.float32)
            history_obs_new = update_hist(history_obs, current_obs_t, num_obs)

            if policy_counter <= unactuated_policy_steps:
                action_new = np.zeros(num_actions, dtype=np.float32)
            else:
                action_new = policy(history_obs_new).detach().cpu().numpy().squeeze()
            actions_scaled = action_new * action_scale
            if control_mode == "relative_current_pos":
                target_dof_pos_new = qj + actions_scaled
            else:
                target_dof_pos_new = default_angles + actions_scaled
            target_dof_pos_new = np.clip(target_dof_pos_new, target_lower, target_upper)
        else:
            history_obs_new = history_obs
            action_new = action
            target_dof_pos_new = target_dof_pos

        remain = m.opt.timestep - (time.time() - step_start)
        if remain > 0:
            time.sleep(remain)

        do_reset = False
        now_after = time.time()
        if enable_periodic_reset:
            if reset_time_basis == "wall_time" and now_after >= next_reset_wall_time:
                do_reset = True
                next_reset_wall_time += reset_interval_s
            elif reset_time_basis == "sim_time" and d.time >= next_reset_sim_time:
                do_reset = True
                next_reset_sim_time += reset_interval_s
        if do_reset:
            target_dof_pos_new = reset_to_init_state(
                m=m,
                d=d,
                qpos_addr=qpos_addr,
                init_joint_angles=init_joint_angles,
                init_target_angles=init_target_angles,
                init_base_pos=init_base_pos,
                init_base_quat_xyzw=init_base_quat_xyzw,
            )
            action_new = np.zeros(num_actions, dtype=np.float32)
            history_obs_new = torch.zeros(1, num_obs * his_obs, dtype=torch.float32, device=device)
            policy_counter = 0
            print(f"[reset] sim_t={d.time:.3f}s -> init_mode={init_mode}")

        if enable_safety_reset:
            base_xy = float(np.linalg.norm(d.qpos[0:2]))
            base_height = float(d.qpos[2])
            out_of_bounds = (
                (not np.isfinite(d.qpos).all())
                or (not np.isfinite(d.qvel).all())
                or (base_height > safety_max_base_height)
                or (base_xy > safety_max_base_xy)
            )
            if out_of_bounds:
                target_dof_pos_new = reset_to_init_state(
                    m=m,
                    d=d,
                    qpos_addr=qpos_addr,
                    init_joint_angles=init_joint_angles,
                    init_target_angles=init_target_angles,
                    init_base_pos=init_base_pos,
                    init_base_quat_xyzw=init_base_quat_xyzw,
                )
                action_new = np.zeros(num_actions, dtype=np.float32)
                history_obs_new = torch.zeros(1, num_obs * his_obs, dtype=torch.float32, device=device)
                policy_counter = 0
                print(
                    f"[safety_reset] sim_t={d.time:.3f}s, base_height={base_height:.3f}, base_xy={base_xy:.3f}"
                )

        return (
            action_new,
            target_dof_pos_new,
            history_obs_new,
            counter,
            policy_counter,
            next_push_wall_time,
            next_push_sim_time,
            next_reset_wall_time,
            next_reset_sim_time,
        )

    if args.headless:
        print("Running in headless mode (no viewer).")
        while True:
            (
                action,
                target_dof_pos,
                history_obs,
                counter,
                policy_counter,
                next_push_wall_time,
                next_push_sim_time,
                next_reset_wall_time,
                next_reset_sim_time,
            ) = step_once(
                action,
                target_dof_pos,
                history_obs,
                counter,
                policy_counter,
                next_push_wall_time,
                next_push_sim_time,
                next_reset_wall_time,
                next_reset_sim_time,
            )
            if args.max_steps > 0 and counter >= args.max_steps:
                break
            if counter % 500 == 0:
                print(f"step={counter}, sim_time={d.time:.3f}s")
        print("Headless run finished.")
    else:
        with mujoco.viewer.launch_passive(m, d) as viewer:
            while viewer.is_running():
                (
                    action,
                    target_dof_pos,
                    history_obs,
                    counter,
                    policy_counter,
                    next_push_wall_time,
                    next_push_sim_time,
                    next_reset_wall_time,
                    next_reset_sim_time,
                ) = step_once(
                    action,
                    target_dof_pos,
                    history_obs,
                    counter,
                    policy_counter,
                    next_push_wall_time,
                    next_push_sim_time,
                    next_reset_wall_time,
                    next_reset_sim_time,
                )
                viewer.sync()
