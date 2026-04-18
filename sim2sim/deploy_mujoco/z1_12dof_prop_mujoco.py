# import os, sys
# print("PYTHONPATH =", os.environ.get("PYTHONPATH"))
# print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
# print("PATH =", os.environ.get("PATH"))
# print("sys.path =", sys.path)

import time
import os
import pdb
from collections import deque
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys
from scipy.spatial.transform import Rotation as R
import gamepad_reader_btp
np.set_printoptions(precision=2, suppress=True)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, dq, kd):
    """Calculates torques from position commands"""
    tau = (target_q - q) * kp - dq * kd
    return tau


def apply_instantaneous_velocity(m, d, body_name, velocity_change, point=None):
    """
    Apply an instantaneous velocity change to a specific body
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        body_name: Name of the body to apply velocity change to
        velocity_change: 3D velocity change vector [vx, vy, vz]
        point: Point of application in body coordinates (optional, defaults to body center)
    """
    try:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            print(f"Warning: Body '{body_name}' not found")
            return False

        # Get the body's position and orientation
        body_pos = d.xpos[body_id]
        body_quat = d.xquat[body_id]

        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]]).as_matrix()

        # Transform velocity change to world coordinates
        world_velocity_change = rot_matrix @ velocity_change

        # Apply velocity change to the body's center of mass
        # For root body (body_id = 0), we apply to the first 3 DOFs (linear motion)
        if body_id == 0:
            d.qvel[0:3] += world_velocity_change
        else:
            # For other bodies, we need to compute the equivalent velocity change
            # This is a simplified approach - for more complex cases, you might need
            # to compute the Jacobian and apply the appropriate generalized velocities
            print(f"Warning: Velocity change to non-root body '{body_name}' not fully implemented")
            return False

        return True
    except Exception as e:
        print(f"Error applying velocity change to body '{body_name}': {e}")
        return False


def apply_instantaneous_velocity_to_root(m, d, velocity_change, point=None):
    """
    Apply an instantaneous velocity change to the root body (body 0)
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        velocity_change: 3D velocity change vector [vx, vy, vz]
        point: Point of application in body coordinates (optional)
    """
    try:
        # Apply velocity change directly to the root body's linear DOFs
        # The first 3 elements of qvel correspond to linear motion of the root body
        d.qvel[0:3] += velocity_change
        return True
    except Exception as e:
        print(f"Error applying velocity change to root body: {e}")
        return False


def create_periodic_velocity_impulse(simulation_time, velocity_magnitude=2.0, velocity_direction=[1, 0, 0],
                                     period=2.0, duration=0.0002):
    """
    Create a periodic velocity impulse that activates every 'period' seconds for 'duration' seconds
    
    Args:
        simulation_time: Current simulation time
        velocity_magnitude: Magnitude of the velocity change
        velocity_direction: Direction of the velocity change (will be normalized)
        period: How often the velocity impulse is applied (seconds)
        duration: How long each velocity impulse lasts (seconds)
    
    Returns:
        velocity_vector: 3D velocity change vector or None if no velocity should be applied
    """
    # Normalize velocity direction
    velocity_direction = np.array(velocity_direction, dtype=np.float32)
    velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)

    # Calculate if we're in an active period
    time_in_period = simulation_time % period
    if time_in_period < duration:
        return velocity_magnitude * velocity_direction
    else:
        return None
    
# --- update history ---
def update_hist(hist_buf, cur, dim):
    cur = cur.to(hist_buf.device)
    res = torch.cat([hist_buf[:, dim:], cur], dim=-1)
    return res


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    gamepad = gamepad_reader_btp.Gamepad()
    command_function = gamepad.get_command

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/p2_prop.yaml",
                        help="config file name in the config folder")
    parser.add_argument("--enable_push_forces", default="false", action="store_true",
                        help="Enable periodic velocity impulses")
    parser.add_argument("--push_force_magnitude", type=float, default=1.0,
                        help="Magnitude of velocity impulses (m/s)")
    parser.add_argument("--push_force_period", type=float, default=2.0,
                        help="Period between velocity impulses (seconds)")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        # odom_path = config["odom_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        rpy_scale = config["rpy_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        # num_odom_output = config["num_odom_output"]
        num_obs = config["num_obs"]
        # num_odom_obs = config["num_odom_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # Set gait cycle time (default 0.8s, can be added to config if needed)
        cycle_time = 0.6
        his_obs = config["his_obs"]
        # his_odom_obs = config["his_odom_obs"]
        controller_scale = config["controller_scale"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    # odom_output = np.zeros(num_odom_output, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    # odom_obs = np.zeros(num_odom_obs, dtype=np.float32)  # signal_obs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    history_obs = torch.zeros(1, num_obs*his_obs, dtype=torch.float32, device=device)
    history = {
        "ang_vel": torch.zeros((1, 3 * his_obs), device=device),
        "gravity": torch.zeros((1, 3 * his_obs), device=device),
        "cmd": torch.zeros((1, 3 * his_obs), device=device),
        "qj": torch.zeros((1, num_actions * his_obs), device=device),
        "dqj": torch.zeros((1, num_actions * his_obs), device=device),
        "actions": torch.zeros((1, num_actions * his_obs), device=device),
        "gait": torch.zeros((1, 2 * his_obs), device=device),
    }

    counter = 0
    common_iter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # 设置初始角度
    d.qpos[7:7+len(default_angles)] = default_angles
    d.qvel[:] = 0.0
    print(f"Initial angles: {d.qpos[7:]}")

    m.opt.timestep = simulation_dt

    # load policy
    # Check if the policy path exists before loading the model
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    policy = torch.jit.load(policy_path)
    policy.to(device)  # 将policy模型移到GPU
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        last_log_time = start  # Track time for 2-second intervals
        log_interval = 2.0  # Log every 2 seconds

        while viewer.is_running():
            step_start = time.time()

            # Apply PD control to 12 actuated joints
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # Read command from gamepad
            if True:
                lin_speed, ang_speed, e_stop, _ = command_function()
                # lin_speed = [0.0,0,0]
                if e_stop:
                    gamepad.read_thread.join()
                    sys.exit(0)
                if (lin_speed[0]<=0):
                    cmd[0] = -1. * lin_speed[0] * (-0.5) * controller_scale[0]
                else:
                    cmd[0] = lin_speed[0] * 0.5 * controller_scale[0]

                if (lin_speed[1]<=0):
                    cmd[1] = -1. * lin_speed[1] * (-0.5) * controller_scale[1]
                else:
                    cmd[1] = lin_speed[1] * 0.5 * controller_scale[1]
                
                if (ang_speed<=0):
                    cmd[2] = -1. * ang_speed * (-1.0) * controller_scale[2]
                else:
                    cmd[2] = ang_speed * (1.0) * controller_scale[2]

            # cmd= [0.0 ,0, 0]
            print(f"cmd: {cmd}")

            counter += 1
            common_iter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]  # dof_pos (12 actuated joints)
                dqj = d.qvel[6:]  # dof_vel (12 actuated joints)
                quat = d.qpos[3:7]  # orientation
                omega = d.qvel[3:6]  # ang_vel
                gravity_orientation = get_gravity_orientation(quat)

                phase = (counter * simulation_dt) / cycle_time
                sin_phase = np.sin(2 * np.pi * phase)
                stance_mask = np.zeros(2, dtype=np.float32)
                stance_mask[0] = sin_phase >= 0
                # right foot stance
                stance_mask[1] = sin_phase < 0
                cmd_l2 = np.linalg.norm(cmd)
                if (cmd_l2 < 0.01):
                    stance_mask[0] = 1
                    stance_mask[1] = 1
                    

                cur_ang_vel = torch.from_numpy(omega * ang_vel_scale).float().unsqueeze(0)
                cur_gravity = torch.from_numpy(gravity_orientation * rpy_scale).float().unsqueeze(0)
                cur_cmd = torch.from_numpy(cmd * cmd_scale).float().unsqueeze(0)

                cur_qj = torch.from_numpy((qj - default_angles) * dof_pos_scale).float().unsqueeze(0)
                cur_dqj = torch.from_numpy(dqj * dof_vel_scale).float().unsqueeze(0)
                cur_action = torch.from_numpy(action).float().unsqueeze(0)
                cur_gait = torch.from_numpy(stance_mask).float().unsqueeze(0)
            

                history["ang_vel"] = update_hist(history["ang_vel"], cur_ang_vel, 3)
                history["gravity"] = update_hist(history["gravity"], cur_gravity, 3)
                history["cmd"] = update_hist(history["cmd"], cur_cmd, 3)
                history["qj"] = update_hist(history["qj"], cur_qj, num_actions)
                history["dqj"] = update_hist(history["dqj"], cur_dqj, num_actions)
                history["actions"] = update_hist(history["actions"], cur_action, num_actions)
                history["gait"] = update_hist(history["gait"], cur_gait, 2)


                history_obs = torch.cat([
                    history["ang_vel"],
                    history["gravity"],
                    history["cmd"],
                    history["qj"],
                    history["dqj"],
                    history["actions"],
                    history["gait"],
                ], dim=-1)

                # print("\n===== His Input (obs) =====")
                # print(f"{history_obs.cpu().detach().numpy().squeeze()}")

                # Policy inference on GPU
                action_tensor = policy(history_obs)
                action = action_tensor.cpu().detach().numpy().squeeze() # 将policy模型移回cpu，并转换为numpy
                # print("\n===== Output =====")
                # print(f"{action}")
                
                # print("#############################")
                # print("\n")

                # transform action to target_dof_pos
                actions_scaled = action * action_scale
                target_dof_pos = actions_scaled + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
# import os, sys
# print("PYTHONPATH =", os.environ.get("PYTHONPATH"))
# print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))
# print("PATH =", os.environ.get("PATH"))
# print("sys.path =", sys.path)

import time
import os
import pdb
from collections import deque
import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import sys
from scipy.spatial.transform import Rotation as R
import gamepad_reader_btp
np.set_printoptions(precision=2, suppress=True)


def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, dq, kd):
    """Calculates torques from position commands"""
    tau = (target_q - q) * kp - dq * kd
    return tau


def apply_instantaneous_velocity(m, d, body_name, velocity_change, point=None):
    """
    Apply an instantaneous velocity change to a specific body
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        body_name: Name of the body to apply velocity change to
        velocity_change: 3D velocity change vector [vx, vy, vz]
        point: Point of application in body coordinates (optional, defaults to body center)
    """
    try:
        body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            print(f"Warning: Body '{body_name}' not found")
            return False

        # Get the body's position and orientation
        body_pos = d.xpos[body_id]
        body_quat = d.xquat[body_id]

        # Convert quaternion to rotation matrix
        rot_matrix = R.from_quat([body_quat[1], body_quat[2], body_quat[3], body_quat[0]]).as_matrix()

        # Transform velocity change to world coordinates
        world_velocity_change = rot_matrix @ velocity_change

        # Apply velocity change to the body's center of mass
        # For root body (body_id = 0), we apply to the first 3 DOFs (linear motion)
        if body_id == 0:
            d.qvel[0:3] += world_velocity_change
        else:
            # For other bodies, we need to compute the equivalent velocity change
            # This is a simplified approach - for more complex cases, you might need
            # to compute the Jacobian and apply the appropriate generalized velocities
            print(f"Warning: Velocity change to non-root body '{body_name}' not fully implemented")
            return False

        return True
    except Exception as e:
        print(f"Error applying velocity change to body '{body_name}': {e}")
        return False


def apply_instantaneous_velocity_to_root(m, d, velocity_change, point=None):
    """
    Apply an instantaneous velocity change to the root body (body 0)
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        velocity_change: 3D velocity change vector [vx, vy, vz]
        point: Point of application in body coordinates (optional)
    """
    try:
        # Apply velocity change directly to the root body's linear DOFs
        # The first 3 elements of qvel correspond to linear motion of the root body
        d.qvel[0:3] += velocity_change
        return True
    except Exception as e:
        print(f"Error applying velocity change to root body: {e}")
        return False


def create_periodic_velocity_impulse(simulation_time, velocity_magnitude=2.0, velocity_direction=[1, 0, 0],
                                     period=2.0, duration=0.0002):
    """
    Create a periodic velocity impulse that activates every 'period' seconds for 'duration' seconds
    
    Args:
        simulation_time: Current simulation time
        velocity_magnitude: Magnitude of the velocity change
        velocity_direction: Direction of the velocity change (will be normalized)
        period: How often the velocity impulse is applied (seconds)
        duration: How long each velocity impulse lasts (seconds)
    
    Returns:
        velocity_vector: 3D velocity change vector or None if no velocity should be applied
    """
    # Normalize velocity direction
    velocity_direction = np.array(velocity_direction, dtype=np.float32)
    velocity_direction = velocity_direction / np.linalg.norm(velocity_direction)

    # Calculate if we're in an active period
    time_in_period = simulation_time % period
    if time_in_period < duration:
        return velocity_magnitude * velocity_direction
    else:
        return None
    
# --- update history ---
def update_hist(hist_buf, cur, dim):
    cur = cur.to(hist_buf.device)
    res = torch.cat([hist_buf[:, dim:], cur], dim=-1)
    return res


if __name__ == "__main__":
    # get config file name from command line
    import argparse
    gamepad = gamepad_reader_btp.Gamepad()
    command_function = gamepad.get_command

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/p2_prop.yaml",
                        help="config file name in the config folder")
    parser.add_argument("--enable_push_forces", default="false", action="store_true",
                        help="Enable periodic velocity impulses")
    parser.add_argument("--push_force_magnitude", type=float, default=1.0,
                        help="Magnitude of velocity impulses (m/s)")
    parser.add_argument("--push_force_period", type=float, default=2.0,
                        help="Period between velocity impulses (seconds)")
    args = parser.parse_args()
    config_file = args.config_file
    with open(f"{config_file}", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"]
        # odom_path = config["odom_path"]
        xml_path = config["xml_path"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        lin_vel_scale = config["lin_vel_scale"]
        ang_vel_scale = config["ang_vel_scale"]
        rpy_scale = config["rpy_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        # num_odom_output = config["num_odom_output"]
        num_obs = config["num_obs"]
        # num_odom_obs = config["num_odom_obs"]

        cmd = np.array(config["cmd_init"], dtype=np.float32)
        # Set gait cycle time (default 0.8s, can be added to config if needed)
        cycle_time = 0.6
        his_obs = config["his_obs"]
        # his_odom_obs = config["his_odom_obs"]
        controller_scale = config["controller_scale"]

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    # odom_output = np.zeros(num_odom_output, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    # odom_obs = np.zeros(num_odom_obs, dtype=np.float32)  # signal_obs
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    history_obs = torch.zeros(1, num_obs*his_obs, dtype=torch.float32, device=device)
    history = {
        "ang_vel": torch.zeros((1, 3 * his_obs), device=device),
        "gravity": torch.zeros((1, 3 * his_obs), device=device),
        "cmd": torch.zeros((1, 3 * his_obs), device=device),
        "qj": torch.zeros((1, num_actions * his_obs), device=device),
        "dqj": torch.zeros((1, num_actions * his_obs), device=device),
        "actions": torch.zeros((1, num_actions * his_obs), device=device),
        "gait": torch.zeros((1, 2 * his_obs), device=device),
    }

    counter = 0
    common_iter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # 设置初始角度
    d.qpos[7:7+len(default_angles)] = default_angles
    d.qvel[:] = 0.0
    print(f"Initial angles: {d.qpos[7:]}")

    m.opt.timestep = simulation_dt

    # load policy
    # Check if the policy path exists before loading the model
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found: {policy_path}")
    policy = torch.jit.load(policy_path)
    policy.to(device)  # 将policy模型移到GPU
    policy.eval()

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        last_log_time = start  # Track time for 2-second intervals
        log_interval = 2.0  # Log every 2 seconds

        while viewer.is_running():
            step_start = time.time()

            # Apply PD control to 12 actuated joints
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, d.qvel[6:], kds)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # Read command from gamepad
            if True:
                lin_speed, ang_speed, e_stop, _ = command_function()
                # lin_speed = [0.0,0,0]
                if e_stop:
                    gamepad.read_thread.join()
                    sys.exit(0)
                if (lin_speed[0]<=0):
                    cmd[0] = -1. * lin_speed[0] * (-0.5) * controller_scale[0]
                else:
                    cmd[0] = lin_speed[0] * 0.5 * controller_scale[0]

                if (lin_speed[1]<=0):
                    cmd[1] = -1. * lin_speed[1] * (-0.5) * controller_scale[1]
                else:
                    cmd[1] = lin_speed[1] * 0.5 * controller_scale[1]
                
                if (ang_speed<=0):
                    cmd[2] = -1. * ang_speed * (-1.0) * controller_scale[2]
                else:
                    cmd[2] = ang_speed * (1.0) * controller_scale[2]

            # cmd= [0.0 ,0, 0]
            print(f"cmd: {cmd}")

            counter += 1
            common_iter += 1
            if counter % control_decimation == 0:
                qj = d.qpos[7:]  # dof_pos (12 actuated joints)
                dqj = d.qvel[6:]  # dof_vel (12 actuated joints)
                quat = d.qpos[3:7]  # orientation
                omega = d.qvel[3:6]  # ang_vel
                gravity_orientation = get_gravity_orientation(quat)

                phase = (counter * simulation_dt) / cycle_time
                sin_phase = np.sin(2 * np.pi * phase)
                stance_mask = np.zeros(2, dtype=np.float32)
                stance_mask[0] = sin_phase >= 0
                # right foot stance
                stance_mask[1] = sin_phase < 0
                cmd_l2 = np.linalg.norm(cmd)
                if (cmd_l2 < 0.01):
                    stance_mask[0] = 1
                    stance_mask[1] = 1
                    

                cur_ang_vel = torch.from_numpy(omega * ang_vel_scale).float().unsqueeze(0)
                cur_gravity = torch.from_numpy(gravity_orientation * rpy_scale).float().unsqueeze(0)
                cur_cmd = torch.from_numpy(cmd * cmd_scale).float().unsqueeze(0)

                cur_qj = torch.from_numpy((qj - default_angles) * dof_pos_scale).float().unsqueeze(0)
                cur_dqj = torch.from_numpy(dqj * dof_vel_scale).float().unsqueeze(0)
                cur_action = torch.from_numpy(action).float().unsqueeze(0)
                cur_gait = torch.from_numpy(stance_mask).float().unsqueeze(0)
            

                history["ang_vel"] = update_hist(history["ang_vel"], cur_ang_vel, 3)
                history["gravity"] = update_hist(history["gravity"], cur_gravity, 3)
                history["cmd"] = update_hist(history["cmd"], cur_cmd, 3)
                history["qj"] = update_hist(history["qj"], cur_qj, num_actions)
                history["dqj"] = update_hist(history["dqj"], cur_dqj, num_actions)
                history["actions"] = update_hist(history["actions"], cur_action, num_actions)
                history["gait"] = update_hist(history["gait"], cur_gait, 2)


                history_obs = torch.cat([
                    history["ang_vel"],
                    history["gravity"],
                    history["cmd"],
                    history["qj"],
                    history["dqj"],
                    history["actions"],
                    history["gait"],
                ], dim=-1)

                # print("\n===== His Input (obs) =====")
                # print(f"{history_obs.cpu().detach().numpy().squeeze()}")

                # Policy inference on GPU
                action_tensor = policy(history_obs)
                action = action_tensor.cpu().detach().numpy().squeeze() # 将policy模型移回cpu，并转换为numpy
                # print("\n===== Output =====")
                # print(f"{action}")
                
                # print("#############################")
                # print("\n")

                # transform action to target_dof_pos
                actions_scaled = action * action_scale
                target_dof_pos = actions_scaled + default_angles

            viewer.sync()
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
