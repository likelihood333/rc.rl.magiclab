import numpy as np
import time
import torch
import logging
import signal
import sys
from enum import Enum

import magicbot_z1_python as magicbot
import gamepad_reader_btp
import threading

from config_z1 import Config
from deploy_z1_base import (
    connect_robot,
    get_gravity_orientation,
    load_policy,
    publish_all_zero_torque,
    switch_to_low_level,
    update_hist,
)

np.set_printoptions(precision=2, suppress=True, floatmode='fixed')


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Global variables
robot = None
running = True

# State enumeration
class ControllerState(Enum):
    ZERO_TORQUE = 0
    MOVE_TO_DEFAULT = 1
    DEFAULT_POSITION = 2
    ACTIVE_CONTROL = 3
    DAMPING = 4

def signal_handler(signum, frame):
    """Signal handler for graceful exit"""
    global robot, running
    running = False
    logging.info("Received interrupt signal (%s), exiting...", signum)
    
    if robot:
        # Send damping command for safe stop
        robot.enter_damping_state()
        robot.disconnect()
        logging.info("Robot disconnected")
        robot.shutdown()
        logging.info("Robot shutdown")
    sys.exit(0)

class MagicBotController:
    def __init__(self, config):
        self.config = config
        self.robot = None
        self.controller = None
        self.state = ControllerState.ZERO_TORQUE
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = load_policy(
            policy_path=config.policy_path,
            device=self.device,
            obs_dim=int(config.num_obs) * int(config.his_obs),
            action_dim=int(config.num_actions),
            policy_format=str(config.policy_format),
            actor_activation=str(config.actor_activation),
        )
        
        # Initialize state variables
        self.num_actions = self.config.num_actions
        self.num_obs = self.config.num_obs
        self.his_obs = self.config.his_obs
        
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.config.default_leg_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.cmd = np.array(self.config.cmd_init, dtype=np.float32)
        self.cmd_btn = (False, False, False, False)
        self.e_stop_active = False
        self.control_dt =self.config.control_dt
        self.cycle_time = 0.6
        
        self.history_obs = torch.zeros(1, self.num_obs * self.his_obs, 
                                     dtype=torch.float32, device=self.device)
        self.history = {
            "ang_vel": torch.zeros((1, 3 * self.his_obs), device=self.device),
            "gravity": torch.zeros((1, 3 * self.his_obs), device=self.device),
            "cmd": torch.zeros((1, 3 * self.his_obs), device=self.device),
            "qj": torch.zeros((1, self.num_actions * self.his_obs), device=self.device),
            "dqj": torch.zeros((1, self.num_actions * self.his_obs), device=self.device),
            "actions": torch.zeros((1, self.num_actions * self.his_obs), device=self.device),
            "gait": torch.zeros((1, 2 * self.his_obs), device=self.device),
        }
        self.cur_ang_vel = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cur_gravity = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cur_cmd = torch.zeros(3, dtype=torch.float32, device=self.device)
        self.cur_qj = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        self.cur_dqj = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)
        self.cur_action = torch.zeros(self.num_actions, dtype=torch.float32, device=self.device)

        self.counter = 0
        self.common_iter = 0

        self.gamepad = gamepad_reader_btp.Gamepad(vel_scale_x=config.controller_scale[0], vel_scale_y=config.controller_scale[1], vel_scale_rot=config.controller_scale[2])
        self.joystick_command = self.gamepad.get_command
        self.running = True
        self.interp_initialized = False
        
        # Joint state cache
        self.leg_joint_state = None
        self.arm_joint_state = None
        self.waist_joint_state = None
        self.head_joint_state = None
        self.body_imu_data = None
        self.wireless_remote_data = None
        
        # Initialize robot connection
        self.initialize_robot()

    def initialize_robot(self):
        """Initialize robot connection"""
        global robot
        try:
            local_ip = str(self.config.local_ip)
            self.robot = connect_robot(local_ip)
            robot = self.robot
            logging.info("Successfully connected to robot")

            switch_to_low_level(self.robot)
            logging.info("Switched to low-level motion controller")

            # Get low-level motion controller
            self.controller = self.robot.get_low_level_motion_controller()

            # Subscribe to sensor data
            self.controller.subscribe_body_imu(self.body_imu_callback)
            logging.info("Subscribed to body IMU data")
            self.controller.subscribe_leg_state(self.leg_state_callback)
            logging.info("Subscribed to leg joint state")
            self.controller.subscribe_arm_state(self.arm_state_callback)
            logging.info("Subscribed to arm joint state")
            self.controller.subscribe_waist_state(self.waist_state_callback)
            logging.info("Subscribed to waist joint state")
            self.controller.subscribe_head_state(self.head_state_callback)
            logging.info("Subscribed to head joint state")

            # Add joystick reading thread
            self.joystick_thread = None
            self.joystick_running = True  
            logging.info("Subscribed to wireless remote data") 
            
            # Wait for sensor data
            self.wait_for_sensor_data()
            
            logging.info("Subscribed to all sensor data")
            self.init_dof_pos_leg = np.zeros(magicbot.LEG_JOINT_NUM, dtype=np.float32)
            self.init_dof_pos_arm = np.zeros(magicbot.ARM_JOINT_NUM, dtype=np.float32)
            self.init_dof_pos_waist = np.zeros(magicbot.WAIST_JOINT_NUM, dtype=np.float32)
            self.init_dof_pos_head = np.zeros(magicbot.HEAD_JOINT_NUM, dtype=np.float32)
            logging.info(f"nums: leg {int(magicbot.LEG_JOINT_NUM)}, arm {int(magicbot.ARM_JOINT_NUM)}, waist {int(magicbot.WAIST_JOINT_NUM)}, head {int(magicbot.HEAD_JOINT_NUM)}")
            return True

        except RuntimeError as e:
            logging.error("%s", e)
            return False
        except Exception as e:
            logging.error("Error initializing robot: %s", e)
            return False

    def wait_for_sensor_data(self):
        """Wait for sensor data reception"""
        logging.info("Waiting for sensor data...")
        while (self.leg_joint_state is None or 
               self.body_imu_data is None):
            time.sleep(self.control_dt * 10)
        logging.info("Successfully received sensor data")

    def body_imu_callback(self, imu_data):
        """IMU data callback"""
        self.body_imu_data = imu_data

    def leg_state_callback(self, joint_state):
        """Leg joint state callback"""        
        self.leg_joint_state = joint_state

    def arm_state_callback(self, joint_state):
        """Arm joint state callback"""
        self.arm_joint_state = joint_state

    def waist_state_callback(self, joint_state):
        """Waist joint state callback"""
        self.waist_joint_state = joint_state

    def head_state_callback(self, joint_state):
        """Head joint state callback"""
        self.head_joint_state = joint_state
    def get_joint_positions(self):
        """Get position information from joint state"""
        if self.leg_joint_state is None:
            return np.zeros(self.num_actions, dtype=np.float32)
        
        qj = np.zeros(self.num_actions, dtype=np.float32)
        for i in range(min(self.num_actions, magicbot.LEG_JOINT_NUM)):
            if i < len(self.leg_joint_state.joints):
                qj[i] = self.leg_joint_state.joints[i].posL
        # print(f"qj: {qj}")
        
        return qj

    def get_joint_velocities(self):
        """Get velocity information from joint state"""
        if self.leg_joint_state is None:
            return np.zeros(self.num_actions, dtype=np.float32)
        
        dqj = np.zeros(self.num_actions, dtype=np.float32)
        for i in range(min(self.num_actions, magicbot.LEG_JOINT_NUM)):
            if i < len(self.leg_joint_state.joints):
                dqj[i] = self.leg_joint_state.joints[i].vel
        
        return dqj

    def get_imu_data(self):
        """Get orientation and angular velocity from IMU data"""
        ori_ori = list(self.body_imu_data.orientation)
        orientation = np.array(ori_ori, dtype=np.float32)
          
        # orientation = np.zeros(4, dtype=np.float32)
        # orientation[0] = ori_ori[3]
        # orientation[1] = ori_ori[0]
        # orientation[2] = ori_ori[1]
        # orientation[3] = ori_ori[2]

        # orientation = np.array(self.body_imu_data.orientation, dtype=float)
        angular_velocity = np.array(self.body_imu_data.angular_velocity, dtype=float)

        # orientation = list(self.body_imu_data.orientation)
        # angular_velocity = list(self.body_imu_data.angular_velocity)

        return orientation, angular_velocity
    
    def joystick_loop(self):
        """Joystick reading loop"""

        loop_count = 0
        while self.joystick_running and self.running:
            try:
                loop_count += 1
                lin_speed, ang_speed, e_stop, btn = self.joystick_command()
                if e_stop:
                    logging.info("Emergency stop triggered from gamepad")
                    self.e_stop_active = True
                    self.running = False
                    break
                
                # Update command
                self.cmd[0] = lin_speed[0]
                self.cmd[1] = lin_speed[1]
                self.cmd[2] = ang_speed
                self.cmd_btn = btn

                # logging.info(f"JS cmd vx: {self.cmd[0]:.3f} vy: {self.cmd[1]:.3f} wz: {self.cmd[2]:.3f} cmd_btn{self.cmd_btn}")
                
                # Control reading frequency
                time.sleep(0.01)  # 100Hz
                
            except Exception as e:
                logging.error("Error in joystick loop: %s", e)
                time.sleep(0.1)
    def send_zero_torque_commands(self):
        """Send zero torque commands"""
        if self.controller is None:
            return

        try:
            publish_all_zero_torque(self.controller)
        except Exception as e:
            logging.error("Error sending zero torque commands: %s", e)

    def send_position_commands(self, leg_target_positions, leg_params, arm_target_positions, waist_target_positions, head_target_positions):
        """Send position control commands"""
        if self.controller is None:
            return

        try:
            # Split
            leg_num   = magicbot.LEG_JOINT_NUM
            arm_num   = magicbot.ARM_JOINT_NUM
            waist_num = magicbot.WAIST_JOINT_NUM
            head_num  = magicbot.HEAD_JOINT_NUM

            leg_pos   = leg_target_positions
            arm_pos   = arm_target_positions
            waist_pos = waist_target_positions
            head_pos  = head_target_positions

            # ---- leg ----
            leg_cmd = magicbot.JointCommand()
            for i in range(leg_num):
                j = magicbot.SingleJointCommand()
                j.operation_mode = int(self.config.leg_mode[i])
                j.pos = float(leg_pos[i])
                j.kp  = float(leg_params[0][i])
                j.kd  = float(leg_params[1][i])
                j.vel = 0.0
                j.toq = 0.0
                leg_cmd.joints.append(j)

            # ---- arm ----
            arm_cmd = magicbot.JointCommand()
            for i in range(arm_num):
                j = magicbot.SingleJointCommand()
                j.operation_mode = int(self.config.arm_mode[i])
                j.pos = float(arm_pos[i])
                j.kp  = float(self.config.arm_kp[i])
                j.kd  = float(self.config.arm_kd[i])
                j.vel = 0.0
                j.toq = 0.0
                arm_cmd.joints.append(j)

            # ---- waist ----
            waist_cmd = magicbot.JointCommand()
            for i in range(waist_num):
                j = magicbot.SingleJointCommand()
                j.operation_mode = int(self.config.waist_mode[i])
                j.pos = float(waist_pos[i])
                j.kp  = float(self.config.waist_kp[i])
                j.kd  = float(self.config.waist_kd[i])
                j.vel = 0.0
                j.toq = 0.0
                waist_cmd.joints.append(j)

            # ---- head ----
            head_cmd = magicbot.JointCommand()
            for i in range(head_num):
                j = magicbot.SingleJointCommand()
                j.operation_mode = int(self.config.head_mode[i])
                j.pos = float(head_pos[i])
                j.kp  = float(self.config.head_kp[i])
                j.kd  = float(self.config.head_kd[i])
                j.vel = 0.0
                j.toq = 0.0
                head_cmd.joints.append(j)

            # ---- publish ----
            self.controller.publish_leg_command(leg_cmd)
            self.controller.publish_arm_command(arm_cmd)
            self.controller.publish_waist_command(waist_cmd)
            self.controller.publish_head_command(head_cmd)

            # logging.info("Sent position commands")

            
        except Exception as e:
            logging.error("Error sending position commands: %s", e)

    def enter_damping_state(self):
        """Enter damping state (safe stop)"""
        logging.info("Entering damping state")
        self.state = ControllerState.DAMPING
        self.send_zero_torque_commands()

    def zero_torque_state(self):
        """Zero torque state, waiting for start signal"""
        logging.info("Enter zero torque state.")
        logging.info("Waiting for the start signal...")
        
        self.state = ControllerState.ZERO_TORQUE
        
        while ((self.cmd_btn[0]==False and 
               running and self.state == ControllerState.ZERO_TORQUE) or self.e_stop_active):
            self.send_zero_torque_commands()
            time.sleep(self.config.control_dt)
            
        if running:
            logging.info("Start signal received, moving to default position.")

    def move_to_default_pos(self):
        # Interpolation duration (seconds)
        self.interp_duration = 2
        num_step = int(self.interp_duration / self.config.control_dt)

        # Get current joint positions
        self.init_dof_pos_leg = self.get_joint_positions().copy()
        for i in range(magicbot.ARM_JOINT_NUM):
            self.init_dof_pos_arm[i] = self.arm_joint_state.joints[i].posL
        for i in range(magicbot.WAIST_JOINT_NUM):
            self.init_dof_pos_waist[i] = self.waist_joint_state.joints[i].posL
        for i in range(magicbot.HEAD_JOINT_NUM):
            self.init_dof_pos_head[i] = self.head_joint_state.joints[i].posL

        # Target positions
        self.interp_target_leg_pos = self.config.default_leg_angles.copy()
        self.arm_target_positions = self.config.default_arm_angles.copy()
        self.waist_target_positions = self.config.default_waist_angles.copy()
        self.head_target_positions = self.config.default_head_angles.copy()

        logging.info(f"move_to_default_pos in {self.interp_duration} seconds")

        interval = 0.002
        next_t = time.perf_counter() + interval

        for i in range(num_step + 1):
            
            if self.e_stop_active:
                logging.info("Emergency stop triggered during move_to_default_pos")
                self.state = ControllerState.ZERO_TORQUE
                self.send_zero_torque_commands()
                return

            # -------- Calculate interpolation ratio alpha --------
            alpha = i / num_step

            # -------- Linear interpolation target = start*(1-alpha) + end*alpha --------
            leg_target_positions = (
                self.init_dof_pos_leg * (1 - alpha)
                + self.interp_target_leg_pos * alpha
            )
            arm_target_positions = (
                self.init_dof_pos_arm * (1 - alpha)
                + self.arm_target_positions * alpha
            )
            waist_target_positions = (
                self.init_dof_pos_waist * (1 - alpha)
                + self.waist_target_positions * alpha
            )
            head_target_positions = (
                self.init_dof_pos_head * (1 - alpha)
                + self.head_target_positions * alpha
            )

            # logging.info(f"###############alpha: {alpha}#########################")
            # logging.info(f"leg_target_positions: {leg_target_positions}")
            # logging.info(f"arm_target_positions: {arm_target_positions}")
            # logging.info(f"waist_target_positions: {waist_target_positions}")
            # logging.info(f"head_target_positions: {head_target_positions}")


            # -------- Send position control commands --------
            self.send_position_commands(leg_target_positions, [self.config.rec_leg_kp, self.config.rec_leg_kd], arm_target_positions, waist_target_positions, head_target_positions)

            next_t += interval
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            # logging.info(f"sleep time: {sleep_time:.4f}")

        logging.info("move_to_default_pos done")
        self.state = ControllerState.DEFAULT_POSITION



    def default_pos_state(self):
        """Default position state, waiting for Button B signal"""
        logging.info("Enter default position state.")
        logging.info("Waiting for the Button B signal...")
        
        self.state = ControllerState.DEFAULT_POSITION

        interval = 0.002
        next_t = time.perf_counter() + interval

        while (self.cmd_btn[1]==False and 
            running and self.state == ControllerState.DEFAULT_POSITION and not self.e_stop_active):

            # Check emergency stop
            if self.e_stop_active:
                logging.info("Emergency stop triggered in default_pos_state")
                self.state = ControllerState.ZERO_TORQUE
                self.send_zero_torque_commands()
                return

            self.send_position_commands(self.config.default_leg_angles, [self.config.rec_leg_kp, self.config.rec_leg_kd], self.config.default_arm_angles, self.config.default_waist_angles, self.config.default_head_angles)

            next_t += interval
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            time.sleep(self.config.control_dt)

            # qj = self.get_joint_positions()
            # dqj = self.get_joint_velocities()
            # quat, omega = self.get_imu_data()
            
            # gravity_orientation = get_gravity_orientation(quat)
            
            # logging.info("#######################################################")
            # logging.info(f"omega: {omega * self.config.ang_vel_scale}")
            # logging.info(f"gravity_orientation: {gravity_orientation * self.config.rpy_scale}")
            # logging.info(f"cmd: {self.cmd * self.config.cmd_scale}")
            # logging.info(f"qj: {(qj - self.config.default_leg_angles) * self.config.dof_pos_scale}")
            # logging.info(f"dqj: {dqj * self.config.dof_vel_scale}")
            
            
        if running:
            logging.info("Button B pressed, starting active control.")
            self.state = ControllerState.ACTIVE_CONTROL

    def start_joystick_thread(self):
        """Start joystick reading thread"""
        self.joystick_thread = threading.Thread(target=self.joystick_loop)
        self.joystick_thread.daemon = True
        self.joystick_thread.start()
        logging.info("Joystick thread started")

    def run_active_control(self):
        """Run active control"""

        # Check emergency stop
        if self.e_stop_active:
            logging.info("Emergency stop triggered in active control")
            self.state = ControllerState.ZERO_TORQUE
            self.send_zero_torque_commands()

        self.counter += 1
        self.common_iter += 1

        # Control frequency decimation
        if self.counter % self.config.control_decimation == 0:
            # Get sensor data
            qj = self.get_joint_positions()
            dqj = self.get_joint_velocities()
            quat, omega = self.get_imu_data()
            
            gravity_orientation = get_gravity_orientation(quat)

            phase = (self.counter * self.control_dt) / self.cycle_time
            sin_phase = np.sin(2 * np.pi * phase)
            stance_mask = np.zeros(2, dtype=np.float32)
            stance_mask[0] = sin_phase >= 0
            # right foot stance
            stance_mask[1] = sin_phase < 0
            cmd_l2 = np.linalg.norm(self.cmd)
            if (cmd_l2 < 0.01):
                stance_mask[0] = 1
                stance_mask[1] = 1
    

            # --- Current frame obs extraction ---
            self.cur_ang_vel = torch.from_numpy(omega * self.config.ang_vel_scale).float().unsqueeze(0)
            self.cur_gravity = torch.from_numpy(gravity_orientation * self.config.rpy_scale).float().unsqueeze(0)
            self.cur_cmd = torch.from_numpy(self.cmd * self.config.cmd_scale).float().unsqueeze(0)
            self.cur_qj = torch.from_numpy((qj - self.config.default_leg_angles) * self.config.dof_pos_scale).float().unsqueeze(0)
            self.cur_dqj = torch.from_numpy(dqj * self.config.dof_vel_scale).float().unsqueeze(0)
            self.cur_action = torch.from_numpy(self.action).float().unsqueeze(0)
            self.cur_gait = torch.from_numpy(stance_mask).float().unsqueeze(0)

            self.history["ang_vel"] = update_hist(self.history["ang_vel"], self.cur_ang_vel, 3)
            self.history["gravity"] = update_hist(self.history["gravity"], self.cur_gravity, 3)
            self.history["cmd"] = update_hist(self.history["cmd"], self.cur_cmd, 3)
            self.history["qj"] = update_hist(self.history["qj"], self.cur_qj, self.num_actions)
            self.history["dqj"] = update_hist(self.history["dqj"], self.cur_dqj, self.num_actions)
            self.history["actions"] = update_hist(self.history["actions"], self.cur_action, self.num_actions)
            self.history["gait"] = update_hist(self.history["gait"], self.cur_gait, 2)

            self.history_obs = torch.cat([
                self.history["ang_vel"],
                self.history["gravity"],
                self.history["cmd"],
                self.history["qj"],
                self.history["dqj"],
                self.history["actions"],
                self.history["gait"]
            ], dim=-1)

            # Policy inference      
            with torch.no_grad():
                action_tensor = self.policy(self.history_obs)
                self.action = action_tensor.cpu().detach().numpy().squeeze()
            
            # Calculate target position
            actions_scaled = self.action * self.config.action_scale
            self.target_dof_pos = actions_scaled + self.config.default_leg_angles

            # logging.info(f"bef send")

            # Send control commands
            self.send_position_commands(self.target_dof_pos, [self.config.leg_kp, self.config.leg_kd], self.config.default_arm_angles, self.config.default_waist_angles, self.config.default_head_angles)

    def shutdown(self):
        if self.robot:
            self.enter_damping_state()
            self.robot.disconnect()
            self.robot.shutdown()

def main():
    """Main function"""
    global running
    
    # Bind signal handler
    signal.signal(signal.SIGINT, signal_handler)

    logging.info("Robot model: %s", magicbot.get_robot_model())  # Print robot model
    
    # Load configuration
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config file name in the configs folder", default="g1.yaml")
    args = parser.parse_args()
    
    # Load configuration file
    config_path = f"configs/{args.config}"  # Adjust according to actual path
    config = Config(config_path)
    
    # Create controller
    controller = MagicBotController(config)
    
    if not controller.robot:
        logging.error("Failed to initialize robot controller")
        return -1
    
    try:
        # Start joystick reading thread
        # while running:
        controller.start_joystick_thread()

        # Enter zero torque state, press start button to continue
        controller.zero_torque_state()
        
      
        if not controller.running:
            return 0
            
        # Move to default position
        controller.move_to_default_pos()
        
        if not controller.running:
            return 0
  
        # Enter default position state, press Button A to continue
        controller.default_pos_state()
        
        if not controller.running:
            return 0
        
        
        interval = 0.002
        next_t = time.perf_counter() + interval
            
        
        # Main control loop
        logging.info("Starting active control loop, press Button Y to exit")
        while running and controller.state == ControllerState.ACTIVE_CONTROL:
            # Run active control
            controller.run_active_control()
            
            # logging.info(f"c button :{controller.cmd_btn[2]}")
            # Check exit condition
            if controller.cmd_btn[3]==True:
                logging.info("stop button pressed, exiting...")
                break
                
            next_t += interval
            sleep_time = next_t - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Move to default position
        controller.move_to_default_pos()
        
        if not running:
            return 0
        
        # Enter default position state, press Button A to continue
        controller.default_pos_state()
        
        if not running:
            return 0
    
                
    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received")
    except Exception as e:
        logging.error("Exception in control loop: %s", e)
        
    finally:
        logging.info("Shutting down controller...")
        controller.shutdown()
        logging.info("Exit")


if __name__ == "__main__":
    main()