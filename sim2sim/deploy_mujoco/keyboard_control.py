# SPDX-FileCopyrightText: Copyright (c) 2023, HUAWEI TECHNOLOGIES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from absl import app
import keyboard
import threading
import time
import signal
import sys
import warnings

# Maximum velocities for each direction
MAX_VEL_X = 1.2  # Maximum forward/backward speed (W/S keys)
MAX_VEL_Y = 0.6  # Maximum left/right speed (A/D keys)
MAX_VEL_ROT = 1.57  # Maximum rotational speed (Q/E keys)

# Velocity adjustment parameters
VEL_INCREMENT = 0.01  # Amount to increase/decrease velocity per key press
VEL_MIN = 0.0  # Minimum allowed velocity
VEL_MAX = 5.0  # Maximum allowed velocity

class KeyboardControl:
    """Interface for reading commands from keyboard.
    The control works as follows:
    1) Press 'p' for emergency stop.
    2) Use 'w'/'s' for forward/backward movement.
    3) Use 'a'/'d' for left/right movement.
    4) Use 'e'/'q' for counterclockwise/clockwise rotation.
    5) Use 'u'/'j' to increase/decrease forward/backward speed.
    6) Use 'i'/'k' to increase/decrease left/right speed.
    7) Use 'o'/'l' to increase/decrease rotational speed.
    Writes commands to a file for external processes to read.
    """

    def __init__(self, command_file="commands.txt"):
        """Initialize the keyboard controller.
        Args:
            command_file: Path to the file where commands are written.
        """
        self._max_vel_x = MAX_VEL_X
        self._max_vel_y = MAX_VEL_Y
        self._max_vel_rot = MAX_VEL_ROT
        self._command_file = command_file

        # Controller states
        self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        self.estop_flagged = False
        self.is_running = True

        try:
            import keyboard
        except ImportError:
            warnings.warn("Keyboard library not installed. Please install it using 'pip install keyboard'.")
            self.is_running = False
            return

        signal.signal(signal.SIGINT, self.signal_handler)
        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for keyboard events.
        This function runs in a separate thread for continuous event recording.
        """
        while self.is_running and not self.estop_flagged:
            self.update_command()
            time.sleep(0.01)  # Small sleep to prevent CPU overuse

    def signal_handler(self, sig, frame):
        """Handle Ctrl+C signal for graceful shutdown."""
        print("Ctrl+C detected, shutting down...")
        self.stop()
        sys.exit(0)

    def update_command(self):
        """Update command based on keyboard inputs and write to file."""
        # Check emergency stop key
        if keyboard.is_pressed('p'):
            self.estop_flagged = True
            self.vx, self.vy, self.wz = 0.0, 0.0, 0.0
        else:
            # Reset velocities
            self.vx, self.vy, self.wz = 0.0, 0.0, 0.0

            # Forward/backward movement (w/s keys)
            if keyboard.is_pressed('w'):
                self.vx = self._max_vel_x
            elif keyboard.is_pressed('s'):
                self.vx = -self._max_vel_x

            # Left/right movement (a/d keys)
            if keyboard.is_pressed('a'):
                self.vy = self._max_vel_y
            elif keyboard.is_pressed('d'):
                self.vy = -self._max_vel_y

            # Rotation (e/q keys: e for counterclockwise, q for clockwise)
            if keyboard.is_pressed('e'):
                self.wz = self._max_vel_rot  # Counterclockwise
            elif keyboard.is_pressed('q'):
                self.wz = -self._max_vel_rot  # Clockwise

            # Adjust maximum velocities
            if keyboard.is_pressed('u'):
                self._max_vel_x = min(self._max_vel_x + VEL_INCREMENT, VEL_MAX)
                print(f"MAX_VEL_X increased to {self._max_vel_x:.2f}")
            elif keyboard.is_pressed('j'):
                self._max_vel_x = max(self._max_vel_x - VEL_INCREMENT, VEL_MIN)
                print(f"MAX_VEL_X decreased to {self._max_vel_x:.2f}")

            if keyboard.is_pressed('i'):
                self._max_vel_y = min(self._max_vel_y + VEL_INCREMENT, VEL_MAX)
                print(f"MAX_VEL_Y increased to {self._max_vel_y:.2f}")
            elif keyboard.is_pressed('k'):
                self._max_vel_y = max(self._max_vel_y - VEL_INCREMENT, VEL_MIN)
                print(f"MAX_VEL_Y decreased to {self._max_vel_y:.2f}")

            if keyboard.is_pressed('o'):
                self._max_vel_rot = min(self._max_vel_rot + VEL_INCREMENT, VEL_MAX)
                print(f"MAX_VEL_ROT increased to {self._max_vel_rot:.2f}")
            elif keyboard.is_pressed('l'):
                self._max_vel_rot = max(self._max_vel_rot - VEL_INCREMENT, VEL_MIN)
                print(f"MAX_VEL_ROT decreased to {self._max_vel_rot:.2f}")

        # Write command to file
        try:
            with open(self._command_file, 'w') as f:
                f.write(f"{self.vx},{self.vy},{self.wz},{self.estop_flagged}")
        except Exception as e:
            print(f"Error writing to command file: {e}")

    def get_command(self):
        """Return the current command in the format (linear_speed, angular_speed, estop).
        Linear speed is a tuple (vx, vy, 0), angular speed is wz, and estop is a boolean.
        """
        return (1.0 * self.vx, 0.5 * self.vy, 0), -1.5 * self.wz, self.estop_flagged

    def stop(self):
        """Stop the keyboard controller."""
        self.is_running = False
        # Write a final stopped command
        try:
            with open(self._command_file, 'w') as f:
                f.write("0.0,0.0,0.0,False")
        except Exception:
            pass

def main(_):
    """Main function to demonstrate keyboard control usage."""
    keyboard_ctrl = KeyboardControl()
    while True:
        lin_speed, ang_speed, e_stop = keyboard_ctrl.get_command()
        print(f"Linear: {lin_speed}, Angular: {ang_speed}, Estop: {e_stop}")
        time.sleep(0.1)

if __name__ == "__main__":
    app.run(main)