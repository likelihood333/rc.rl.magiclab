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
from absl import flags
from inputs import get_gamepad
import threading
import time

FLAGS = flags.FLAGS
MAX_ABS_RX = 256#32768
MAX_ABS_RY = 256#32768


# def _interpolate(raw_reading, max_raw_reading, new_scale):
#     return raw_reading / max_raw_reading * new_scale

def _interpolate(raw_reading, max_raw_reading, new_scale):
    normalized = -(raw_reading - (max_raw_reading/2 - 1)) / (max_raw_reading/2 - 1)
    return normalized * new_scale


class Gamepad:
    """Interface for reading commands from Logitech F710 Gamepad.
      The control works as following:
      1) Press LB+RB at any time for emergency stop
      2) Use the left joystick for forward/backward/left/right walking.
      3) Use the right joystick for rotation around the z-axis.
    """
    def __init__(self, vel_scale_x=1, vel_scale_y=1, vel_scale_rot=1):
        """Initialize the gamepad controller.
      Args:
        vel_scale_x: maximum absolute x-velocity command.
        vel_scale_y: maximum absolute y-velocity command.
        vel_scale_rot: maximum absolute yaw-dot command.
      """
        self._vel_scale_x = vel_scale_x
        self._vel_scale_y = vel_scale_y
        self._vel_scale_rot = vel_scale_rot
        self._lb_pressed = False
        self._rb_pressed = False
        self.btn_a = False
        self.btn_b = False
        self.btn_x = False
        self.btn_y = False

        # Controller states
        self.vx, self.vy, self.wz = 0., 0., 0.
        self.estop_flagged = False
        self.is_running = True

        self.read_thread = threading.Thread(target=self.read_loop)
        self.read_thread.start()

    def read_loop(self):
        """The read loop for events.
        This funnction should be executed in a separate thread for continuous
        event recording.
        """
        while self.is_running and not self.estop_flagged:
            events = get_gamepad()
            for event in events:
                self.update_command(event)

    def update_command(self, event):
        """Update command based on event readings."""
        if event.ev_type == 'Key' and event.code == 'BTN_TL':
            self._lb_pressed = bool(event.state)
            # print(f"BTN_TL: {event.state}")
        elif event.ev_type == 'Key' and event.code == 'BTN_TR':
            self._rb_pressed = bool(event.state)
            # print(f"BTN_TR: {event.state}")
        elif event.ev_type == 'Key' and event.code == 'BTN_SOUTH':   # A
            self.btn_a = bool(event.state)
        elif event.ev_type == 'Key' and event.code == 'BTN_EAST':    # B
            self.btn_b = bool(event.state)
        elif event.ev_type == 'Key' and event.code == 'BTN_NORTH':   # X
            self.btn_x = bool(event.state)
        elif event.ev_type == 'Key' and event.code == 'BTN_WEST':    # Y
            self.btn_y = bool(event.state)
        elif event.ev_type == 'Absolute' and event.code == 'ABS_X':
            # Left Joystick L/R axis
            self.vy = _interpolate(event.state, MAX_ABS_RX, self._vel_scale_y)
            # print(f"left joystick x: {self.vy}")
            # print(f"raw left joystick x: {event.state}")
        elif event.ev_type == 'Absolute' and event.code == 'ABS_Y':
            # Left Joystick F/B axis; need to flip sign for consistency
            self.vx = _interpolate(event.state, MAX_ABS_RY, self._vel_scale_x)
            # print(f"left joystick x: {self.vx}")
            # print(f"raw left joystick x: {event.state}")
        elif event.ev_type == 'Absolute' and event.code == 'ABS_Z':
            self.wz = _interpolate(event.state, MAX_ABS_RX,
                                   self._vel_scale_rot)
            # print(f"right joystick x: {self.wz}")
            # print(f"raw right joystick x: {event.state}")

        if self._lb_pressed and self._rb_pressed:
            self.estop_flagged = True
            self.vx, self.vy, self.wz = 0., 0., 0.

    def get_command(self):
        # del time_since_reset  # unused

        return (self.vx, self.vy,
                0), self.wz, self.estop_flagged, (self.btn_a, self.btn_b, self.btn_x, self.btn_y)

    def stop(self):
        self.is_running = False


def main(_):
    gamepad = Gamepad()
    # gamepad = HandstandGamepad()
    while True:
        print(f"Vx: {gamepad.vx:.3f}, Vy: {gamepad.vy:.3f}, Wz: {gamepad.wz:.3f}, Estop: {gamepad.estop_flagged}, abxy:{gamepad.btn_a} {gamepad.btn_b}  {gamepad.btn_x} {gamepad.btn_y} ")
        # print(f"Vx: {gamepad.vx}, Vy: {gamepad.vy}, Wz: {gamepad.wz}, x_press: {gamepad._x_pressed}, Estop: {gamepad.estop_flagged}, _lb_pressed: {gamepad._lb_pressed}, _rb_pressed: {gamepad._rb_pressed}")
        time.sleep(0.1)


if __name__ == "__main__":
    app.run(main)
