from rclpy.node import Node
from threading import Event
import math, time
import rclpy
from unitree_go.msg._wireless_controller import WirelessController
from .motion_backend import GrootHttpDiscreteBackend


class StepControlClient(Node):
    def __init__(self, cfg=None):
        super().__init__('step_control_client')
        self.get_logger().info("StepControl client initialized...")

        self.motion_backend = GrootHttpDiscreteBackend(cfg, log=None)
        self.publisher = None
        if not self.motion_backend.enabled:
            self.publisher = self.create_publisher(
                WirelessController, '/wirelesscontroller', 10)
        else:
            self.get_logger().info("StepControl uses GR00T HTTP discrete backend.")

        self.get_logger().info("StepControl client launched...")

        self.task = 'StepControl '
        self._stop_flag = Event()
        self.code2msg = {
            0: self.task + 'undefined.',
            1: self.task + 'success.',
            -1: self.task + 'fail.',
            -2: self.task + 'canceled.',
            2: self.task + 'still excuting',
            3: self.task + 'paused',
            4: self.task + 'paused failed',
            5: self.task + 'excuting failed'
        }

    @staticmethod
    def _zero_motion_msg():
        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = 0.0
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0
        return msg

    def _publish_motion(self, msg):
        if self.motion_backend.enabled:
            self.get_logger().warning("Ignoring continuous WirelessController command in GR00T backend mode.")
            return False
        if not rclpy.ok():
            return False
        try:
            self.publisher.publish(msg)
            return True
        except Exception as exc:
            if "context is invalid" in str(exc):
                self.get_logger().debug(f"Motion command skipped because ROS context is invalid: {exc}")
            else:
                self.get_logger().warning(f"Failed to publish motion command: {exc}")
            return False

    def _publish_zero_motion(self, repeat=1, interval=0.02):
        if self.motion_backend.enabled:
            success, _, _ = self.motion_backend.stop()
            return success
        ok = True
        repeat = max(1, int(repeat))
        for index in range(repeat):
            ok = self._publish_motion(self._zero_motion_msg()) and ok
            if index < repeat - 1:
                time.sleep(interval)
        return ok


    def stop(self):
        self._stop_flag.set()
        if self.motion_backend.enabled:
            return self.motion_backend.stop()
        self._publish_zero_motion(repeat=5)
        return True, 'success.', 'stop success.'

    def forward(self, distance = 1.0, speed = 1.0):
        if self.motion_backend.enabled:
            self._stop_flag.clear()
            return self.motion_backend.forward(float(distance))

        distance = distance*2.5
        self._stop_flag.clear()
        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = speed if distance > 0 else -speed
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0

        duration = abs(distance) / speed
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
            if self._stop_flag.is_set():
                self.get_logger().info('[StepControl] Movement interrupted by stop().')
                break
            if not self._publish_motion(msg):
                break
            time.sleep(0.01)

        self._publish_zero_motion(repeat=2)

        if not self._stop_flag.is_set():
            result_holder = {
                'success_flag': True,
                'msg': 'success.',
                'code': 1
            }
        else:
            result_holder = {
                'success_flag': False,
                'msg': 'stopped.',
                'code': -2
            }
        return result_holder['success_flag'], result_holder['msg'], result_holder['code']
    def shift(self, distance = 1.0, speed = 0.3):
        if self.motion_backend.enabled:
            self._stop_flag.clear()
            return self.motion_backend.shift(float(distance))

        self._stop_flag.clear()
        msg = WirelessController()
        msg.lx = -speed if distance > 0 else speed
        msg.ly = 0.0
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0
        self._publish_motion(msg)

        duration = abs(distance) / speed
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
            if self._stop_flag.is_set():
                self.get_logger().info('[StepControl] Movement interrupted by stop().')
                break
            if not self._publish_motion(msg):
                break
            time.sleep(0.01)

        self._publish_zero_motion(repeat=2)

        if not self._stop_flag.is_set():
            result_holder = {
                'success_flag': True,
                'msg': 'success.',
                'code': 1
            }
        else:
            result_holder = {
                'success_flag': False,
                'msg': 'stopped.',
                'code': -2
            }
        return result_holder['success_flag'], result_holder['msg'], result_holder['code']

    def rotate(self, angle = math.pi/2, speed = math.pi/2):
        if self.motion_backend.enabled:
            self._stop_flag.clear()
            return self.motion_backend.rotate(float(angle))

        self._stop_flag.clear()
        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = 0.0
        msg.rx = -speed if angle > 0 else speed
        msg.ry = 0.0
        msg.keys = 0
        self._publish_motion(msg)

        duration = abs(angle) / speed
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds / 1e9 < duration:
            if self._stop_flag.is_set():
                self.get_logger().info('[StepControl] Movement interrupted by stop().')
                break
            if not self._publish_motion(msg):
                break
            time.sleep(0.01)

        self._publish_zero_motion(repeat=2)

        if not self._stop_flag.is_set():
            result_holder = {
                'success_flag': True,
                'msg': 'success.',
                'code': 1
            }
        else:
            result_holder = {
                'success_flag': False,
                'msg': 'stopped.',
                'code': -2
            }
        return result_holder['success_flag'], result_holder['msg'], result_holder['code']
