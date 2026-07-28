import cv2
import ctypes
ctypes.CDLL("/lib/aarch64-linux-gnu/libffi.so.7", mode=ctypes.RTLD_GLOBAL)
from sensor_msgs.msg import Image
import rclpy
from rclpy.node import Node
import pyrealsense2 as rs
from threading import Lock, Thread
import numpy as np
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import logging
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

logger = logging.getLogger(__name__)

class RGBDClient(Node):
    def __init__(self, cfg, visualization=None):
        super().__init__('rgbd_client')
        # config the camera
        self.cfg = cfg
        self.visualization = visualization
        self.bridge = CvBridge()
        if cfg.rgbd_server.launch_rgbd_server:
            rgb_topic = cfg.rgbd_server.publish_topic.rgb
            depth_topic = cfg.rgbd_server.publish_topic.depth

            serial_number = "419222302306"
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(serial_number)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.pipeline.start(config)

            self.publisher_rgb = self.create_publisher(Image, rgb_topic, qos_profile_sensor_data)
            self.publisger_depth = self.create_publisher(Image, depth_topic, qos_profile_sensor_data)
            self.thread = Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
        else:
            # Subscribe to RGB and Depth topics
            self.rgbd_group = MutuallyExclusiveCallbackGroup()
            self.rgb_subscription = Subscriber(self, Image, cfg.rgbd_server.subscrib_topic.rgb, qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
            self.depth_subscription = Subscriber(self, Image, cfg.rgbd_server.subscrib_topic.depth, qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
            self.timesynchronizer = ApproximateTimeSynchronizer([self.rgb_subscription, self.depth_subscription], queue_size=10, slop=0.05)
            self.timesynchronizer.registerCallback(self.rgbd_callback)
            self.timesynchronizer = ApproximateTimeSynchronizer([self.rgb_subscription, self.depth_subscription], queue_size=10, slop=0.05)
            self.timesynchronizer.registerCallback(self.rgbd_callback)

        logger.info("RGBD Node ready!")
        self.rgb_frame = None
        self.depth_frame = None
        self._lock = Lock()

    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            with self._lock:
                self.rgb_frame = rgb_frame
                self.depth_frame = depth_frame
            self._publish_visualization_rgbd(rgb_frame, depth_frame)
        except Exception as e:
            logger.error(f"Failed to process rgb or depth frame: {str(e)}")
    
    def _capture_loop(self):
        while rclpy.ok():
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if not color_frame or not depth_frame:
                    continue
                color_image = np.asanyarray(color_frame.get_data())
                depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) / 1000.0  # scale = 0.001 m/unit, mm->m
                with self._lock:
                    self.rgb_frame = color_image
                    self.depth_frame = depth_image
                self._publish_visualization_rgbd(color_image, depth_image)
                ros_image = self.bridge.cv2_to_imgmsg(color_image, encoding='bgr8')
                ros_depth = self.bridge.cv2_to_imgmsg(depth_image, encoding='32FC1')
                current_time = self.get_clock().now().to_msg()
                ros_image.header.stamp = current_time
                ros_depth.header.stamp = current_time
                self.publisher_rgb.publish(ros_image)
                self.publisger_depth.publish(ros_depth)
                

            except Exception as e:
                logger.error(f"Camera error: {e}")
                continue
        self.pipeline.stop()
        logger.info("RealSense pipeline stopped.")
    
    def get_image(self,):
        while rclpy.ok():
            with self._lock:
                rgb_frame = self.rgb_frame
                depth_frame = self.depth_frame
            if rgb_frame is None or depth_frame is None:
                logger.error('Got rgb_frame None!')
                return None
            return rgb_frame, depth_frame

    def _publish_visualization_rgbd(self, rgb_frame, depth_frame):
        if not self.visualization:
            return
        frame_pipeline = self.visualization.get("frame_pipeline")
        if frame_pipeline is not None:
            frame_pipeline.submit_frame("rgb_latest", rgb_frame)


# def main(args=None):
#     rclpy.init(args=args)
#     node = RGBDClient()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


# def list_devices():
#     ctx = rs.context()
#     devices = ctx.query_devices()
#     if len(devices) == 0:
#         print("❌ 未检测到任何 RealSense 设备。")
#         return None

#     print(f"✅ 检测到 {len(devices)} 个 RealSense 设备：")
#     for i, dev in enumerate(devices):
#         print(f"  [{i}] Name: {dev.get_info(rs.camera_info.name)}")
#         print(f"      Serial: {dev.get_info(rs.camera_info.serial_number)}")
#         print(f"      Firmware: {dev.get_info(rs.camera_info.firmware_version)}")
#         print(f"      Product Line: {dev.get_info(rs.camera_info.product_line)}")
#         print(f"      USB Type: {dev.get_info(rs.camera_info.usb_type_descriptor)}")
#     return devices

# if __name__ == '__main__':
#     list_devices()
#     main()
