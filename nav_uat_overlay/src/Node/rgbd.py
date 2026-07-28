import ctypes
import logging
from threading import Event, Thread

import numpy as np
import rclpy
from camera_runtime import CaptureErrorBackoff
from frame_hub import CameraFrameHub
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from runtime_config import config_get, resolve_capture_depth

logger = logging.getLogger(__name__)


class RGBDClient(Node):
    """Single camera ingress for the navigation process.

    Direct RealSense frames stay in process through ``CameraFrameHub``. Raw ROS
    image publication is opt-in for external consumers and is never used by the
    in-process localization or planner nodes.
    """

    def __init__(self, cfg, visualization=None, frame_hub=None, require_depth=False):
        super().__init__("rgbd_client")
        self.cfg = cfg
        self.visualization = visualization
        self.frame_hub = frame_hub or CameraFrameHub()
        self.require_depth = bool(require_depth)
        self._stop_event = Event()
        self.thread = None
        self.pipeline = None
        self.bridge = None
        self.depth_scale = 1.0
        self._camera_error_backoff = CaptureErrorBackoff()

        rgbd_cfg = cfg.rgbd_server
        self.publish_ros_topics = bool(config_get(rgbd_cfg, "publish_ros_topics", False))
        self.capture_depth = resolve_capture_depth(
            rgbd_cfg,
            require_depth=self.require_depth,
        )

        if rgbd_cfg.launch_rgbd_server:
            self._initialize_direct_camera(rgbd_cfg)
        else:
            self._initialize_ros_input(rgbd_cfg)

        logger.info(
            "RGBD node ready (direct=%s, depth=%s, publish_ros=%s)",
            bool(rgbd_cfg.launch_rgbd_server),
            self.capture_depth,
            self.publish_ros_topics,
        )

    @staticmethod
    def _load_cv_bridge():
        from cv_bridge import CvBridge

        return CvBridge

    @staticmethod
    def _load_image_type():
        from sensor_msgs.msg import Image

        return Image

    @staticmethod
    def _load_realsense():
        try:
            ctypes.CDLL("/lib/aarch64-linux-gnu/libffi.so.7", mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        import pyrealsense2 as rs

        return rs

    def _initialize_direct_camera(self, rgbd_cfg):
        rs = self._load_realsense()
        width = int(config_get(rgbd_cfg, "width", 640))
        height = int(config_get(rgbd_cfg, "height", 480))
        fps = int(config_get(rgbd_cfg, "fps", 15))
        serial_number = str(config_get(rgbd_cfg, "serial_number", "419222302306"))

        self._rs = rs
        self.pipeline = rs.pipeline()
        camera_config = rs.config()
        camera_config.enable_device(serial_number)
        camera_config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self.capture_depth:
            camera_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        pipeline_profile = self.pipeline.start(camera_config)
        if self.capture_depth:
            try:
                self.depth_scale = float(
                    pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
                )
            except Exception as exc:
                self.depth_scale = 0.001
                logger.warning(
                    "Failed to read RealSense depth scale; using %.4f: %s",
                    self.depth_scale,
                    exc,
                )

        self.publisher_rgb = None
        self.publisher_depth = None
        if self.publish_ros_topics:
            Image = self._load_image_type()
            CvBridge = self._load_cv_bridge()
            self.bridge = CvBridge()
            self.publisher_rgb = self.create_publisher(
                Image,
                rgbd_cfg.publish_topic.rgb,
                qos_profile_sensor_data,
            )
            if self.capture_depth:
                self.publisher_depth = self.create_publisher(
                    Image,
                    rgbd_cfg.publish_topic.depth,
                    qos_profile_sensor_data,
                )

        self.thread = Thread(target=self._capture_loop, name="nav_camera_capture", daemon=True)
        self.thread.start()

    def _initialize_ros_input(self, rgbd_cfg):
        Image = self._load_image_type()
        CvBridge = self._load_cv_bridge()
        self.bridge = CvBridge()
        self.rgbd_group = MutuallyExclusiveCallbackGroup()

        if self.capture_depth:
            from message_filters import ApproximateTimeSynchronizer, Subscriber

            self.rgb_subscription = Subscriber(
                self,
                Image,
                rgbd_cfg.subscrib_topic.rgb,
                qos_profile=qos_profile_sensor_data,
                callback_group=self.rgbd_group,
            )
            self.depth_subscription = Subscriber(
                self,
                Image,
                rgbd_cfg.subscrib_topic.depth,
                qos_profile=qos_profile_sensor_data,
                callback_group=self.rgbd_group,
            )
            self.timesynchronizer = ApproximateTimeSynchronizer(
                [self.rgb_subscription, self.depth_subscription],
                queue_size=2,
                slop=0.05,
            )
            self.timesynchronizer.registerCallback(self.rgbd_callback)
            return

        self.rgb_subscription = self.create_subscription(
            Image,
            rgbd_cfg.subscrib_topic.rgb,
            self.rgb_callback,
            qos_profile_sensor_data,
            callback_group=self.rgbd_group,
        )

    def rgb_callback(self, rgb_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            self._accept_frame(rgb_frame, None, source="ros")
        except Exception as exc:
            logger.error("Failed to process RGB frame: %s", exc)

    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
            self._accept_frame(rgb_frame, depth_frame, source="ros")
        except Exception as exc:
            logger.error("Failed to process RGB-D frame: %s", exc)

    def _capture_loop(self):
        try:
            while rclpy.ok() and not self._stop_event.is_set():
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame() if self.capture_depth else None
                    if not color_frame or (self.capture_depth and not depth_frame):
                        if self._wait_after_camera_failure(
                            "Camera returned an incomplete frameset"
                        ):
                            break
                        continue

                    color_image = np.asanyarray(color_frame.get_data()).copy()
                    depth_image = None
                    if depth_frame is not None:
                        depth_image = np.asanyarray(depth_frame.get_data()).copy()

                    self._accept_frame(
                        color_image,
                        depth_image,
                        source="realsense",
                        depth_scale=self.depth_scale,
                    )
                    if self.publish_ros_topics:
                        self._publish_ros_frame(
                            color_image,
                            depth_image,
                            depth_scale=self.depth_scale,
                        )
                    self._camera_error_backoff.reset()
                except Exception as exc:
                    if self._stop_event.is_set():
                        break
                    if self._wait_after_camera_failure(f"Camera error: {exc}"):
                        break
        finally:
            if self.pipeline is not None:
                self.pipeline.stop()
            logger.info("RealSense pipeline stopped.")

    def _wait_after_camera_failure(self, message):
        delay_s, should_log = self._camera_error_backoff.failure()
        if should_log:
            logger.error("%s; retrying in %.2fs", message, delay_s)
        return self._stop_event.wait(delay_s)

    def _accept_frame(self, rgb_frame, depth_frame, *, source, depth_scale=1.0):
        frame = self.frame_hub.publish(
            rgb_frame,
            depth_frame,
            source=source,
            depth_scale=depth_scale,
        )
        self._publish_visualization_rgbd(frame.rgb, frame.depth)
        return frame

    def _publish_ros_frame(self, rgb_frame, depth_frame, depth_scale=1.0):
        ros_image = self.bridge.cv2_to_imgmsg(rgb_frame, encoding="bgr8")
        current_time = self.get_clock().now().to_msg()
        ros_image.header.stamp = current_time
        self.publisher_rgb.publish(ros_image)

        if depth_frame is None or self.publisher_depth is None:
            return
        depth_meters = np.asarray(depth_frame, dtype=np.float32) * float(depth_scale)
        ros_depth = self.bridge.cv2_to_imgmsg(depth_meters, encoding="32FC1")
        ros_depth.header.stamp = current_time
        self.publisher_depth.publish(ros_depth)

    def get_image(self, require_depth=None):
        require_depth = self.require_depth if require_depth is None else bool(require_depth)
        frame = self.frame_hub.latest(require_depth=require_depth)
        if frame is None:
            logger.error("No suitable camera frame is available")
            return None
        depth_frame = frame.depth
        if depth_frame is not None and frame.depth_scale != 1.0:
            depth_frame = np.asarray(depth_frame, dtype=np.float32) * frame.depth_scale
        return frame.rgb, depth_frame

    def _publish_visualization_rgbd(self, rgb_frame, depth_frame):
        if not self.visualization:
            return
        frame_pipeline = self.visualization.get("frame_pipeline")
        if frame_pipeline is not None:
            frame_pipeline.submit_frame("rgb_latest", rgb_frame)

    def shutdown(self, timeout_sec=1.0):
        self._stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=timeout_sec)
