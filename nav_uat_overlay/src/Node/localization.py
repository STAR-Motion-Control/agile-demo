#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path, Odometry
from std_srvs.srv import Trigger
import math, os
from utils.utils import ModelServiceError, convert_image2pose
from cv_bridge import CvBridge
import cv2
from threading import Lock
from sensor_msgs.msg import Image
import numpy as np
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
import logging
from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)

class LocalizationClient(Node):
    CONTINUOUS_VPR_MODE = "continuous_vpr"
    VPR_ONCE_ODOM_MODE = "vpr_once_odom"

    def __init__(self, cfg, visualization=None):
        super().__init__('localization_client')
        self.cfg = cfg
        self.visualization = visualization
        self.bridge = CvBridge()

        self.rgbd_group = MutuallyExclusiveCallbackGroup()
        self.odom_group = MutuallyExclusiveCallbackGroup()
        self.localization_group = MutuallyExclusiveCallbackGroup()
        self.position_group = MutuallyExclusiveCallbackGroup()

        self.rgb_subscription = Subscriber(self, Image, cfg.rgbd_server.subscrib_topic.rgb, qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
        self.depth_subscription = Subscriber(self, Image, cfg.rgbd_server.subscrib_topic.depth, qos_profile=qos_profile_sensor_data, callback_group=self.rgbd_group)
        self.timesynchronizer = ApproximateTimeSynchronizer([self.rgb_subscription, self.depth_subscription], queue_size=10, slop=0.05)
        self.timesynchronizer.registerCallback(self.rgbd_callback)

        qos_profile = QoSProfile(
        depth = 1, 
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        )
        self.odom_subscription = self.create_subscription(Odometry, '/dog_odom', self.odom_callback, qos_profile, callback_group=self.odom_group)
        self.T_odom2map = None
        self.T_base2odom = None

        self.rgb_frame = None
        self.depth_frame = None
        
        self.rgbd_lock = Lock()
        self.odom_lock = Lock()
        self.transform_lock = Lock()
        self.navigation_goal_lock = Lock()
        self.navigation_goal = None
        self.vpr_transform_frozen_near_goal = False
        self.near_goal_vpr_freeze_enabled, self.near_goal_vpr_freeze_distance = (
            self._load_near_goal_vpr_freeze_config(cfg)
        )
        
        self.localization_mode = self.config_get(
            self.cfg.localization, "mode", self.CONTINUOUS_VPR_MODE
        )
        supported_modes = {self.CONTINUOUS_VPR_MODE, self.VPR_ONCE_ODOM_MODE}
        if self.localization_mode not in supported_modes:
            raise ValueError(
                f"Unsupported localization mode: {self.localization_mode!r}. "
                f"Expected one of {sorted(supported_modes)}"
            )

        self.global_localization_timer = None
        if self.cfg.localization.enable_vpr:
            self.global_localization_timer = self.create_timer(
                self.cfg.localization.vpr.time_interval, self.localization_callback, callback_group=self.localization_group
            )
        self.publish_position_timer = self.create_timer(
            self.cfg.localization.position_update_interval, self.publish_position_callback, callback_group=self.position_group
        )
        self.position_publisher = self.create_publisher(PoseStamped, self.cfg.localization.position_topic, qos_profile)
        self.vpr_timeout = float(
            self.config_get(self.config_get(self.cfg.localization, "vpr", {}), "timeout", 5.0)
        )

        self.img_id = 0

    def get_position(self,):
        with self.odom_lock:
            if self.T_base2odom is None:
                logger.warning("T_base2odom not ready, skip replanning service")
                return None
            T_base2odom = self.T_base2odom

        with self.transform_lock:
            if self.T_odom2map is None:
                logger.warning("T_odom2map not ready, skip replanning service")
                return None
            T_odom2map = self.T_odom2map

        estimated_base2map_from_odom = self.T_transform(T_odom2map, T_base2odom)
        current_position = self.T2x_y_theta(estimated_base2map_from_odom)
        return current_position

    @classmethod
    def _load_near_goal_vpr_freeze_config(cls, cfg):
        localization_cfg = getattr(cfg, "localization", None)
        freeze_cfg = cls.config_get(localization_cfg, "near_goal_vpr_freeze", None)
        enabled = bool(cls.config_get(freeze_cfg, "enable", True))

        closed_loop_cfg = getattr(getattr(cfg, "motion_control", None), "closed_loop", None)
        final_position_tolerance = float(
            cls.config_get(closed_loop_cfg, "final_position_tolerance", 0.12)
        )
        default_distance = max(0.5, final_position_tolerance * 4.0)
        distance = float(cls.config_get(freeze_cfg, "distance", default_distance))
        return enabled, max(distance, final_position_tolerance)

    def set_navigation_goal(self, goal):
        if goal is None:
            self.clear_navigation_goal()
            return
        if len(goal) != 3:
            raise ValueError("navigation goal must be [x, y, theta]")
        with self.navigation_goal_lock:
            self.navigation_goal = (float(goal[0]), float(goal[1]), float(goal[2]))
            self.vpr_transform_frozen_near_goal = False

    def clear_navigation_goal(self):
        with self.navigation_goal_lock:
            was_frozen = self.vpr_transform_frozen_near_goal
            self.navigation_goal = None
            self.vpr_transform_frozen_near_goal = False
        if was_frozen:
            self._append_visualization_event(
                "vpr_updates_unfrozen",
                payload={"reason": "navigation_finished"},
            )

    def _append_visualization_event(self, event_name, level="info", payload=None):
        if not self.visualization:
            return
        state_hub = self.visualization.get("state_hub")
        if state_hub is None:
            return
        state_hub.append_event(
            event_name,
            level=level,
            task_id=state_hub.get_active_task_id(),
            payload=payload or {},
        )

    def _maybe_freeze_vpr_near_goal(self, current_position):
        if not self.near_goal_vpr_freeze_enabled or current_position is None:
            return False
        with self.navigation_goal_lock:
            if self.navigation_goal is None:
                return False
            if self.vpr_transform_frozen_near_goal:
                return True
            goal_x, goal_y, _ = self.navigation_goal
            distance_to_goal = math.hypot(goal_x - current_position[0], goal_y - current_position[1])
            if distance_to_goal > self.near_goal_vpr_freeze_distance:
                return False
            self.vpr_transform_frozen_near_goal = True
            freeze_distance = self.near_goal_vpr_freeze_distance
        self._append_visualization_event(
            "vpr_updates_frozen_near_goal",
            payload={
                "distance_to_goal": distance_to_goal,
                "freeze_distance": freeze_distance,
                "pose": {
                    "x": float(current_position[0]),
                    "y": float(current_position[1]),
                    "theta": float(current_position[2]),
                },
            },
        )
        return True

    def _vpr_transform_update_frozen(self):
        with self.navigation_goal_lock:
            return self.vpr_transform_frozen_near_goal
    
    def publish_position_callback(self,):
        current_position = self.get_position()
        if current_position is not None:
            self._maybe_freeze_vpr_near_goal(current_position)
            x, y, theta = current_position
            if self.visualization:
                state_hub = self.visualization.get("state_hub")
                if state_hub is not None:
                    state_hub.update_pose(pose={"x": x, "y": y, "theta": theta})
            msg = PoseStamped()
            msg.header.frame_id = "map"
            msg.pose.position.x = float(x)
            msg.pose.position.y = float(y)
            msg.pose.position.z = 0.0

            qx, qy, qz, qw = 0.0, 0.0, math.sin(theta / 2.0), math.cos(theta / 2.0)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            self.position_publisher.publish(msg)

    def odom_callback(self, msg):
        try:
            position = msg.pose.pose.position
            orientation = msg.pose.pose.orientation
            T_base2odom = self.quaternion2T(position,orientation)
            with self.odom_lock:
                self.T_base2odom = T_base2odom
            
        except Exception as e:
            logger.error(f"Failed to process odom information: {str(e)}")
    
    def rgbd_callback(self, rgb_msg, depth_msg):
        try:
            rgb_frame = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_frame = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')
            # print("RGB shape:", rgb_frame.shape, "Depth shape:", depth_frame.shape)
            with self.rgbd_lock:
                self.rgb_frame = rgb_frame
                self.depth_frame = depth_frame
        except Exception as e:
            logger.error(f"Failed to process rgb or depth frame: {str(e)}")
    
    def localization_callback(self,):
        rgb_frame = None
        logger.info(f"Localization service callback triggered, robot_id={self.cfg.localization.vpr.robot_id}")
        with self.rgbd_lock:
            if self.rgb_frame is None:
                logger.error(f"Get rgb_frame None, skip localization")
                return 
            rgb_frame = self.rgb_frame
        with self.odom_lock:
            if self.T_base2odom is None:
                logger.error("T_base2odom not ready, skip localization")
                return
            T_base2odom = self.T_base2odom

        if self.visualization:
            state_hub = self.visualization.get("state_hub")
            if state_hub is not None:
                state_hub.append_event(
                    "vpr_request_started",
                    task_id=state_hub.get_active_task_id(),
                )

        # this operation is relatively time-consuming across different devices
        start_time = self.get_clock().now().nanoseconds / 1e6
        try:
            x, y, theta = convert_image2pose(
                self.cfg.localization.vpr.endpoint,
                rgb_frame,
                depth_frame=None,
                timeout=self.vpr_timeout,
                robot_id=self.cfg.localization.vpr.robot_id
            )
        except ModelServiceError as exc:
            logger.warning("VPR localization failed, keep last transform: %s", exc)
            if self.visualization:
                state_hub = self.visualization.get("state_hub")
                if state_hub is not None:
                    state_hub.append_event(
                        "vpr_request_failed",
                        level="warning",
                        task_id=state_hub.get_active_task_id(),
                        payload={"message": str(exc)},
                    )
            return
        end_time = self.get_clock().now().nanoseconds / 1e6

        self._publish_visualization_localization(
            rgb_frame=rgb_frame,
            pose={"x": x, "y": y, "theta": theta},
            latency_ms=end_time - start_time,
        )

        current_position = self.get_position()
        frozen = self._maybe_freeze_vpr_near_goal(current_position)
        if frozen or self._vpr_transform_update_frozen():
            distance_to_goal = None
            with self.navigation_goal_lock:
                goal = self.navigation_goal
            if current_position is not None and goal is not None:
                distance_to_goal = math.hypot(goal[0] - current_position[0], goal[1] - current_position[1])
            self._append_visualization_event(
                "vpr_transform_update_skipped",
                payload={
                    "reason": "near_goal_freeze",
                    "distance_to_goal": distance_to_goal,
                    "vpr_pose": {"x": x, "y": y, "theta": theta},
                },
            )
            return

        T_odom2map = self.T_transform(self.x_y_theta2T(x, y, theta), np.linalg.inv(T_base2odom))
        with self.transform_lock:
            self.T_odom2map = T_odom2map
        self._stop_vpr_after_initial_fix()

        if self.cfg.rgbd_server.visualization:
            os.makedirs(f'./img/localization_rgb',exist_ok=True)
            cv2.imwrite(f'./img/localization_rgb/{self.img_id}.jpg', rgb_frame)
            self.img_id += 1
    
    @staticmethod
    def quaternion2T(position, orientation):
        x, y, z = position.x, position.y, position.z
        o_x, o_y, o_z, o_w = orientation.x, orientation.y, orientation.z, orientation.w
        R_base = R.from_quat(np.array([o_x, o_y, o_z, o_w ])).as_matrix()
        T = np.eye(4)
        T[0:3, 0:3] = R_base
        T[0:3, 3] = [x, y, 0]
        return T

    @staticmethod
    def x_y_theta2T(x, y, theta):
        c, s = np.cos(theta), np.sin(theta)
        T = np.eye(4)
        T[0:3, 0:3] = [[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]]
        T[0:3, 3] = [x, y, 0]
        return T

    @staticmethod
    def T2x_y_theta(T):
        x, y = T[0:2, 3]
        theta = np.arctan2(T[1, 0], T[0, 0])
        return x, y, theta
    
    @staticmethod
    def T_transform(matrix_A, matrix_B):
        return matrix_A @ matrix_B

    @staticmethod
    def config_get(config, key, default=None):
        if hasattr(config, "get") and callable(config.get):
            return config.get(key, default)

        return getattr(config, key, default)

    def _stop_vpr_after_initial_fix(self):
        if (
            self.localization_mode == self.VPR_ONCE_ODOM_MODE
            and self.global_localization_timer is not None
        ):
            self.global_localization_timer.cancel()
            logger.info("Initial VPR fix acquired; subsequent positions use odometry only")

    def _publish_visualization_localization(self, rgb_frame, pose, latency_ms):
        if not self.visualization:
            return
        state_hub = self.visualization.get("state_hub")
        if state_hub is not None:
            state_hub.update_pose(vpr_pose=pose)
            state_hub.append_event(
                "vpr_result",
                task_id=state_hub.get_active_task_id(),
                payload={"pose": pose, "latency_ms": latency_ms},
            )
