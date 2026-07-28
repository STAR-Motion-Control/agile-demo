#!/usr/bin/env python
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
import math, os
from utils.utils import ModelServiceError, convert_image_depth2localaction
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
from .path_planner import PathPlanner
from visualization.adapters import PlannerVisualizationAdapter

logger = logging.getLogger(__name__)

NAVDP_ACTION_METADATA_MARKER = -1.0

class ActionPlanner(Node):
    def __init__(self, cfg, img, origin, resolution, visualization=None):
        super().__init__('action_planner_client')
        self.cfg = cfg
        self.visualization = visualization
        self.viz = PlannerVisualizationAdapter(visualization=visualization)
        self.path_planner = PathPlanner(img, origin, resolution)
        self.action_pub = self.create_publisher(Path, '/planned_action', 1)
        self.bridge = CvBridge()

        self.rgbd_group = MutuallyExclusiveCallbackGroup()
        self.replanning_group = MutuallyExclusiveCallbackGroup()
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
        self.position_subscription = self.create_subscription(PoseStamped, self.cfg.localization.position_topic, self.position_callback, qos_profile, callback_group=self.position_group)

        self.rgb_frame = None
        self.depth_frame = None
        self.current_position = None

        self.rgbd_lock = Lock()
        self.position_lock = Lock()
        self.navigation_lock = Lock()
        self._navigation_active = False

        self.max_path_threshold = int(cfg.local_planner.line_segment_planner.max_path_threshold)
        self.interval = int(cfg.local_planner.line_segment_planner.interval)
        self.final_position_tolerance = self.resolve_final_position_tolerance(cfg)
        self.near_goal_yaw_only_distance = self.resolve_near_goal_yaw_only_distance(cfg)
        self.near_goal_alignment_distance = float(
            self.config_get(
                cfg.local_planner.line_segment_planner,
                "near_goal_alignment_distance",
                1.0,
            )
        )
        self.localplanner_interval = int(cfg.local_planner.model_planner.interval)
        self.max_action_allowed = int(cfg.local_planner.model_planner.max_action_allowed)
        self.model_planner_timeout = float(
            self.config_get(cfg.local_planner.model_planner, "action_timeout", 8.0)
        )
        self._navdp_plan_sequence = 0
        self.camera_intrinsics = self._normalize_camera_intrinsics(
            cfg.local_planner.model_planner.camera_intrinsics
        )

        self.discrete_action_map = {
            1: (0.0, 0.25),
            2: (math.pi/12, 0.0),
            3: (-math.pi/12, 0.0)
        }

        area_cfg = self.cfg.only_global_planner_area
        self.only_global_planner_area_polygons = self.normalize_polygons(
            self.config_get(area_cfg, "polygons", self.config_get(area_cfg, "polygon", []))
        )


        self.srv = self.create_service(Trigger, '/trigger_replanning', self.replanning_trigger_call_back, callback_group=self.replanning_group)

    def set_navigation_active(self, active):
        with self.navigation_lock:
            self._navigation_active = bool(active)

    def is_navigation_active(self):
        with self.navigation_lock:
            return self._navigation_active

    @classmethod
    def resolve_final_position_tolerance(cls, cfg):
        closed_loop_cfg = getattr(getattr(cfg, "motion_control", None), "closed_loop", None)
        return float(cls.config_get(closed_loop_cfg, "final_position_tolerance", 0.15))

    @classmethod
    def resolve_near_goal_yaw_only_distance(cls, cfg):
        line_segment_cfg = getattr(getattr(cfg, "local_planner", None), "line_segment_planner", None)
        configured_distance = float(cls.config_get(line_segment_cfg, "near_goal_yaw_only_distance", 0.12))
        final_position_tolerance = cls.resolve_final_position_tolerance(cfg)
        return max(configured_distance, final_position_tolerance)
    
    def position_callback(self, msg):
        try:
            position = msg.pose.position
            orientation = msg.pose.orientation
            current_position = self.quaternion2yaw(position, orientation)
            with self.position_lock:
                self.current_position = current_position
            
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
    

    def replanning_trigger_call_back(self, request, response):
        logger.info("replanning_trigger callback triggered")
        if not self.is_navigation_active():
            response.success = False
            response.message = 'navigation inactive, skip replanning'
            return response
        try:
            planned = self.plan_action()
        except Exception as exc:
            logger.exception("Replanning callback failed, keep executor alive: %s", exc)
            planned = False
        response.success = bool(planned)
        response.message = 'planned new path!' if planned else 'failed to plan a new path'
        return response
    
    def plan_action(self, goal = None):
        if not self.is_navigation_active():
            logger.info("Navigation inactive, skip planning request.")
            return False
        if goal is not None:
            assert len(goal) == 3, f'Goal must be in the format of [x, y, theta]! but got length = {len(goal)}'
            self.goal = goal
        else:
            goal = self.goal
        with self.position_lock:
            current_position = self.current_position
        if current_position is None:
            logger.warning("Current position unavailable, skip planning")
            return False
        self.path_planner.set_pose(*current_position)
        return self.get_action(*goal)
    

    def path_to_actions(self, path, goal_theta = None):
        actions = []
        theta0 = path[0][-1]
        for i in range(len(path)-1):
            x0, y0, _ = path[i]
            x1, y1, _ = path[i+1]

            expected_theta = math.atan2(y1 - y0, x1 - x0)
            delta_theta = self.normalize_angle(expected_theta - theta0)
            dist = math.hypot(x1 - x0, y1 - y0)
            actions.append((delta_theta, dist))
            theta0 = expected_theta
        if goal_theta is not None:
            delta_theta = self.normalize_angle(goal_theta - theta0)
            actions.append((delta_theta, 0.0))
        return actions

    def actions_to_local_path(self, actions):
        x, y, theta = self.path_planner.current_position
        local_path = [[x, y]]
        for delta_theta, dist in actions:
            theta = self.normalize_angle(theta + delta_theta)
            x += dist * math.cos(theta)
            y += dist * math.sin(theta)
            local_path.append([x, y])
        return local_path

    @staticmethod
    def _normalize_camera_intrinsics(camera_intrinsics):
        if camera_intrinsics is None:
            return None
        if len(camera_intrinsics) != 3:
            raise ValueError("camera_intrinsics must have 3 rows")
        normalized_intrinsics = []
        for row in camera_intrinsics:
            if len(row) != 3:
                raise ValueError("camera_intrinsics rows must have 3 values")
            normalized_intrinsics.append([float(value) for value in row])
        return normalized_intrinsics

    def _publish_navdp_input_frame(self, rgb_frame):
        if not self.visualization:
            return
        frame_pipeline = self.visualization.get("frame_pipeline")
        if frame_pipeline is None:
            return
        frame_pipeline.submit_frame("rgb_navdp", rgb_frame)

    def _next_navdp_plan_id(self):
        current_sequence = getattr(self, "_navdp_plan_sequence", 0) + 1
        self._navdp_plan_sequence = current_sequence
        return f"navdp-{current_sequence}"
    
    def get_action_from_action_model(self, path_world, select_localplanner_interval):
        if not self.is_navigation_active():
            logger.info("Navigation inactive, skip local planner publish.")
            return None
        rgb_frame = None
        depth_frame = None
        logger.info(f"Localplanner triggered!", extra={'color': 'BLUE'})
        with self.rgbd_lock:
            if self.rgb_frame is not None and self.depth_frame is not None:
                rgb_frame = self.rgb_frame
                depth_frame = self.depth_frame
            else:
                logger.error(f"Get rgb_frame None!")
                return None
        self._publish_navdp_input_frame(rgb_frame)
        
        x0, y0, theta0 = self.path_planner.current_position
        plan_pose = (float(x0), float(y0), float(theta0))
        plan_id = self._next_navdp_plan_id()
        plan_stamp = self.get_clock().now().to_msg()
        x1, y1, _ = path_world[select_localplanner_interval]
        abs_distance = math.hypot(x1 - x0, y1 - y0)
        expected_theta = math.atan2(y1 - y0, x1 - x0)
        delta_theta = self.normalize_angle(expected_theta - theta0)
        relative_x = abs_distance*math.cos(delta_theta)
        relative_y = abs_distance*math.sin(delta_theta)
        goal = [relative_x, relative_y]
        # logger.info(f"relative point goal = {goal}", extra={'color': 'BLUE'})
        try:
            full_action_sequence = convert_image_depth2localaction(
                self.cfg.local_planner.model_planner.action_endpoint,
                rgb_frame,
                depth_frame,
                goal,
                self.cfg.local_planner.model_planner.use_discrete_action,
                timeout=self.model_planner_timeout,
            )
        except ModelServiceError as exc:
            logger.warning("NavDP request failed, falling back to line-segment planner: %s", exc)
            self.viz.append_event(
                "navdp_request_failed",
                level="warning",
                payload={"message": str(exc)},
            )
            return None
        if not full_action_sequence:
            logger.error(f"localplanner returned empty action sequence!")
            return None
        executed_action_sequence = full_action_sequence[:self.max_action_allowed]
        logger.info(
            f"navdp preview actions = {full_action_sequence}, executed actions = {executed_action_sequence}",
            extra={'color': 'BLUE'},
        )
        action_sequence2publish = Path()
        action_sequence2publish.header.frame_id = f"map|planner=navdp|plan_id={plan_id}"
        action_sequence2publish.header.stamp = plan_stamp
        for index, p in enumerate(executed_action_sequence):
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.header.stamp = plan_stamp
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            if index == 0:
                pose.pose.position.z = NAVDP_ACTION_METADATA_MARKER
                pose.pose.orientation.x = plan_pose[0]
                pose.pose.orientation.y = plan_pose[1]
                pose.pose.orientation.z = plan_pose[2]
            pose.pose.orientation.w = 0.0
            action_sequence2publish.poses.append(pose)

        if not self.is_navigation_active():
            logger.info("Navigation inactive before local planner publish, drop actions.")
            return None
        self.action_pub.publish(action_sequence2publish)
        self.viz.append_event(
            "actions_published",
            payload={
                "planner_mode": "navdp",
                "actions": executed_action_sequence,
                "action_limit": self.max_action_allowed,
                "preview_action_count": len(full_action_sequence),
                "plan_id": plan_id,
                "plan_pose": {
                    "x": plan_pose[0],
                    "y": plan_pose[1],
                    "theta": plan_pose[2],
                },
            },
        )
        return {
            "preview_actions": full_action_sequence,
            "executed_actions": executed_action_sequence,
            "camera_image_size": [int(rgb_frame.shape[1]), int(rgb_frame.shape[0])],
            "plan_id": plan_id,
            "plan_pose": plan_pose,
        }

    def _line_segment_waypoints(
        self,
        path_world,
        select_interval,
        select_waypoint,
        yaw_only=False,
        include_final_position=False,
    ):
        selected_path = [] if yaw_only else path_world[select_interval:select_waypoint:select_interval]
        if include_final_position and not yaw_only and path_world:
            final_index = min(max(select_waypoint - 1, 0), len(path_world) - 1)
            final_waypoint = path_world[final_index]
            if not selected_path or selected_path[-1][:2] != final_waypoint[:2]:
                selected_path.append(final_waypoint)
        return selected_path

    def _line_segment_actions(
        self,
        path_world,
        select_interval,
        select_waypoint,
        goal_theta,
        yaw_only=False,
        include_final_position=False,
    ):
        selected_path = self._line_segment_waypoints(
            path_world,
            select_interval,
            select_waypoint,
            yaw_only=yaw_only,
            include_final_position=include_final_position,
        )
        return self.path_to_actions([self.path_planner.current_position] + selected_path, goal_theta)

    def get_action_from_line_segment_planner(
        self,
        path_world,
        select_interval,
        select_waypoint,
        goal_theta,
        moving_flag,
        final_goal=None,
        yaw_only=False,
        include_final_position=False,
    ):
        if not self.is_navigation_active():
            logger.info("Navigation inactive, skip line-segment planner publish.")
            return False
        logger.info(f"line segment planner triggered!", extra={'color': 'GREEN'})
        actions = self._line_segment_actions(
            path_world,
            select_interval,
            select_waypoint,
            goal_theta,
            yaw_only=yaw_only,
            include_final_position=include_final_position,
        )
        if not actions:
            logger.warning("Line-segment planner produced no executable actions.")
            self.viz.append_event(
                "empty_action_sequence",
                level="warning",
                payload={
                    "planner_mode": "line_segment",
                    "moving_flag": moving_flag,
                    "path_length": len(path_world),
                },
            )
            return False
        # logger.info(f"all actions = {actions}", extra={'color': 'GREEN'})
        action_squence = Path()
        action_squence.header.frame_id = "map"
        for p in actions:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.orientation.w = moving_flag
            if moving_flag == 1.0 and final_goal is not None:
                pose.pose.position.z = 1.0
                pose.pose.orientation.x = float(final_goal[0])
                pose.pose.orientation.y = float(final_goal[1])
                pose.pose.orientation.z = float(final_goal[2])
            action_squence.poses.append(pose)

        action_squence.header.stamp = self.get_clock().now().to_msg()
        if not self.is_navigation_active():
            logger.info("Navigation inactive before line-segment publish, drop actions.")
            return False
        self.action_pub.publish(action_squence)
        self.viz.append_event(
            "actions_published",
            payload={"planner_mode": "line_segment", "actions": actions},
        )
        return True

    def in_only_global_planner_area(self, x, y):
        return any(
            self.point_in_polygon(x, y, polygon)
            for polygon in self.only_global_planner_area_polygons
            if len(polygon) >= 3
        )
    
    def get_action(self, x, y, theta):
        if not self.is_navigation_active():
            logger.info("Navigation inactive, skip path planning.")
            return False
        path_world, path_pixel = self.path_planner.get_path(x, y, theta)
        if path_world is None or path_pixel is None:
            logger.warning(f"No valid path found for goal {[x, y, theta]}")
            self.viz.append_event(
                "global_path_computed",
                level="warning",
                payload={
                    "success": False,
                    "mode": "unavailable",
                    "global_path": [],
                    "waypoints": [],
                    "local_goal": None,
                    "actions": [],
                    "message": "path planning failed",
                },
            )
            return False
        select_waypoint = self.max_path_threshold
        select_interval = self.interval
        select_localplanner_interval = self.localplanner_interval
        moving_flag = 0.0 # 0 -> still moving  1 -> near the goal(the final move)
        goal_theta = None

        with self.position_lock:
            if self.current_position is None:
                return
            x0, y0, _ = self.current_position

        distance_to_goal = math.hypot(x - x0, y - y0)
        final_position_tolerance = float(getattr(self, "final_position_tolerance", 0.15))
        near_goal_by_distance = distance_to_goal <= float(
            getattr(self, "near_goal_alignment_distance", 1.0)
        )
        if len(path_world) <= select_waypoint or near_goal_by_distance:
            select_waypoint = len(path_world)
            moving_flag = 1.0
            if distance_to_goal <= final_position_tolerance:
                goal_theta = theta
        if len(path_world) <= select_interval:
            select_interval = max(len(path_world) - 1, 1)
        if len(path_world) <= select_localplanner_interval:
            select_localplanner_interval = max(len(path_world) - 1, 0)

        near_goal_yaw_only_distance = float(getattr(self, "near_goal_yaw_only_distance", 0.12))
        yaw_only_distance = min(near_goal_yaw_only_distance, final_position_tolerance)
        yaw_only_final_goal = (
            moving_flag == 1.0
            and distance_to_goal <= yaw_only_distance
        )
        include_final_position = moving_flag == 1.0 and goal_theta is None
        line_segment_waypoints = self._line_segment_waypoints(
            path_world,
            select_interval,
            select_waypoint,
            yaw_only=yaw_only_final_goal,
            include_final_position=include_final_position,
        )
        if not self.is_navigation_active():
            logger.info("Navigation inactive after path planning, drop planner update.")
            return False
        self.viz.append_event(
            "global_path_computed",
            payload={
                "success": True,
                "global_path": path_world,
                "waypoints": line_segment_waypoints,
                "distance_to_goal": distance_to_goal,
                "final_position_tolerance": final_position_tolerance,
                "final_yaw_enabled": goal_theta is not None,
                "include_final_position": include_final_position,
            },
        )
        
        if not self.cfg.local_planner.enbale_model_planner or moving_flag == 1.0:
            published = self.get_action_from_line_segment_planner(
                path_world,
                select_interval,
                select_waypoint,
                goal_theta,
                moving_flag,
                final_goal=(x, y, theta) if moving_flag == 1.0 else None,
                yaw_only=yaw_only_final_goal,
                include_final_position=include_final_position,
            )
            if not published:
                return False
            self.path_planner.plot(np.array(path_pixel).T)
            if not self.is_navigation_active():
                logger.info("Navigation inactive before visualization plan update, drop planner result.")
                return False
            self.viz.publish_plan(
                mode="line_segment",
                path_world=path_world,
                waypoints=line_segment_waypoints,
                local_path=[],
                local_goal=None,
                actions=self._line_segment_actions(
                    path_world,
                    select_interval,
                    select_waypoint,
                    goal_theta,
                    yaw_only=yaw_only_final_goal,
                    include_final_position=include_final_position,
                ),
                action_limit=None,
            )
            return True
        elif self.cfg.only_global_planner_area.enable_only_global_planner_area and self.in_only_global_planner_area(x0, y0):
            
            published = self.get_action_from_line_segment_planner(
                path_world,
                select_interval,
                select_waypoint,
                goal_theta,
                moving_flag,
                final_goal=(x, y, theta) if moving_flag == 1.0 else None,
                yaw_only=yaw_only_final_goal,
                include_final_position=include_final_position,
            )
            if not published:
                return False
            self.path_planner.plot(np.array(path_pixel).T)
            if not self.is_navigation_active():
                logger.info("Navigation inactive before visualization plan update, drop planner result.")
                return False
            self.viz.publish_plan(
                mode="line_segment",
                path_world=path_world,
                waypoints=line_segment_waypoints,
                local_path=[],
                local_goal=None,
                actions=self._line_segment_actions(
                    path_world,
                    select_interval,
                    select_waypoint,
                    goal_theta,
                    yaw_only=yaw_only_final_goal,
                    include_final_position=include_final_position,
                ),
                action_limit=None,
            )
            return True
        else:
            navdp_result = self.get_action_from_action_model(path_world, select_localplanner_interval)
            if navdp_result is not None:
                preview_actions = navdp_result["preview_actions"]
                executed_actions = navdp_result["executed_actions"]
                self.path_planner.plot(np.array(path_pixel).T, interval=self.localplanner_interval-1, max_threshold=self.localplanner_interval)
                if not self.is_navigation_active():
                    logger.info("Navigation inactive before visualization plan update, drop planner result.")
                    return False
                self.viz.publish_plan(
                    mode="navdp",
                    path_world=path_world,
                    waypoints=path_world[: self.max_path_threshold],
                    local_path=self.actions_to_local_path(preview_actions),
                    local_goal=[relative for relative in path_world[select_localplanner_interval][:2]],
                    actions=executed_actions,
                    action_limit=self.max_action_allowed,
                    preview_actions=preview_actions,
                    camera_intrinsics=self.camera_intrinsics,
                    camera_image_size=navdp_result["camera_image_size"],
                )
                # self.path_planner.save_fig()
                return True
            else:
                published = self.get_action_from_line_segment_planner(
                    path_world,
                    select_interval,
                    select_waypoint,
                    goal_theta,
                    moving_flag,
                    final_goal=(x, y, theta) if moving_flag == 1.0 else None,
                    yaw_only=yaw_only_final_goal,
                    include_final_position=include_final_position,
                )
                if not published:
                    return False
                self.path_planner.plot(np.array(path_pixel).T)
                if not self.is_navigation_active():
                    logger.info("Navigation inactive before visualization plan update, drop planner result.")
                    return False
                self.viz.publish_plan(
                    mode="line_segment",
                    path_world=path_world,
                    waypoints=line_segment_waypoints,
                    local_path=[],
                    local_goal=None,
                    actions=self._line_segment_actions(
                        path_world,
                        select_interval,
                        select_waypoint,
                        goal_theta,
                        yaw_only=yaw_only_final_goal,
                        include_final_position=include_final_position,
                    ),
                    action_limit=None,
                )
                return True

    @staticmethod
    def quaternion2yaw(position, orientation):
        x, y, z = position.x, position.y, position.z
        o_x, o_y, o_z, o_w = orientation.x, orientation.y, orientation.z, orientation.w
        R_base = R.from_quat(np.array([o_x, o_y, o_z, o_w ])).as_matrix()
        theta = np.arctan2(R_base[1, 0], R_base[0, 0])
        return x, y, theta
    
    @staticmethod
    def normalize_angle(angle):
        angle = (angle + math.pi)%(2*math.pi) - math.pi
        return angle

    @staticmethod
    def config_get(config, key, default=None):
        if hasattr(config, "get") and callable(config.get):
            return config.get(key, default)

        return getattr(config, key, default)

    @staticmethod
    def normalize_polygons(value):
        if value is None:
            return []

        value = list(value)
        if not value:
            return []

        first = value[0]
        if len(first) >= 2 and all(isinstance(item, (int, float)) for item in first[:2]):
            return [value]

        return [list(polygon) for polygon in value]

    @staticmethod
    def point_in_polygon(x, y, polygon):
        inside = False
        n = len(polygon)
        px1, py1 = polygon[0]
        for i in range(n + 1):
            px2, py2 = polygon[i % n]
            if y > min(py1, py2):
                if y <= max(py1, py2):
                    if x <= max(px1, px2):
                        if py1 != py2:
                            xinters = (y - py1) * (px2 - px1) / (py2 - py1) + px1
                        if px1 == px2 or x <= xinters:
                            inside = not inside

            px1, py1 = px2, py2
        # if inside == True:
        #     print("inside-----------------------------------")
        # else:
        #     print("not inside+++++++++++++++++++++++++++++++")
        return inside
