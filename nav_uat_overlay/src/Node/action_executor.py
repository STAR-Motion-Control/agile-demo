import rclpy
from rclpy.node import Node
from threading import Event, Lock, Thread
import time
import math
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from std_srvs.srv import Trigger
from unitree_go.msg._wireless_controller import WirelessController
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import logging
from visualization.adapters import ExecutorVisualizationAdapter
from .motion_backend import CombinedCancelToken, GrootHttpDiscreteBackend

logger = logging.getLogger(__name__)

NAVDP_ACTION_METADATA_MARKER = -1.0
PREEMPTED = "preempted"

class ActionExecutorClient(Node):
    def __init__(self, cfg=None, visualization=None):
        super().__init__('action_excuter_client')
        self.cfg = cfg
        self.viz = ExecutorVisualizationAdapter(visualization=visualization)
        self.subscriber = self.create_subscription(Path, '/planned_action', self._action_callback, qos_profile_sensor_data)
        self.motion_backend = GrootHttpDiscreteBackend(cfg, log=None)
        self.publisher = None
        if not self.motion_backend.enabled:
            self.publisher = self.create_publisher(
                WirelessController, '/wirelesscontroller', 10)
        else:
            logger.info("Action executor uses GR00T HTTP discrete backend.")
        self.motion_control = self._load_motion_control_config(cfg)
        self.pose_lock = Lock()
        self.current_pose = None
        self.current_pose_time = None
        position_topic = self.motion_control["position_topic"]
        pose_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.position_subscription = self.create_subscription(
            PoseStamped,
            position_topic,
            self._position_callback,
            pose_qos,
        )

        logger.info("Action client launched...")

        self.replanning_client = self.create_client(Trigger, '/trigger_replanning')
        while not self.replanning_client.wait_for_service(timeout_sec=5.0):
            logger.info('Waiting for localization service...')
        self.task = 'StepControl '
        self._stop_flag = Event()
        self._pause_flag = Event()
        self._finish_flag = Event()
        self._new_action_event = Event()
        self._preempt_action_event = Event()
        self._shutdown_flag = Event()
        self.lock = Lock()
        self._pause_reason = None
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
        self.actions = []
        self.action_plan_metadata = None
        self._dry_run = False
        self._navigation_active = False
        self.thread = Thread(target=self.execute_action)
        self.thread.daemon = True
        self.thread.start()

    @classmethod
    def _load_motion_control_config(cls, cfg):
        defaults = {
            "enable": True,
            "position_topic": "/global_position",
            "pose_timeout": 0.5,
            "position_tolerance": 0.06,
            "forward_timeout_acceptance_tolerance": 0.12,
            "forward_timeout_acceptance_heading_tolerance": 0.1,
            "angle_tolerance": 0.05,
            "final_position_tolerance": 0.15,
            "final_angle_tolerance": 0.2,
            "final_pose_timeout": 4.0,
            "final_pose_max_forward_speed": 0.18,
            "final_pose_max_lateral_command": 0.30,
            "final_pose_max_angular_speed": 0.35,
            "final_pose_min_linear_command": 0.03,
            "final_pose_min_angular_command": 0.04,
            "max_forward_speed": 0.50,
            "max_lateral_command": 0.30,
            "max_angular_speed": 0.60,
            "min_linear_command": 0.08,
            "min_angular_command": 0.12,
            "forward_command_scale": 2.6,
            "backward_command_scale": 1.9,
            "left_command_scale": 6.0,
            "right_command_scale": 7.5,
            "positive_angular_command_scale": 1.23,
            "negative_angular_command_scale": 1.06,
            "angular_stop_prediction_time": 0.8,
            "angular_stop_velocity_tolerance": 0.03,
            "angular_stop_stable_time": 0.25,
            "angular_velocity_filter_alpha": 0.35,
            "linear_stop_prediction_time": 1.05,
            "linear_stop_velocity_tolerance": 0.03,
            "linear_stop_stable_time": 0.25,
            "linear_velocity_filter_alpha": 0.35,
            "waypoint_turn_in_place_threshold": 0.5,
            "waypoint_max_forward_speed": 0.50,
            "waypoint_min_forward_command": 0.08,
            "waypoint_timeout_per_meter": 4.0,
            "forward_timeout_per_meter": 8.0,
            "rotate_timeout_per_rad": 3.0,
            "min_timeout": 1.0,
            "stop_command_repeat": 8,
            "stop_command_interval": 0.02,
            "final_settle_time": 0.25,
            "final_validation_samples": 3,
            "final_validation_interval": 0.05,
            "final_discrete_alignment_enable": True,
            "final_discrete_alignment_passes": 2,
            "final_discrete_alignment_position_tolerance": None,
            "final_discrete_alignment_angle_tolerance": None,
            "final_discrete_alignment_min_translation": 0.02,
            "final_discrete_alignment_min_angle": 0.03,
            "final_discrete_alignment_settle_time": 0.2,
            "final_pose_jump_filter_enable": True,
            "final_pose_max_position_jump": 0.15,
            "final_pose_max_angle_jump": 0.35,
            "plan_stale_position_tolerance": 0.25,
            "plan_stale_angle_tolerance": 0.5,
            "plan_stale_max_age": 2.0,
        }
        if cfg is None:
            return defaults

        localization_cfg = getattr(cfg, "localization", None)
        if localization_cfg is not None:
            defaults["position_topic"] = str(getattr(localization_cfg, "position_topic", defaults["position_topic"]))

        motion_cfg = getattr(cfg, "motion_control", None)
        closed_loop_cfg = getattr(motion_cfg, "closed_loop", None) if motion_cfg is not None else None
        if closed_loop_cfg is None:
            return defaults

        for key, default_value in list(defaults.items()):
            value = getattr(closed_loop_cfg, key, default_value)
            if isinstance(default_value, bool):
                defaults[key] = bool(value)
            elif isinstance(default_value, str):
                defaults[key] = str(value)
            elif default_value is None:
                defaults[key] = None if value is None else float(value)
            else:
                defaults[key] = float(value)
        return defaults

    @staticmethod
    def normalize_angle(angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    @staticmethod
    def _clamp(value, low, high):
        return max(low, min(high, value))

    def _motion_backend_enabled(self):
        return bool(getattr(getattr(self, "motion_backend", None), "enabled", False))

    def _motion_backend_supports_continuous_velocity(self):
        backend = getattr(self, "motion_backend", None)
        if backend is None or not bool(getattr(backend, "enabled", False)):
            return False
        supports = getattr(backend, "supports_continuous_velocity", None)
        if not callable(supports):
            return False
        try:
            return bool(supports())
        except Exception as exc:
            logger.warning("Failed to check motion backend continuous velocity support: %s", exc)
            return False

    @staticmethod
    def _scaled_directional_error(value, positive_scale, negative_scale):
        scale = positive_scale if value >= 0.0 else negative_scale
        return value * scale

    def _scaled_angular_error(self, error):
        return self._scaled_directional_error(
            error,
            self.motion_control["positive_angular_command_scale"],
            self.motion_control["negative_angular_command_scale"],
        )

    @staticmethod
    def _command_with_min(value, limit, minimum):
        value = ActionExecutorClient._clamp(value, -limit, limit)
        if math.isclose(value, 0.0, abs_tol=1e-9):
            return 0.0
        if abs(value) < minimum:
            return math.copysign(minimum, value)
        return value

    @staticmethod
    def _pose_from_msg(msg):
        position = msg.pose.position
        orientation = msg.pose.orientation
        siny_cosp = 2.0 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        return float(position.x), float(position.y), theta

    @staticmethod
    def _pose_payload(pose):
        if pose is None:
            return None
        x, y, theta = pose
        return {"x": x, "y": y, "theta": theta}

    def _position_callback(self, msg):
        pose = self._pose_from_msg(msg)
        with self.pose_lock:
            self.current_pose = pose
            self.current_pose_time = time.monotonic()

    def _get_current_pose(self):
        with self.pose_lock:
            if self.current_pose is None or self.current_pose_time is None:
                return None
            age = time.monotonic() - self.current_pose_time
            if age > self.motion_control["pose_timeout"]:
                return None
            return self.current_pose

    def _get_current_pose_observation(self):
        """Return a pose with its sensor update time, including test/fallback clients."""
        pose = self._get_current_pose()
        if pose is None:
            return None, None
        pose_lock = getattr(self, "pose_lock", None)
        if pose_lock is not None:
            with pose_lock:
                observation_time = getattr(self, "current_pose_time", None)
            if observation_time is not None:
                return pose, observation_time
        return pose, time.monotonic()

    def _filter_final_control_pose(self, pose, reference_pose):
        if (
            not self.motion_control.get("final_pose_jump_filter_enable", True)
            or pose is None
            or reference_pose is None
        ):
            return pose, False, None

        position_jump = math.hypot(pose[0] - reference_pose[0], pose[1] - reference_pose[1])
        angle_jump = abs(self.normalize_angle(pose[2] - reference_pose[2]))
        max_position_jump = self.motion_control["final_pose_max_position_jump"]
        max_angle_jump = self.motion_control["final_pose_max_angle_jump"]
        if position_jump <= max_position_jump and angle_jump <= max_angle_jump:
            return pose, False, None

        return reference_pose, True, {
            "raw_pose": self._pose_payload(pose),
            "held_pose": self._pose_payload(reference_pose),
            "position_jump": position_jump,
            "angle_jump": angle_jump,
            "max_position_jump": max_position_jump,
            "max_angle_jump": max_angle_jump,
        }

    def _closed_loop_enabled(self):
        if self._motion_backend_enabled() and not self._motion_backend_supports_continuous_velocity():
            return False
        return bool(self.motion_control.get("enable", True))

    def _shutdown_requested(self):
        shutdown_flag = getattr(self, "_shutdown_flag", None)
        return bool(shutdown_flag is not None and shutdown_flag.is_set())

    def _motion_cancel_token(self):
        return CombinedCancelToken(
            getattr(self, "_pause_flag", None),
            getattr(self, "_stop_flag", None),
            getattr(self, "_shutdown_flag", None),
            self._ensure_preempt_action_event(),
        )

    def _ensure_preempt_action_event(self):
        preempt_event = getattr(self, "_preempt_action_event", None)
        if preempt_event is None:
            preempt_event = Event()
            self._preempt_action_event = preempt_event
        return preempt_event

    def _request_action_preempt(self):
        self._ensure_preempt_action_event().set()
        if hasattr(self, "publisher") and hasattr(self, "motion_control"):
            self._publish_stop_motion()

    def _clear_action_preempt(self):
        self._ensure_preempt_action_event().clear()

    def _action_preempt_requested(self):
        return self._ensure_preempt_action_event().is_set()

    @staticmethod
    def _stamp_to_seconds(stamp):
        if stamp is None:
            return None
        sec = getattr(stamp, "sec", None)
        nanosec = getattr(stamp, "nanosec", None)
        if sec is None or nanosec is None:
            return None
        return float(sec) + float(nanosec) / 1e9

    def _clock_now_seconds(self):
        try:
            now = self.get_clock().now()
            nanoseconds = getattr(now, "nanoseconds", None)
            if nanoseconds is not None:
                return float(nanoseconds) / 1e9
            return self._stamp_to_seconds(now.to_msg())
        except Exception:
            return None

    def _extract_plan_metadata(self, msg):
        frame_id = getattr(getattr(msg, "header", None), "frame_id", "") or ""
        if "planner=navdp" not in frame_id or not getattr(msg, "poses", None):
            return None

        first_pose = msg.poses[0].pose
        if not math.isclose(first_pose.position.z, NAVDP_ACTION_METADATA_MARKER, abs_tol=1e-9):
            return None

        plan_id = None
        for field in frame_id.split("|"):
            if field.startswith("plan_id="):
                plan_id = field.split("=", 1)[1]
                break

        return {
            "planner_mode": "navdp",
            "plan_id": plan_id,
            "plan_pose": (
                float(first_pose.orientation.x),
                float(first_pose.orientation.y),
                float(first_pose.orientation.z),
            ),
            "plan_time": self._stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None)),
        }

    def _validate_plan_freshness(self, plan_metadata):
        if not plan_metadata:
            return True
        current_pose = self._get_current_pose()
        plan_pose = plan_metadata.get("plan_pose")
        if current_pose is None or plan_pose is None:
            self.viz.append_event(
                "plan_stale_rejected",
                level="warning",
                payload={
                    "plan_id": plan_metadata.get("plan_id"),
                    "reason": "pose_unavailable",
                },
            )
            return False

        position_delta = math.hypot(current_pose[0] - plan_pose[0], current_pose[1] - plan_pose[1])
        angle_delta = abs(self.normalize_angle(current_pose[2] - plan_pose[2]))
        max_position_delta = self.motion_control["plan_stale_position_tolerance"]
        max_angle_delta = self.motion_control["plan_stale_angle_tolerance"]
        max_age = self.motion_control["plan_stale_max_age"]
        plan_time = plan_metadata.get("plan_time")
        now = self._clock_now_seconds()
        age = None if plan_time is None or now is None else max(0.0, now - plan_time)

        stale_reasons = []
        if position_delta > max_position_delta:
            stale_reasons.append("position_delta")
        if angle_delta > max_angle_delta:
            stale_reasons.append("angle_delta")
        if age is not None and age > max_age:
            stale_reasons.append("age")

        if not stale_reasons:
            return True

        self.viz.append_event(
            "plan_stale_rejected",
            level="warning",
            payload={
                "plan_id": plan_metadata.get("plan_id"),
                "reason": ",".join(stale_reasons),
                "plan_pose": self._pose_payload(plan_pose),
                "current_pose": self._pose_payload(current_pose),
                "position_delta": position_delta,
                "angle_delta": angle_delta,
                "age": age,
                "max_position_delta": max_position_delta,
                "max_angle_delta": max_angle_delta,
                "max_age": max_age,
            },
        )
        return False

    def set_dry_run(self, enabled):
        with self.lock:
            self._dry_run = bool(enabled)

    def is_dry_run(self):
        with self.lock:
            return self._dry_run

    def set_navigation_active(self, active):
        with self.lock:
            self._navigation_active = bool(active)
            if not active:
                self.actions = []
                self.action_plan_metadata = None
                self._clear_action_preempt()

    def is_navigation_active(self):
        with self.lock:
            return self._navigation_active

    def set_safety_pause(self, paused, reason=None):
        paused = bool(paused)
        with self.lock:
            was_paused = self._pause_flag.is_set()
            self._pause_reason = reason if paused else None
            if paused:
                self._pause_flag.set()
            else:
                self._pause_flag.clear()

        if paused:
            self._publish_zero_motion()
            if not was_paused:
                logger.warning("LiDAR safety pause engaged: %s", reason or "unknown")
        elif was_paused:
            logger.info("LiDAR safety pause cleared.")
        return was_paused != paused

    def is_safety_paused(self):
        return self._pause_flag.is_set()

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
        if self._motion_backend_enabled():
            backend = getattr(self, "motion_backend", None)
            publish_velocity = getattr(backend, "publish_velocity", None)
            if not callable(publish_velocity) or not self._motion_backend_supports_continuous_velocity():
                logger.warning("Ignoring continuous WirelessController command in GR00T backend mode.")
                return False
            cancel_token = self._motion_cancel_token()
            if cancel_token.is_set():
                return False
            success, message, _ = publish_velocity(
                forward=float(msg.ly),
                lateral=-float(msg.lx),
                yaw=-float(msg.rx),
                cancel_event=cancel_token,
            )
            if not success:
                logger.warning("Failed to publish GR00T continuous velocity command: %s", message)
            return bool(success)
        try:
            self.publisher.publish(msg)
            return True
        except Exception as exc:
            if "context is invalid" in str(exc):
                logger.debug("Motion command skipped because ROS context is invalid: %s", exc)
            else:
                logger.warning("Failed to publish motion command: %s", exc)
            return False

    def _publish_zero_motion(self, repeat=1, interval=0.02):
        if self._motion_backend_enabled():
            success, _, _ = self.motion_backend.stop()
            return success
        ok = True
        repeat = max(1, int(repeat))
        for index in range(repeat):
            ok = self._publish_motion(self._zero_motion_msg()) and ok
            if index < repeat - 1:
                time.sleep(interval)
        return ok

    def _publish_stop_motion(self):
        return self._publish_zero_motion(
            repeat=self.motion_control["stop_command_repeat"],
            interval=self.motion_control["stop_command_interval"],
        )

    def _wait_for_final_settle(self):
        settle_time = max(0.0, float(self.motion_control["final_settle_time"]))
        if settle_time <= 0.0:
            return
        deadline = time.monotonic() + settle_time
        while time.monotonic() < deadline:
            if self._stop_flag.is_set() or self._shutdown_requested():
                return
            if not self._publish_zero_motion():
                return
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _final_goal_error(self, final_goal):
        pose = self._get_current_pose()
        if pose is None:
            return None
        goal_x, goal_y, goal_theta = final_goal
        x, y, theta = pose
        return {
            "position": math.hypot(goal_x - x, goal_y - y),
            "angle": self.normalize_angle(goal_theta - theta),
            "pose": {"x": x, "y": y, "theta": theta},
            "goal": {"x": goal_x, "y": goal_y, "theta": goal_theta},
        }

    def _collect_final_goal_errors(self, final_goal):
        sample_count = max(1, int(self.motion_control["final_validation_samples"]))
        interval = max(0.0, float(self.motion_control["final_validation_interval"]))
        errors = []
        for index in range(sample_count):
            error = self._final_goal_error(final_goal)
            if error is None:
                return None
            errors.append(error)
            if index < sample_count - 1:
                if self._stop_flag.is_set() or self._shutdown_requested():
                    return None
                self._publish_zero_motion()
                time.sleep(interval)
        return errors

    def _validate_final_goal(self, final_goal):
        self._publish_stop_motion()
        self._wait_for_final_settle()
        errors = self._collect_final_goal_errors(final_goal)
        if errors is None:
            self._last_final_goal_validation = None
            self.viz.append_event(
                "final_goal_validation_failed",
                level="warning",
                payload={"reason": "pose_unavailable"},
            )
            return False

        max_position_error = max(error["position"] for error in errors)
        max_angle_error = max(abs(error["angle"]) for error in errors)
        last_error = errors[-1]
        position_ok = max_position_error <= self.motion_control["final_position_tolerance"]
        angle_ok = max_angle_error <= self.motion_control["final_angle_tolerance"]
        self._last_final_goal_validation = {
            "max_position_error": max_position_error,
            "max_angle_error": max_angle_error,
            "last_position_error": last_error["position"],
            "last_angle_error": last_error["angle"],
            "pose": last_error["pose"],
            "goal": last_error["goal"],
            "sample_count": len(errors),
        }
        event_name = "final_goal_validated" if position_ok and angle_ok else "final_goal_validation_failed"
        self.viz.append_event(
            event_name,
            level="info" if position_ok and angle_ok else "warning",
            payload={
                "position_error": max_position_error,
                "angle_error": last_error["angle"],
                "max_angle_error": max_angle_error,
                "position_tolerance": self.motion_control["final_position_tolerance"],
                "angle_tolerance": self.motion_control["final_angle_tolerance"],
                "pose": last_error["pose"],
                "goal": last_error["goal"],
                "sample_count": len(errors),
            },
        )
        return position_ok and angle_ok

    def _run_discrete_final_alignment(self, final_goal):
        if not self.motion_control.get("final_discrete_alignment_enable", True):
            return True

        passes = max(1, int(self.motion_control["final_discrete_alignment_passes"]))
        position_tolerance = self.motion_control["final_discrete_alignment_position_tolerance"]
        if position_tolerance is None:
            position_tolerance = self.motion_control["final_position_tolerance"]
        angle_tolerance = self.motion_control["final_discrete_alignment_angle_tolerance"]
        if angle_tolerance is None:
            angle_tolerance = self.motion_control["final_angle_tolerance"]
        min_translation = max(0.0, float(self.motion_control["final_discrete_alignment_min_translation"]))
        min_angle = max(0.0, float(self.motion_control["final_discrete_alignment_min_angle"]))
        settle_time = max(0.0, float(self.motion_control["final_discrete_alignment_settle_time"]))

        self.viz.append_event(
            "final_discrete_alignment_started",
            level="info",
            payload={
                "goal": {
                    "x": float(final_goal[0]),
                    "y": float(final_goal[1]),
                    "theta": float(final_goal[2]),
                },
                "passes": passes,
                "position_tolerance": position_tolerance,
                "angle_tolerance": angle_tolerance,
                "min_translation": min_translation,
                "min_angle": min_angle,
            },
        )

        for pass_index in range(passes):
            if self._stop_flag.is_set() or self._shutdown_requested():
                return False

            self._wait_for_final_alignment_settle(settle_time)
            if not self._align_final_yaw(final_goal, angle_tolerance, min_angle, pass_index):
                return False

            self._wait_for_final_alignment_settle(settle_time)
            if not self._align_final_position(final_goal, position_tolerance, min_translation, pass_index):
                return False

        self.viz.append_event(
            "final_discrete_alignment_completed",
            level="info",
            payload={"passes": passes},
        )
        return True

    def _wait_for_final_alignment_settle(self, settle_time):
        if settle_time <= 0.0:
            return
        deadline = time.monotonic() + settle_time
        while time.monotonic() < deadline:
            if self._stop_flag.is_set() or self._shutdown_requested():
                return
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _align_final_yaw(self, final_goal, angle_tolerance, min_angle, pass_index):
        error = self._final_goal_error(final_goal)
        if error is None:
            self.viz.append_event(
                "final_discrete_alignment_failed",
                level="warning",
                payload={"stage": "yaw", "reason": "pose_unavailable", "pass": pass_index + 1},
            )
            return False

        angle_error = error["angle"]
        if abs(angle_error) <= angle_tolerance or abs(angle_error) < min_angle:
            self.viz.append_event(
                "final_discrete_alignment_step_skipped",
                level="info",
                payload={
                    "stage": "yaw",
                    "pass": pass_index + 1,
                    "angle_error": angle_error,
                    "angle_tolerance": angle_tolerance,
                    "min_angle": min_angle,
                    "pose": error["pose"],
                },
            )
            return True

        self.viz.append_event(
            "final_discrete_alignment_step",
            level="info",
            payload={
                "stage": "yaw",
                "pass": pass_index + 1,
                "command": angle_error,
                "pose": error["pose"],
                "goal": error["goal"],
            },
        )
        success, message, _ = self.rotate(angle_error, near_goal=True)
        if not success:
            self.viz.append_event(
                "final_discrete_alignment_failed",
                level="warning",
                payload={
                    "stage": "yaw",
                    "pass": pass_index + 1,
                    "command": angle_error,
                    "message": message,
                },
            )
            return False
        return True

    def _align_final_position(self, final_goal, position_tolerance, min_translation, pass_index):
        error = self._final_goal_error(final_goal)
        if error is None:
            self.viz.append_event(
                "final_discrete_alignment_failed",
                level="warning",
                payload={"stage": "position", "reason": "pose_unavailable", "pass": pass_index + 1},
            )
            return False

        if error["position"] <= position_tolerance:
            self.viz.append_event(
                "final_discrete_alignment_step_skipped",
                level="info",
                payload={
                    "stage": "position",
                    "pass": pass_index + 1,
                    "position_error": error["position"],
                    "position_tolerance": position_tolerance,
                    "pose": error["pose"],
                },
            )
            return True

        pose = error["pose"]
        goal = error["goal"]
        theta = pose["theta"]
        dx = goal["x"] - pose["x"]
        dy = goal["y"] - pose["y"]
        forward_error = math.cos(theta) * dx + math.sin(theta) * dy
        lateral_error = -math.sin(theta) * dx + math.cos(theta) * dy

        commands = [
            ("lateral", lateral_error, self.shift),
            ("forward", forward_error, self.forward),
        ]
        for stage, command, action_fn in commands:
            if self._stop_flag.is_set() or self._shutdown_requested():
                return False
            if abs(command) < min_translation:
                self.viz.append_event(
                    "final_discrete_alignment_step_skipped",
                    level="info",
                    payload={
                        "stage": stage,
                        "pass": pass_index + 1,
                        "command": command,
                        "min_translation": min_translation,
                    },
                )
                continue
            self.viz.append_event(
                "final_discrete_alignment_step",
                level="info",
                payload={
                    "stage": stage,
                    "pass": pass_index + 1,
                    "command": command,
                    "position_error": error["position"],
                    "pose": pose,
                    "goal": goal,
                },
            )
            success, message, _ = action_fn(command)
            if not success:
                self.viz.append_event(
                    "final_discrete_alignment_failed",
                    level="warning",
                    payload={
                        "stage": stage,
                        "pass": pass_index + 1,
                        "command": command,
                        "message": message,
                    },
                )
                return False
            self._wait_for_final_alignment_settle(
                self.motion_control["final_discrete_alignment_settle_time"]
            )
        return True

    def _run_closed_loop_final_pose(self, final_goal):
        if not self._closed_loop_enabled():
            return None

        initial_pose = self._get_current_pose()
        if initial_pose is None:
            self.viz.append_event(
                "closed_loop_motion_fallback",
                level="warning",
                payload={"motion": "final_pose", "reason": "pose_unavailable"},
            )
            return None

        goal_x, goal_y, goal_theta = final_goal
        timeout = max(self.motion_control["min_timeout"], self.motion_control["final_pose_timeout"])
        sample_count = max(1, int(self.motion_control["final_validation_samples"]))
        sample_interval = max(0.0, float(self.motion_control["final_validation_interval"]))
        next_sample_at = 0.0
        good_samples = 0
        max_position_error = 0.0
        max_angle_error = 0.0
        last_error = None
        accepted_pose = initial_pose
        jump_rejections = 0
        start_time = time.monotonic()

        self.viz.append_event(
            "final_goal_pose_settle_started",
            level="info",
            payload={
                "goal": {"x": goal_x, "y": goal_y, "theta": goal_theta},
                "position_tolerance": self.motion_control["final_position_tolerance"],
                "angle_tolerance": self.motion_control["final_angle_tolerance"],
                "timeout": timeout,
            },
        )

        while time.monotonic() - start_time < timeout:
            if self._stop_flag.is_set():
                return False
            paused_duration = self._wait_while_paused()
            if paused_duration > 0:
                start_time += paused_duration
                continue

            raw_pose = self._get_current_pose()
            if raw_pose is None:
                self.viz.append_event(
                    "closed_loop_motion_fallback",
                    level="warning",
                    payload={"motion": "final_pose", "reason": "pose_stale"},
                )
                return None
            pose, rejected_jump, jump_payload = self._filter_final_control_pose(raw_pose, accepted_pose)
            if rejected_jump:
                jump_rejections += 1
                self._publish_zero_motion()
                if jump_rejections == 1:
                    jump_payload["motion"] = "final_pose"
                    self.viz.append_event(
                        "final_pose_jump_rejected",
                        level="warning",
                        payload=jump_payload,
                    )
            else:
                accepted_pose = pose

            x, y, theta = pose
            dx = goal_x - x
            dy = goal_y - y
            position_error = math.hypot(dx, dy)
            angle_error = self.normalize_angle(goal_theta - theta)
            max_position_error = max(max_position_error, position_error)
            max_angle_error = max(max_angle_error, abs(angle_error))
            last_error = {
                "position": position_error,
                "angle": angle_error,
                "pose": {"x": x, "y": y, "theta": theta},
                "goal": {"x": goal_x, "y": goal_y, "theta": goal_theta},
            }

            now = time.monotonic()
            position_ok = position_error <= self.motion_control["final_position_tolerance"]
            angle_ok = abs(angle_error) <= self.motion_control["final_angle_tolerance"]
            if position_ok and angle_ok:
                self._publish_stop_motion()
                if now >= next_sample_at:
                    good_samples += 1
                    next_sample_at = now + sample_interval
                    if good_samples >= sample_count:
                        self._last_final_goal_validation = {
                            "max_position_error": max_position_error,
                            "max_angle_error": max_angle_error,
                            "last_position_error": position_error,
                            "last_angle_error": angle_error,
                            "pose": last_error["pose"],
                            "goal": last_error["goal"],
                            "sample_count": good_samples,
                        }
                        self.viz.append_event(
                            "final_goal_validated",
                            level="info",
                            payload={
                                "position_error": position_error,
                                "angle_error": angle_error,
                                "max_position_error": max_position_error,
                                "max_angle_error": max_angle_error,
                                "position_tolerance": self.motion_control["final_position_tolerance"],
                                "angle_tolerance": self.motion_control["final_angle_tolerance"],
                                "pose": last_error["pose"],
                                "goal": last_error["goal"],
                                "sample_count": good_samples,
                            },
                        )
                        return True
                time.sleep(min(0.02, max(0.0, next_sample_at - now)))
                continue

            if rejected_jump:
                time.sleep(0.02)
                continue

            good_samples = 0
            forward_error = math.cos(theta) * dx + math.sin(theta) * dy
            lateral_error = -math.sin(theta) * dx + math.cos(theta) * dy
            forward_cmd = self._command_with_min(
                self._scaled_directional_error(
                    forward_error,
                    self.motion_control["forward_command_scale"],
                    self.motion_control["backward_command_scale"],
                ),
                self.motion_control["final_pose_max_forward_speed"],
                self.motion_control["final_pose_min_linear_command"],
            )
            lateral_cmd = self._command_with_min(
                self._scaled_directional_error(
                    lateral_error,
                    self.motion_control["left_command_scale"],
                    self.motion_control["right_command_scale"],
                ),
                self.motion_control["final_pose_max_lateral_command"],
                self.motion_control["final_pose_min_linear_command"],
            )
            angular_cmd = self._command_with_min(
                self._scaled_angular_error(angle_error),
                self.motion_control["final_pose_max_angular_speed"],
                self.motion_control["final_pose_min_angular_command"],
            )

            msg = WirelessController()
            msg.lx = -lateral_cmd
            msg.ly = forward_cmd
            msg.rx = -angular_cmd
            msg.ry = 0.0
            msg.keys = 0
            if not self._publish_motion(msg):
                return False
            time.sleep(0.02)

        self._publish_stop_motion()
        if last_error is None:
            self._last_final_goal_validation = None
            return False

        self._last_final_goal_validation = {
            "max_position_error": max_position_error,
            "max_angle_error": max_angle_error,
            "last_position_error": last_error["position"],
            "last_angle_error": last_error["angle"],
            "pose": last_error["pose"],
            "goal": last_error["goal"],
            "sample_count": good_samples,
        }
        self.viz.append_event(
            "final_goal_validation_failed",
            level="warning",
            payload={
                "position_error": last_error["position"],
                "angle_error": last_error["angle"],
                "max_position_error": max_position_error,
                "max_angle_error": max_angle_error,
                "position_tolerance": self.motion_control["final_position_tolerance"],
                "angle_tolerance": self.motion_control["final_angle_tolerance"],
                "pose": last_error["pose"],
                "goal": last_error["goal"],
                "sample_count": good_samples,
                "reason": "timeout",
                "jump_rejections": jump_rejections,
            },
        )
        return False

    def _resolve_final_goal(self, final_goal):
        if self._motion_backend_enabled():
            aligned = self._run_discrete_final_alignment(final_goal)
            if not aligned:
                return "replan"
            return "validated" if self._validate_final_goal(final_goal) else "replan"
        result = self._run_closed_loop_final_pose(final_goal)
        if result is True:
            return "validated"
        return "replan"

    def _wait_while_paused(self, motion=None):
        pause_started = None
        while self._pause_flag.is_set() and not self._stop_flag.is_set() and rclpy.ok():
            if pause_started is None:
                pause_started = time.monotonic()
            if not self._publish_zero_motion():
                break
            time.sleep(0.05)
        if pause_started is None:
            return 0.0
        paused_duration = time.monotonic() - pause_started
        if motion:
            self.viz.append_event(
                f"closed_loop_{motion}_paused",
                level="warning",
                payload={
                    "duration": paused_duration,
                    "reason": self._pause_reason,
                },
            )
        return paused_duration

    def _run_timed_motion(self, msg, duration):
        start_time = time.monotonic()
        while time.monotonic() - start_time < duration:
            if self._stop_flag.is_set() or self._action_preempt_requested():
                logger.info('[StepControl] Movement interrupted by stop().')
                break
            paused_duration = self._wait_while_paused()
            if paused_duration > 0:
                start_time += paused_duration
                continue
            if not self._publish_motion(msg):
                break
            time.sleep(0.01)

    def _run_closed_loop_rotate(self, angle, speed, near_goal=False):
        if not self._closed_loop_enabled():
            return None
        start_pose, start_pose_observation_time = self._get_current_pose_observation()
        if start_pose is None:
            self.viz.append_event(
                "closed_loop_motion_fallback",
                level="warning",
                payload={"motion": "rotate", "reason": "pose_unavailable"},
            )
            return None

        target_theta = self.normalize_angle(start_pose[2] + angle)
        max_speed = min(abs(speed), self.motion_control["max_angular_speed"])
        min_angular_command = self.motion_control["min_angular_command"]
        positive_angular_scale = self.motion_control["positive_angular_command_scale"]
        negative_angular_scale = self.motion_control["negative_angular_command_scale"]
        stop_prediction_time = self.motion_control["angular_stop_prediction_time"]
        stop_velocity_tolerance = self.motion_control["angular_stop_velocity_tolerance"]
        stop_stable_time = self.motion_control["angular_stop_stable_time"]
        velocity_filter_alpha = self.motion_control["angular_velocity_filter_alpha"]
        angle_tolerance = self.motion_control["angle_tolerance"]
        timeout_margin = 0.5
        if near_goal:
            max_speed = min(
                max_speed,
                self.motion_control.get("near_goal_max_angular_speed", max_speed),
            )
            min_angular_command = min(
                min_angular_command,
                self.motion_control.get("near_goal_min_angular_command", min_angular_command),
            )
            angle_tolerance = self.motion_control["final_angle_tolerance"]
            timeout_margin = self.motion_control.get("near_goal_rotate_timeout_margin", timeout_margin)
        timeout = max(
            self.motion_control["min_timeout"],
            abs(angle) * self.motion_control["rotate_timeout_per_rad"]
            + timeout_margin
            + stop_prediction_time
            + stop_stable_time,
        )
        start_time = time.monotonic()
        last_error = None
        accepted_pose = start_pose
        jump_rejections = 0
        previous_yaw = start_pose[2]
        previous_pose_time = start_pose_observation_time
        filtered_yaw_velocity = 0.0
        stable_since = None
        braking = False
        rotation_direction = 1.0 if angle >= 0.0 else -1.0
        self.viz.append_event(
            "closed_loop_rotate_started",
            level="info",
            payload={
                "target": angle,
                "start_pose": self._pose_payload(start_pose),
                "target_theta": target_theta,
                "speed": speed,
                "max_speed": max_speed,
                "min_command": min_angular_command,
                "positive_command_scale": positive_angular_scale,
                "negative_command_scale": negative_angular_scale,
                "stop_prediction_time": stop_prediction_time,
                "tolerance": angle_tolerance,
                "timeout": timeout,
                "near_goal": bool(near_goal),
            },
        )
        while time.monotonic() - start_time < timeout:
            if self._action_preempt_requested():
                self._publish_stop_motion()
                self.viz.append_event(
                    "closed_loop_rotate_preempted",
                    level="info",
                    payload={
                        "target": angle,
                        "target_theta": target_theta,
                        "remaining_error": last_error,
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                    },
                )
                return PREEMPTED
            if self._stop_flag.is_set():
                self.viz.append_event(
                    "closed_loop_rotate_completed",
                    level="warning",
                    payload={
                        "success": False,
                        "target": angle,
                        "target_theta": target_theta,
                        "remaining_error": last_error,
                        "tolerance": angle_tolerance,
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                        "reason": "stopped",
                    },
                )
                return False
            paused_duration = self._wait_while_paused(motion="rotate")
            if paused_duration > 0:
                start_time += paused_duration
                continue
            raw_pose, pose_observation_time = self._get_current_pose_observation()
            if raw_pose is None:
                self.viz.append_event(
                    "closed_loop_motion_fallback",
                    level="warning",
                    payload={"motion": "rotate", "reason": "pose_stale"},
                )
                return None
            if near_goal:
                pose, rejected_jump, jump_payload = self._filter_final_control_pose(raw_pose, accepted_pose)
                if rejected_jump:
                    jump_rejections += 1
                    self._publish_zero_motion()
                    if jump_rejections == 1:
                        jump_payload["motion"] = "rotate"
                        self.viz.append_event(
                            "final_pose_jump_rejected",
                            level="warning",
                            payload=jump_payload,
                        )
                else:
                    accepted_pose = pose
            else:
                pose = raw_pose
                rejected_jump = False
            pose_time = time.monotonic()
            pose_dt = pose_observation_time - previous_pose_time
            if pose_dt >= 1e-3:
                measured_yaw_velocity = self.normalize_angle(pose[2] - previous_yaw) / pose_dt
                measured_yaw_velocity = self._clamp(measured_yaw_velocity, -4.0, 4.0)
                filtered_yaw_velocity = (
                    velocity_filter_alpha * measured_yaw_velocity
                    + (1.0 - velocity_filter_alpha) * filtered_yaw_velocity
                )
                previous_yaw = pose[2]
                previous_pose_time = pose_observation_time
            error = self.normalize_angle(target_theta - pose[2])
            yaw_velocity_toward_target = rotation_direction * filtered_yaw_velocity
            remaining_angle = rotation_direction * error
            predicted_stop_angle = max(0.0, yaw_velocity_toward_target) * stop_prediction_time
            last_error = error
            within_tolerance = abs(error) <= angle_tolerance
            nearly_stopped = abs(filtered_yaw_velocity) <= stop_velocity_tolerance
            if within_tolerance and nearly_stopped:
                if stable_since is None:
                    stable_since = pose_time
                if pose_time - stable_since < stop_stable_time:
                    self._publish_zero_motion()
                    time.sleep(0.02)
                    continue
                self._publish_stop_motion()
                self.viz.append_event(
                    "closed_loop_rotate_completed",
                    level="info",
                    payload={
                        "success": True,
                        "target": angle,
                        "target_theta": target_theta,
                        "current_pose": self._pose_payload(pose),
                        "remaining_error": error,
                        "tolerance": angle_tolerance,
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                        "yaw_velocity": filtered_yaw_velocity,
                        "reason": "settled_within_tolerance",
                    },
                )
                if near_goal:
                    self.viz.append_event(
                        "closed_loop_motion_accepted",
                        level="warning",
                        payload={
                            "motion": "rotate",
                            "target": angle,
                            "remaining_error": error,
                            "tolerance": angle_tolerance,
                            "reason": "within_final_tolerance",
                        },
                    )
                return True
            stable_since = None

            if rejected_jump:
                time.sleep(0.02)
                continue

            should_brake = (
                yaw_velocity_toward_target > stop_velocity_tolerance
                and remaining_angle <= predicted_stop_angle + angle_tolerance
            )
            if should_brake:
                angular_cmd = 0.0
                if not braking:
                    self.viz.append_event(
                        "closed_loop_rotate_braking",
                        level="info",
                        payload={
                            "remaining_angle": remaining_angle,
                            "yaw_velocity": filtered_yaw_velocity,
                            "predicted_stop_angle": predicted_stop_angle,
                        },
                    )
                braking = True
            else:
                braking = False
                angular_cmd = self._command_with_min(
                    self._scaled_angular_error(error),
                    max_speed,
                    min_angular_command,
                )
            msg = WirelessController()
            msg.lx = 0.0
            msg.ly = 0.0
            msg.rx = -angular_cmd
            msg.ry = 0.0
            msg.keys = 0
            if not self._publish_motion(msg):
                return False
            time.sleep(0.02)

        self._publish_stop_motion()
        if near_goal:
            self._wait_for_final_settle()
            settled_pose = self._get_current_pose()
            if settled_pose is not None:
                settled_error = self.normalize_angle(target_theta - settled_pose[2])
                if abs(settled_error) <= self.motion_control["final_angle_tolerance"]:
                    self.viz.append_event(
                        "closed_loop_rotate_completed",
                        level="info",
                        payload={
                            "success": True,
                            "target": angle,
                            "target_theta": target_theta,
                            "current_pose": self._pose_payload(settled_pose),
                            "remaining_error": settled_error,
                            "tolerance": self.motion_control["final_angle_tolerance"],
                            "elapsed": time.monotonic() - start_time,
                            "near_goal": bool(near_goal),
                            "reason": "settled_after_timeout",
                        },
                    )
                    self.viz.append_event(
                        "closed_loop_motion_accepted",
                        level="warning",
                        payload={
                            "motion": "rotate",
                            "target": angle,
                            "remaining_error": settled_error,
                            "tolerance": self.motion_control["final_angle_tolerance"],
                            "reason": "settled_after_timeout",
                        },
                    )
                    return True
                last_error = settled_error

        if near_goal and last_error is not None and abs(last_error) <= self.motion_control["final_angle_tolerance"]:
            self.viz.append_event(
                "closed_loop_rotate_completed",
                level="info",
                payload={
                    "success": True,
                    "target": angle,
                    "target_theta": target_theta,
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["final_angle_tolerance"],
                    "elapsed": time.monotonic() - start_time,
                    "near_goal": bool(near_goal),
                    "reason": "within_final_tolerance_after_timeout",
                },
            )
            self.viz.append_event(
                "closed_loop_motion_accepted",
                level="warning",
                payload={
                    "motion": "rotate",
                    "target": angle,
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["final_angle_tolerance"],
                },
            )
            return True
        if last_error is not None and abs(last_error) <= angle_tolerance:
            self.viz.append_event(
                "closed_loop_rotate_completed",
                level="warning",
                payload={
                    "success": True,
                    "target": angle,
                    "target_theta": target_theta,
                    "remaining_error": last_error,
                    "tolerance": angle_tolerance,
                    "elapsed": time.monotonic() - start_time,
                    "near_goal": bool(near_goal),
                    "reason": "within_tolerance_after_timeout",
                },
            )
            self.viz.append_event(
                "closed_loop_motion_accepted",
                level="warning",
                payload={
                    "motion": "rotate",
                    "target": angle,
                    "remaining_error": last_error,
                    "tolerance": angle_tolerance,
                    "reason": "within_tolerance_after_timeout",
                },
            )
            return True
        self.viz.append_event(
            "closed_loop_rotate_timeout",
            level="warning",
            payload={
                "target": angle,
                "target_theta": target_theta,
                "remaining_error": last_error,
                "tolerance": angle_tolerance,
                "timeout": timeout,
                "elapsed": time.monotonic() - start_time,
                "near_goal": bool(near_goal),
            },
        )
        self.viz.append_event(
            "closed_loop_motion_timeout",
            level="warning",
            payload={
                "motion": "rotate",
                "target": angle,
                "remaining_error": last_error,
                "timeout": timeout,
            },
        )
        return False

    def _should_use_combined_waypoint(self, theta, distance):
        return (
            self._closed_loop_enabled()
            and abs(float(distance)) > 1e-6
            and abs(float(theta)) < self.motion_control["waypoint_turn_in_place_threshold"]
        )

    def _run_closed_loop_waypoint(self, delta_theta, distance, speed=1.0, near_goal=False):
        """Drive toward one planned waypoint with simultaneous linear/yaw control."""
        if not self._closed_loop_enabled():
            return None
        start_pose, start_pose_observation_time = self._get_current_pose_observation()
        if start_pose is None:
            self.viz.append_event(
                "closed_loop_motion_fallback",
                level="warning",
                payload={"motion": "waypoint", "reason": "pose_unavailable"},
            )
            return None

        start_x, start_y, start_theta = start_pose
        target_theta = self.normalize_angle(start_theta + delta_theta)
        target_x = start_x + distance * math.cos(target_theta)
        target_y = start_y + distance * math.sin(target_theta)
        max_forward = min(abs(speed), self.motion_control["waypoint_max_forward_speed"])
        min_forward = self.motion_control["waypoint_min_forward_command"]
        turn_threshold = self.motion_control["waypoint_turn_in_place_threshold"]
        position_tolerance = (
            self.motion_control["final_position_tolerance"]
            if near_goal
            else self.motion_control["position_tolerance"]
        )
        stop_prediction_time = self.motion_control["linear_stop_prediction_time"]
        stop_velocity_tolerance = self.motion_control["linear_stop_velocity_tolerance"]
        stop_stable_time = self.motion_control["linear_stop_stable_time"]
        velocity_filter_alpha = self.motion_control["linear_velocity_filter_alpha"]
        timeout = max(
            self.motion_control["min_timeout"],
            abs(distance) * self.motion_control["waypoint_timeout_per_meter"],
            abs(delta_theta) * self.motion_control["rotate_timeout_per_rad"],
        ) + 0.5 + stop_prediction_time + stop_stable_time
        start_time = time.monotonic()
        previous_pose = start_pose
        previous_pose_time = start_pose_observation_time
        previous_distance = abs(distance)
        filtered_approach_velocity = 0.0
        stable_since = None
        braking = False
        last_error = None

        self.viz.append_event(
            "closed_loop_waypoint_started",
            payload={
                "target": {"x": target_x, "y": target_y, "theta": target_theta},
                "delta_theta": delta_theta,
                "distance": distance,
                "turn_in_place_threshold": turn_threshold,
                "timeout": timeout,
                "near_goal": bool(near_goal),
            },
        )

        while time.monotonic() - start_time < timeout:
            if self._action_preempt_requested():
                self._publish_stop_motion()
                return PREEMPTED
            if self._stop_flag.is_set() or self._shutdown_requested():
                self._publish_stop_motion()
                return False
            paused_duration = self._wait_while_paused(motion="waypoint")
            if paused_duration > 0:
                start_time += paused_duration
                continue
            pose, pose_observation_time = self._get_current_pose_observation()
            if pose is None:
                self._publish_stop_motion()
                self.viz.append_event(
                    "closed_loop_motion_fallback",
                    level="warning",
                    payload={"motion": "waypoint", "reason": "pose_stale"},
                )
                return None

            x, y, theta = pose
            dx = target_x - x
            dy = target_y - y
            position_error = math.hypot(dx, dy)
            bearing = math.atan2(dy, dx) if position_error > 1e-9 else theta
            bearing_error = self.normalize_angle(bearing - theta)
            pose_dt = pose_observation_time - previous_pose_time
            if pose_dt >= 1e-3:
                measured_velocity = (previous_distance - position_error) / pose_dt
                measured_velocity = self._clamp(measured_velocity, -2.0, 2.0)
                filtered_approach_velocity = (
                    velocity_filter_alpha * measured_velocity
                    + (1.0 - velocity_filter_alpha) * filtered_approach_velocity
                )
                previous_pose = pose
                previous_pose_time = pose_observation_time
                previous_distance = position_error
            predicted_stop_distance = max(0.0, filtered_approach_velocity) * stop_prediction_time
            last_error = {
                "position": position_error,
                "bearing": bearing_error,
                "approach_velocity": filtered_approach_velocity,
                "predicted_stop_distance": predicted_stop_distance,
            }

            within_tolerance = position_error <= position_tolerance
            nearly_stopped = abs(filtered_approach_velocity) <= stop_velocity_tolerance
            now = time.monotonic()
            if within_tolerance and nearly_stopped:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= stop_stable_time:
                    self._publish_stop_motion()
                    self.viz.append_event(
                        "closed_loop_waypoint_completed",
                        payload={
                            "target": {"x": target_x, "y": target_y},
                            "remaining_error": last_error,
                            "elapsed": now - start_time,
                            "reason": "settled_within_tolerance",
                        },
                    )
                    return True
            else:
                stable_since = None

            should_brake = (
                filtered_approach_velocity > stop_velocity_tolerance
                and position_error <= predicted_stop_distance + position_tolerance
            )
            if should_brake or within_tolerance or abs(bearing_error) >= turn_threshold:
                forward_cmd = 0.0
                if should_brake and not braking:
                    self.viz.append_event(
                        "closed_loop_waypoint_braking",
                        payload={"remaining_error": last_error},
                    )
                braking = should_brake
            else:
                braking = False
                raw_forward = self._scaled_directional_error(
                    position_error,
                    self.motion_control["forward_command_scale"],
                    self.motion_control["backward_command_scale"],
                )
                heading_scale = max(0.0, math.cos(bearing_error))
                forward_cmd = self._command_with_min(
                    raw_forward * heading_scale,
                    max_forward,
                    min_forward,
                )
            angular_cmd = self._command_with_min(
                self._scaled_angular_error(bearing_error),
                self.motion_control["max_angular_speed"],
                self.motion_control["min_angular_command"],
            )
            if within_tolerance:
                angular_cmd = 0.0

            msg = WirelessController()
            msg.lx = 0.0
            msg.ly = forward_cmd
            msg.rx = -angular_cmd
            msg.ry = 0.0
            msg.keys = 0
            if not self._publish_motion(msg):
                return False
            time.sleep(0.02)

        self._publish_stop_motion()
        if last_error is not None and last_error["position"] <= position_tolerance:
            self.viz.append_event(
                "closed_loop_waypoint_completed",
                level="warning",
                payload={
                    "target": {"x": target_x, "y": target_y},
                    "remaining_error": last_error,
                    "elapsed": time.monotonic() - start_time,
                    "reason": "within_tolerance_after_timeout",
                },
            )
            return True
        self.viz.append_event(
            "closed_loop_waypoint_timeout",
            level="warning",
            payload={
                "target": {"x": target_x, "y": target_y},
                "remaining_error": last_error,
                "timeout": timeout,
            },
        )
        self.viz.append_event(
            "closed_loop_motion_timeout",
            level="warning",
            payload={"motion": "waypoint", "remaining_error": last_error, "timeout": timeout},
        )
        return False

    def _run_closed_loop_forward(self, distance, speed, near_goal=False):
        if not self._closed_loop_enabled():
            return None
        start_pose, start_pose_observation_time = self._get_current_pose_observation()
        if start_pose is None:
            self.viz.append_event(
                "closed_loop_motion_fallback",
                level="warning",
                payload={"motion": "forward", "reason": "pose_unavailable"},
            )
            return None

        start_x, start_y, start_theta = start_pose
        target_x = start_x + distance * math.cos(start_theta)
        target_y = start_y + distance * math.sin(start_theta)
        max_forward = min(abs(speed), self.motion_control["max_forward_speed"])
        max_lateral = self.motion_control["max_lateral_command"]
        timeout = max(
            self.motion_control["min_timeout"],
            abs(distance) * self.motion_control["forward_timeout_per_meter"]
            + 0.5
            + self.motion_control["linear_stop_prediction_time"]
            + self.motion_control["linear_stop_stable_time"],
        )
        start_time = time.monotonic()
        last_error = None
        accepted_pose = start_pose
        jump_rejections = 0
        previous_pose = start_pose
        previous_pose_time = start_pose_observation_time
        filtered_path_velocity = 0.0
        stable_since = None
        braking = False
        motion_direction = 1.0 if distance >= 0.0 else -1.0
        stop_prediction_time = self.motion_control["linear_stop_prediction_time"]
        stop_velocity_tolerance = self.motion_control["linear_stop_velocity_tolerance"]
        stop_stable_time = self.motion_control["linear_stop_stable_time"]
        velocity_filter_alpha = self.motion_control["linear_velocity_filter_alpha"]
        self.viz.append_event(
            "closed_loop_forward_started",
            level="info",
            payload={
                "target": distance,
                "start_pose": self._pose_payload(start_pose),
                "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                "speed": speed,
                "max_forward": max_forward,
                "max_lateral": max_lateral,
                "tolerance": self.motion_control["position_tolerance"],
                "stop_prediction_time": stop_prediction_time,
                "stop_velocity_tolerance": stop_velocity_tolerance,
                "timeout": timeout,
                "near_goal": bool(near_goal),
            },
        )
        while time.monotonic() - start_time < timeout:
            if self._action_preempt_requested():
                self._publish_stop_motion()
                self.viz.append_event(
                    "closed_loop_forward_preempted",
                    level="info",
                    payload={
                        "target": distance,
                        "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                        "remaining_error": last_error,
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                    },
                )
                return PREEMPTED
            if self._stop_flag.is_set():
                self.viz.append_event(
                    "closed_loop_forward_completed",
                    level="warning",
                    payload={
                        "success": False,
                        "target": distance,
                        "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                        "remaining_error": last_error,
                        "tolerance": self.motion_control["position_tolerance"],
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                        "reason": "stopped",
                    },
                )
                return False
            paused_duration = self._wait_while_paused(motion="forward")
            if paused_duration > 0:
                start_time += paused_duration
                continue
            raw_pose, pose_observation_time = self._get_current_pose_observation()
            if raw_pose is None:
                self.viz.append_event(
                    "closed_loop_motion_fallback",
                    level="warning",
                    payload={"motion": "forward", "reason": "pose_stale"},
                )
                return None
            if near_goal:
                pose, rejected_jump, jump_payload = self._filter_final_control_pose(raw_pose, accepted_pose)
                if rejected_jump:
                    jump_rejections += 1
                    self._publish_zero_motion()
                    if jump_rejections == 1:
                        jump_payload["motion"] = "forward"
                        self.viz.append_event(
                            "final_pose_jump_rejected",
                            level="warning",
                            payload=jump_payload,
                        )
                else:
                    accepted_pose = pose
            else:
                pose = raw_pose
                rejected_jump = False

            x, y, theta = pose
            pose_time = time.monotonic()
            pose_dt = pose_observation_time - previous_pose_time
            if pose_dt >= 1e-3:
                path_delta = (
                    math.cos(start_theta) * (x - previous_pose[0])
                    + math.sin(start_theta) * (y - previous_pose[1])
                )
                measured_path_velocity = motion_direction * path_delta / pose_dt
                # Reject impossible localization spikes without hiding normal gait response.
                measured_path_velocity = self._clamp(measured_path_velocity, -2.0, 2.0)
                filtered_path_velocity = (
                    velocity_filter_alpha * measured_path_velocity
                    + (1.0 - velocity_filter_alpha) * filtered_path_velocity
                )
                previous_pose = pose
                previous_pose_time = pose_observation_time
            dx = target_x - x
            dy = target_y - y
            position_error = math.hypot(dx, dy)
            heading_error = self.normalize_angle(start_theta - theta)
            remaining_path_distance = motion_direction * (
                math.cos(start_theta) * dx + math.sin(start_theta) * dy
            )
            predicted_stop_distance = max(0.0, filtered_path_velocity) * stop_prediction_time
            last_error = {
                "position": position_error,
                "heading": heading_error,
                "remaining_path_distance": remaining_path_distance,
                "path_velocity": filtered_path_velocity,
                "predicted_stop_distance": predicted_stop_distance,
            }
            within_tolerance = position_error <= self.motion_control["position_tolerance"]
            nearly_stopped = abs(filtered_path_velocity) <= stop_velocity_tolerance
            if within_tolerance and nearly_stopped:
                if stable_since is None:
                    stable_since = pose_time
                if pose_time - stable_since < stop_stable_time:
                    self._publish_zero_motion()
                    time.sleep(0.02)
                    continue
                self._publish_stop_motion()
                self.viz.append_event(
                    "closed_loop_forward_completed",
                    level="info",
                    payload={
                        "success": True,
                        "target": distance,
                        "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                        "current_pose": self._pose_payload(pose),
                        "remaining_error": last_error,
                        "tolerance": self.motion_control["position_tolerance"],
                        "elapsed": time.monotonic() - start_time,
                        "near_goal": bool(near_goal),
                        "reason": "settled_within_tolerance",
                    },
                )
                return True
            stable_since = None

            if rejected_jump:
                time.sleep(0.02)
                continue

            forward_error = math.cos(theta) * dx + math.sin(theta) * dy
            lateral_error = -math.sin(theta) * dx + math.cos(theta) * dy
            should_brake = (
                filtered_path_velocity > stop_velocity_tolerance
                and remaining_path_distance
                <= predicted_stop_distance + self.motion_control["position_tolerance"]
            )
            if should_brake:
                forward_cmd = 0.0
                if not braking:
                    self.viz.append_event(
                        "closed_loop_forward_braking",
                        level="info",
                        payload={
                            "remaining_path_distance": remaining_path_distance,
                            "path_velocity": filtered_path_velocity,
                            "predicted_stop_distance": predicted_stop_distance,
                        },
                    )
                braking = True
            else:
                braking = False
                forward_cmd = self._command_with_min(
                    self._scaled_directional_error(
                        forward_error,
                        self.motion_control["forward_command_scale"],
                        self.motion_control["backward_command_scale"],
                    ),
                    max_forward,
                    self.motion_control["min_linear_command"],
                )
            lateral_cmd = self._clamp(
                self._scaled_directional_error(
                    lateral_error,
                    self.motion_control["left_command_scale"],
                    self.motion_control["right_command_scale"],
                ),
                -max_lateral,
                max_lateral,
            )
            angular_cmd = self._clamp(
                self._scaled_angular_error(heading_error),
                -self.motion_control["max_angular_speed"],
                self.motion_control["max_angular_speed"],
            )

            msg = WirelessController()
            msg.lx = -lateral_cmd
            msg.ly = forward_cmd
            msg.rx = -angular_cmd
            msg.ry = 0.0
            msg.keys = 0
            if not self._publish_motion(msg):
                return False
            time.sleep(0.02)

        self._publish_stop_motion()
        if (
            not near_goal
            and last_error is not None
            and last_error["position"] <= self.motion_control["forward_timeout_acceptance_tolerance"]
            and abs(last_error["heading"]) <= self.motion_control["forward_timeout_acceptance_heading_tolerance"]
        ):
            self.viz.append_event(
                "closed_loop_forward_completed",
                level="warning",
                payload={
                    "success": True,
                    "target": distance,
                    "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["forward_timeout_acceptance_tolerance"],
                    "elapsed": time.monotonic() - start_time,
                    "near_goal": bool(near_goal),
                    "reason": "accepted_after_timeout",
                    "jump_rejections": jump_rejections,
                },
            )
            self.viz.append_event(
                "closed_loop_motion_accepted",
                level="warning",
                payload={
                    "motion": "forward",
                    "target": distance,
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["forward_timeout_acceptance_tolerance"],
                    "heading_tolerance": self.motion_control["forward_timeout_acceptance_heading_tolerance"],
                    "reason": "accepted_after_timeout",
                },
            )
            return True
        if (
            near_goal
            and last_error is not None
            and last_error["position"] <= self.motion_control["final_position_tolerance"]
        ):
            self.viz.append_event(
                "closed_loop_forward_completed",
                level="info",
                payload={
                    "success": True,
                    "target": distance,
                    "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["final_position_tolerance"],
                    "elapsed": time.monotonic() - start_time,
                    "near_goal": bool(near_goal),
                    "reason": "within_final_tolerance_after_timeout",
                },
            )
            self.viz.append_event(
                "closed_loop_motion_accepted",
                level="warning",
                payload={
                    "motion": "forward",
                    "target": distance,
                    "remaining_error": last_error,
                    "tolerance": self.motion_control["final_position_tolerance"],
                },
            )
            return True
        self.viz.append_event(
            "closed_loop_forward_timeout",
            level="warning",
            payload={
                "target": distance,
                "target_pose": {"x": target_x, "y": target_y, "theta": start_theta},
                "remaining_error": last_error,
                "tolerance": self.motion_control["position_tolerance"],
                "timeout": timeout,
                "elapsed": time.monotonic() - start_time,
                "near_goal": bool(near_goal),
                "jump_rejections": jump_rejections,
            },
        )
        self.viz.append_event(
            "closed_loop_motion_timeout",
            level="warning",
            payload={
                "motion": "forward",
                "target": distance,
                "remaining_error": last_error,
                "timeout": timeout,
            },
        )
        return False

    def _action_callback(self, msg):
        if not self.is_navigation_active():
            logger.info("Ignoring planned actions because navigation is inactive.")
            return
        logger.info("Received planned actions.")
        self._request_action_preempt()
        plan_metadata = self._extract_plan_metadata(msg)
        actions = []
        for pose_stamped in msg.poses:
            delta_theta = pose_stamped.pose.position.x
            dist = pose_stamped.pose.position.y
            near_goal = pose_stamped.pose.orientation.w == 1.0
            final_goal = None
            if near_goal and pose_stamped.pose.position.z == 1.0:
                orientation = pose_stamped.pose.orientation
                final_goal = (
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                )
            actions.append((delta_theta, dist, near_goal, final_goal))
        if not actions:
            logger.warning("Received empty planned action sequence; finishing navigation to avoid waiting forever.")
            self.viz.append_event(
                "empty_action_sequence",
                level="warning",
                payload={"source": "executor"},
            )
            self._stop_flag.set()
            self._finish_flag.set()
            return
        # logger.info(f"near_goal = {near_goal}")
        with self.lock:
            self.actions = actions
            self.action_plan_metadata = plan_metadata
        self._new_action_event.set()
        self.viz.publish_actions_started(
            [(theta, distance, near_goal) for theta, distance, near_goal, _ in actions],
            dry_run=self.is_dry_run(),
        )

    def execute_action(self):
        while rclpy.ok() and not self._shutdown_flag.is_set():
            self._new_action_event.wait()
            self._new_action_event.clear()
            if self._shutdown_flag.is_set():
                break
            with self.lock:
                action_to_execute = list(self.actions)
                plan_metadata = self.action_plan_metadata
                dry_run = self._dry_run
                navigation_active = self._navigation_active
            if not navigation_active:
                continue
            if not action_to_execute:
                if self._stop_flag.is_set():
                    self._finish_flag.set()
                continue
            near_goal = bool(action_to_execute[-1][2])
            if dry_run:
                logger.info("Dry-run enabled, skipping execution of %d planned actions.", len(action_to_execute))
                self.viz.publish_actions_skipped(action_to_execute)
            else:
                self._clear_action_preempt()
                if not self._validate_plan_freshness(plan_metadata):
                    self._handle_action_failure(near_goal=near_goal)
                    continue
                action_failed = False
                action_preempted = False
                final_goal = None
                for action in action_to_execute:
                    theta, distance, near_goal = action[:3]
                    action_final_goal = action[3] if len(action) > 3 else None
                    if action_final_goal is not None:
                        final_goal = action_final_goal
                    self._wait_while_paused()
                    if self._stop_flag.is_set() or self._new_action_event.is_set():
                        break
                    use_combined_waypoint = self._should_use_combined_waypoint(theta, distance)
                    self.viz.append_event(
                        "action_motion_mode_selected",
                        payload={
                            "mode": "combined" if use_combined_waypoint else "turn_in_place",
                            "theta": theta,
                            "distance": distance,
                            "threshold": self.motion_control["waypoint_turn_in_place_threshold"],
                        },
                    )
                    if use_combined_waypoint:
                        waypoint_result = self._run_closed_loop_waypoint(
                            theta,
                            distance,
                            near_goal=near_goal,
                        )
                        if waypoint_result is True:
                            continue
                        if waypoint_result == PREEMPTED:
                            action_preempted = True
                            break
                        if waypoint_result is False:
                            logger.warning("Combined waypoint action failed.")
                            action_failed = True
                            break
                        # Missing/stale pose falls back to the established sequential path.
                    rotate_success, rotate_msg, _ = self.rotate(theta, near_goal=near_goal)
                    if not rotate_success:
                        if rotate_msg == "preempted.":
                            action_preempted = True
                            break
                        if self._stop_flag.is_set() or self._shutdown_requested():
                            break
                        logger.warning("Rotate action failed: %s", rotate_msg)
                        action_failed = True
                        break
                    if self._stop_flag.is_set() or self._new_action_event.is_set():
                        break
                    self._wait_while_paused()
                    if self._stop_flag.is_set() or self._new_action_event.is_set():
                        break
                    forward_success, forward_msg, _ = self.forward(distance, near_goal=near_goal)
                    if not forward_success:
                        if forward_msg == "preempted.":
                            action_preempted = True
                            break
                        if self._stop_flag.is_set() or self._shutdown_requested():
                            break
                        logger.warning("Forward action failed: %s", forward_msg)
                        action_failed = True
                        break
                    if self._stop_flag.is_set() or self._new_action_event.is_set():
                        break
                if action_preempted:
                    continue
                if action_failed:
                    self._handle_action_failure(near_goal=near_goal)
                    continue
                if self._new_action_event.is_set():
                    continue
                finish_segment = getattr(self.motion_backend, "finish_segment", None)
                if (not self._stop_flag.is_set() and not self._shutdown_requested()
                        and self._motion_backend_enabled() and callable(finish_segment)):
                    finish_success, finish_message, _ = finish_segment()
                    if not finish_success:
                        logger.warning("Navigation segment recovery failed: %s", finish_message)
                        self._handle_action_failure(near_goal=near_goal)
                        continue
                if near_goal and final_goal is not None:
                    final_status = self._resolve_final_goal(final_goal)
                    if final_status == "validated":
                        pass
                    elif final_status == "replan":
                        self._handle_action_failure(near_goal=True)
                        continue
                    else:
                        continue
            if self._stop_flag.is_set():
                self._finish_flag.set()
                continue
            if not near_goal:
                # to stable the camera
                # time.sleep(0.1)
                self.call_replanning_service()
            else:
                logger.info(f'The navigation process have been done!')
                self._finish_flag.set()

    def call_replanning_service(self):
        if self._stop_flag.is_set() or self._shutdown_requested() or not rclpy.ok():
            return
        logger.info('calling localization function')
        req = Trigger.Request()
        future = self.replanning_client.call_async(req)
        future.add_done_callback(self.replanning_response_callback)

    def _handle_action_failure(self, near_goal=False):
        if self._stop_flag.is_set() or self._shutdown_requested():
            return
        self.viz.append_event(
            "action_execution_replan_requested",
            level="warning",
            payload={"near_goal": bool(near_goal)},
        )
        self.call_replanning_service()

    def replanning_response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                logger.info(f'replanning service success! {response.message}')
                self.viz.publish_replanning_result(success=True, message=response.message)
            else:
                logger.warning(f'replanning service failed! {response.message}')
                self.viz.publish_replanning_result(success=False, message=response.message)
                self._stop_flag.set()
                self._finish_flag.set()
        except Exception as e:
            logger.error(f'failed to call replanning service {e}')

    def stop(self):
        self.set_navigation_active(False)
        self._stop_flag.set()
        self._finish_flag.set()
        self._new_action_event.set()
        self._publish_zero_motion(repeat=5)
        return True, 'success.', 'stop success.'

    def shutdown(self, timeout_sec=1.0):
        self.set_navigation_active(False)
        self._stop_flag.set()
        self._finish_flag.set()
        self._shutdown_flag.set()
        self._new_action_event.set()
        self._publish_zero_motion(repeat=5)
        if self.thread.is_alive():
            self.thread.join(timeout=timeout_sec)
        backend = getattr(self, "motion_backend", None)
        shutdown_backend = getattr(backend, "shutdown", None)
        if callable(shutdown_backend):
            try:
                shutdown_backend()
            except Exception as exc:
                logger.warning("GR00T backend shutdown failed: %s", exc)

    def forward(self, distance = 1.0, speed = 1.0, near_goal=False):
        if self._motion_backend_enabled():
            cancel_token = self._motion_cancel_token()
            if cancel_token.is_set():
                return False, 'stopped.', -2
            return self.motion_backend.forward(
                float(distance), cancel_event=cancel_token
            )

        closed_loop_result = self._run_closed_loop_forward(distance, speed, near_goal=near_goal)
        if closed_loop_result is True:
            return True, 'success.', 1
        if closed_loop_result == PREEMPTED:
            return False, 'preempted.', 2
        if closed_loop_result is False:
            if self._stop_flag.is_set() or self._shutdown_requested():
                return False, 'stopped.', -2
            return False, 'closed-loop forward timeout.', -1

        distance = distance*2.5
        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = speed if distance > 0 else -speed
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0

        duration = abs(distance) / speed
        self._run_timed_motion(msg, duration)

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

    def shift(self, distance = 1.0, speed = 0.5):
        if self._motion_backend_enabled():
            cancel_token = self._motion_cancel_token()
            if cancel_token.is_set():
                return False, 'stopped.', -2
            return self.motion_backend.shift(
                float(distance), cancel_event=cancel_token
            )

        msg = WirelessController()
        msg.lx = -speed if distance > 0 else speed
        msg.ly = 0.0
        msg.rx = 0.0
        msg.ry = 0.0
        msg.keys = 0

        duration = abs(distance) / speed
        self._run_timed_motion(msg, duration)

        self._publish_zero_motion()

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

    def rotate(self, angle = math.pi/2, speed = math.pi/2, near_goal=False):
        if self._motion_backend_enabled():
            cancel_token = self._motion_cancel_token()
            if cancel_token.is_set():
                return False, 'stopped.', -2
            return self.motion_backend.rotate(
                float(angle), cancel_event=cancel_token
            )

        closed_loop_result = self._run_closed_loop_rotate(angle, speed, near_goal=near_goal)
        if closed_loop_result is True:
            return True, 'success.', 1
        if closed_loop_result == PREEMPTED:
            return False, 'preempted.', 2
        if closed_loop_result is False:
            if self._stop_flag.is_set() or self._shutdown_requested():
                return False, 'stopped.', -2
            return False, 'closed-loop rotate timeout.', -1

        msg = WirelessController()
        msg.lx = 0.0
        msg.ly = 0.0
        msg.rx = -speed if angle > 0 else speed
        msg.ry = 0.0
        msg.keys = 0

        duration = abs(angle) / speed
        self._run_timed_motion(msg, duration)

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
