import sys
import threading
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path as RosPath


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from Node.action_executor import ActionExecutorClient  # noqa: E402
from Node.action_planner import ActionPlanner  # noqa: E402
from Node.localization import LocalizationClient  # noqa: E402
from Node.rgbd import RGBDClient  # noqa: E402
from frame_hub import CameraFrameHub  # noqa: E402
from visualization.adapters import ExecutorVisualizationAdapter, PlannerVisualizationAdapter  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


class FakeFramePipeline:
    def __init__(self):
        self.submissions = []

    def submit_frame(self, key, frame, captured_at=None):
        self.submissions.append((key, frame, captured_at))


class FakePublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(msg)


class FakeContinuousMotionBackend:
    enabled = True

    def __init__(self):
        self.commands = []
        self.stop_count = 0

    def supports_continuous_velocity(self):
        return True

    def publish_velocity(self, forward, lateral=0.0, yaw=0.0):
        self.commands.append(
            {
                "forward": float(forward),
                "lateral": float(lateral),
                "yaw": float(yaw),
            }
        )
        return True, "success.", 1

    def stop(self):
        self.stop_count += 1
        return True, "success.", 1


class FakeTimer:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def make_visualization_bundle():
    return {
        "state_hub": VisualizationStateHub(max_event_buffer=20),
        "frame_pipeline": FakeFramePipeline(),
    }


def test_rgbd_client_publish_visualization_rgbd_submits_latest_rgb_to_frame_pipeline():
    client = object.__new__(RGBDClient)
    client.visualization = make_visualization_bundle()

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    depth = np.zeros((8, 8), dtype=np.float32)

    client._publish_visualization_rgbd(rgb, depth)

    assert client.visualization["frame_pipeline"].submissions[0][0] == "rgb_latest"


def test_localization_client_publish_visualization_localization_updates_pose_and_event():
    client = object.__new__(LocalizationClient)
    client.visualization = make_visualization_bundle()
    client.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    client._publish_visualization_localization(
        rgb_frame=rgb,
        pose={"x": 1.0, "y": 2.0, "theta": 0.3},
        latency_ms=123,
    )

    snapshot = client.visualization["state_hub"].build_snapshot()
    assert snapshot["robot"]["vpr_pose"]["x"] == 1.0
    assert snapshot["events"][-1]["event_name"] == "vpr_result"
    assert snapshot["events"][-1]["task_id"] == "task-1"


def test_localization_client_vpr_once_odom_cancels_timer_after_initial_fix():
    client = object.__new__(LocalizationClient)
    client.localization_mode = LocalizationClient.VPR_ONCE_ODOM_MODE
    client.global_localization_timer = FakeTimer()

    client._stop_vpr_after_initial_fix()

    assert client.global_localization_timer.cancelled is True


def test_localization_client_continuous_vpr_keeps_timer_running():
    client = object.__new__(LocalizationClient)
    client.localization_mode = LocalizationClient.CONTINUOUS_VPR_MODE
    client.global_localization_timer = FakeTimer()

    client._stop_vpr_after_initial_fix()

    assert client.global_localization_timer.cancelled is False


def test_localization_client_freezes_vpr_updates_near_navigation_goal():
    client = object.__new__(LocalizationClient)
    client.visualization = make_visualization_bundle()
    client.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    client.navigation_goal_lock = threading.Lock()
    client.navigation_goal = None
    client.vpr_transform_frozen_near_goal = False
    client.near_goal_vpr_freeze_enabled = True
    client.near_goal_vpr_freeze_distance = 0.5

    client.set_navigation_goal([1.0, 2.0, 0.3])

    assert client._maybe_freeze_vpr_near_goal((0.0, 2.0, 0.0)) is False
    assert client._vpr_transform_update_frozen() is False

    assert client._maybe_freeze_vpr_near_goal((0.7, 2.0, 0.0)) is True
    assert client._vpr_transform_update_frozen() is True
    snapshot = client.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "vpr_updates_frozen_near_goal"
    assert abs(snapshot["events"][-1]["payload"]["distance_to_goal"] - 0.3) < 1e-9

    client.clear_navigation_goal()

    assert client._vpr_transform_update_frozen() is False
    snapshot = client.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "vpr_updates_unfrozen"


def test_localization_client_skips_vpr_transform_update_after_near_goal_freeze(monkeypatch):
    client = object.__new__(LocalizationClient)
    client.visualization = make_visualization_bundle()
    client.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    client.frame_hub = CameraFrameHub()
    client.frame_hub.publish(np.zeros((8, 8, 3), dtype=np.uint8), source="test")
    client.vpr_frame_max_age = None
    client.odom_lock = threading.Lock()
    client.transform_lock = threading.Lock()
    client.navigation_goal_lock = threading.Lock()
    client.T_base2odom = np.eye(4)
    client.T_odom2map = np.eye(4)
    client.T_odom2map[0:2, 3] = [1.0, 2.0]
    client.vpr_transform_frozen_near_goal = False
    client.near_goal_vpr_freeze_enabled = True
    client.near_goal_vpr_freeze_distance = 0.5
    client.navigation_goal = (1.1, 2.0, 0.0)
    client.vpr_timeout = 1.0
    client.localization_mode = LocalizationClient.CONTINUOUS_VPR_MODE
    client.global_localization_timer = FakeTimer()
    client.img_id = 0
    client.save_vpr_images = False
    client.cfg = SimpleNamespace(
        localization=SimpleNamespace(
            vpr=SimpleNamespace(
                endpoint="http://vpr",
                robot_id="robot",
            )
        ),
        rgbd_server=SimpleNamespace(visualization=False),
    )
    client.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=123_000_000))
    monkeypatch.setattr(
        "Node.localization.convert_image2pose",
        lambda *args, **kwargs: (9.0, 9.0, 1.0),
    )

    client.localization_callback()

    assert np.allclose(client.T_odom2map[0:2, 3], [1.0, 2.0])
    snapshot = client.visualization["state_hub"].build_snapshot()
    event_names = [event["event_name"] for event in snapshot["events"]]
    assert "vpr_result" in event_names
    assert event_names[-1] == "vpr_transform_update_skipped"


def test_action_planner_publish_visualization_plan_updates_planner_state():
    visualization = make_visualization_bundle()
    visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    adapter = PlannerVisualizationAdapter(visualization=visualization)

    adapter.publish_plan(
        mode="navdp",
        path_world=[[0.0, 0.0], [1.0, 1.0]],
        waypoints=[[1.0, 1.0]],
        local_path=[[0.0, 0.0], [0.2, 0.1]],
        local_goal=[0.5, 0.2],
        actions=[[0.1, 0.25]],
        action_limit=2,
        preview_actions=[[0.1, 0.25], [0.0, 0.25]],
        camera_intrinsics=[[387.0, 0.0, 320.0], [0.0, 386.0, 243.0], [0.0, 0.0, 1.0]],
        camera_image_size=[640, 480],
    )

    snapshot = visualization["state_hub"].build_snapshot()
    assert snapshot["planner"]["mode"] == "navdp"
    assert snapshot["planner"]["global_path"] == [[0.0, 0.0], [1.0, 1.0]]
    assert snapshot["planner"]["local_path"] == [[0.0, 0.0], [0.2, 0.1]]
    assert snapshot["planner"]["action_limit"] == 2
    assert snapshot["planner"]["preview_actions"] == [[0.1, 0.25], [0.0, 0.25]]
    assert snapshot["planner"]["camera_image_size"] == [640, 480]
    assert [event["event_name"] for event in snapshot["events"][-2:]] == ["planner_selected", "navdp_result"]


def test_action_planner_actions_to_local_path_builds_world_frame_preview():
    planner = object.__new__(ActionPlanner)
    planner.path_planner = type("PathPlanner", (), {"current_position": (1.0, 2.0, 0.0)})()

    local_path = planner.actions_to_local_path([(0.0, 1.0), (np.pi / 2, 1.0)])

    assert len(local_path) == 3
    assert local_path[0] == [1.0, 2.0]
    assert np.allclose(local_path[1], [2.0, 2.0])
    assert np.allclose(local_path[2], [2.0, 3.0])


def test_action_planner_preview_path_can_be_longer_than_executed_actions():
    planner = object.__new__(ActionPlanner)
    planner.path_planner = type("PathPlanner", (), {"current_position": (1.0, 2.0, 0.0)})()

    preview_path = planner.actions_to_local_path([(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)])
    executed_path = planner.actions_to_local_path([(0.0, 1.0), (0.0, 1.0)])

    assert len(preview_path) == 4
    assert len(executed_path) == 3
    assert np.allclose(preview_path[-1], [4.0, 2.0])
    assert np.allclose(executed_path[-1], [3.0, 2.0])


def test_action_planner_encodes_navdp_plan_metadata(monkeypatch):
    planner = object.__new__(ActionPlanner)
    planner.action_pub = FakePublisher()
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.visualization = None
    planner.frame_hub = CameraFrameHub()
    planner.frame_hub.publish(
        np.zeros((8, 8, 3), dtype=np.uint8),
        np.zeros((8, 8), dtype=np.float32),
        source="test",
    )
    planner.model_planner_enabled = True
    planner.model_planner_frame_max_age = None
    planner.path_planner = SimpleNamespace(current_position=(1.0, 2.0, 0.3))
    planner.model_planner_timeout = 1.0
    planner.max_action_allowed = 2
    planner.cfg = SimpleNamespace(
        local_planner=SimpleNamespace(
            model_planner=SimpleNamespace(
                action_endpoint="http://navdp/action",
                use_discrete_action=True,
            )
        )
    )
    planner.viz = PlannerVisualizationAdapter(visualization=make_visualization_bundle())
    planner.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time(sec=12, nanosec=345)))
    monkeypatch.setattr(
        "Node.action_planner.convert_image_depth2localaction",
        lambda *args, **kwargs: [[0.0, 0.25], [0.261799, 0.0], [0.0, 0.25]],
    )

    result = planner.get_action_from_action_model(
        path_world=[(1.0, 2.0, 0.3), (2.0, 2.0, 0.3)],
        select_localplanner_interval=1,
    )

    assert result["plan_id"] == "navdp-1"
    message = planner.action_pub.messages[-1]
    assert message.header.frame_id == "map|planner=navdp|plan_id=navdp-1"
    assert message.header.stamp.sec == 12
    assert len(message.poses) == 2
    first_pose = message.poses[0].pose
    assert first_pose.position.z == -1.0
    assert first_pose.orientation.x == 1.0
    assert first_pose.orientation.y == 2.0
    assert first_pose.orientation.z == 0.3
    assert message.poses[1].pose.position.z == 0.0


def test_action_executor_publish_visualization_actions_appends_event():
    visualization = make_visualization_bundle()
    visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    adapter = ExecutorVisualizationAdapter(visualization=visualization)

    adapter.publish_actions_started([(0.1, 0.25, False)], dry_run=True)

    snapshot = visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "action_execution_started"
    assert snapshot["events"][-1]["payload"]["actions"] == [(0.1, 0.25, False)]
    assert snapshot["events"][-1]["payload"]["dry_run"] is True
    assert snapshot["events"][-1]["task_id"] == "task-1"


def test_action_executor_action_callback_preempts_current_motion():
    executor = object.__new__(ActionExecutorClient)
    executor.lock = threading.Lock()
    executor._navigation_active = True
    executor._dry_run = False
    executor.actions = [(0.0, 0.25, False, None)]
    executor.action_plan_metadata = None
    executor._new_action_event = threading.Event()
    executor._preempt_action_event = threading.Event()
    executor.motion_control = ActionExecutorClient._load_motion_control_config(None)
    executor.publisher = FakePublisher()
    executor.viz = ExecutorVisualizationAdapter(visualization=make_visualization_bundle())

    message = RosPath()
    pose = PoseStamped()
    pose.pose.position.x = 0.1
    pose.pose.position.y = 0.25
    message.poses.append(pose)

    executor._action_callback(message)

    assert executor._preempt_action_event.is_set()
    assert executor._new_action_event.is_set()
    assert executor.actions == [(0.1, 0.25, False, None)]
    assert executor.publisher.messages
    assert executor.publisher.messages[-1].lx == 0.0
    assert executor.publisher.messages[-1].ly == 0.0
    assert executor.publisher.messages[-1].rx == 0.0


def test_action_executor_rejects_stale_navdp_plan_metadata():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["plan_stale_position_tolerance"] = 0.2
    executor.motion_control["plan_stale_angle_tolerance"] = 0.2
    executor._get_current_pose = lambda: (1.0, 0.0, 0.4)
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    fresh = executor._validate_plan_freshness(
        {
            "planner_mode": "navdp",
            "plan_id": "navdp-1",
            "plan_pose": (0.0, 0.0, 0.0),
            "plan_time": None,
        }
    )

    assert fresh is False
    event = executor.viz.visualization["state_hub"].build_snapshot()["events"][-1]
    assert event["event_name"] == "plan_stale_rejected"
    assert "position_delta" in event["payload"]["reason"]
    assert "angle_delta" in event["payload"]["reason"]


def test_action_executor_safety_pause_publishes_zero_motion_and_resumes():
    executor = object.__new__(ActionExecutorClient)
    executor.lock = threading.Lock()
    executor._pause_flag = threading.Event()
    executor._pause_reason = None
    executor.publisher = FakePublisher()

    assert executor.set_safety_pause(True, reason="lidar blocked") is True
    assert executor.is_safety_paused() is True
    assert executor._pause_reason == "lidar blocked"
    assert executor.publisher.messages[-1].lx == 0.0
    assert executor.publisher.messages[-1].ly == 0.0
    assert executor.publisher.messages[-1].rx == 0.0
    assert executor.publisher.messages[-1].ry == 0.0

    assert executor.set_safety_pause(False, reason="clear") is True
    assert executor.is_safety_paused() is False
    assert executor._pause_reason is None


def _make_executor_for_closed_loop():
    executor = object.__new__(ActionExecutorClient)
    executor.motion_control = ActionExecutorClient._load_motion_control_config(None)
    executor.motion_control.update(
        {
            "position_tolerance": 0.03,
            "angle_tolerance": 0.03,
            "pose_timeout": 10.0,
            "max_forward_speed": 0.4,
            "max_lateral_command": 0.12,
            "max_angular_speed": 0.8,
            "min_timeout": 0.2,
            "forward_timeout_per_meter": 2.0,
            "rotate_timeout_per_rad": 1.0,
        }
    )
    executor._stop_flag = threading.Event()
    executor._pause_flag = threading.Event()
    executor._shutdown_flag = threading.Event()
    executor._finish_flag = threading.Event()
    executor.motion_backend = SimpleNamespace(enabled=False)
    executor.publisher = FakePublisher()
    executor.viz = ExecutorVisualizationAdapter(visualization=make_visualization_bundle())
    return executor


def test_action_executor_directional_command_scales():
    executor = _make_executor_for_closed_loop()

    assert np.isclose(
        executor._scaled_directional_error(
            0.2,
            executor.motion_control["forward_command_scale"],
            executor.motion_control["backward_command_scale"],
        ),
        0.52,
    )
    assert np.isclose(
        executor._scaled_directional_error(
            -0.2,
            executor.motion_control["forward_command_scale"],
            executor.motion_control["backward_command_scale"],
        ),
        -0.38,
    )
    assert "kp_linear" not in executor.motion_control
    assert "kp_lateral" not in executor.motion_control
    assert "kp_angular" not in executor.motion_control


def test_action_executor_closed_loop_forward_publishes_correction_commands():
    executor = _make_executor_for_closed_loop()
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.12, 0.02, 0.01),
            (0.25, 0.0, 0.0),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (0.25, 0.0, 0.0))

    result = executor._run_closed_loop_forward(0.25, speed=1.0)

    assert result is True
    assert len(executor.publisher.messages) >= 2
    moving = executor.publisher.messages[0]
    assert moving.ly > 0.0
    assert any(not np.isclose(msg.lx, 0.0) for msg in executor.publisher.messages)
    stop = executor.publisher.messages[-1]
    assert stop.lx == 0.0
    assert stop.ly == 0.0
    assert stop.rx == 0.0


def test_action_executor_combined_waypoint_publishes_forward_and_yaw_together():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["linear_stop_stable_time"] = 0.0
    target_x = 0.25 * np.cos(0.2)
    target_y = 0.25 * np.sin(0.2)
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.12, 0.015, 0.08),
            (target_x, target_y, 0.2),
            (target_x, target_y, 0.2),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (target_x, target_y, 0.2))

    result = executor._run_closed_loop_waypoint(0.2, 0.25, speed=1.0)

    assert result is True
    assert any(msg.ly > 0.0 and not np.isclose(msg.rx, 0.0) for msg in executor.publisher.messages)
    assert executor.publisher.messages[-1].ly == 0.0
    assert executor.publisher.messages[-1].rx == 0.0


def test_action_executor_combined_waypoint_uses_groot_continuous_velocity_backend():
    executor = _make_executor_for_closed_loop()
    executor.motion_backend = FakeContinuousMotionBackend()
    executor.motion_control["linear_stop_stable_time"] = 0.0
    target_x = 0.25 * np.cos(0.2)
    target_y = 0.25 * np.sin(0.2)
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.12, 0.015, 0.08),
            (target_x, target_y, 0.2),
            (target_x, target_y, 0.2),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (target_x, target_y, 0.2))

    assert executor._should_use_combined_waypoint(0.2, 0.25) is True
    result = executor._run_closed_loop_waypoint(0.2, 0.25, speed=1.0)

    assert result is True
    assert any(
        command["forward"] > 0.0 and not np.isclose(command["yaw"], 0.0)
        for command in executor.motion_backend.commands
    )
    assert executor.motion_backend.stop_count >= 1


def test_action_executor_combined_waypoint_stops_forward_for_large_live_heading_error():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["waypoint_timeout_per_meter"] = 0.0
    executor.motion_control["rotate_timeout_per_rad"] = 0.0
    executor.motion_control["linear_stop_prediction_time"] = 0.0
    executor.motion_control["linear_stop_stable_time"] = 0.0
    poses = iter([(0.0, 0.0, 0.0)])
    executor._get_current_pose = lambda: next(poses, (0.0, 0.0, -0.6))

    result = executor._run_closed_loop_waypoint(0.0, 0.25, speed=1.0)

    assert result is False
    moving_commands = [msg for msg in executor.publisher.messages if not np.isclose(msg.rx, 0.0)]
    assert moving_commands
    assert all(np.isclose(msg.ly, 0.0) for msg in moving_commands)


def test_action_executor_combined_waypoint_selection_keeps_large_turn_in_place():
    executor = _make_executor_for_closed_loop()

    assert executor._should_use_combined_waypoint(0.49, 0.3) is True
    assert executor._should_use_combined_waypoint(0.5, 0.3) is False
    assert executor._should_use_combined_waypoint(0.0, 0.0) is False


def test_action_executor_closed_loop_rotate_uses_pose_error_sign():
    executor = _make_executor_for_closed_loop()
    poses = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)])
    executor._get_current_pose = lambda: next(poses, (0.0, 0.0, 0.2))

    result = executor._run_closed_loop_rotate(0.2, speed=1.0)

    assert result is True
    assert executor.publisher.messages[0].rx < 0.0
    assert executor.publisher.messages[-1].rx == 0.0


def test_action_executor_closed_loop_rotate_applies_directional_angular_scale():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["positive_angular_command_scale"] = 2.0
    poses = iter([(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.2)])
    executor._get_current_pose = lambda: next(poses, (0.0, 0.0, 0.2))

    result = executor._run_closed_loop_rotate(0.2, speed=1.0)

    assert result is True
    assert np.isclose(executor.publisher.messages[0].rx, -0.4)


def test_action_executor_final_control_filter_rejects_near_goal_pose_jump():
    executor = object.__new__(ActionExecutorClient)
    executor.motion_control = ActionExecutorClient._load_motion_control_config(None)

    pose, rejected, payload = executor._filter_final_control_pose(
        pose=(0.16, 0.0, 0.01),
        reference_pose=(0.0, 0.0, 0.0),
    )

    assert rejected is True
    assert pose == (0.0, 0.0, 0.0)
    assert payload["position_jump"] > executor.motion_control["final_pose_max_position_jump"]


def test_action_executor_closed_loop_falls_back_when_pose_unavailable():
    executor = _make_executor_for_closed_loop()
    executor._get_current_pose = lambda: None
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_forward(0.25, speed=1.0)

    assert result is None
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "closed_loop_motion_fallback"
    assert snapshot["events"][-1]["payload"]["reason"] == "pose_unavailable"


def test_action_executor_accepts_near_goal_forward_timeout_within_final_tolerance():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["forward_timeout_per_meter"] = 0.0
    executor.motion_control["final_position_tolerance"] = 0.15
    executor._get_current_pose = lambda: (0.0, 0.0, 0.0)

    result = executor._run_closed_loop_forward(0.1, speed=1.0, near_goal=True)

    assert result is True
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "closed_loop_motion_accepted"
    assert snapshot["events"][-1]["payload"]["motion"] == "forward"


def test_action_executor_keeps_non_near_goal_forward_timeout_as_failure():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["forward_timeout_per_meter"] = 0.0
    executor.motion_control["final_position_tolerance"] = 0.15
    executor._get_current_pose = lambda: (0.0, 0.0, 0.0)

    result = executor._run_closed_loop_forward(0.1, speed=1.0, near_goal=False)

    assert result is False
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "closed_loop_motion_timeout"


def test_action_executor_accepts_non_near_goal_forward_timeout_with_small_residual():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["forward_timeout_per_meter"] = 0.0
    executor.motion_control["position_tolerance"] = 0.08
    executor.motion_control["forward_timeout_acceptance_tolerance"] = 0.12
    executor.motion_control["forward_timeout_acceptance_heading_tolerance"] = 0.1
    executor._get_current_pose = lambda: (0.16, 0.0, 0.0)
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_forward(0.25, speed=1.0, near_goal=False)

    assert result is True
    events = executor.viz.visualization["state_hub"].build_snapshot()["events"]
    assert events[-2]["event_name"] == "closed_loop_forward_completed"
    assert events[-2]["payload"]["reason"] == "accepted_after_timeout"
    assert events[-1]["event_name"] == "closed_loop_motion_accepted"


def test_action_executor_accepts_near_goal_rotate_timeout_within_final_tolerance():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["rotate_timeout_per_rad"] = 0.0
    executor.motion_control["final_angle_tolerance"] = 0.2
    executor._get_current_pose = lambda: (0.0, 0.0, 0.0)

    result = executor._run_closed_loop_rotate(0.1, speed=1.0, near_goal=True)

    assert result is True
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "closed_loop_motion_accepted"
    assert snapshot["events"][-1]["payload"]["motion"] == "rotate"


def test_action_executor_accepts_regular_rotate_timeout_within_angle_tolerance():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["rotate_timeout_per_rad"] = 0.0
    executor.motion_control["angular_stop_prediction_time"] = 0.0
    executor.motion_control["angular_stop_stable_time"] = 0.0
    executor.motion_control["angular_stop_velocity_tolerance"] = -1.0
    poses = iter([(0.0, 0.0, 0.0)])
    executor._get_current_pose = lambda: next(poses, (0.0, 0.0, 0.18))

    result = executor._run_closed_loop_rotate(0.2, speed=1.0, near_goal=False)

    assert result is True
    events = executor.viz.visualization["state_hub"].build_snapshot()["events"]
    assert events[-2]["event_name"] == "closed_loop_rotate_completed"
    assert events[-2]["payload"]["reason"] == "within_tolerance_after_timeout"
    assert events[-1]["event_name"] == "closed_loop_motion_accepted"


def test_action_executor_rejects_final_goal_after_angle_drift():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["final_settle_time"] = 0.0
    executor.motion_control["stop_command_repeat"] = 1
    executor.motion_control["final_position_tolerance"] = 0.1
    executor.motion_control["final_angle_tolerance"] = 0.07
    executor._get_current_pose = lambda: (-0.922409757744374, -0.9020476545634879, 0.4658214586467734)
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._validate_final_goal((-0.93, -0.82, -0.28))

    assert result is False
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "final_goal_validation_failed"
    assert snapshot["events"][-1]["payload"]["position_error"] <= 0.1
    assert abs(snapshot["events"][-1]["payload"]["angle_error"]) > 0.7


def test_action_executor_final_goal_validation_requires_consecutive_good_samples():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["final_settle_time"] = 0.0
    executor.motion_control["stop_command_repeat"] = 1
    executor.motion_control["final_validation_samples"] = 3
    executor.motion_control["final_validation_interval"] = 0.0
    executor.motion_control["final_position_tolerance"] = 0.1
    executor.motion_control["final_angle_tolerance"] = 0.07
    poses = iter(
        [
            (1.0, 2.0, 0.01),
            (1.0, 2.0, 0.4),
            (1.0, 2.0, 0.01),
        ]
    )
    executor._get_current_pose = lambda: next(poses)
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._validate_final_goal((1.0, 2.0, 0.0))

    assert result is False
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["payload"]["sample_count"] == 3
    assert snapshot["events"][-1]["payload"]["max_angle_error"] > 0.3


def test_action_executor_final_pose_settle_commands_translation_and_rotation():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["final_position_tolerance"] = 0.03
    executor.motion_control["final_angle_tolerance"] = 0.03
    executor.motion_control["final_validation_samples"] = 2
    executor.motion_control["final_validation_interval"] = 0.0
    executor.motion_control["final_pose_timeout"] = 0.5
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.1, 0.1, 0.2),
            (0.1, 0.1, 0.2),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (0.1, 0.1, 0.2))
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_final_pose((0.1, 0.1, 0.2))

    assert result is True
    moving = next(msg for msg in executor.publisher.messages if not np.isclose(msg.ly, 0.0))
    assert moving.ly > 0.0
    assert moving.lx < 0.0
    assert moving.rx < 0.0
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "final_goal_validated"


def test_action_executor_final_pose_rejects_large_pose_jump_without_chasing_it():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["final_validation_samples"] = 1
    executor.motion_control["final_validation_interval"] = 0.0
    executor.motion_control["final_position_tolerance"] = 0.1
    executor.motion_control["final_angle_tolerance"] = 0.1
    executor.motion_control["final_pose_max_position_jump"] = 0.2
    executor.motion_control["final_pose_max_angle_jump"] = 0.4
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.8, 0.8, 1.2),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (0.8, 0.8, 1.2))
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_final_pose((0.0, 0.0, 0.0))

    assert result is True
    assert all(np.isclose(msg.ly, 0.0) for msg in executor.publisher.messages)
    events = executor.viz.visualization["state_hub"].build_snapshot()["events"]
    assert any(event["event_name"] == "final_pose_jump_rejected" for event in events)
    assert events[-1]["event_name"] == "final_goal_validated"


def test_action_executor_near_goal_forward_rejects_large_pose_jump():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["forward_timeout_per_meter"] = 0.0
    executor.motion_control["final_position_tolerance"] = 0.2
    executor.motion_control["final_pose_max_position_jump"] = 0.2
    poses = iter(
        [
            (0.0, 0.0, 0.0),
            (0.8, 0.8, 0.0),
        ]
    )
    executor._get_current_pose = lambda: next(poses, (0.8, 0.8, 0.0))
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_forward(0.1, speed=1.0, near_goal=True)

    assert result is True
    assert all(np.isclose(msg.ly, 0.0) for msg in executor.publisher.messages)
    events = executor.viz.visualization["state_hub"].build_snapshot()["events"]
    assert any(event["event_name"] == "final_pose_jump_rejected" for event in events)
    assert events[-1]["event_name"] == "closed_loop_motion_accepted"


def test_action_executor_final_pose_timeout_keeps_position_and_yaw_coupled():
    executor = _make_executor_for_closed_loop()
    executor.motion_control["final_position_tolerance"] = 0.1
    executor.motion_control["final_angle_tolerance"] = 0.07
    executor.motion_control["min_timeout"] = 0.03
    executor.motion_control["final_pose_timeout"] = 0.03
    executor._get_current_pose = lambda: (0.22, 0.0, 0.0)
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")

    result = executor._run_closed_loop_final_pose((0.0, 0.0, 0.0))

    assert result is False
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "final_goal_validation_failed"
    assert snapshot["events"][-1]["payload"]["position_error"] > 0.1
    assert abs(snapshot["events"][-1]["payload"]["angle_error"]) <= 0.07


def test_action_executor_resolves_final_goal_with_pose_settle():
    executor = _make_executor_for_closed_loop()
    calls = []
    executor._run_closed_loop_final_pose = lambda goal: calls.append(goal) or True

    result = executor._resolve_final_goal((1.0, 2.0, 0.0))

    assert result == "validated"
    assert calls == [(1.0, 2.0, 0.0)]
    assert not executor._finish_flag.is_set()


def test_action_planner_encodes_final_goal_on_near_goal_actions():
    planner = object.__new__(ActionPlanner)
    planner.action_pub = FakePublisher()
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.viz = PlannerVisualizationAdapter(visualization=make_visualization_bundle())
    planner.path_planner = SimpleNamespace(current_position=(0.0, 0.0, 0.0))
    planner.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))

    published = planner.get_action_from_line_segment_planner(
        path_world=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        select_interval=1,
        select_waypoint=2,
        goal_theta=0.5,
        moving_flag=1.0,
        final_goal=(1.0, 0.0, 0.5),
    )

    assert published is True
    message = planner.action_pub.messages[-1]
    assert message.poses
    for pose in message.poses:
        assert pose.pose.position.z == 1.0
        assert pose.pose.orientation.w == 1.0
        assert pose.pose.orientation.x == 1.0
        assert pose.pose.orientation.y == 0.0
        assert pose.pose.orientation.z == 0.5


def test_action_planner_yaw_only_near_goal_action_skips_forward_segment():
    planner = object.__new__(ActionPlanner)
    planner.action_pub = FakePublisher()
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.viz = PlannerVisualizationAdapter(visualization=make_visualization_bundle())
    planner.path_planner = SimpleNamespace(current_position=(0.0, 0.0, 0.2))
    planner.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))

    published = planner.get_action_from_line_segment_planner(
        path_world=[(0.0, 0.0, 0.2), (0.08, 0.0, 0.0)],
        select_interval=1,
        select_waypoint=2,
        goal_theta=0.5,
        moving_flag=1.0,
        final_goal=(0.08, 0.0, 0.5),
        yaw_only=True,
    )

    assert published is True
    message = planner.action_pub.messages[-1]
    assert len(message.poses) == 1
    assert np.isclose(message.poses[0].pose.position.x, 0.3)
    assert message.poses[0].pose.position.y == 0.0


def test_action_planner_near_goal_delays_final_yaw_until_position_tolerance():
    planner = object.__new__(ActionPlanner)
    visualization = make_visualization_bundle()
    visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    planner.viz = PlannerVisualizationAdapter(visualization=visualization)
    planner.max_path_threshold = 20
    planner.interval = 10
    planner.localplanner_interval = 60
    planner.final_position_tolerance = 0.2
    planner.near_goal_yaw_only_distance = 0.2
    planner.current_position = (0.0, 0.0, 0.0)
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.action_pub = FakePublisher()
    planner.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time()))
    planner.cfg = SimpleNamespace(
        local_planner=SimpleNamespace(enbale_model_planner=False),
        only_global_planner_area=SimpleNamespace(enable_only_global_planner_area=False),
    )

    class DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyPathPlanner:
        current_position = (0.0, 0.0, 0.0)

        def get_path(self, x, y, theta):
            return [(0.3, 0.0, theta)], [(1, 1)]

        def plot(self, _points):
            return None

    planner.position_lock = DummyLock()
    planner.path_planner = DummyPathPlanner()

    success = planner.get_action(0.3, 0.0, 1.2)

    assert success is True
    message = planner.action_pub.messages[-1]
    assert len(message.poses) == 1
    assert np.isclose(message.poses[0].pose.position.x, 0.0)
    assert np.isclose(message.poses[0].pose.position.y, 0.3)
    events = visualization["state_hub"].build_snapshot()["events"]
    global_path_event = next(event for event in events if event["event_name"] == "global_path_computed")
    assert global_path_event["payload"]["final_yaw_enabled"] is False
    assert global_path_event["payload"]["include_final_position"] is True


def test_action_planner_yaw_only_distance_uses_final_position_tolerance_floor():
    cfg = SimpleNamespace(
        local_planner=SimpleNamespace(
            line_segment_planner=SimpleNamespace(near_goal_yaw_only_distance=0.12)
        ),
        motion_control=SimpleNamespace(
            closed_loop=SimpleNamespace(final_position_tolerance=0.2)
        ),
    )

    distance = ActionPlanner.resolve_near_goal_yaw_only_distance(cfg)

    assert distance == 0.2


def test_action_planner_rejects_empty_line_segment_actions():
    planner = object.__new__(ActionPlanner)
    planner.action_pub = FakePublisher()
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.viz = PlannerVisualizationAdapter(visualization=make_visualization_bundle())
    planner.path_planner = SimpleNamespace(current_position=(0.0, 0.0, 0.0))

    published = planner.get_action_from_line_segment_planner(
        path_world=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)],
        select_interval=2,
        select_waypoint=2,
        goal_theta=None,
        moving_flag=0.0,
    )

    assert published is False
    assert planner.action_pub.messages == []
    events = planner.viz.visualization["state_hub"].build_snapshot()["events"]
    assert events[-1]["event_name"] == "empty_action_sequence"


def test_action_executor_empty_planned_actions_releases_navigation_wait():
    executor = object.__new__(ActionExecutorClient)
    executor._stop_flag = threading.Event()
    executor._finish_flag = threading.Event()
    executor.lock = threading.Lock()
    executor.actions = []
    executor._navigation_active = True
    executor.viz = ExecutorVisualizationAdapter(visualization=make_visualization_bundle())

    executor._action_callback(SimpleNamespace(poses=[]))

    assert executor._stop_flag.is_set()
    assert executor._finish_flag.is_set()
    events = executor.viz.visualization["state_hub"].build_snapshot()["events"]
    assert events[-1]["event_name"] == "empty_action_sequence"


def test_action_executor_near_goal_failure_requests_replan_without_stopping():
    executor = object.__new__(ActionExecutorClient)
    executor._stop_flag = threading.Event()
    executor._finish_flag = threading.Event()
    executor.viz = ExecutorVisualizationAdapter(visualization=make_visualization_bundle())
    executor.viz.visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    calls = []
    executor.call_replanning_service = lambda: calls.append("replan")

    executor._handle_action_failure(near_goal=True)

    assert calls == ["replan"]
    assert not executor._stop_flag.is_set()
    assert not executor._finish_flag.is_set()
    snapshot = executor.viz.visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["event_name"] == "action_execution_replan_requested"
    assert snapshot["events"][-1]["payload"]["near_goal"] is True


def test_action_executor_motion_control_config_uses_localization_topic_by_default():
    cfg = SimpleNamespace(
        localization=SimpleNamespace(position_topic="/pose"),
        motion_control=SimpleNamespace(
            closed_loop=SimpleNamespace(enable=False, max_forward_speed=0.2)
        ),
    )

    control = ActionExecutorClient._load_motion_control_config(cfg)

    assert control["enable"] is False
    assert control["position_topic"] == "/pose"
    assert control["max_forward_speed"] == 0.2


def test_action_planner_handles_single_point_path_without_zero_slice_step():
    planner = object.__new__(ActionPlanner)
    visualization = make_visualization_bundle()
    visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    planner.viz = PlannerVisualizationAdapter(visualization=visualization)
    planner.max_path_threshold = 20
    planner.interval = 10
    planner.localplanner_interval = 60
    planner.position_lock = None
    planner.current_position = (0.0, 0.0, 0.0)
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.cfg = type(
        "Cfg",
        (),
        {
            "local_planner": type("LocalPlanner", (), {"enbale_model_planner": False})(),
            "only_global_planner_area": type(
                "OnlyGlobalPlannerArea",
                (),
                {"enable_only_global_planner_area": False},
            )(),
        },
    )()

    class DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyPathPlanner:
        current_position = (0.0, 0.0, 0.0)

        def get_path(self, x, y, theta):
            return [[x, y, theta]], [[1, 1]]

        def plot(self, _points):
            return None

    planner.position_lock = DummyLock()
    planner.path_planner = DummyPathPlanner()
    planner.get_action_from_line_segment_planner = lambda *args, **kwargs: True

    planner.get_action(1.0, 2.0, 0.5)

    snapshot = visualization["state_hub"].build_snapshot()
    assert snapshot["planner"]["mode"] == "line_segment"
    assert snapshot["planner"]["actions"] == [(0.5, 0.0)]


def test_action_planner_returns_false_when_path_is_unavailable():
    planner = object.__new__(ActionPlanner)
    visualization = make_visualization_bundle()
    visualization["state_hub"].start_task_recording(task_id="task-1", goal_text="door")
    planner.viz = PlannerVisualizationAdapter(visualization=visualization)
    planner.max_path_threshold = 20
    planner.interval = 10
    planner.localplanner_interval = 60
    planner.navigation_lock = threading.Lock()
    planner._navigation_active = True
    planner.cfg = type(
        "Cfg",
        (),
        {
            "local_planner": type("LocalPlanner", (), {"enbale_model_planner": False})(),
            "only_global_planner_area": type(
                "OnlyGlobalPlannerArea",
                (),
                {"enable_only_global_planner_area": False},
            )(),
        },
    )()

    class DummyLock:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class DummyPathPlanner:
        current_position = (0.0, 0.0, 0.0)

        def get_path(self, x, y, theta):
            return None, None

    planner.position_lock = DummyLock()
    planner.current_position = (0.0, 0.0, 0.0)
    planner.path_planner = DummyPathPlanner()

    success = planner.get_action(1.0, 2.0, 0.5)

    assert success is False
    snapshot = visualization["state_hub"].build_snapshot()
    assert snapshot["events"][-1]["level"] == "warning"
