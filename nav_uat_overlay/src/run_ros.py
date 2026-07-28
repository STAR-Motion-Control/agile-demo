import datetime
import json
import logging
import math
import os
import socket

from nav_runtime_guard import NavigationStartupError, authorized_navigation_process


_NAV_PROCESS_GUARD = None
if __name__ == "__main__":
    try:
        _NAV_PROCESS_GUARD = authorized_navigation_process()
        _NAV_PROCESS_GUARD.__enter__()
    except NavigationStartupError as exc:
        raise SystemExit(str(exc)) from exc


import cv2
import hydra
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from Node import NodeManager
from navigation import NavigationTaskService
from utils.logger import ColorFormatter
from utils.utils import reset_embedding_model
from visualization.image_store import ImageStore
from visualization.map_utils import build_map_metadata
from visualization.server import create_app, launch_server_in_thread
from visualization.state_hub import VisualizationStateHub


logger = logging.getLogger()
NAV_CONFIG_NAME = os.environ.get("NAV_CONFIG_NAME", "config_bk")
if NAV_CONFIG_NAME not in ("config", "config_bk"):
    raise RuntimeError("NAV_CONFIG_NAME must be 'config' or 'config_bk'")


def config_get(config, key, default=None):
    if hasattr(config, "get") and callable(config.get):
        return config.get(key, default)

    return getattr(config, key, default)


def _resolve_visualization_urls(host, port):
    port = int(port)
    normalized_host = str(host).strip()
    urls = []

    def add_url(candidate_host):
        candidate = str(candidate_host).strip()
        if not candidate:
            return
        url = f"http://{candidate}:{port}/viz"
        if url not in urls:
            urls.append(url)

    if normalized_host in {"0.0.0.0", "::", ""}:
        add_url("127.0.0.1")
        add_url("localhost")
        try:
            hostname = socket.gethostname()
            for family, _, _, _, sockaddr in socket.getaddrinfo(hostname, None, family=socket.AF_INET):
                if family != socket.AF_INET:
                    continue
                ip = sockaddr[0]
                if ip.startswith("127."):
                    continue
                add_url(ip)
        except OSError:
            pass
    else:
        add_url(normalized_host)

    return urls


def prepare_visualization(cfg, map_ctx):
    if not cfg.visualization.enable:
        return None

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    static_dir = os.path.join(repo_root, "webviz", "dist")
    map_image_path = os.path.join(cfg.map_path, "map.png")
    map_config_path = os.path.join(cfg.map_path, "config.yaml")
    with open(map_config_path, "r", encoding="utf-8") as handle:
        map_cfg = yaml.safe_load(handle)

    save_replay = bool(config_get(cfg.visualization, "save_replay", False))
    replay_store = None
    if save_replay:
        from visualization.replay_store import ReplayStore

        replay_store = ReplayStore(
            history_dir=cfg.visualization.history_dir,
            max_replay_mb=cfg.visualization.max_replay_mb,
        )
    state_hub = VisualizationStateHub(
        max_event_buffer=cfg.visualization.max_event_buffer,
        replay_store=replay_store,
        save_replay=save_replay,
    )
    image_store = ImageStore(
        jpeg_quality=cfg.visualization.jpeg_quality,
        rgb_preview_fps=cfg.visualization.rgb_preview_fps,
        depth_preview_fps=cfg.visualization.depth_preview_fps,
    )
    frame_pipeline = None
    if bool(config_get(cfg.visualization, "enable_live_frames", False)):
        from visualization.frame_pipeline import FramePipeline

        frame_pipeline = FramePipeline(
            image_store=image_store,
            state_hub=state_hub,
            stream_fps={
                "rgb_latest": {
                    "fps": cfg.visualization.rgb_preview_fps,
                    "preview_size": [
                        cfg.visualization.rgb_preview_width,
                        cfg.visualization.rgb_preview_height,
                    ],
                },
                "rgb_navdp": {
                    "fps": cfg.visualization.rgb_preview_fps,
                    "preview_size": [
                        cfg.visualization.rgb_preview_width,
                        cfg.visualization.rgb_preview_height,
                    ],
                }
            },
        )
    area_cfg = cfg.only_global_planner_area
    only_global_planner_area = config_get(
        area_cfg,
        "polygons",
        config_get(area_cfg, "polygon", []),
    )
    map_metadata = build_map_metadata(
        map_image_path=map_image_path,
        resolution=map_cfg["resolution"],
        origin=map_cfg["origin"],
        frame_id="map",
        map_url="/viz/api/map/image",
        only_global_planner_area=only_global_planner_area,
        labels=map_ctx.get("labels", []),
    )
    app = create_app(
        state_hub=state_hub,
        image_store=image_store,
        map_metadata=map_metadata,
        replay_store=replay_store,
        static_dir=static_dir,
    )
    server, thread = launch_server_in_thread(app, cfg.visualization.host, cfg.visualization.port)
    logger.info(
        "Visualization web is available at %s",
        ", ".join(_resolve_visualization_urls(cfg.visualization.host, cfg.visualization.port)),
    )
    return {
        "state_hub": state_hub,
        "image_store": image_store,
        "frame_pipeline": frame_pipeline,
        "replay_store": replay_store,
        "map_metadata": map_metadata,
        "app": app,
        "server": server,
        "thread": thread,
    }


def prepare4log():
    os.makedirs("./log", exist_ok=True)
    timestamp_log = datetime.datetime.now().strftime("./log/ros_%Y-%m-%d_%H-%M-%S.log")
    fixed_log = "./log_ros.txt"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = ColorFormatter("[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    timestamp_handler = logging.FileHandler(timestamp_log, mode="w", encoding="utf-8")
    timestamp_handler.setFormatter(formatter)

    fixed_handler = logging.FileHandler(fixed_log, mode="w", encoding="utf-8")
    fixed_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(timestamp_handler)
    logger.addHandler(fixed_handler)
    return logger


def prepare_reference_coordinates(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        coordinates = [[round(x, 2) for x in item["position"]] for item in data]
        caption = [item["description"] for item in data]
    return coordinates, caption


def load_map_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = []
    seen = set()
    for item in data:
        label = item.get("label")
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


def prepare_map(cfg):
    image_path = os.path.join(cfg.map_path, "map.png")
    yaml_path = os.path.join(cfg.map_path, "config.yaml")
    sp_path = os.path.join(cfg.map_path, "prompt.txt")
    label_path = os.path.join(cfg.map_path, "label.json")

    map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if map_image is None:
        raise FileNotFoundError(f"Failed to load map image: {image_path}")

    window_size = 10
    free_points = []
    for i in range(map_image.shape[0]):
        for j in range(map_image.shape[1]):
            top = max(0, i - window_size)
            bottom = min(map_image.shape[0], i + window_size)
            left = max(0, j - window_size)
            right = min(map_image.shape[1], j + window_size)
            window = map_image[top:bottom, left:right]
            if np.all(window > 220):
                free_points.append((i, j))

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        resolution = data["resolution"]
        origin = data["origin"]

    map_ctx = {
        "map_image": map_image,
        "origin": origin,
        "resolution": resolution,
        "window_size": window_size,
        "free_points": free_points,
        "labels": load_map_labels(label_path) if os.path.exists(label_path) else [],
    }

    if cfg.goal_recognition.enable_embedding:
        ref_coords, ref_caption = prepare_reference_coordinates(label_path)
        logger.info("Loaded %d reference labels for embedding retrieval", len(ref_caption))
        reset_embedding_model(cfg.goal_recognition.embedding.reset_endpoint, ref_caption)
        map_ctx["reference_coordinates"] = ref_coords
        map_ctx["reference_caption"] = ref_caption
    else:
        with open(sp_path, "r", encoding="utf-8") as f:
            map_ctx["system_prompt"] = f.read()

    return map_ctx


class NavRosBridge(Node):
    def __init__(self, cfg, map_ctx, node_manager):
        super().__init__("nav_ros_bridge")
        self.cfg = cfg
        self.node_manager = node_manager
        self.task_service = NavigationTaskService(
            cfg=cfg,
            map_ctx=map_ctx,
            node_manager=node_manager,
            visualization=node_manager.visualization,
            publish_status=self._publish_status,
        )
        lidar_safety_cfg = getattr(cfg, "lidar_safety", None)
        self.lidar_safety_enabled = bool(getattr(lidar_safety_cfg, "enable", True))
        self.lidar_safety_topic = str(getattr(lidar_safety_cfg, "state_topic", "/safety/lidar_state"))
        self._lidar_state = "unknown"
        self._lidar_pause_active = False

        self.status_pub = self.create_publisher(String, "/nav/status", 10)
        self.create_subscription(String, "/nav/text_nav", self._on_text_nav, 10)
        self.create_subscription(Empty, "/nav/stop_cmd", self._on_stop_cmd, 10)
        self.create_subscription(Twist, "/nav/forward_cmd", self._on_forward_cmd, 10)
        self.create_subscription(Twist, "/nav/rotate_cmd", self._on_rotate_cmd, 10)
        self.create_subscription(String, "/nav/relative_cmd", self._on_relative_cmd, 10)
        if self.lidar_safety_enabled:
            self.create_subscription(String, self.lidar_safety_topic, self._on_lidar_state, 10)
        self.create_service(Trigger, "/nav/get_pose", self._on_get_pose)
        self.create_timer(1.0, self._publish_status)

        self._publish_status()
        self.get_logger().info("ROS bridge ready.")

    def _get_status_payload(self):
        payload = self.task_service.get_status_payload()
        if hasattr(self, "_lidar_state"):
            payload["lidar_safety"] = {
                "enabled": bool(getattr(self, "lidar_safety_enabled", False)),
                "state": self._lidar_state,
                "obstacle_blocked": self._lidar_state == "blocked",
                "paused": bool(getattr(self, "_lidar_pause_active", False)),
                "topic": getattr(self, "lidar_safety_topic", "/safety/lidar_state"),
            }
        return payload

    def _publish_status(self):
        if not rclpy.ok():
            return
        payload = json.dumps(self._get_status_payload(), ensure_ascii=False)
        msg = String()
        msg.data = payload
        try:
            self.status_pub.publish(msg)
        except Exception as exc:
            if "context is invalid" in str(exc):
                self.get_logger().debug(f"Skipping status publish during shutdown: {exc}")
            else:
                raise

    def _on_text_nav(self, msg):
        self.task_service.submit_navigation_request(msg.data, dry_run=False, source="ros_text_nav")

    def _on_stop_cmd(self, _msg):
        self.task_service.handle_stop_request(source="ros_stop_cmd")

    def _on_lidar_state(self, msg):
        state = str(getattr(msg, "data", "")).strip().lower()
        if state not in {"clear", "blocked", "stale"}:
            self.get_logger().warning(f"Ignoring unknown LiDAR safety state: {state!r}")
            return

        pause_active = state in {"blocked", "stale"}
        self._lidar_state = state
        self._lidar_pause_active = pause_active

        if hasattr(self.node_manager, "set_lidar_safety_pause"):
            changed = self.node_manager.set_lidar_safety_pause(
                pause_active,
                reason=f"{self.lidar_safety_topic}={state}",
            )
        else:
            changed = False

        if changed:
            self._append_lidar_safety_event(state=state, paused=pause_active)
            self.get_logger().info(f"LiDAR safety state changed to {state}")
            self._publish_status()

    def _append_lidar_safety_event(self, state, paused):
        get_state_hub = getattr(self.task_service, "get_state_hub", None)
        state_hub = get_state_hub() if callable(get_state_hub) else None
        if state_hub is None:
            visualization = getattr(self.node_manager, "visualization", None)
            get_from_visualization = getattr(visualization, "get", None)
            state_hub = get_from_visualization("state_hub") if callable(get_from_visualization) else None
        if state_hub is None:
            return

        get_active_task_id = getattr(state_hub, "get_active_task_id", None)
        task_id = get_active_task_id() if callable(get_active_task_id) else None
        state_hub.append_event(
            "lidar_safety_changed",
            task_id=task_id,
            level="warning" if paused else "info",
            payload={
                "state": state,
                "paused": bool(paused),
                "topic": self.lidar_safety_topic,
            },
        )

    def stop_navigation(self):
        return self.task_service.stop_navigation()

    def submit_text_navigation(self, goal_text, dry_run=False):
        return self.task_service.submit_text_navigation(goal_text, dry_run=dry_run)

    def _is_lidar_safety_paused(self):
        if getattr(self, "_lidar_pause_active", False):
            return True
        if hasattr(self.node_manager, "is_lidar_safety_paused"):
            return bool(self.node_manager.is_lidar_safety_paused())
        return False

    def _reject_manual_move_if_lidar_paused(self):
        if not self._is_lidar_safety_paused():
            return False
        self.task_service.set_terminal_status(
            "failed",
            success_flag=False,
            message=f"LiDAR safety pause active ({getattr(self, '_lidar_state', 'unknown')})",
            state=-4,
        )
        return True

    def _on_forward_cmd(self, msg):
        if self.task_service.is_busy():
            self.task_service.record_command_rejection(
                "Navigation task is running, manual move rejected",
                state=-3,
            )
            return
        if self._reject_manual_move_if_lidar_paused():
            return

        x = float(msg.linear.x)
        y = float(msg.linear.y)

        if math.isclose(x, 0.0, abs_tol=1e-6) and math.isclose(y, 0.0, abs_tol=1e-6):
            self.task_service.set_terminal_status(
                "failed",
                success_flag=False,
                message="linear.x/y are both zero",
                state=-2,
            )
            return

        if not math.isclose(x, 0.0, abs_tol=1e-6) and not math.isclose(y, 0.0, abs_tol=1e-6):
            self.task_service.set_terminal_status(
                "failed",
                success_flag=False,
                message="Set only one of linear.x or linear.y per command",
                state=-2,
            )
            return

        if not math.isclose(x, 0.0, abs_tol=1e-6):
            success_flag, info, state = self.node_manager.forward(x)
        else:
            success_flag, info, state = self.node_manager.shift(y)

        self.task_service.set_terminal_status(
            "completed" if success_flag else "failed",
            success_flag=bool(success_flag),
            message=str(info),
            state=self.task_service.normalize_state(state),
        )

    def _on_rotate_cmd(self, msg):
        if self.task_service.is_busy():
            self.task_service.record_command_rejection(
                "Navigation task is running, rotate command rejected",
                state=-3,
            )
            return
        if self._reject_manual_move_if_lidar_paused():
            return

        theta = float(msg.angular.z)
        if math.isclose(theta, 0.0, abs_tol=1e-6):
            self.task_service.set_terminal_status(
                "failed",
                success_flag=False,
                message="angular.z is zero",
                state=-2,
            )
            return

        success_flag, info, state = self.node_manager.rotate(theta)
        self.task_service.set_terminal_status(
            "completed" if success_flag else "failed",
            success_flag=bool(success_flag),
            message=str(info),
            state=self.task_service.normalize_state(state),
        )

    @staticmethod
    def _parse_relative_cmd_payload(raw):
        raw = str(raw or "").strip()
        if not raw:
            raise ValueError("relative command payload is empty")
        try:
            payload = json.loads(raw)
            return float(payload["distance_m"]), float(payload["direction_deg"])
        except json.JSONDecodeError:
            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                raise ValueError("expected JSON with distance_m/direction_deg or '<distance_m> <direction_deg>'")
            return float(parts[0]), float(parts[1])

    def _on_relative_cmd(self, msg):
        if self.task_service.is_busy():
            self.task_service.record_command_rejection(
                "Navigation task is running, relative move rejected",
                state=-3,
            )
            return
        if self._reject_manual_move_if_lidar_paused():
            return

        try:
            distance_m, direction_deg = self._parse_relative_cmd_payload(msg.data)
        except Exception as exc:
            self.task_service.set_terminal_status(
                "failed",
                success_flag=False,
                message=f"Invalid relative command: {exc}",
                state=-2,
            )
            return

        theta = math.radians(direction_deg)
        if math.isclose(distance_m, 0.0, abs_tol=1e-6) and math.isclose(theta, 0.0, abs_tol=1e-6):
            self.task_service.set_terminal_status(
                "failed",
                success_flag=False,
                message="distance_m and direction_deg are both zero",
                state=-2,
            )
            return

        if not math.isclose(theta, 0.0, abs_tol=1e-6):
            success_flag, info, state = self.node_manager.rotate(theta)
            if not success_flag:
                self.task_service.set_terminal_status(
                    "failed",
                    success_flag=False,
                    message=f"relative rotate failed: {info}",
                    state=self.task_service.normalize_state(state),
                )
                return

        if not math.isclose(distance_m, 0.0, abs_tol=1e-6):
            success_flag, info, state = self.node_manager.forward(distance_m)
        else:
            success_flag, info, state = True, "success.", 1

        self.task_service.set_terminal_status(
            "completed" if success_flag else "failed",
            success_flag=bool(success_flag),
            message=(
                f"relative move distance_m={distance_m:.3f}, "
                f"direction_deg={direction_deg:.1f}: {info}"
            ),
            state=self.task_service.normalize_state(state),
        )

    def _on_get_pose(self, _request, response):
        pose = self.node_manager.get_position()
        if pose is None:
            response.success = False
            response.message = json.dumps({"pose": None, "message": "pose unavailable"}, ensure_ascii=False)
            return response

        x, y, theta = pose
        response.success = True
        response.message = json.dumps({"pose": [x, y, theta]}, ensure_ascii=False)
        return response


@hydra.main(version_base=None, config_path=".", config_name=NAV_CONFIG_NAME)
def main(cfg):
    if _NAV_PROCESS_GUARD is None:
        raise NavigationStartupError(
            "navigation main requires the guarded __main__ launcher path"
        )
    prepare4log()
    logger.info("Starting nav ROS bridge...")
    logger.info("Hydra config loaded: %s", NAV_CONFIG_NAME)

    map_ctx = prepare_map(cfg)
    visualization = prepare_visualization(cfg, map_ctx)
    node_manager = NodeManager(
        cfg,
        map_ctx["map_image"],
        map_ctx["origin"],
        map_ctx["resolution"],
        visualization=visualization,
    )
    bridge = NavRosBridge(cfg, map_ctx, node_manager)
    if visualization is not None:
        visualization["app"].state.stop_navigation = bridge.task_service.stop_navigation
        visualization["app"].state.submit_text_navigation = bridge.task_service.submit_text_navigation

    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        logger.info("Received KeyboardInterrupt, shutting down.")
    finally:
        try:
            node_manager.request_stop()
        except Exception:
            pass
        try:
            bridge.task_service.join_task(timeout=1.0)
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        try:
            node_manager.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        if visualization is not None:
            frame_pipeline = visualization.get("frame_pipeline")
            if frame_pipeline is not None:
                try:
                    frame_pipeline.stop(timeout=1.0)
                except Exception:
                    pass
            try:
                visualization["server"].should_exit = True
            except Exception:
                pass
            try:
                visualization["thread"].join(timeout=1.0)
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    finally:
        _NAV_PROCESS_GUARD.__exit__(None, None, None)
