from collections import deque
from threading import Lock
from time import time

from .models import (
    SCHEMA_VERSION,
    build_snapshot,
    default_goal_state,
    default_images_state,
    default_planner_state,
    default_robot_state,
    default_task_state,
)

_UNSET = object()


class VisualizationStateHub:
    def __init__(self, max_event_buffer=2000, replay_store=None, save_replay=False):
        self._lock = Lock()
        self._task = default_task_state()
        self._goal = default_goal_state()
        self._robot = default_robot_state()
        self._planner = default_planner_state()
        self._images = default_images_state()
        self._events = deque(maxlen=max_event_buffer)
        self.schema_version = SCHEMA_VERSION
        self._replay_store = replay_store
        self._save_replay = save_replay
        self._active_task_id = None
        self._broadcast = None

    def set_broadcast(self, broadcast):
        self._broadcast = broadcast

    def _emit(self, message):
        if self._broadcast is not None:
            self._broadcast(message)

    def _resolve_replay_task_id(self, task_id=None, allow_active_task=False):
        if not self._save_replay or self._replay_store is None:
            return None

        with self._lock:
            active_task_id = self._active_task_id

        if active_task_id is None:
            return None
        if task_id is not None and task_id != active_task_id:
            return None
        if task_id is None and not allow_active_task:
            return None
        return active_task_id

    def _append_replay_message(self, message, task_id=None, allow_active_task=False, replay_message=None):
        replay_task_id = self._resolve_replay_task_id(task_id=task_id, allow_active_task=allow_active_task)
        if replay_task_id is None:
            return
        self._replay_store.append_event(replay_task_id, replay_message or message)

    def _build_replay_image_message(self, key, version, updated_at, replay_url):
        if replay_url is None:
            return None
        return {
            "type": "image_update",
            "schema_version": self.schema_version,
            "timestamp": updated_at,
            "images": {
                key: {
                    "url": replay_url,
                    "version": version,
                    "updated_at": updated_at,
                }
            },
        }

    def _build_message(self, message_type, **payload):
        return {
            "type": message_type,
            "schema_version": self.schema_version,
            "timestamp": time(),
            **payload,
        }

    def update_task_status(
        self,
        task_id=_UNSET,
        status=_UNSET,
        result_status=_UNSET,
        goal_text=_UNSET,
        success_flag=_UNSET,
        message=_UNSET,
        state=_UNSET,
        dry_run=_UNSET,
    ):
        with self._lock:
            if task_id is not _UNSET:
                self._task["task_id"] = task_id
            if status is not _UNSET:
                self._task["status"] = status
            if result_status is not _UNSET:
                self._task["result_status"] = result_status
            if goal_text is not _UNSET:
                self._task["goal_text"] = goal_text
            if success_flag is not _UNSET:
                self._task["success_flag"] = success_flag
            if message is not _UNSET:
                self._task["message"] = message
            if state is not _UNSET:
                self._task["state"] = state
            if dry_run is not _UNSET:
                self._task["dry_run"] = bool(dry_run)
            task_payload = dict(self._task)
        message = self._build_message("task_status", **task_payload)
        self._append_replay_message(message, task_id=message.get("task_id"), allow_active_task=True)
        self._emit(message)

    def update_goal(self, x, y, theta, source=None):
        with self._lock:
            self._goal["pose"] = {"x": x, "y": y, "theta": theta}
            self._goal["source"] = source
            goal_payload = dict(self._goal)
        message = self._build_message("goal_update", goal=goal_payload)
        self._append_replay_message(message, allow_active_task=True)
        self._emit(message)

    def update_pose(self, pose=None, vpr_pose=None):
        with self._lock:
            if pose is not None:
                self._robot["pose"] = dict(pose)
            if vpr_pose is not None:
                self._robot["vpr_pose"] = dict(vpr_pose)
            robot_payload = dict(self._robot)
        message = self._build_message("pose_update", **robot_payload)
        self._append_replay_message(message, allow_active_task=True)
        self._emit(message)

    def update_robot(self, pose=None, vpr_pose=None):
        self.update_pose(pose=pose, vpr_pose=vpr_pose)

    def update_planner(
        self,
        mode=None,
        global_path=None,
        waypoints=None,
        local_path=None,
        local_goal=_UNSET,
        actions=None,
        action_limit=_UNSET,
        preview_actions=None,
        camera_intrinsics=_UNSET,
        camera_image_size=_UNSET,
    ):
        with self._lock:
            if mode is not None:
                self._planner["mode"] = mode
            if global_path is not None:
                self._planner["global_path"] = list(global_path)
            if waypoints is not None:
                self._planner["waypoints"] = list(waypoints)
            if local_path is not None:
                self._planner["local_path"] = list(local_path)
            if local_goal is not _UNSET:
                self._planner["local_goal"] = list(local_goal) if local_goal is not None else None
            if actions is not None:
                self._planner["actions"] = list(actions)
            if action_limit is not _UNSET:
                self._planner["action_limit"] = int(action_limit) if action_limit is not None else None
            if preview_actions is not None:
                self._planner["preview_actions"] = list(preview_actions)
            if camera_intrinsics is not _UNSET:
                self._planner["camera_intrinsics"] = (
                    [list(row) for row in camera_intrinsics]
                    if camera_intrinsics is not None
                    else None
                )
            if camera_image_size is not _UNSET:
                self._planner["camera_image_size"] = (
                    [int(value) for value in camera_image_size]
                    if camera_image_size is not None
                    else None
                )
            planner_payload = dict(self._planner)
        message = self._build_message("planner_update", **planner_payload)
        self._append_replay_message(message, allow_active_task=True)
        self._emit(message)

    def persist_image_frame(self, key, version, content, content_type="image/jpeg", updated_at=None):
        replay_task_id = self._resolve_replay_task_id(allow_active_task=True)
        if replay_task_id is None:
            return None
        return self._replay_store.store_frame(
            replay_task_id,
            key=key,
            version=version,
            content=content,
            content_type=content_type,
            updated_at=updated_at,
        )

    def update_image(self, key, url, version, updated_at=None, replay_url=None):
        with self._lock:
            image_updated_at = updated_at if updated_at is not None else time()
            self._images[key] = {
                "url": url,
                "version": version,
                "updated_at": image_updated_at,
            }
            image_payload = {key: dict(self._images[key])}
        message = self._build_message("image_update", images=image_payload)
        replay_message = self._build_replay_image_message(
            key=key,
            version=version,
            updated_at=image_updated_at,
            replay_url=replay_url,
        )
        if replay_message is not None:
            replay_message["timestamp"] = message["timestamp"]
        self._append_replay_message(message, allow_active_task=True, replay_message=replay_message)
        self._emit(message)

    def update_images(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                self._images[key] = value
            images_payload = {
                key: (dict(value) if isinstance(value, dict) else value)
                for key, value in kwargs.items()
            }
        message = self._build_message("image_update", images=images_payload)
        self._append_replay_message(message, allow_active_task=True)
        self._emit(message)

    def append_event(self, event_name, task_id=None, payload=None, level="info"):
        event = {
            "type": "event",
            "schema_version": self.schema_version,
            "timestamp": time(),
            "event_name": event_name,
            "level": level,
            "task_id": task_id,
            "payload": payload or {},
        }
        with self._lock:
            self._events.append(event)
        self._append_replay_message(event, task_id=task_id)
        self._emit(event)
        return event

    def build_snapshot(self):
        with self._lock:
            return build_snapshot(
                task=self._task,
                goal=self._goal,
                robot=self._robot,
                planner=self._planner,
                images=self._images,
                events=list(self._events),
                timestamp=time(),
            )

    def start_task_recording(self, task_id, goal_text):
        with self._lock:
            self._active_task_id = task_id
        if self._save_replay and self._replay_store is not None:
            return self._replay_store.start_task(task_id=task_id, goal_text=goal_text)
        return None

    def finalize_task_recording(self, task_id, status):
        with self._lock:
            is_active = task_id == self._active_task_id
            snapshot = None
            if is_active:
                snapshot = build_snapshot(
                    task=self._task,
                    goal=self._goal,
                    robot=self._robot,
                    planner=self._planner,
                    images=self._images,
                    events=list(self._events),
                    timestamp=time(),
                )
                self._active_task_id = None
        if self._save_replay and self._replay_store is not None and is_active:
            self._replay_store.finalize_task(task_id=task_id, status=status, snapshot=snapshot)

    def get_active_task_id(self):
        with self._lock:
            return self._active_task_id
