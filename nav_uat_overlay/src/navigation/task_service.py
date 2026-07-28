import json
import logging
import threading
import uuid

from utils.utils import convert_text2coordinate


logger = logging.getLogger(__name__)


class NavigationTaskService:
    def __init__(self, cfg, map_ctx, node_manager, visualization=None, publish_status=None):
        self.cfg = cfg
        self.map_ctx = map_ctx
        self.node_manager = node_manager
        self.visualization = visualization
        self.publish_status = publish_status or (lambda: None)

        self._status_lock = threading.Lock()
        self._task_thread = None
        self._status = {
            "status": "idle",
            "result_status": None,
            "success_flag": None,
            "message": "ready",
            "state": None,
            "task_id": None,
            "goal_text": None,
            "dry_run": False,
        }

    def join_task(self, timeout=None):
        task_thread = self._task_thread
        if task_thread is None or not task_thread.is_alive():
            return False
        task_thread.join(timeout=timeout)
        return True

    def get_status_payload(self):
        with self._status_lock:
            return dict(self._status)

    @staticmethod
    def normalize_state(state):
        if state is None:
            return None
        if isinstance(state, (int, float)):
            return int(state)
        try:
            return int(state)
        except (TypeError, ValueError):
            return str(state)

    def build_goal_kwargs(self):
        kwargs = {
            "map_image": self.map_ctx["map_image"],
            "origin": self.map_ctx["origin"],
            "resolution": self.map_ctx["resolution"],
            "window_size": self.map_ctx["window_size"],
            "free_points": self.map_ctx["free_points"],
        }
        if self.cfg.goal_recognition.enable_embedding:
            kwargs["reference_coordinates"] = self.map_ctx["reference_coordinates"]
            kwargs["reference_caption"] = self.map_ctx["reference_caption"]
        else:
            kwargs["system_prompt"] = self.map_ctx["system_prompt"]
        return kwargs

    def is_busy(self):
        with self._status_lock:
            status = self._status.get("status")
        return status == "processing" and self._task_thread is not None and self._task_thread.is_alive()

    def get_state_hub(self):
        if not self.visualization:
            return None
        return self.visualization.get("state_hub")

    def set_status(self, **kwargs):
        with self._status_lock:
            self._status.update(kwargs)
            status_payload = dict(self._status)
        state_hub = self.get_state_hub()
        if state_hub is not None:
            state_hub.update_task_status(**status_payload)
        try:
            self.publish_status()
        except Exception as exc:
            if "context is invalid" in str(exc):
                logger.debug("Skipping status publish because ROS context is invalid: %s", exc)
            else:
                raise

    def record_command_rejection(self, message, state=-3, task_id=None):
        state_hub = self.get_state_hub()
        if task_id is None and state_hub is not None:
            task_id = state_hub.get_active_task_id()
        if state_hub is not None:
            state_hub.append_event(
                "command_rejected",
                task_id=task_id,
                level="warning",
                payload={"message": str(message), "state": self.normalize_state(state)},
            )
        return {
            **self.get_status_payload(),
            "accepted": False,
            "message": str(message),
            "state": self.normalize_state(state),
        }

    def set_terminal_status(
        self,
        terminal_status,
        *,
        success_flag,
        message,
        state,
        task_id=None,
        goal_text=None,
        dry_run=None,
    ):
        payload = {
            "status": "idle",
            "result_status": terminal_status,
            "success_flag": bool(success_flag),
            "message": str(message),
            "state": self.normalize_state(state),
        }
        if task_id is not None:
            payload["task_id"] = task_id
        if goal_text is not None:
            payload["goal_text"] = goal_text
        if dry_run is not None:
            payload["dry_run"] = bool(dry_run)
        self.set_status(**payload)

    def start_visualization_task(self, task_id, goal_text, dry_run=False, source="unknown"):
        state_hub = self.get_state_hub()
        if state_hub is None:
            return
        state_hub.start_task_recording(task_id=task_id, goal_text=goal_text)
        state_hub.append_event(
            "task_received",
            task_id=task_id,
            payload={"goal_text": goal_text, "dry_run": bool(dry_run), "source": source},
        )

    def finalize_visualization_task(self, task_id, status):
        state_hub = self.get_state_hub()
        if state_hub is None:
            return
        terminal_event = {
            "completed": "task_completed",
            "failed": "task_failed",
            "stopped": "task_stopped",
        }.get(status)
        if terminal_event is not None:
            state_hub.append_event(
                terminal_event,
                task_id=task_id,
                payload={"status": status},
            )
        state_hub.finalize_task_recording(task_id=task_id, status=status)

    def process_navigation(self, task_id, goal_text, dry_run=False):
        try:
            goal_kwargs = self.build_goal_kwargs()
            x, y, theta = convert_text2coordinate(self.cfg.goal_recognition, goal_text, **goal_kwargs)
            if x is None or y is None or theta is None:
                self.set_terminal_status(
                    "failed",
                    success_flag=False,
                    message="Skipping navigation due to invalid pose data.",
                    state=-2,
                    task_id=task_id,
                    goal_text=goal_text,
                    dry_run=dry_run,
                )
                self.finalize_visualization_task(task_id=task_id, status="failed")
                return

            state_hub = self.get_state_hub()
            if state_hub is not None:
                state_hub.update_goal(x=x, y=y, theta=theta, source="goal_recognition")
                state_hub.append_event(
                    "goal_resolved",
                    task_id=task_id,
                    payload={"pose": {"x": x, "y": y, "theta": theta}, "dry_run": dry_run},
                )
                if dry_run:
                    state_hub.append_event(
                        "dry_run_enabled",
                        task_id=task_id,
                        level="warning",
                        payload={"goal_text": goal_text},
                    )

            success_flag, info, state = self.node_manager.navigation([x, y, theta], dry_run=dry_run)
            status_text = "completed" if success_flag else "failed"
            if state == -1:
                status_text = "stopped"

            self.set_terminal_status(
                status_text,
                success_flag=bool(success_flag),
                message=str(info),
                state=self.normalize_state(state),
                task_id=task_id,
                goal_text=goal_text,
                dry_run=dry_run,
            )
            self.finalize_visualization_task(task_id=task_id, status=status_text)
        except Exception as exc:
            logger.exception("Navigation task failed")
            self.set_terminal_status(
                "failed",
                success_flag=False,
                message=str(exc),
                state=-1,
                task_id=task_id,
                goal_text=goal_text,
                dry_run=dry_run,
            )
            self.finalize_visualization_task(task_id=task_id, status="failed")

    def submit_navigation_request(self, goal_text, dry_run=False, source="unknown"):
        goal_text = goal_text.strip()
        dry_run = bool(dry_run)

        if not goal_text:
            self.set_terminal_status(
                "failed",
                success_flag=False,
                message="Goal text is required",
                state=-2,
                dry_run=dry_run,
            )
            return {"accepted": False, **self.get_status_payload()}

        if self.is_busy():
            return self.record_command_rejection(
                "Another navigation task is already running",
                state=-3,
            )

        task_id = str(uuid.uuid4())
        self.start_visualization_task(task_id=task_id, goal_text=goal_text, dry_run=dry_run, source=source)
        self.set_status(
            status="processing",
            result_status=None,
            success_flag=None,
            message="Navigation task accepted",
            state=None,
            task_id=task_id,
            goal_text=goal_text,
            dry_run=dry_run,
        )

        self._task_thread = threading.Thread(
            target=self.process_navigation,
            args=(task_id, goal_text, dry_run),
            daemon=True,
        )
        self._task_thread.start()
        return {"accepted": True, **self.get_status_payload()}

    def handle_stop_request(self, source):
        state_hub = self.get_state_hub()
        active_task_id = state_hub.get_active_task_id() if state_hub is not None else None
        if state_hub is not None and active_task_id is not None:
            state_hub.append_event(
                "stop_requested",
                task_id=active_task_id,
                payload={"source": source},
                level="warning",
            )

        success_flag, info, state = self.node_manager.stop()
        self.set_terminal_status(
            "stopped",
            success_flag=bool(success_flag),
            message=str(info),
            state=self.normalize_state(state),
        )

        if active_task_id is not None and not self.is_busy():
            self.finalize_visualization_task(task_id=active_task_id, status="stopped")

        return {
            "success": bool(success_flag),
            "message": str(info),
            "state": self.normalize_state(state),
            "task_id": active_task_id,
        }

    def submit_text_navigation(self, goal_text, dry_run=False):
        return self.submit_navigation_request(goal_text, dry_run=dry_run, source="webviz")

    def stop_navigation(self):
        return self.handle_stop_request(source="webviz")
