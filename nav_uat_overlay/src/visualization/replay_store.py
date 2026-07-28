from copy import deepcopy
import json
import logging
import os
import shutil
from pathlib import Path
from threading import Lock
from time import time


logger = logging.getLogger(__name__)


class ReplayStore:
    def __init__(self, history_dir, max_replay_mb=512):
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.max_replay_bytes = int(max_replay_mb * 1024 * 1024)
        self._lock = Lock()
        self._tasks = {}

    def start_task(self, task_id, goal_text):
        task_dir = self.history_dir / f"{int(time() * 1000)}-{task_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._tasks[task_id] = {
                "dir": task_dir,
                "goal_text": goal_text,
                "started_at": time(),
                "latest_frames": {},
            }
        return task_dir

    def append_event(self, task_id, event):
        if task_id is None:
            return False
        with self._lock:
            task_info = self._tasks.get(task_id)
        if task_info is None:
            return False
        task_dir = task_info["dir"]
        events_path = task_dir / "events.jsonl"
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True

    def store_frame(self, task_id, key, version, content, content_type="image/jpeg", updated_at=None):
        with self._lock:
            task_info = self._tasks.get(task_id)
            if task_info is None:
                return None

        version = int(version)
        frames_dir = task_info["dir"] / "frames" / key
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_path = self._frame_path(frames_dir, version, content_type)
        frame_path.write_bytes(content)

        metadata = {
            "version": version,
            "content_type": content_type,
            "updated_at": time() if updated_at is None else float(updated_at),
        }
        self._write_json_file(self._frame_metadata_path(frames_dir, version), metadata)

        with self._lock:
            latest_frames = self._tasks.get(task_id, {}).get("latest_frames")
            if latest_frames is not None:
                latest_frames[key] = metadata

        return self.build_frame_url(task_id, key, version)

    def finalize_task(self, task_id, status, snapshot=None):
        with self._lock:
            task_info = self._tasks.pop(task_id, None)
        if task_info is None:
            return False

        meta = {
            "task_id": task_id,
            "goal_text": task_info["goal_text"],
            "status": status,
            "started_at": task_info["started_at"],
            "ended_at": time(),
        }
        self._write_json_file(task_info["dir"] / "task_meta.json", meta)
        if snapshot is not None:
            snapshot = self._rewrite_snapshot_image_urls(task_id, snapshot, task_info["latest_frames"])
            self._write_json_file(task_info["dir"] / "snapshot.json", snapshot)
        self._prune_oldest_if_needed()
        return True

    def list_tasks(self, limit=50):
        tasks = []
        for task_dir in self.history_dir.iterdir():
            if not task_dir.is_dir():
                continue
            meta_path = task_dir / "task_meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_json_file(meta_path, "task metadata")
            if meta is None:
                continue
            summary = dict(meta)
            summary["has_snapshot"] = self._read_json_file(task_dir / "snapshot.json", "task snapshot") is not None
            summary["event_count"] = self._count_events(task_dir / "events.jsonl")
            tasks.append(summary)
        tasks.sort(key=lambda item: item.get("ended_at") or item.get("started_at") or 0, reverse=True)
        return tasks[: int(limit)]

    def get_task(self, task_id):
        task_dir = self._find_task_dir(task_id)
        if task_dir is None:
            return None
        meta_path = task_dir / "task_meta.json"
        if not meta_path.exists():
            return None
        meta = self._read_json_file(meta_path, "task metadata")
        if meta is None:
            return None
        meta["has_snapshot"] = self._read_json_file(task_dir / "snapshot.json", "task snapshot") is not None
        meta["event_count"] = self._count_events(task_dir / "events.jsonl")
        return meta

    def load_events(self, task_id):
        task_dir = self._find_task_dir(task_id)
        if task_dir is None:
            return None
        events_path = task_dir / "events.jsonl"
        if not events_path.exists():
            return []
        events = []
        for line_no, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping invalid replay event in %s at line %d", events_path, line_no)
        return events

    def load_snapshot(self, task_id):
        task_dir = self._find_task_dir(task_id)
        if task_dir is None:
            return None
        snapshot_path = task_dir / "snapshot.json"
        if not snapshot_path.exists():
            return None
        return self._read_json_file(snapshot_path, "task snapshot")

    def load_frame(self, task_id, key, version):
        task_dir = self._find_task_dir(task_id)
        if task_dir is None:
            return None
        version = int(version)
        frame_dir = task_dir / "frames" / key
        metadata_path = self._frame_metadata_path(frame_dir, version)
        if not metadata_path.exists():
            return None
        metadata = self._read_json_file(metadata_path, "frame metadata")
        if metadata is None:
            return None
        frame_path = self._frame_path(
            frame_dir,
            version,
            metadata.get("content_type", "application/octet-stream"),
        )
        if not frame_path.exists():
            return None
        return {
            "version": version,
            "content_type": metadata.get("content_type", "application/octet-stream"),
            "updated_at": metadata.get("updated_at"),
            "bytes": frame_path.read_bytes(),
        }

    def _prune_oldest_if_needed(self):
        task_dirs = sorted(
            [path for path in self.history_dir.iterdir() if path.is_dir()],
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        if self.max_replay_bytes <= 0:
            while len(task_dirs) > 1:
                oldest = task_dirs.pop(0)
                shutil.rmtree(oldest, ignore_errors=True)
            return
        while len(task_dirs) > 1 and self._directory_size(self.history_dir) > self.max_replay_bytes:
            oldest = task_dirs.pop(0)
            shutil.rmtree(oldest, ignore_errors=True)

    def _find_task_dir(self, task_id):
        for task_dir in self.history_dir.iterdir():
            if not task_dir.is_dir():
                continue
            meta_path = task_dir / "task_meta.json"
            if not meta_path.exists():
                continue
            meta = self._read_json_file(meta_path, "task metadata")
            if meta is None:
                continue
            if meta.get("task_id") == task_id:
                return task_dir
        return None

    @staticmethod
    def _write_json_file(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        temp_path.replace(path)

    @staticmethod
    def _read_json_file(path, description):
        path = Path(path)
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s from %s: %s", description, path, exc)
            return None
        if not content.strip():
            logger.warning("Skipping empty %s file: %s", description, path)
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            logger.warning("Skipping invalid %s file %s: %s", description, path, exc)
            return None

    @staticmethod
    def _count_events(events_path):
        if not events_path.exists():
            return 0
        return len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line.strip()])

    @staticmethod
    def _directory_size(path):
        total = 0
        for child in path.rglob("*"):
            if child.is_file():
                total += child.stat().st_size
        return total

    @staticmethod
    def build_frame_url(task_id, key, version):
        group, _, name = key.partition("_")
        if not group or not name:
            raise ValueError(f"invalid frame key: {key}")
        version = int(version)
        return f"/viz/api/replay/tasks/{task_id}/frame/{group}/{name}/{version}.jpg"

    def _rewrite_snapshot_image_urls(self, task_id, snapshot, latest_frames):
        rewritten = deepcopy(snapshot)
        images = rewritten.get("images", {})
        for key, frame_info in latest_frames.items():
            image_ref = images.get(key)
            if image_ref is None:
                continue
            image_ref["url"] = self.build_frame_url(task_id, key, frame_info["version"])
            image_ref["version"] = frame_info["version"]
            image_ref["updated_at"] = frame_info["updated_at"]
        return rewritten

    @staticmethod
    def _frame_metadata_path(frame_dir, version):
        return frame_dir / f"{int(version)}.json"

    @staticmethod
    def _frame_path(frame_dir, version, content_type):
        suffix = ".jpg" if content_type == "image/jpeg" else ""
        return frame_dir / f"{int(version)}{suffix}"
