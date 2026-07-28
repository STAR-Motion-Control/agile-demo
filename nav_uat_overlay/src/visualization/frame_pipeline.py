from threading import Event, RLock, Thread
from time import time

import cv2


class FramePipeline:
    def __init__(self, image_store, state_hub, stream_fps=None, poll_interval=0.02, autostart=True):
        self._image_store = image_store
        self._state_hub = state_hub
        self._poll_interval = float(poll_interval)
        self._lock = RLock()
        self._stop_event = Event()
        self._wake_event = Event()
        self._thread = None
        self._mailboxes = {}

        for key, config in (stream_fps or {}).items():
            if isinstance(config, dict):
                self.register_stream(
                    key,
                    fps=config.get("fps", 0.0),
                    preview_size=config.get("preview_size"),
                )
                continue
            self.register_stream(key, fps=config)

        if autostart:
            self.start()

    def register_stream(self, key, fps=0.0, preview_size=None):
        min_interval = 0.0 if float(fps) <= 0 else 1.0 / float(fps)
        normalized_preview_size = self._normalize_preview_size(preview_size)
        with self._lock:
            if key in self._mailboxes:
                self._mailboxes[key]["min_interval"] = min_interval
                self._mailboxes[key]["preview_size"] = normalized_preview_size
                return
            self._mailboxes[key] = {
                "frame": None,
                "captured_at": 0.0,
                "sequence": 0,
                "encoded_sequence": 0,
                "last_encoded_at": 0.0,
                "min_interval": min_interval,
                "preview_size": normalized_preview_size,
                "failure_count": 0,
            }

    def submit_frame(self, key, frame, captured_at=None):
        timestamp = time() if captured_at is None else float(captured_at)
        with self._lock:
            if key not in self._mailboxes:
                self.register_stream(key)
            mailbox = self._mailboxes[key]
            mailbox["frame"] = frame
            mailbox["captured_at"] = timestamp
            mailbox["sequence"] += 1
            self._wake_event.set()
            return mailbox["sequence"]

    def process_once(self):
        processed = False
        now = time()
        with self._lock:
            pending = []
            for key, mailbox in self._mailboxes.items():
                if mailbox["frame"] is None:
                    continue
                if mailbox["sequence"] == mailbox["encoded_sequence"]:
                    continue
                if mailbox["min_interval"] > 0 and now - mailbox["last_encoded_at"] < mailbox["min_interval"]:
                    continue
                pending.append(
                    (
                        key,
                        mailbox["frame"],
                        mailbox["captured_at"],
                        mailbox["sequence"],
                        mailbox["preview_size"],
                        now,
                    )
                )

        for key, frame, captured_at, sequence, preview_size, encode_started_at in pending:
            processed = True
            try:
                frame = self._prepare_frame(frame, preview_size)
                success, encoded = cv2.imencode(
                    ".jpg",
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, self._image_store.jpeg_quality],
                )
                if not success:
                    raise RuntimeError(f"failed to encode frame for {key}")
                version = self._image_store.store_encoded(
                    key,
                    encoded.tobytes(),
                    content_type="image/jpeg",
                    updated_at=captured_at,
                )
                replay_url = self._state_hub.persist_image_frame(
                    key,
                    version=version,
                    content=encoded.tobytes(),
                    content_type="image/jpeg",
                    updated_at=captured_at,
                )
                self._state_hub.update_image(
                    key,
                    url=self._build_frame_url(key, version),
                    version=version,
                    updated_at=captured_at,
                    replay_url=replay_url,
                )
                with self._lock:
                    mailbox = self._mailboxes.get(key)
                    if mailbox is None:
                        continue
                    mailbox["encoded_sequence"] = sequence
                    mailbox["last_encoded_at"] = encode_started_at
                    mailbox["failure_count"] = 0
            except Exception as exc:
                with self._lock:
                    mailbox = self._mailboxes.get(key)
                    if mailbox is None:
                        continue
                    mailbox["encoded_sequence"] = sequence
                    mailbox["failure_count"] += 1
                    failure_count = mailbox["failure_count"]
                if failure_count == 1 or failure_count % 10 == 0:
                    self._state_hub.append_event(
                        "image_encoding_failed",
                        task_id=self._state_hub.get_active_task_id(),
                        level="warning",
                        payload={"key": key, "message": str(exc), "failure_count": failure_count},
                    )
        return processed

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout=1.0):
        self._stop_event.set()
        self._wake_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self):
        while not self._stop_event.is_set():
            processed = self.process_once()
            if processed:
                continue
            timeout = self._get_wait_timeout()
            self._wake_event.wait(timeout)
            self._wake_event.clear()

    def _get_wait_timeout(self):
        now = time()
        next_wait = None
        with self._lock:
            for mailbox in self._mailboxes.values():
                if mailbox["frame"] is None:
                    continue
                if mailbox["sequence"] == mailbox["encoded_sequence"]:
                    continue
                if mailbox["min_interval"] <= 0:
                    return 0.0
                remaining = mailbox["min_interval"] - (now - mailbox["last_encoded_at"])
                if remaining <= 0:
                    return 0.0
                next_wait = remaining if next_wait is None else min(next_wait, remaining)
        if next_wait is None:
            return self._poll_interval
        return max(0.0, next_wait)

    @staticmethod
    def _normalize_preview_size(preview_size):
        if not preview_size:
            return None
        width, height = preview_size
        width = int(width)
        height = int(height)
        if width <= 0 or height <= 0:
            return None
        return (width, height)

    @staticmethod
    def _prepare_frame(frame, preview_size):
        if preview_size is None:
            return frame
        target_width, target_height = preview_size
        height, width = frame.shape[:2]
        if width == target_width and height == target_height:
            return frame
        interpolation = cv2.INTER_AREA if target_width < width or target_height < height else cv2.INTER_LINEAR
        return cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

    @staticmethod
    def _build_frame_url(key, version):
        group, _, name = key.partition("_")
        if not group or not name:
            raise ValueError(f"invalid frame key: {key}")
        return f"/viz/api/frame/{group}/{name}.jpg?t={version}"
