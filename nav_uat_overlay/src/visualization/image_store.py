from threading import Lock
from time import time

import cv2


class ImageStore:
    def __init__(self, jpeg_quality=80, rgb_preview_fps=2.0, depth_preview_fps=1.0):
        self.jpeg_quality = int(jpeg_quality)
        self._lock = Lock()
        self._versions = {}
        self._payloads = {}
        self._last_update_at = {}

    def _get_existing_version(self, key):
        existing = self._payloads.get(key)
        if existing is None:
            return None
        return existing["version"]

    def store_encoded(self, key, content, content_type="image/jpeg", updated_at=None):
        timestamp = time() if updated_at is None else float(updated_at)
        with self._lock:
            version = self._versions.get(key, 0) + 1
            self._versions[key] = version
            self._last_update_at[key] = timestamp
            self._payloads[key] = {
                "version": version,
                "content_type": content_type,
                "bytes": content,
                "updated_at": timestamp,
            }
            return version

    def _update_jpeg(self, key, frame):
        success, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not success:
            raise RuntimeError(f"failed to encode frame for {key}")
        return self.store_encoded(key, encoded.tobytes(), content_type="image/jpeg")

    def update_rgb_latest(self, frame):
        return self._update_jpeg("rgb_latest", frame)

    def update_frame(self, key, frame):
        return self._update_jpeg(key, frame)

    def update_depth_latest(self, frame):
        return self._update_jpeg("depth_latest", frame)

    def update_rgb_vpr(self, frame):
        return self._update_jpeg("rgb_vpr", frame)

    def update_rgb_navdp(self, frame):
        return self._update_jpeg("rgb_navdp", frame)

    def get(self, key):
        with self._lock:
            return self._payloads.get(key)
