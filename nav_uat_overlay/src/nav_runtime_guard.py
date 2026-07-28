"""Process-level safety guard for the navigation runtime entry point."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Iterator


LAUNCH_AUTH_ENV = "GROOT_NAV_LAUNCH_AUTHORIZED"
LOCK_FILE_ENV = "GROOT_NAV_LOCK_FILE"
DEFAULT_LOCK_FILE = "/tmp/groot_nav_run_ros.lock"


class NavigationStartupError(RuntimeError):
    """Raised before ROS initialization when navigation must not start."""


def require_authorized_launch(environ: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environ is None else environ
    if environment.get(LAUNCH_AUTH_ENV) != "1":
        raise NavigationStartupError(
            "direct run_ros.py startup is locked; use "
            "nav_uat_overlay/start_nav_refactored.sh with explicit human approval"
        )


class NavigationInstanceLock:
    """Nonblocking flock retained for the complete navigation process lifetime."""

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_LOCK_FILE):
        self.path = Path(path).expanduser()
        self._stream: IO[str] | None = None

    def acquire(self) -> "NavigationInstanceLock":
        if self._stream is not None:
            return self
        try:
            stream = self.path.open("a+", encoding="utf-8")
        except OSError as exc:
            raise NavigationStartupError(
                f"cannot open navigation instance lock {self.path}: {exc}"
            ) from exc
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise NavigationStartupError(
                f"another run_ros.py instance owns {self.path}"
            ) from exc
        except OSError as exc:
            stream.close()
            raise NavigationStartupError(
                f"cannot acquire navigation instance lock {self.path}: {exc}"
            ) from exc

        try:
            stream.seek(0)
            stream.truncate()
            stream.write(f"{os.getpid()}\n")
            stream.flush()
        except OSError as exc:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            finally:
                stream.close()
            raise NavigationStartupError(
                f"cannot initialize navigation instance lock {self.path}: {exc}"
            ) from exc
        self._stream = stream
        return self

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> "NavigationInstanceLock":
        return self.acquire()

    def __exit__(self, *_exc: object) -> None:
        self.close()


@contextmanager
def authorized_navigation_process(
    *,
    environ: Mapping[str, str] | None = None,
    lock_path: str | os.PathLike[str] | None = None,
) -> Iterator[NavigationInstanceLock]:
    environment = os.environ if environ is None else environ
    require_authorized_launch(environment)
    resolved_lock = lock_path or environment.get(LOCK_FILE_ENV, DEFAULT_LOCK_FILE)
    with NavigationInstanceLock(resolved_lock) as instance_lock:
        yield instance_lock
