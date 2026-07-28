"""Adapter from the legacy GrootMover writer signature to MotionBusClient."""

from __future__ import annotations

import threading
from typing import Any

from .motion_bus import MotionBusClient


_registry_lock = threading.Lock()
_clients: dict[tuple[str, str], MotionBusClient] = {}


def _shared_client(source: str, socket_path: str) -> MotionBusClient:
    key = (source, socket_path)
    with _registry_lock:
        client = _clients.get(key)
        if client is None:
            client = MotionBusClient(source, socket_path)
            _clients[key] = client
        return client


class MotionBusCommandSink:
    """Callable accepted by the refactored GrootMover."""

    def __init__(
        self,
        *,
        source: str,
        socket_path: str = "/tmp/groot_motion_bus.sock",
        lease_s: float = 0.35,
    ):
        self.source = source
        self.socket_path = socket_path
        self.lease_s = max(0.10, float(lease_s))
        self._client = _shared_client(source, socket_path)
        self._hold_lock = threading.Lock()
        self._hold_stop = threading.Event()
        self._hold_thread: threading.Thread | None = None
        self._hold_error: Exception | None = None

    def _stop_hold(self) -> None:
        with self._hold_lock:
            thread = self._hold_thread
            self._hold_stop.set()
            self._hold_thread = None
        if (
            thread is not None
            and thread is not threading.current_thread()
            and thread.is_alive()
        ):
            thread.join(timeout=0.5)

    def raise_if_hold_failed(self) -> None:
        if self._hold_error is not None:
            raise RuntimeError("motion-bus held mode lost") from self._hold_error

    def __call__(
        self,
        _cmd_file: str,
        fsm: str | None,
        forward: float = 0.0,
        lateral: float = 0.0,
        yaw: float = 0.0,
        height: float | None = None,
        duration: float | None = None,
        estop: bool = False,
        limp: bool = False,
        allow_recovery: bool = False,
        defer_recovery: bool = False,
        **_ignored: Any,
    ) -> None:
        del duration
        self._stop_hold()
        self.raise_if_hold_failed()
        self._client.publish(
            fsm=fsm or "RL_FULL",
            forward=forward,
            lateral=lateral,
            yaw=yaw,
            height=0.76 if height is None else height,
            lease_s=5.0 if estop or limp else self.lease_s,
            allow_recovery=allow_recovery,
            defer_recovery=defer_recovery,
            estop=estop,
            limp=limp,
        )

    def hold(
        self,
        *,
        fsm: str,
        height: float,
        allow_recovery: bool = False,
        defer_recovery: bool = False,
    ) -> None:
        self._stop_hold()
        self._hold_error = None
        self._hold_stop.clear()

        def publish_once() -> None:
            self._client.publish(
                fsm=fsm,
                height=height,
                lease_s=self.lease_s,
                allow_recovery=allow_recovery,
                defer_recovery=defer_recovery,
            )

        publish_once()

        def refresh() -> None:
            period = max(0.05, self.lease_s / 3.0)
            while not self._hold_stop.wait(period):
                try:
                    publish_once()
                except Exception as exc:
                    self._hold_error = exc
                    return

        thread = threading.Thread(
            target=refresh,
            daemon=True,
            name=f"motion_hold_{self.source}",
        )
        with self._hold_lock:
            self._hold_thread = thread
        thread.start()

    def release(self) -> None:
        self._stop_hold()
        self._client.release()
