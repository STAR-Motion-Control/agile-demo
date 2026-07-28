import os
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "Node" / "motion_backend.py"
SPEC = importlib.util.spec_from_file_location("motion_backend_under_test", MODULE_PATH)
motion_backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_backend
SPEC.loader.exec_module(motion_backend)
GrootHttpDiscreteBackend = motion_backend.GrootHttpDiscreteBackend
CombinedCancelToken = motion_backend.CombinedCancelToken
resolve_stop_hold = motion_backend.resolve_stop_hold


class FakeMover:
    fwd_max = 0.50
    lat_max = 0.20
    yaw_max = 0.40
    _height = 0.74

    def __init__(self):
        self.calls = []
        self.moves = []
        self.continuous_moves = []
        self.stops = 0
        self.finishes = 0
        self.releases = 0
        self.lifecycle = []

    def _get(self, path, **params):
        self.calls.append((path, params))
        return {"ok": True}

    def move_forward(self, value, *, cancel_event=None):
        self.moves.append(("forward", value, cancel_event))

    def move_left(self, value, *, cancel_event=None):
        self.moves.append(("left", value, cancel_event))

    def rotate(self, value, *, cancel_event=None):
        self.moves.append(("rotate", value, cancel_event))

    def publish_velocity(self, forward, lateral, yaw, **kwargs):
        self.continuous_moves.append((forward, lateral, yaw, kwargs))

    def stop(self):
        self.stops += 1
        self.lifecycle.append("stop")

    def finish_segment(self):
        self.finishes += 1

    def release(self):
        self.releases += 1
        self.lifecycle.append("release")


def make_backend(linear_rate, yaw_rate):
    backend = GrootHttpDiscreteBackend.__new__(GrootHttpDiscreteBackend)
    backend.enabled = True
    backend._mover = FakeMover()
    backend._initialized = True
    backend._initialize_before_motion = False
    backend._continuous_velocity_enabled = True
    backend._continuous_command_interval = 0.10
    backend._continuous_hold_duration = 0.25
    backend._continuous_command_epsilon = 0.01
    backend._continuous_fsm = "RL_FULL"
    backend._last_continuous_command = None
    backend._last_continuous_send_time = 0.0
    backend._continuous_applied_command = (0.0, 0.0, 0.0)
    backend._continuous_applied_time = None
    backend._continuous_linear_slew_rate = linear_rate
    backend._continuous_yaw_slew_rate = yaw_rate
    backend.log = mock.Mock()
    return backend


class TaptapBackendTest(unittest.TestCase):
    def test_combined_cancel_token_reflects_every_source(self):
        events = [mock.Mock() for _ in range(4)]
        for event in events:
            event.is_set.return_value = False
        token = CombinedCancelToken(*events)

        self.assertFalse(token.is_set())
        for event in events:
            event.is_set.return_value = True
            self.assertTrue(token.is_set())
            event.is_set.return_value = False

    def test_taptap_uses_standard_stop_hold(self):
        self.assertEqual(resolve_stop_hold({"stop_hold_s": 0.4}), 0.4)
        with mock.patch.dict(os.environ, {"GROOT_TAPTAP_ADAPTIVE": "1"}, clear=True):
            self.assertEqual(resolve_stop_hold({"stop_hold_s": 0.4}), 0.4)

    def test_taptap_continuous_command_is_not_slew_limited(self):
        backend = make_backend(0.0, 0.0)
        with mock.patch.object(motion_backend.time, "monotonic", return_value=10.0):
            self.assertTrue(backend.publish_velocity(0.40, 0.20, 0.40)[0])
        first = backend._mover.calls[0][1]
        self.assertEqual((first["vx"], first["vy"], first["wz"]), (0.40, 0.20, 0.40))
        self.assertEqual(first["allow_recovery"], 0)
        self.assertEqual(first["defer_recovery"], 1)

    def test_standard_path_applies_target_immediately(self):
        backend = make_backend(0.0, 0.0)
        with mock.patch.object(motion_backend.time, "monotonic", return_value=10.0):
            self.assertTrue(backend.publish_velocity(0.40, 0.20, 0.40)[0])
        params = backend._mover.calls[0][1]
        self.assertEqual((params["vx"], params["vy"], params["wz"]), (0.40, 0.20, 0.40))

    def test_bus_continuous_command_forwards_cancel_token(self):
        backend = make_backend(0.0, 0.0)
        backend._bus_enabled = True
        token = mock.Mock()
        token.is_set.return_value = False

        self.assertTrue(
            backend.publish_velocity(0.40, 0.20, 0.40, cancel_event=token)[0]
        )

        self.assertEqual(len(backend._mover.continuous_moves), 1)
        _, _, _, kwargs = backend._mover.continuous_moves[0]
        self.assertIs(kwargs["cancel_event"], token)

    def test_stop_bypasses_slew_and_resets_state(self):
        backend = make_backend(0.80, 1.20)
        backend._continuous_applied_command = (0.32, 0.16, 0.24)
        backend._continuous_applied_time = 10.0
        self.assertTrue(backend.stop()[0])
        self.assertEqual(backend._mover.stops, 1)
        self.assertEqual(backend._continuous_applied_command, (0.0, 0.0, 0.0))
        self.assertIsNone(backend._continuous_applied_time)

    def test_discrete_calls_forward_the_same_cancel_token_to_mover(self):
        backend = make_backend(0.0, 0.0)
        token = mock.Mock()

        self.assertTrue(backend.forward(0.2, cancel_event=token)[0])
        self.assertTrue(backend.shift(0.1, cancel_event=token)[0])
        self.assertTrue(backend.rotate(0.4, cancel_event=token)[0])

        self.assertEqual(
            backend._mover.moves,
            [
                ("forward", 0.2, token),
                ("left", 0.1, token),
                ("rotate", 0.4, token),
            ],
        )

    def test_finish_segment_is_the_only_normal_recovery_boundary(self):
        backend = make_backend(0.0, 0.0)
        self.assertTrue(backend.finish_segment()[0])
        self.assertEqual(backend._mover.finishes, 1)
        self.assertEqual(backend._mover.stops, 0)

    def test_shutdown_stops_before_releasing_motion_lease(self):
        backend = make_backend(0.0, 0.0)
        self.assertTrue(backend.shutdown()[0])
        self.assertEqual(backend._mover.lifecycle, ["stop", "release"])


if __name__ == "__main__":
    unittest.main()
