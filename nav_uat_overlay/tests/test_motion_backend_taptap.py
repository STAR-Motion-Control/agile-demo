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
resolve_stop_hold = motion_backend.resolve_stop_hold


class FakeMover:
    fwd_max = 0.50
    lat_max = 0.20
    yaw_max = 0.40
    _height = 0.74

    def __init__(self):
        self.calls = []
        self.stops = 0
        self.finishes = 0

    def _get(self, path, **params):
        self.calls.append((path, params))
        return {"ok": True}

    def stop(self):
        self.stops += 1

    def finish_segment(self):
        self.finishes += 1


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

    def test_stop_bypasses_slew_and_resets_state(self):
        backend = make_backend(0.80, 1.20)
        backend._continuous_applied_command = (0.32, 0.16, 0.24)
        backend._continuous_applied_time = 10.0
        self.assertTrue(backend.stop()[0])
        self.assertEqual(backend._mover.stops, 1)
        self.assertEqual(backend._continuous_applied_command, (0.0, 0.0, 0.0))
        self.assertIsNone(backend._continuous_applied_time)

    def test_finish_segment_is_the_only_normal_recovery_boundary(self):
        backend = make_backend(0.0, 0.0)
        self.assertTrue(backend.finish_segment()[0])
        self.assertEqual(backend._mover.finishes, 1)
        self.assertEqual(backend._mover.stops, 0)


if __name__ == "__main__":
    unittest.main()
