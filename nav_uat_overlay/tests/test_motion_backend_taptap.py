import os
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "Node" / "motion_backend.py"
SPEC = importlib.util.spec_from_file_location("motion_backend_under_test", MODULE_PATH)
motion_backend = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = motion_backend
SPEC.loader.exec_module(motion_backend)
GrootHttpDiscreteBackend = motion_backend.GrootHttpDiscreteBackend
resolve_taptap_limits = motion_backend.resolve_taptap_limits


class FakeMover:
    fwd_max = 0.50
    lat_max = 0.20
    yaw_max = 0.40
    _height = 0.74

    def __init__(self):
        self.calls = []
        self.stops = 0

    def _get(self, path, **params):
        self.calls.append((path, params))
        return {"ok": True}

    def stop(self):
        self.stops += 1


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
    def test_limits_are_enabled_only_by_wrapper_environment(self):
        cfg = {"lat_max": 0.40, "yaw_max": 0.60}
        with mock.patch.dict(os.environ, {}, clear=True):
            standard = resolve_taptap_limits(cfg)
        with mock.patch.dict(os.environ, {"GROOT_NAV_TAPTAP_LIMITS": "1"}, clear=True):
            taptap = resolve_taptap_limits(cfg)
        self.assertEqual(standard["linear_slew_rate"], 0.0)
        self.assertEqual(standard["yaw_slew_rate"], 0.0)
        self.assertEqual(taptap["lat_max"], 0.20)
        self.assertEqual(taptap["yaw_max"], 0.40)
        self.assertEqual(taptap["linear_slew_rate"], 0.80)
        self.assertEqual(taptap["yaw_slew_rate"], 1.20)

    def test_taptap_marker_survives_a_separate_navigation_shell(self):
        cfg = {"lat_max": 0.40, "yaw_max": 0.60}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as marker:
            json.dump({"taptap_optimized": True}, marker)
            marker.flush()
            with mock.patch.dict(
                os.environ,
                {"GROOT_NAV_MOTION_PROFILE_FILE": marker.name},
                clear=True,
            ):
                resolved = resolve_taptap_limits(cfg)
        self.assertTrue(resolved["enabled"])
        self.assertEqual(resolved["linear_slew_rate"], 0.80)

    def test_taptap_continuous_command_is_slew_limited(self):
        backend = make_backend(0.80, 1.20)
        with mock.patch.object(motion_backend.time, "monotonic", side_effect=[10.0, 10.1]):
            self.assertTrue(backend.publish_velocity(0.40, 0.20, 0.40)[0])
            self.assertTrue(backend.publish_velocity(0.40, 0.20, 0.40)[0])
        first = backend._mover.calls[0][1]
        second = backend._mover.calls[1][1]
        self.assertAlmostEqual(first["vx"], 0.08)
        self.assertAlmostEqual(first["vy"], 0.08)
        self.assertAlmostEqual(first["wz"], 0.12)
        self.assertAlmostEqual(second["vx"], 0.16)
        self.assertAlmostEqual(second["vy"], 0.16)
        self.assertAlmostEqual(second["wz"], 0.24)

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


if __name__ == "__main__":
    unittest.main()
