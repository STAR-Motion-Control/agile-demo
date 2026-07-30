import importlib.util
import os
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir

from onboard_runtime import mover_transport
from onboard_runtime.nav_profile import build_runtime_profile, write_profile


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "src"
REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_PATH = CONFIG_ROOT / "Node" / "motion_backend.py"


def load_motion_backend():
    spec = importlib.util.spec_from_file_location("g001_motion_backend_test", BACKEND_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_g001_profile_preserves_live_navigation_contract():
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="config_g001")

    assert cfg.map_path == "map/sustech_demo_g001"
    assert cfg.rgbd_server.launch_rgbd_server is False
    assert cfg.rgbd_server.publish_ros_topics is False
    assert cfg.rgbd_server.capture_depth is True
    assert cfg.rgbd_server.fps == 5
    assert cfg.rgbd_server.subscrib_topic.rgb == "/camera/captured_image"
    assert cfg.rgbd_server.subscrib_topic.depth == "/camera/captured_depth"
    assert cfg.motion_control.closed_loop.enable is False
    assert cfg.motion_backend.type == "groot_motion_bus"
    assert cfg.motion_backend.fwd_cruise == 0.40
    assert cfg.motion_backend.back_cruise == 0.20
    assert cfg.motion_backend.lat_cruise == 0.20
    assert cfg.motion_backend.yaw_cruise == 0.40
    assert cfg.visualization.enable_live_frames is True
    assert cfg.visualization.save_replay is True


def test_g001_runtime_profile_resolves_to_legacy_mover_parameters(
    tmp_path, monkeypatch
):
    class FakeMotionBusCommandSink:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, *_args, **_kwargs):
            raise AssertionError("parameter resolution must not publish a command")

    profile_path = tmp_path / "g001-nav-profile.json"
    profile = build_runtime_profile(
        motion_bus_socket=f"/tmp/g001-profile-{os.getpid()}.sock",
        stand_height=0.76,
        walk_min_height=0.72,
        fwd_max=0.50,
        lat_max=0.30,
        yaw_max=0.60,
        fwd_cruise=0.40,
        back_cruise=0.20,
        lat_cruise=0.20,
        yaw_cruise=0.40,
        updated_at=1234.5,
    )
    write_profile(profile_path, profile)

    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_ROOT)):
        cfg = compose(config_name="config_g001")
    cfg.motion_backend.fwd_cruise = 0.11
    cfg.motion_backend.back_cruise = 0.12
    cfg.motion_backend.lat_cruise = 0.13
    cfg.motion_backend.yaw_cruise = 0.14
    cfg.motion_backend.profile_file = str(profile_path)
    cfg.motion_backend.runtime_module_path = str(REPO_ROOT)
    cfg.motion_backend.box_demo_module_path = str(REPO_ROOT / "box_demo_groot")
    cfg.motion_backend.verbose = False
    monkeypatch.delenv("GROOT_NAV_MOTION_PROFILE_FILE", raising=False)
    monkeypatch.delenv("GROOT_NAV_MOTION_PROFILE", raising=False)
    monkeypatch.delenv("GROOT_MOTION_BUS_SOCKET", raising=False)
    monkeypatch.setattr(
        mover_transport, "MotionBusCommandSink", FakeMotionBusCommandSink
    )

    mover = load_motion_backend().GrootHttpDiscreteBackend(cfg)._mover

    assert mover._height == 0.76
    assert mover.walk_min_height == 0.72
    assert mover.fwd_cruise == 0.40
    assert mover.back_cruise == 0.20
    assert mover.lat_cruise == 0.20
    assert mover.yaw_cruise == 0.40
    assert mover.fwd_max == 0.50
    assert mover.back_max == 0.20
    assert mover.lat_max == 0.30
    assert mover.yaw_max == 0.60
    assert mover.min_duration == 1.0
    assert mover.min_distance == 0.08
    assert mover.warmup_time == 0.0
    assert mover.warmup_speed == 0.15
    assert mover.v_floor == 0.12
    assert mover.w_floor == 0.10
