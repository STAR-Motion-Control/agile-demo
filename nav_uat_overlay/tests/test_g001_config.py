from pathlib import Path

from hydra import compose, initialize_config_dir


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "src"


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
