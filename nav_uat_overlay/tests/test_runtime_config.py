import sys
from pathlib import Path
from types import SimpleNamespace


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from runtime_config import (  # noqa: E402
    diagnostics_enabled,
    model_planner_enabled,
    resolve_capture_depth,
    resolve_executor_threads,
    resolve_frame_max_age,
)


def test_executor_thread_count_is_explicit_and_bounded():
    assert resolve_executor_threads(SimpleNamespace()) == 2
    assert resolve_executor_threads({"runtime": {"executor_threads": 1}}) == 1
    assert resolve_executor_threads({"runtime": {"executor_threads": 99}}) == 4


def test_model_planner_and_diagnostics_default_to_disabled():
    cfg = SimpleNamespace(
        local_planner=SimpleNamespace(enbale_model_planner=False),
        runtime=SimpleNamespace(),
    )

    assert model_planner_enabled(cfg) is False
    assert diagnostics_enabled(cfg, "record_path_gif") is False
    assert diagnostics_enabled(cfg, "save_vpr_images") is False


def test_depth_capture_tracks_navdp_or_explicit_external_publication():
    assert resolve_capture_depth({}, require_depth=False) is True
    assert resolve_capture_depth({"publish_ros_topics": True}, require_depth=False) is True
    assert resolve_capture_depth({"publish_ros_topics": False}, require_depth=False) is False
    assert resolve_capture_depth({"capture_depth": True}, require_depth=False) is True
    assert resolve_capture_depth({"capture_depth": False}, require_depth=True) is True


def test_frame_age_limit_supports_attribute_and_mapping_configs():
    cfg = SimpleNamespace(vpr=SimpleNamespace(frame_max_age=0.75))

    assert resolve_frame_max_age(cfg, "vpr") == 0.75
    assert resolve_frame_max_age({"vpr": {"frame_max_age": None}}, "vpr") is None
