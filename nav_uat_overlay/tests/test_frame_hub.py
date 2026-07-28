import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from frame_hub import CameraFrameHub  # noqa: E402


def test_latest_frame_replaces_previous_frame_without_queueing():
    hub = CameraFrameHub()

    first = hub.publish("rgb-1", source="camera")
    second = hub.publish("rgb-2", "depth-2", source="camera", depth_scale=0.001)

    assert first.sequence == 1
    assert second.sequence == 2
    assert hub.latest() is second
    assert second.rgb == "rgb-2"
    assert second.depth == "depth-2"
    assert second.depth_scale == 0.001


def test_latest_can_require_depth_and_reject_stale_frames():
    now = [10.0]
    hub = CameraFrameHub(clock=lambda: now[0])

    hub.publish("rgb-only", captured_at=10.0)
    assert hub.latest(require_depth=True) is None

    frame = hub.publish("rgb", "depth", captured_at=10.0)
    assert hub.latest(require_depth=True, max_age_s=0.5) is frame

    now[0] = 10.6
    assert hub.latest(max_age_s=0.5) is None


def test_wait_for_frame_returns_only_a_newer_matching_frame():
    hub = CameraFrameHub()
    rgb_only = hub.publish("rgb-only")
    rgbd = hub.publish("rgb", "depth")

    assert hub.wait_for_frame(after_sequence=rgb_only.sequence, require_depth=True, timeout=0.01) is rgbd
    assert hub.wait_for_frame(after_sequence=rgbd.sequence, timeout=0.01) is None
