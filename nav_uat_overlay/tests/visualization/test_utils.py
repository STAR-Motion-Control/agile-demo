import json
import sys
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from utils import utils  # noqa: E402


def test_reset_model_planner_serializes_omegaconf_intrinsics(monkeypatch):
    captured = {}

    def fake_post(url, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout

        class Response:
            @staticmethod
            def json():
                return {"message": "ok"}

        return Response()

    monkeypatch.setattr(utils.requests, "post", fake_post)

    intrinsic = OmegaConf.create(
        [
            [387.00686646, 0.0, 319.60223389],
            [0.0, 386.47955322, 242.79476929],
            [0.0, 0.0, 1.0],
        ]
    )

    utils.reset_model_planner("http://example/reset", intrinsic=intrinsic)

    assert captured["url"] == "http://example/reset"
    assert captured["timeout"] == utils.DEFAULT_HTTP_TIMEOUT
    assert json.loads(captured["data"]["intrinsic"]) == [
        [387.00686646, 0.0, 319.60223389],
        [0.0, 386.47955322, 242.79476929],
        [0.0, 0.0, 1.0],
    ]


def test_navdp_request_uses_timeout_and_wraps_request_error(monkeypatch):
    import numpy as np
    import pytest
    import requests

    captured = {}

    def fake_post(url, timeout=None, **kwargs):
        captured["url"] = url
        captured["timeout"] = timeout
        raise requests.Timeout("slow")

    monkeypatch.setattr(utils.requests, "post", fake_post)

    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    depth = np.zeros((4, 4), dtype=np.float32)

    with pytest.raises(utils.ModelServiceError):
        utils.convert_image_depth2localaction(
            "http://example/get_action",
            rgb,
            depth,
            [1.0, 0.0],
            True,
            timeout=2.5,
        )

    assert captured["url"] == "http://example/get_action"
    assert captured["timeout"] == (2.5, 2.5)
