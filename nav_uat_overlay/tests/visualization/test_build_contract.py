import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.image_store import ImageStore  # noqa: E402
from visualization.server import create_app  # noqa: E402
from visualization.state_hub import VisualizationStateHub  # noqa: E402


def _find_endpoint(app, path: str):
    for route in app.router.routes:
        if getattr(route, "path", None) == path:
            return route.endpoint
    raise AssertionError(f"route not found: {path}")


def test_static_frontend_serving_keeps_api_and_websocket_routes(tmp_path):
    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    (dist_dir / "index.html").write_text("<html><body>webviz</body></html>", encoding="utf-8")

    hub = VisualizationStateHub()
    image_store = ImageStore()
    image_store.update_rgb_latest(np.zeros((8, 8, 3), dtype=np.uint8))

    app = create_app(
        state_hub=hub,
        image_store=image_store,
        map_metadata={"frame_id": "map", "resolution": 0.05, "origin": [0, 0, 0], "width": 8, "height": 6},
        static_dir=dist_dir,
    )
    route_paths = {getattr(route, "path", None) for route in app.router.routes}
    page_response = _find_endpoint(app, "/viz")("")
    metadata_response = _find_endpoint(app, "/viz/api/map/metadata")()
    image_response = _find_endpoint(app, "/viz/api/frame/{group}/{name}.jpg")("rgb", "latest")

    assert page_response.status_code == 200
    assert "webviz" in page_response.body.decode("utf-8")
    assert metadata_response.status_code == 200
    assert json.loads(metadata_response.body)["frame_id"] == "map"
    assert image_response.status_code == 200
    assert hasattr(app.state, "ws_manager")
    assert "/viz" in route_paths
    assert "/viz/api/map/metadata" in route_paths
    assert "/viz/api/frame/{group}/{name}.jpg" in route_paths
    assert "/viz/ws" in route_paths
