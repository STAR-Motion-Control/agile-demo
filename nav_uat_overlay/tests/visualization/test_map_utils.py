import json
import sys
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.map_utils import build_map_metadata  # noqa: E402


def test_build_map_metadata_returns_dimensions_origin_and_resolution(tmp_path):
    map_image = np.full((6, 8), 255, dtype=np.uint8)
    map_path = tmp_path / "map.png"
    cv2.imwrite(str(map_path), map_image)

    metadata = build_map_metadata(
        map_image_path=map_path,
        resolution=0.05,
        origin=[-1.0, -2.0, 0.0],
        frame_id="map",
        map_url="/viz/api/map/image",
        labels=["办公桌", "玻璃大门门前"],
    )

    assert metadata["frame_id"] == "map"
    assert metadata["resolution"] == 0.05
    assert metadata["origin"] == [-1.0, -2.0, 0.0]
    assert metadata["width"] == 8
    assert metadata["height"] == 6
    assert metadata["image_url"] == "/viz/api/map/image"
    assert metadata["labels"] == ["办公桌", "玻璃大门门前"]


def test_build_map_metadata_converts_omegaconf_containers_to_plain_json(tmp_path):
    map_image = np.full((4, 4), 255, dtype=np.uint8)
    map_path = tmp_path / "map.png"
    cv2.imwrite(str(map_path), map_image)

    cfg = OmegaConf.create(
        {
            "origin": [-1.0, -2.0, 0.0],
            "polygons": [
                [[7.5, 0.42], [10.7, 0.066], [10.7, -12.8], [7.49, -12.9]],
                [[1.0, 1.0], [2.0, 1.0], [2.0, 0.0]],
            ],
        }
    )

    metadata = build_map_metadata(
        map_image_path=map_path,
        resolution=0.05,
        origin=cfg.origin,
        frame_id="map",
        map_url="/viz/api/map/image",
        only_global_planner_area=cfg.polygons,
    )

    assert metadata["origin"] == [-1.0, -2.0, 0.0]
    assert metadata["only_global_planner_area"][0][0] == [7.5, 0.42]
    assert metadata["only_global_planner_area"][1][0] == [1.0, 1.0]
    json.dumps(metadata)


def test_build_map_metadata_accepts_legacy_single_polygon(tmp_path):
    map_image = np.full((4, 4), 255, dtype=np.uint8)
    map_path = tmp_path / "map.png"
    cv2.imwrite(str(map_path), map_image)

    metadata = build_map_metadata(
        map_image_path=map_path,
        resolution=0.05,
        origin=[0, 0, 0],
        only_global_planner_area=[[7.5, 0.42], [10.7, 0.066], [10.7, -12.8]],
    )

    assert metadata["only_global_planner_area"][0][0] == [7.5, 0.42]
