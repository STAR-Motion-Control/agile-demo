from pathlib import Path

import cv2
from omegaconf import OmegaConf


def to_plain_data(value):
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)

    if isinstance(value, dict):
        return {key: to_plain_data(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain_data(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    return value


def normalize_polygons(value):
    value = to_plain_data(value or [])
    if not value:
        return []

    first = value[0]
    if isinstance(first, (list, tuple)) and len(first) >= 2 and all(
        isinstance(item, (int, float)) for item in first[:2]
    ):
        return [value]

    return value


def build_map_metadata(
    map_image_path,
    resolution,
    origin,
    frame_id="map",
    map_url=None,
    only_global_planner_area=None,
    labels=None,
):
    map_image_path = Path(map_image_path)
    image = cv2.imread(str(map_image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"failed to load map image: {map_image_path}")

    height, width = image.shape[:2]
    return {
        "frame_id": frame_id,
        "resolution": float(resolution),
        "origin": to_plain_data(origin),
        "width": int(width),
        "height": int(height),
        "image_url": map_url,
        "image_path": str(map_image_path),
        "only_global_planner_area": normalize_polygons(only_global_planner_area),
        "labels": to_plain_data(labels or []),
    }
