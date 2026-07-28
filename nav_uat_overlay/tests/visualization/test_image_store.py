import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from visualization.image_store import ImageStore  # noqa: E402


def test_image_store_updates_version_when_frame_changes():
    store = ImageStore(jpeg_quality=80)

    version1 = store.update_rgb_latest(np.zeros((8, 8, 3), dtype=np.uint8))
    version2 = store.update_rgb_latest(np.ones((8, 8, 3), dtype=np.uint8) * 255)

    assert version2 > version1


def test_image_store_returns_jpeg_bytes_and_metadata():
    store = ImageStore(jpeg_quality=80)

    version = store.update_rgb_latest(np.zeros((8, 8, 3), dtype=np.uint8))
    payload = store.get("rgb_latest")

    assert payload["version"] == version
    assert payload["content_type"] == "image/jpeg"
    assert isinstance(payload["bytes"], bytes)
    assert payload["bytes"][:2] == b"\xff\xd8"
    assert isinstance(payload["updated_at"], float)


def test_image_store_store_encoded_records_metadata():
    store = ImageStore(jpeg_quality=80)

    version = store.store_encoded("rgb_latest", b"abc", content_type="image/jpeg", updated_at=123.0)
    payload = store.get("rgb_latest")

    assert version == 1
    assert payload["updated_at"] == 123.0
