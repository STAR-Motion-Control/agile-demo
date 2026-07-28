import os
import subprocess
import sys
from pathlib import Path

import pytest


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from nav_runtime_guard import (  # noqa: E402
    LAUNCH_AUTH_ENV,
    NavigationStartupError,
    authorized_navigation_process,
    require_authorized_launch,
)


@pytest.mark.parametrize("value", [None, "", "0", "true", "yes"])
def test_direct_start_requires_exact_launcher_authorization(value):
    environment = {} if value is None else {LAUNCH_AUTH_ENV: value}

    with pytest.raises(NavigationStartupError, match="direct run_ros.py startup is locked"):
        require_authorized_launch(environment)


def test_authorized_guard_is_exclusive_and_releases_cleanly(tmp_path):
    lock_path = tmp_path / "run_ros.lock"
    environment = {LAUNCH_AUTH_ENV: "1"}

    with authorized_navigation_process(environ=environment, lock_path=lock_path):
        assert lock_path.read_text(encoding="utf-8").strip().isdigit()
        with pytest.raises(NavigationStartupError, match="another run_ros.py instance"):
            with authorized_navigation_process(
                environ=environment,
                lock_path=lock_path,
            ):
                pass

    with authorized_navigation_process(environ=environment, lock_path=lock_path):
        pass


def test_unauthorized_guard_does_not_create_lock_file(tmp_path):
    lock_path = tmp_path / "run_ros.lock"

    with pytest.raises(NavigationStartupError):
        with authorized_navigation_process(environ={}, lock_path=lock_path):
            pass

    assert not lock_path.exists()


def test_run_ros_rejects_direct_execution_before_heavy_imports():
    environment = dict(os.environ)
    environment.pop(LAUNCH_AUTH_ENV, None)
    result = subprocess.run(
        [sys.executable, str(SRC_ROOT / "run_ros.py")],
        cwd=SRC_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "direct run_ros.py startup is locked" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
