import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from onboard_runtime.nav_profile import (
    G001_LEGACY_NAV_DEFAULTS,
    NavProfileError,
    build_runtime_profile,
    read_profile,
    validate_g001_legacy_profile,
    write_profile,
)


LEGACY_LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "box_demo_groot"
    / "start_g1_onboard_nav.sh"
)
RUNTIME_LAUNCHER = (
    Path(__file__).resolve().parents[2]
    / "box_demo_groot"
    / "start_g1_onboard_runtime.sh"
)


def default_profile(**overrides):
    values = {
        "motion_bus_socket": "/tmp/groot_motion_bus.sock",
        "stand_height": 0.76,
        "walk_min_height": 0.72,
        "fwd_max": 0.50,
        "lat_max": 0.30,
        "yaw_max": 0.60,
        "lat_cruise": 0.20,
        "updated_at": 1234.5,
    }
    values.update(overrides)
    return build_runtime_profile(**values)


def launcher_config(path: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["CONDA_ENV"] = "static-profile-test"
    for name in (
        "NAV_MOTION_PROFILE",
        "DWBC_WARMUP_MODE",
        "DWBC_WARMUP_TIME",
        "DWBC_WARMUP_SPEED",
        "STAND_HEIGHT",
        "WALK_FLOOR",
        "FWD_MAX",
        "LAT_MAX",
        "YAW_MAX",
        "LAT_CRUISE",
        "WAIST_RL",
    ):
        environment.pop(name, None)
    output = subprocess.check_output(
        ["bash", str(path), "dwbc", "--print-config"],
        text=True,
        env=environment,
    )
    return {
        key: value
        for line in output.splitlines()
        if "=" in line
        for key, value in [line.split("=", 1)]
    }


def labeled_numbers(value: str) -> dict[str, str]:
    return dict(part.split(":", 1) for part in value.split(","))


def legacy_profile_from_launcher(tmp_path: Path) -> dict:
    launcher = LEGACY_LAUNCHER.read_text(encoding="utf-8")
    command = launcher.index('python3 - "$NAV_PROFILE_FILE"')
    body_start = launcher.index("<<'PY'\n", command) + len("<<'PY'\n")
    body_end = launcher.index("\nPY\n", body_start)
    writer = launcher[body_start:body_end]
    config = launcher_config(LEGACY_LAUNCHER)
    limits = labeled_numbers(config["limits"])
    path = tmp_path / "legacy-profile.json"
    subprocess.run(
        [
            sys.executable,
            "-",
            str(path),
            config["nav_motion_profile"],
            config["stand_height"],
            config["walk_height_floor"],
            config["nav_warmup"],
            config["nav_warmup_time"],
            config["nav_warmup_speed"],
            limits["fwd"],
            limits["lat"],
            limits["yaw"],
            config["nav_lat_cruise"],
            config["waist_to_rl_on_motion"],
        ],
        input=writer,
        text=True,
        check=True,
    )
    return json.loads(path.read_text(encoding="utf-8"))


def runtime_profile_from_launcher_config() -> dict:
    config = launcher_config(RUNTIME_LAUNCHER)
    limits = labeled_numbers(config["limits"])
    heights = labeled_numbers(config["height"])
    return build_runtime_profile(
        motion_bus_socket="/tmp/groot_motion_bus.sock",
        stand_height=float(heights["stand"]),
        walk_min_height=float(heights["walk_floor"]),
        fwd_max=float(limits["fwd"]),
        lat_max=float(limits["lat"]),
        yaw_max=float(limits["yaw"]),
        lat_cruise=float(config["nav_lat_cruise"]),
        updated_at=1234.5,
    )


def test_runtime_profile_matches_preserved_g001_defaults():
    profile = default_profile()

    assert {key: profile[key] for key in G001_LEGACY_NAV_DEFAULTS} == (
        G001_LEGACY_NAV_DEFAULTS
    )
    assert profile["schema_version"] == 3
    assert profile["source"] == "start_g1_onboard_runtime.sh"
    assert profile["motion_backend"] == "groot_motion_bus"
    assert profile["motion_bus_socket"] == "/tmp/groot_motion_bus.sock"
    assert profile["updated_at"] == 1234.5


def test_old_and_new_launcher_default_profiles_are_behaviorally_identical(tmp_path):
    legacy = legacy_profile_from_launcher(tmp_path)
    runtime = runtime_profile_from_launcher_config()
    behavior_keys = tuple(G001_LEGACY_NAV_DEFAULTS)

    assert {key: runtime[key] for key in behavior_keys} == {
        key: legacy[key] for key in behavior_keys
    }
    assert legacy["schema_version"] == 2
    assert legacy["source"] == "start_g1_onboard_nav.sh"
    assert runtime["schema_version"] == 3
    assert runtime["source"] == "start_g1_onboard_runtime.sh"
    assert runtime["motion_backend"] == "groot_motion_bus"
    assert runtime["motion_bus_socket"] == "/tmp/groot_motion_bus.sock"


def test_preserved_launcher_still_contains_the_profile_oracle():
    launcher = LEGACY_LAUNCHER.read_text(encoding="utf-8")

    assert 'NAV_MOTION_PROFILE="${NAV_MOTION_PROFILE:-precise}"' in launcher
    assert 'WARMUP_MODE="${DWBC_WARMUP_MODE:-off}"' in launcher
    assert 'WARMUP_SPEED="${DWBC_WARMUP_SPEED:-0.15}"' in launcher
    assert 'WARMUP_EFFECTIVE="0.0"' in launcher
    assert '"v_floor": 0.12' in launcher
    assert '"w_floor": 0.10' in launcher
    assert '"waist_to_rl_on_motion": waist_rl == "1"' in launcher


def test_runtime_launcher_writes_profile_before_starting_broker():
    launcher = RUNTIME_LAUNCHER.read_text(encoding="utf-8")

    profile_write = launcher.index("onboard_runtime/nav_profile.py\" write")
    broker = launcher.index("# STARTUP_STAGE: broker")
    assert profile_write < broker


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("motion_profile", "keyboard"),
        ("warmup_time", 0.6),
        ("v_floor", 0.10),
        ("lat_max", 0.40),
        ("stand_height", 0.74),
    ],
)
def test_validation_rejects_parameter_drift(key, value):
    profile = default_profile()
    profile[key] = value

    with pytest.raises(NavProfileError, match=key):
        validate_g001_legacy_profile(
            profile, expected_socket="/tmp/groot_motion_bus.sock"
        )


def test_profile_round_trip_is_atomic_and_validated(tmp_path):
    path = tmp_path / "nav-profile.json"
    write_profile(path, default_profile())

    payload = read_profile(path)
    validate_g001_legacy_profile(
        payload, expected_socket="/tmp/groot_motion_bus.sock"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == payload
