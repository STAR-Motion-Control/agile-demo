import json

from onboard_runtime.loop_health import ControlLoopHealthReporter, RuntimeHealthHeartbeat


def test_healthy_after_stable_warmup(tmp_path):
    reporter = ControlLoopHealthReporter(
        str(tmp_path / "health.json"),
        expected_period_s=0.02,
        min_samples=5,
        report_hz=1000.0,
    )
    payload = None
    for index in range(7):
        payload = reporter.record(
            cycle_started_at=10.0 + index * 0.02,
            compute_s=0.008,
            success=True,
            now_wall=100.0 + index * 0.02,
        )
    assert payload is not None
    assert payload["healthy"] is True
    assert abs(payload["cycle_p99_ms"] - 20.0) < 1e-9


def test_timing_spikes_trip_health(tmp_path):
    reporter = ControlLoopHealthReporter(
        str(tmp_path / "health.json"),
        expected_period_s=0.02,
        min_samples=5,
        report_hz=1000.0,
    )
    starts = [1.0, 1.02, 1.04, 1.12, 1.20, 1.28, 1.36]
    payload = None
    for start in starts:
        payload = reporter.record(
            cycle_started_at=start,
            compute_s=0.01,
            success=True,
            now_wall=100.0 + start,
        )
    assert payload is not None
    assert payload["healthy"] is False
    assert "overrun_ratio_high" in payload["reason"]


def test_rare_long_gap_cannot_hide_below_p99_or_overrun_ratio(tmp_path):
    reporter = ControlLoopHealthReporter(
        str(tmp_path / "health.json"),
        expected_period_s=0.02,
        p99_max_s=0.04,
        max_gap_s=0.06,
        min_samples=10,
        window_samples=200,
        report_hz=1000.0,
    )
    start = 10.0
    payload = None
    for index in range(201):
        payload = reporter.record(
            cycle_started_at=start,
            compute_s=0.005,
            success=True,
            now_wall=100.0 + index,
        )
        start += 0.10 if index in (98, 198) else 0.02

    assert payload is not None
    assert payload["healthy"] is False
    assert "cycle_gap_high" in payload["reason"]
    assert abs(payload["cycle_max_ms"] - 100.0) < 1e-6


def test_missing_success_trips_health(tmp_path):
    reporter = ControlLoopHealthReporter(
        str(tmp_path / "health.json"),
        expected_period_s=0.02,
        recent_success_s=0.05,
        min_samples=2,
        report_hz=1000.0,
    )
    reporter.record(cycle_started_at=2.0, compute_s=0.005, success=True)
    reporter.record(cycle_started_at=2.02, compute_s=0.005, success=False)
    payload = reporter.record(cycle_started_at=2.10, compute_s=0.005, success=False)
    assert payload is not None
    assert payload["healthy"] is False
    assert "observation_stale" in payload["reason"]


def test_runtime_heartbeat_writes_on_transition_and_rate_limits(tmp_path):
    path = tmp_path / "merger-health.json"
    reporter = RuntimeHealthHeartbeat(
        str(path),
        component="lowcmd_merger",
        report_hz=2.0,
        runtime_id="candidate-123",
    )
    first = reporter.record(
        healthy=False,
        reason="rl_missing",
        now_mono=10.0,
        now_wall=100.0,
    )
    suppressed = reporter.record(
        healthy=False,
        reason="rl_missing",
        now_mono=10.1,
        now_wall=100.1,
    )
    transitioned = reporter.record(
        healthy=True,
        reason="healthy",
        now_mono=10.2,
        now_wall=100.2,
        rl_age_ms=4.0,
    )

    assert first is not None
    assert suppressed is None
    assert transitioned is not None
    payload = json.loads(path.read_text())
    assert payload["component"] == "lowcmd_merger"
    assert payload["healthy"] is True
    assert payload["rl_age_ms"] == 4.0
    assert payload["runtime_id"] == "candidate-123"


def test_runtime_heartbeat_marks_stopped(tmp_path):
    path = tmp_path / "merger-health.json"
    reporter = RuntimeHealthHeartbeat(str(path), component="lowcmd_merger")
    reporter.record(healthy=True, reason="healthy", now_mono=1.0)
    reporter.mark_stopped("worker_exception")

    payload = json.loads(path.read_text())
    assert payload["healthy"] is False
    assert payload["reason"] == "worker_exception"
