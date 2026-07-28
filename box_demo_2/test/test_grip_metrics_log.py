import json
import sys
from pathlib import Path


BOX_ROOT = Path(__file__).resolve().parents[1]
if str(BOX_ROOT) not in sys.path:
    sys.path.insert(0, str(BOX_ROOT))

from grip_metrics_log import GripMetricsLog


def _append_sample(log: GripMetricsLog) -> None:
    log.append(
        stage_label="hold",
        tau_mean=3.5,
        is_baseline=True,
        gripped=True,
        baseline_tau=3.5,
    )


def test_save_does_not_plot_by_default(tmp_path):
    log = GripMetricsLog(str(tmp_path))
    _append_sample(log)
    log._plot = lambda _path: (_ for _ in ()).throw(
        AssertionError("plotting must be opt-in")
    )

    assert log.save(tag="default") == str(tmp_path)
    payload = json.loads((tmp_path / "grip_metrics_default.json").read_text())
    assert payload[0]["tau_mean"] == 3.5
    assert not (tmp_path / "grip_metrics_default.png").exists()


def test_plot_runs_when_explicitly_enabled(tmp_path):
    log = GripMetricsLog(str(tmp_path), enable_plot=True)
    _append_sample(log)
    plotted = []
    log._plot = plotted.append

    log.save(tag="diagnostic")

    assert plotted == [str(tmp_path / "grip_metrics_diagnostic.png")]
