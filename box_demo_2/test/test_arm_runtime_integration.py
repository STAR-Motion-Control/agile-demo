import sys
from pathlib import Path
from types import SimpleNamespace


BOX_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BOX_ROOT.parent
for path in (REPO_ROOT, BOX_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from arm_natural_hang import ArmNaturalHangKeeper
from dual_arm_target_reach import DualArmController, FixedRateThread


class FakeArmRuntime:
    is_ipc = True

    def __init__(self):
        self.state = SimpleNamespace(
            q=tuple(0.01 * index for index in range(17)),
            tau_est=tuple(float(index) for index in range(17)),
        )
        self.token = "token"
        self.frames = []
        self.releases = []
        self.connected = False
        self.last_error = None

    def connect(self):
        self.connected = True

    def wait_state(self, timeout_s=2.0, max_age_s=0.1):
        return self.state

    def latest_state(self, max_age_s=0.1):
        return self.state

    def publish(self, q, *, weight=1.0, profile="manip_v1"):
        self.frames.append((tuple(q), float(weight), profile))
        self.token = "token"

    def release_to_policy(self, *, duration_s=2.5, timeout_s=0.2):
        self.releases.append(float(duration_s))
        self.token = None
        return True


def test_keeper_uses_one_runtime_lease_without_keepalive_thread():
    runtime = FakeArmRuntime()
    keeper = ArmNaturalHangKeeper(arm_runtime=runtime)
    keeper.move_to_hang(duration=0.04, start_keepalive=True)

    assert keeper.is_running()
    assert keeper._thread is None
    assert runtime.connected
    assert len(runtime.frames) == 3
    assert runtime.frames[-1][2] == "manip_v1"

    keeper.stop()
    assert not keeper.is_running()
    # Stopping one controller phase does not release the process-wide lease.
    assert runtime.token == "token"


def test_keeper_release_is_server_side_and_policy_staleness_cannot_be_hidden():
    runtime = FakeArmRuntime()
    keeper = ArmNaturalHangKeeper(arm_runtime=runtime)
    keeper.release_to_policy(duration=1.25)

    assert runtime.releases == [1.25]
    assert runtime.token is None
    assert runtime.frames[-1][0] == runtime.state.q


def test_controller_runtime_publish_and_grip_state_use_no_dds_objects():
    runtime = FakeArmRuntime()
    controller = DualArmController.__new__(DualArmController)
    controller._arm_runtime = runtime
    controller._runtime_state = runtime.state
    controller._low_state = None

    q = [0.2] * 17
    controller._publish(q, sdk_weight=0.75)
    assert runtime.frames[-1] == (tuple(q), 0.75, "manip_v1")
    assert controller._read_q() == list(runtime.state.q)
    assert controller._sample_grip_tau() == (1.0 + 3.0 + 8.0 + 10.0) / 4.0


def test_closed_controller_abort_is_idempotent_and_cannot_republish_old_frame():
    controller = DualArmController.__new__(DualArmController)
    controller._closed = True
    controller._publish = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("closed controller republished")
    )
    controller.abort()


def test_keeper_reports_runtime_lease_loss_instead_of_silently_continuing():
    runtime = FakeArmRuntime()
    keeper = ArmNaturalHangKeeper(arm_runtime=runtime)
    keeper.move_to_hang(duration=0.02, start_keepalive=True)
    runtime.token = None
    deadline = __import__("time").monotonic() + 0.5
    while keeper._failure is None and __import__("time").monotonic() < deadline:
        __import__("time").sleep(0.01)
    try:
        assert keeper._failure is not None
        try:
            keeper.raise_if_failed()
        except RuntimeError as exc:
            assert "lease lost" in str(exc)
        else:
            raise AssertionError("lost lease was not surfaced")
    finally:
        keeper.stop()


def test_fixed_rate_thread_surfaces_target_exception():
    def fail():
        raise ValueError("publish failed")

    thread = FixedRateThread(0.001, fail, "failing_control")
    thread.Start()
    thread.Wait(timeout=0.5)
    try:
        thread.raise_if_failed()
    except RuntimeError as exc:
        assert isinstance(exc.__cause__, ValueError)
    else:
        raise AssertionError("control-thread exception was swallowed")
