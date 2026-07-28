import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


from navigation import NavigationController  # noqa: E402


class FakeEvent:
    def __init__(self, initial=False):
        self._is_set = bool(initial)
        self.wait_calls = 0

    def set(self):
        self._is_set = True

    def clear(self):
        self._is_set = False

    def is_set(self):
        return self._is_set

    def wait(self, timeout=None):
        self.wait_calls += 1
        return self._is_set


class FakePlanner:
    def __init__(self, planned=True, on_plan=None):
        self.planned = planned
        self.on_plan = on_plan
        self.active_calls = []
        self.plan_calls = []
        self.path_planner = type("PathPlanner", (), {"save_fig": self._save_fig})()
        self.save_fig_calls = 0

    def set_navigation_active(self, active):
        self.active_calls.append(bool(active))

    def plan_action(self, goal):
        self.plan_calls.append(goal)
        if self.on_plan is not None:
            self.on_plan()
        return self.planned

    def _save_fig(self):
        self.save_fig_calls += 1


class FakeExecutor:
    def __init__(self):
        self._stop_flag = FakeEvent()
        self._finish_flag = FakeEvent(initial=True)
        self._new_action_event = FakeEvent()
        self.active_calls = []
        self.dry_run_calls = []
        self.stop_calls = 0
        self.safety_pause_calls = []
        self.safety_paused = False

    def set_navigation_active(self, active):
        self.active_calls.append(bool(active))

    def set_dry_run(self, dry_run):
        self.dry_run_calls.append(bool(dry_run))

    def set_safety_pause(self, paused, reason=None):
        self.safety_pause_calls.append((bool(paused), reason))
        changed = self.safety_paused != bool(paused)
        self.safety_paused = bool(paused)
        return changed

    def is_safety_paused(self):
        return self.safety_paused

    def stop(self):
        self.stop_calls += 1
        return True, "stop success.", "stop success."


class FakeStepControlClient:
    def __init__(self):
        self.stop_calls = 0

    def stop(self):
        self.stop_calls += 1
        return True, "ok", 1


class FakeLocalizationClient:
    def __init__(self):
        self.goal_calls = []
        self.clear_calls = 0

    def set_navigation_goal(self, goal):
        self.goal_calls.append(goal)

    def clear_navigation_goal(self):
        self.clear_calls += 1


def make_cfg(enable_model_planner=False):
    return type(
        "Cfg",
        (),
        {
            "local_planner": type(
                "LocalPlanner",
                (),
                {
                    "enbale_model_planner": enable_model_planner,
                    "model_planner": type("ModelPlanner", (), {"reset_endpoint": "http://reset"})(),
                },
            )(),
        },
    )()


def test_navigation_controller_completes_dry_run_and_cleans_up():
    planner = FakePlanner(planned=True)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()
    reset_calls = []

    def record_reset(endpoint, intrinsic=None):
        reset_calls.append((endpoint, intrinsic))

    controller = NavigationController(
        cfg=make_cfg(enable_model_planner=True),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=record_reset,
    )

    result = controller.navigation([1.0, 2.0, 0.3], dry_run=True)

    assert result == (True, "Dry-run navigation finished without moving robot!", 1)
    assert planner.plan_calls == [[1.0, 2.0, 0.3]]
    assert planner.save_fig_calls == 1
    assert planner.active_calls == [True, False]
    assert executor.active_calls == [True, False]
    assert executor.dry_run_calls == [True, False]
    assert executor._finish_flag.wait_calls == 1
    assert reset_calls == [("http://reset", None)]


def test_navigation_controller_continues_when_navdp_reset_fails():
    planner = FakePlanner(planned=True)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()

    def failed_reset(endpoint, intrinsic=None):
        raise RuntimeError("reset unavailable")

    controller = NavigationController(
        cfg=make_cfg(enable_model_planner=True),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=failed_reset,
    )

    result = controller.navigation([1.0, 2.0, 0.3], dry_run=True)

    assert result == (True, "Dry-run navigation finished without moving robot!", 1)
    assert planner.plan_calls == [[1.0, 2.0, 0.3]]


def test_navigation_controller_returns_stopped_when_executor_stop_flag_is_set():
    executor = FakeExecutor()
    planner = FakePlanner(planned=True, on_plan=executor._stop_flag.set)
    step_control = FakeStepControlClient()

    controller = NavigationController(
        cfg=make_cfg(),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=lambda _endpoint: None,
    )

    result = controller.navigation([0.0, 0.0, 0.0], dry_run=False)

    assert result == (False, "Navigation has been stopped!", -1)
    assert not executor._stop_flag.is_set()
    assert planner.active_calls == [True, False]
    assert executor.active_calls == [True, False]


def test_navigation_controller_returns_failed_when_planning_fails():
    planner = FakePlanner(planned=False)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()

    controller = NavigationController(
        cfg=make_cfg(),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=lambda _endpoint: None,
    )

    result = controller.navigation([3.0, 4.0, 0.0], dry_run=False)

    assert result == (False, "Planning failed!", -1)
    assert planner.save_fig_calls == 0
    assert planner.active_calls == [True, False]
    assert executor.active_calls == [True, False]
    assert executor.dry_run_calls == [False, False]


def test_navigation_controller_sets_and_clears_localization_goal():
    planner = FakePlanner(planned=True)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()
    localization = FakeLocalizationClient()

    controller = NavigationController(
        cfg=make_cfg(),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=lambda _endpoint: None,
        localization_client=localization,
    )

    result = controller.navigation([3.0, 4.0, 0.5], dry_run=False)

    assert result == (True, "Navigation finished!", 1)
    assert localization.goal_calls == [[3.0, 4.0, 0.5]]
    assert localization.clear_calls == 1


def test_navigation_controller_stop_disables_navigation_and_delegates_stop():
    planner = FakePlanner(planned=True)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()
    localization = FakeLocalizationClient()

    controller = NavigationController(
        cfg=make_cfg(),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=lambda _endpoint: None,
        localization_client=localization,
    )

    result = controller.stop()

    assert result == (True, "stop success.", "stop success.")
    assert planner.active_calls == [False]
    assert executor.active_calls == [False]
    assert step_control.stop_calls == 1
    assert executor.stop_calls == 1
    assert localization.clear_calls == 1


def test_navigation_controller_delegates_safety_pause_to_executor():
    planner = FakePlanner(planned=True)
    executor = FakeExecutor()
    step_control = FakeStepControlClient()

    controller = NavigationController(
        cfg=make_cfg(),
        action_planner=planner,
        action_executor=executor,
        step_control_client=step_control,
        reset_model_planner_fn=lambda _endpoint: None,
    )

    changed = controller.set_safety_pause(True, reason="lidar blocked")

    assert changed is True
    assert controller.is_safety_paused() is True
    assert executor.safety_pause_calls == [(True, "lidar blocked")]
