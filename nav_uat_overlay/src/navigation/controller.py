import logging


logger = logging.getLogger(__name__)


class NavigationController:
    def __init__(
        self,
        cfg,
        action_planner,
        action_executor,
        step_control_client,
        reset_model_planner_fn,
        localization_client=None,
    ):
        self.cfg = cfg
        self.action_planner = action_planner
        self.action_executor = action_executor
        self.step_control_client = step_control_client
        self.reset_model_planner_fn = reset_model_planner_fn
        self.localization_client = localization_client

    def navigation(self, goal, dry_run=False):
        self.action_executor._stop_flag.clear()
        self.action_executor._finish_flag.clear()
        self.action_planner.set_navigation_active(True)
        self.action_executor.set_navigation_active(True)
        self.action_executor.set_dry_run(dry_run)
        if (
            self.localization_client is not None
            and hasattr(self.localization_client, "set_navigation_goal")
        ):
            self.localization_client.set_navigation_goal(goal)

        if self.cfg.local_planner.enbale_model_planner:
            try:
                model_planner_cfg = self.cfg.local_planner.model_planner
                kwargs = {
                    "intrinsic": getattr(model_planner_cfg, "camera_intrinsics", None),
                }
                reset_timeout = getattr(model_planner_cfg, "reset_timeout", None)
                if reset_timeout is not None:
                    kwargs["timeout"] = reset_timeout
                self.reset_model_planner_fn(
                    model_planner_cfg.reset_endpoint,
                    **kwargs,
                )
            except Exception as exc:
                logger.warning("Failed to reset NavDP, continue with planner fallback: %s", exc)

        try:
            planned = self.action_planner.plan_action(goal)
            if not planned:
                return False, "Planning failed!", -1

            self.action_executor._finish_flag.wait()
            self.action_executor._finish_flag.clear()
            self.action_planner.path_planner.save_fig()

            if self.action_executor._stop_flag.is_set():
                self.action_executor._stop_flag.clear()
                return False, "Navigation has been stopped!", -1

            message = (
                "Dry-run navigation finished without moving robot!"
                if dry_run
                else "Navigation finished!"
            )
            return True, message, 1
        finally:
            if (
                self.localization_client is not None
                and hasattr(self.localization_client, "clear_navigation_goal")
            ):
                self.localization_client.clear_navigation_goal()
            self.action_planner.set_navigation_active(False)
            self.action_executor.set_navigation_active(False)
            self.action_executor.set_dry_run(False)

    def stop(self):
        if (
            self.localization_client is not None
            and hasattr(self.localization_client, "clear_navigation_goal")
        ):
            self.localization_client.clear_navigation_goal()
        self.action_planner.set_navigation_active(False)
        self.action_executor.set_navigation_active(False)
        self.step_control_client.stop()
        return self.action_executor.stop()

    def set_safety_pause(self, paused, reason=None):
        if hasattr(self.action_executor, "set_safety_pause"):
            return self.action_executor.set_safety_pause(paused, reason=reason)
        return False

    def is_safety_paused(self):
        if hasattr(self.action_executor, "is_safety_paused"):
            return self.action_executor.is_safety_paused()
        return False
