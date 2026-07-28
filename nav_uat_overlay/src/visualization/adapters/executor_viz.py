class ExecutorVisualizationAdapter:
    def __init__(self, visualization=None):
        self.visualization = visualization

    def _get_state_hub(self):
        if not self.visualization:
            return None
        return self.visualization.get("state_hub")

    def _get_active_task_id(self):
        state_hub = self._get_state_hub()
        if state_hub is None:
            return None
        return state_hub.get_active_task_id()

    def append_event(self, event_name, payload=None, level="info"):
        state_hub = self._get_state_hub()
        if state_hub is None:
            return
        state_hub.append_event(
            event_name,
            task_id=self._get_active_task_id(),
            payload=payload,
            level=level,
        )

    def publish_actions_started(self, actions, dry_run):
        self.append_event(
            "action_execution_started",
            payload={"actions": actions, "dry_run": bool(dry_run)},
        )

    def publish_actions_skipped(self, actions, reason="dry_run"):
        self.append_event(
            "action_execution_skipped",
            level="warning",
            payload={"actions": actions, "reason": reason},
        )

    def publish_replanning_result(self, success, message):
        self.append_event(
            "replanning_triggered",
            level="info" if success else "warning",
            payload={"success": bool(success), "message": message},
        )
