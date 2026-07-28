class PlannerVisualizationAdapter:
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

    def publish_plan(
        self,
        mode,
        path_world,
        waypoints,
        local_path,
        local_goal,
        actions,
        action_limit=None,
        preview_actions=None,
        camera_intrinsics=None,
        camera_image_size=None,
    ):
        state_hub = self._get_state_hub()
        if state_hub is None:
            return
        state_hub.update_planner(
            mode=mode,
            global_path=path_world,
            waypoints=waypoints,
            local_path=local_path,
            local_goal=local_goal,
            actions=actions,
            action_limit=action_limit,
            preview_actions=preview_actions,
            camera_intrinsics=camera_intrinsics,
            camera_image_size=camera_image_size,
        )
        self.append_event(
            "planner_selected",
            payload={
                "mode": mode,
                "global_path": path_world,
                "waypoints": waypoints,
                "local_path": local_path,
                "local_goal": local_goal,
                "actions": actions,
                "action_limit": action_limit,
                "preview_actions": preview_actions,
                "camera_image_size": camera_image_size,
            },
        )
        self.append_event(
            "navdp_result" if mode == "navdp" else "line_segment_result",
            payload={
                "mode": mode,
                "local_path": local_path,
                "local_goal": local_goal,
                "actions": actions,
                "action_limit": action_limit,
                "preview_actions": preview_actions,
                "camera_image_size": camera_image_size,
            },
        )
