import logging
from threading import Thread

import rclpy
from frame_hub import CameraFrameHub
from navigation import NavigationController
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from runtime_config import model_planner_enabled, resolve_executor_threads
from utils.utils import reset_model_planner

from .action_executor import ActionExecutorClient
from .action_planner import ActionPlanner
from .localization import LocalizationClient
from .rgbd import RGBDClient
from .step_control import StepControlClient

logger = logging.getLogger(__name__)


class NodeManager:
    def __init__(self, cfg, img, origin, resolution, visualization=None):
        self.cfg = cfg
        self.visualization = visualization
        self._shutdown = False
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        self.executor_threads = resolve_executor_threads(cfg)
        self.executor = MultiThreadedExecutor(num_threads=self.executor_threads)
        self.frame_hub = CameraFrameHub()
        require_depth = model_planner_enabled(cfg)
        
        self.step_control_client = StepControlClient(cfg=cfg)
        self.localization_client = LocalizationClient(
            cfg,
            visualization=visualization,
            frame_hub=self.frame_hub,
        )
        self.action_planner = ActionPlanner(
            cfg,
            img,
            origin,
            resolution,
            visualization=visualization,
            frame_hub=self.frame_hub,
        )
        self.action_excutor = ActionExecutorClient(cfg=cfg, visualization=visualization)
        self.rgbd_client = RGBDClient(
            cfg,
            visualization=visualization,
            frame_hub=self.frame_hub,
            require_depth=require_depth,
        )
        self.navigation_controller = NavigationController(
            cfg=cfg,
            action_planner=self.action_planner,
            action_executor=self.action_excutor,
            step_control_client=self.step_control_client,
            reset_model_planner_fn=reset_model_planner,
            localization_client=self.localization_client,
        )
        self._nodes = [
            self.localization_client,
            self.action_planner,
            self.action_excutor,
            self.step_control_client,
            self.rgbd_client,
        ]

        for node in self._nodes:
            self.executor.add_node(node)

        self.executor_thread = Thread(
            target=self.executor.spin,
            name="nav_ros_executor",
            daemon=True,
        )
        self.executor_thread.start()
        logger.info(
            "Navigation nodes started with %d ROS executor threads (NavDP depth=%s)",
            self.executor_threads,
            require_depth,
        )
        

    def forward(
        self,
        distance : float
    ):
        '''
        Make the robot move forward.

        Args
        ------
            distance : float in meters
        '''
        if self.is_lidar_safety_paused():
            return False, "LiDAR safety pause active.", -4
        success_flag,info,state = self.step_control_client.forward(float(distance))
        return success_flag,info,state

    def rotate(
        self,
        theta : float,
    ):
        '''
        Make the robot rotate.

        Args
        ------
            theta : float in radians
                theta > 0 -> left, theta < 0 -> right
        '''
        if self.is_lidar_safety_paused():
            return False, "LiDAR safety pause active.", -4
        success_flag,info,state = self.step_control_client.rotate(theta)
        return success_flag,info,state

    def shift(
        self,
        distance: float,
    ):
        '''
        Make the robot move laterally.

        Args
        ------
            distance : float in meters
                distance > 0 -> left, distance < 0 -> right
        '''
        if self.is_lidar_safety_paused():
            return False, "LiDAR safety pause active.", -4
        success_flag, info, state = self.step_control_client.shift(float(distance))
        return success_flag, info, state

    def navigation(
        self,
        goal : list[float],
        dry_run: bool = False,
    ):
        '''
        Point Goal Navigation

        Args
        ------
            goal : list or tuple
                in the form of (x, y, theta)
        '''
        return self.navigation_controller.navigation(goal, dry_run=dry_run)


    def stop(
        self,
    ):
        """
        Make the robot stop right now.
        """
        return self.navigation_controller.stop()

    def set_lidar_safety_pause(self, paused, reason=None):
        return self.navigation_controller.set_safety_pause(paused, reason=reason)

    def is_lidar_safety_paused(self):
        return self.navigation_controller.is_safety_paused()

    def request_stop(self):
        try:
            self.stop()
        except Exception as exc:
            logger.debug("Failed to send stop command during shutdown: %s", exc)

        action_executor = getattr(self, "action_excutor", None)
        if action_executor is not None:
            try:
                action_executor._stop_flag.set()
                action_executor._finish_flag.set()
                action_executor._new_action_event.set()
            except Exception as exc:
                logger.debug("Failed to signal action executor shutdown: %s", exc)
    
    
    def get_rgbd(
        self,
    ):
        """
        Get the rgb frame and depth frame of the robot.

        Returns
        -------
            rgb : np.ndarray
                uint8 array of shape (H, W, C), BGR order.
            depth : np.ndarray
                float32 array of shape (H, W), meters.
        """
        rgb, depth = self.rgbd_client.get_image()
        return rgb, depth

    def get_position(
        self,
    ):
        """
        Get the current position of the robot.

        Returns
        -------
            current_position: list or tuple
                in the form of (x, y, theta), or None if unavailable.
        """
        current_position = self.localization_client.get_position()
        return current_position

    def shutdown(self, timeout_sec=1.0):
        if self._shutdown:
            return
        self._shutdown = True

        self.request_stop()

        action_executor = getattr(self, "action_excutor", None)
        if action_executor is not None and hasattr(action_executor, "shutdown"):
            try:
                action_executor.shutdown(timeout_sec=timeout_sec)
            except Exception as exc:
                logger.debug("Action executor shutdown failed: %s", exc)

        rgbd_client = getattr(self, "rgbd_client", None)
        if rgbd_client is not None and hasattr(rgbd_client, "shutdown"):
            try:
                rgbd_client.shutdown(timeout_sec=timeout_sec)
            except Exception as exc:
                logger.debug("RGBD client shutdown failed: %s", exc)

        executor = getattr(self, "executor", None)
        if executor is not None:
            try:
                completed = executor.shutdown(timeout_sec=timeout_sec)
                if completed is False:
                    logger.warning("Node executor did not finish shutdown within %.2fs", timeout_sec)
            except Exception as exc:
                logger.debug("Executor shutdown failed: %s", exc)

            for node in getattr(self, "_nodes", []):
                try:
                    executor.remove_node(node)
                except Exception:
                    pass

        for node in getattr(self, "_nodes", []):
            try:
                node.destroy_node()
            except Exception:
                pass

        thread = getattr(self, "executor_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_sec)
