from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import pyastar2d
import os, cv2, math
import numpy as np
import matplotlib.pylab as plt
import imageio, io
import logging
from threading import Lock

logger = logging.getLogger(__name__)

class PathPlanner:
    def __init__(self, img, origin, resolution):
        self.origin_img = img
        self.img = self.preprocess(img)
        self.current_position = None
        self.origin = origin
        self.resolution = resolution
        self.path_imgs = [] # used to plot path
        self._lock = Lock()
        self.episode_id = 0
        
        
    def world_to_pixel(self, world_coords):
        resolution = self.resolution
        origin = self.origin
        world_coords = np.array(world_coords)
        if len(world_coords.shape) == 1:
            world_coords = world_coords[None,:]
        assert world_coords.shape[-1] == 2
        H = self.img.shape[0]

        T = np.array([
            [1/resolution, 0, -origin[0]/resolution],
            [0, -1/resolution, H + origin[1]/resolution]
        ])

        ones = np.ones((world_coords.shape[0], 1))
        homo_world = np.hstack([world_coords, ones])

        pixels = homo_world @ T.T
        return pixels.astype(int)

    def pixel_to_world(self, pixel_coords):
        resolution = self.resolution
        origin = self.origin

        H = self.img.shape[0]
        pixel_coords = np.array(pixel_coords)
        if len(pixel_coords.shape) == 1:
            pixel_coords = pixel_coords[None,:]
        assert pixel_coords.shape[-1] == 2, f"Got unexpected size = {pixel_coords.shape}"

        T = np.array([
            [resolution, 0, origin[0]],
            [0, -resolution, H * resolution + origin[1]]
        ])
        ones = np.ones((pixel_coords.shape[0], 1))
        homo_pixel = np.hstack([pixel_coords, ones])
        worlds = homo_pixel @ T.T
        return worlds

    def set_pose(self, x, y, theta):
        self.current_position = [x, y, theta]
        logger.info(f'Set current pose to {self.current_position}')
        return True, 'success', 'set pose success!'
    
    @staticmethod
    def preprocess(img):
        dist_transform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        norm_dist = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        # cost_map = 1.0 - norm_dist
        beta = 2
        cost_map = np.exp(-beta * norm_dist) 
        output = cost_map*255
        img = np.clip(output, 0, 255).astype(np.float32)
        img[img==255] = np.inf
        return img

    def get_path(self, x, y, theta):
        
        start_world = self.current_position[:-1]
        start_pixel = self.world_to_pixel(start_world)[0]

        goal_world = [x, y]
        goal_pixel = self.world_to_pixel(goal_world)[0]
        # logger.info(f"start_pixel = {start_pixel}, goal_pixel = {goal_pixel}")

        with self._lock:
            path = pyastar2d.astar_path(self.img.T, start_pixel, goal_pixel, allow_diagonal=False)
        if path is None or len(path) == 0:
            logger.warning(f"Failed to find path from {start_pixel.tolist()} to {goal_pixel.tolist()}")
            return None, None
        # logger.info(f'Get {len(path)} waypoint in the given path!')

        # path_pixel = [[item.x, item.y] for item in path]
        path_pixel = [[item[0], item[1]] for item in path]
        path_pixel = self.smooth_path_moving_average(path_pixel)

        path_world = self.pixel_to_world(path_pixel)
        ones = np.ones((path_world.shape[0], 1))
        path_world = np.hstack([path_world, ones]).tolist()

        # self.plot(np.array(path_pixel).T)

        return path_world, path_pixel
    
    def smooth_path_moving_average(self, path, window=10):
        path = np.array(path)
        smoothed = path.copy()
        for i in range(len(path)):
            start = max(0, i - window)
            end = min(len(path), i + window + 1)
            smoothed[i] = path[start:end].mean(axis=0)
        return smoothed
    
    def plot(self, path_pixel, interval=30, max_threshold=60):
        start_point = self.world_to_pixel(self.current_position[:-1])[0]
        end_point = self.world_to_pixel([self.current_position[0]+1*math.cos(self.current_position[-1]), self.current_position[1]+1*math.sin(self.current_position[-1])])[0]
        path_point_x = path_pixel[0]
        path_point_y = path_pixel[1]
        waypoint_x = path_pixel[0][interval:max_threshold:interval]
        waypoint_y = path_pixel[1][interval:max_threshold:interval]
        fig = plt.figure()
        plt.axis('off')
        plt.imshow(self.origin_img, cmap='gray')
        plt.scatter(start_point[0],start_point[1],c='b')
        plt.scatter(waypoint_x,waypoint_y,c='b')
        plt.plot(path_point_x,path_point_y,c='r',linewidth=2)
        plt.plot([start_point[0],end_point[0]],[start_point[-1],end_point[1]],c='g',linewidth=2)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        img = imageio.imread(buf)
        self.path_imgs.append(img)

    def save_fig(self,):
        os.makedirs('./img',exist_ok=True)
        imageio.mimsave(f"./img/{self.episode_id}.gif", self.path_imgs, duration=1.0)
        self.episode_id += 1
        self.path_imgs = []
