from dataclasses import dataclass
import numpy as np
import cv2, io, requests, json, logging, math
from scipy.spatial import cKDTree
import time
from urllib.parse import urljoin
import zipfile, re
from openai import OpenAI
from typing import List, Optional, Tuple, Dict
from omegaconf import OmegaConf
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_HTTP_TIMEOUT = (1.0, 8.0)


class ModelServiceError(RuntimeError):
    """Raised when an external model service request fails or returns invalid data."""


def _request_timeout(timeout=None):
    if timeout is None:
        return DEFAULT_HTTP_TIMEOUT
    if isinstance(timeout, (int, float)):
        timeout = float(timeout)
        return (timeout, timeout)
    return timeout


def _post_model_service(endpoint, *, timeout=None, **kwargs):
    try:
        response = requests.post(endpoint, timeout=_request_timeout(timeout), **kwargs)
        raise_for_status = getattr(response, "raise_for_status", None)
        if callable(raise_for_status):
            raise_for_status()
        return response
    except requests.RequestException as exc:
        raise ModelServiceError(f"model service request failed: {endpoint}: {exc}") from exc


def world_to_pixel(
    world_coords : List[float], 
    map_image : np.ndarray, 
    origin : List[float], 
    resolution : float, 
) -> Tuple[int, int]:
    """
    Convert world coodinates to pixel coordinates. 
    The origin of the world coordinate system corresponds to the `bottom left` corner of the map, 
    and the x-axis corresponds to the width of the image.
    The origin of the image coordinate system is located at the top left corner of the image.


    Args
    ------
        world_coords : [x, y]
            the goal coordinates in the world coordinates system.
        map_image : np.ndarray
            the global top down binary map
        origin: [x, y]
            the global coordinates corresponding to the bottom left corner of the map
        resolution: float
            the resolution of the map, means meters one pixel correspond to
    Returns
    ------
        position : [pixel_x, pixel_y]
    """
    x, y = world_coords
    x = np.asarray(x)
    y = np.asarray(y)
    pixel_x = ((x - origin[0]) / resolution).astype(int)
    pixel_y = map_image.shape[0] - ((y - origin[1]) / resolution).astype(int) # the Coordinate system of cv2 located in the left top
    return pixel_x, pixel_y

def pixel_to_world(
    pixel_coords : List[int], 
    map_image : np.ndarray,
    origin : List[float], 
    resolution : float, 
) -> Tuple[float, float]:
    """
    Convert pixel coodinates to world coordinates
    The origin of the world coordinate system corresponds to the `bottom left` corner of the map, 
    and the x-axis corresponds to the width of the image.
    The origin of the image coordinate system is located at the top left corner of the image.

    Args
    ------
        pixel_coords : [x, y]
            the goal coordinates in the world coordinates system.
        map_image : np.ndarray
            the global top down binary map
        origin: [x, y]
            the global coordinates corresponding to the bottom left corner of the map
        resolution: float
            the resolution of the map, means meters one pixel correspond to
    Returns
    ------
        position : [world_x, world_y]
    """
    px, py = pixel_coords
    world_x = px * resolution + origin[0]
    world_y = (map_image.shape[0] - py)* resolution + origin[1]
    return world_x, world_y

def find_nearest_feasible_point(pixel_coords, free_points):
    tree = cKDTree(free_points)
    distance, index = tree.query(pixel_coords, k = 1,distance_upper_bound=400)  # find the nearest feasible point
    nearest_pixel = free_points[index]
    return (nearest_pixel[1], nearest_pixel[0])

def theta2Quaternion(x,y,theta):
    z = math.sin(theta / 2)
    w = math.cos(theta / 2)
    return x, y, z, w 

def calculate_theta(
    start_point: Tuple[int, int],
    target_point: Tuple[int, int],
):
    dy = target_point[1] - start_point[1]
    dx = target_point[0] - start_point[0]
    theta = math.atan2(dy, dx)
    return theta

def is_point_feasible(
    world_coords : List[float], 
    map_image : np.ndarray, 
    origin : List[float], 
    resolution : float, 
    window_size : int, 
    free_points : List[int],
) -> Tuple[bool, List[float]]:
    """
    Determine whether a given point is reachable by the robot.

    Args
    ------
        world_coords : [x, y]
            the goal coordinates in the world coordinates system.
        map_image : np.ndarray
            the global top down binary map
        origin: [x, y]
            the global coordinates corresponding to the bottom left corner of the map
        resolution: float
            the resolution of the map, means meters one pixel correspond to
        window_size: int
            the pixel size of the robot's placeholder image on the map
        free_points: List[int]
            all passable points on the pixel map
    Returns
    ------
        feasible: bool
        coords: [x, y]
            if `feasible`, return `world_coords`, otherwise return the nearest point
    """
    # convert to pixel coordinate
    pixel_coords = world_to_pixel(world_coords, map_image, origin, resolution)
    px, py = pixel_coords

    if 0 <= px < map_image.shape[1] and 0 <= py < map_image.shape[0]:
        # if map_image[py, px] > 220:
        if np.all(map_image[py-window_size:py+window_size, px-window_size:px+window_size] > 220):
            return True, world_coords
        else:
            # find the nearest feasible pixel
            nearest_pixel = find_nearest_feasible_point((py, px), free_points)
            nearest_world = pixel_to_world(nearest_pixel, map_image, origin, resolution)
            return False, nearest_world
    else:
        logger.error("Input point is outside the map range!")
        raise ValueError("Input point is outside the map range!")

def convert_text2coordinate(
    config : Dict, 
    query_text : str, 
    **kwargs
)-> List[float]:
    """
    Retrieve matching coordinates from the database based on the given query.

    Args
    ------
        config : Dict
        query_text : str
            the query text representing the place the robot need to arrive
    Returns
    ------
        position: List[float]
            the estimated position in the form of [x, y, theta] in the global coordinate system
    """
    map_image = kwargs['map_image']
    origin = kwargs['origin']
    window_size = kwargs['window_size']
    free_points = kwargs['free_points']
    resolution = kwargs['resolution']
    if not config.enable_embedding:
        system_prompt = kwargs['system_prompt']
        x, y, theta = convert_text2coordinate_by_llm(config.llm.use_api, config.llm.api_key, config.llm.api_endpoint,
                                                     config.llm.local_endpoint, query_text, system_prompt)
    else:
        reference_coordinates = kwargs['reference_coordinates']
        reference_caption = kwargs['reference_caption']
        x, y, theta = convert_text2coordinate_by_embedding_model(config.embedding.query_endpoint,
                                                                 query_text, reference_coordinates, reference_caption)
    if x is None or y is None or theta is None:
        return x, y, theta
    feasible, result_coords = is_point_feasible([x, y], map_image, origin, resolution, window_size, free_points)
    if feasible:
        logger.info(f"The input points {[x, y]} are inside the feasible region.")
    else:
        new_x, new_y = result_coords
        theta = calculate_theta([new_x, new_y], [x, y]) #new -> old
        logger.info(f"The input points {[x, y]} are NOT inside the feasible region, The nearest feasible point is {new_x, new_y}.")
        x, y = new_x, new_y
    position = [x, y, theta]
    return position


def convert_text2coordinate_by_llm(
    use_api : bool, 
    api_key : str, 
    api_endpoint : str, 
    local_endpoint : str, 
    query_text :str, 
    system_prompt: str,
    timeout: Optional[float] = None,
) -> List[float]:
    """
    Use LLM to retrieve matching coordinates from the database based on the given query.

    Args
    ------
        use_api : bool
            whether to use remote cloud LLM service or use locally deployed LLM service
        api_key : str
            the valid api key to use cloud LLM, valid only when `use_api` is `True`
        api_endpoint : str
            endpoint of the cloud service provider
        local_endpoint : str
            endpoint of the ocally deployed LLM service, valid only when `use_api` is `False`
        query_text : str
            the query text representing the place the robot need to arrive
        system_prompt : str
            the prompt use to query the LLM with all coordinates and caption information included
    Returns
    ------
        position: List[float]
            the estimated position in the form of [x, y, theta] in the global coordinate system
    """
    try:
        if use_api:
            api_client = OpenAI(
                api_key=api_key,
                base_url=api_endpoint
            )
            messages = [
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": system_prompt+query_text},
                    ]
                }
            ]
            
            result_from_gpt = api_client.chat.completions.create(
                        model='qwen-max-latest',
                        messages=messages,
                        temperature=0.1
                    )
            result_from_gpt = result_from_gpt.choices[0].message.content
            result_from_gpt = result_from_gpt.replace('\n','')
            logger.info(f'Got response from GPT = {result_from_gpt}')
        else:
            # FIXME to be consistent with api method
            result_from_gpt = _post_model_service(
                local_endpoint,
                json={"instruction": system_prompt, "prompt": query_text},
                timeout=timeout,
            )
            result_from_gpt = result_from_gpt.json()['read_message']
            result_from_gpt = result_from_gpt.replace('```json\n','')
            result_from_gpt = result_from_gpt.replace('```','')
        # re match in order to be more robust to the response
        matches = re.findall(r"-?\d+(?:\.\d+)?", result_from_gpt)
        if matches:
            position = [float(item) for item in matches]
            position = [item if item!=0.0 else 0.0 for item in position]
            position = position[:3]
        else:
            return None, None, None
        logger.info(f'location: {position}')
        return position
        
    except (requests.RequestException, ModelServiceError) as e:
        logger.error(f"Error during request: {e}")
        return [None, None, None]
        
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")
        return [None, None, None]


def convert_text2coordinate_by_embedding_model(
    query_endpoint : str, 
    query_text : str, 
    reference_coordinates : List[List[float]], 
    reference_caption : List[str],
    timeout: Optional[float] = None,
) -> List[float]:
    """
    Use embedding model to retrieve matching coordinates from the database based on the given query.

    Args
    ------
        query_endpoint : str
            the request endpoint
        query_text : str
            the query text representing the place the robot need to arrive
        reference_coordinates : List[List[float]]
            the ordered coordinates corresponding with `reference_caption`, each in the form of [x, y, theta]
        reference_caption
            the reference caption list of the database, used by embedding model to match query and reference caption
    Returns
    ------
        position: List[float]
            the estimated position in the form of [x, y, theta] in the global coordinate system
    """
    try:
        response = _post_model_service(
            query_endpoint,
            data={'query': json.dumps([query_text])},
            timeout=timeout,
        ).json()
    except (ModelServiceError, ValueError, KeyError, TypeError) as exc:
        logger.error("Embedding model request failed: %s", exc)
        return [None, None, None]
    idx_list = response['idx_list']
    success_status = response['success']
    message = response['message']
    logger.info(f"Get response from embedding model = {message}")
    if success_status:
        logger.info(f"Get response from embedding model = {idx_list}, corresponding to {reference_caption[idx_list[0]]}")
        position = reference_coordinates[idx_list[0]]
        return position
    return [None, None, None]

def convert_image2pose(
    vpr_endpoint : str, 
    rgb_frame : Image.Image, 
    depth_frame: np.ndarray = None,
    timeout: Optional[float] = None,
    robot_id: Optional[str] = None,
) -> List[float]:
    """
    Call the visual place recognition (VPS) to get the precise location.

    Args
    ------
        vpr_endpoint : str
            the request endpoint
        rgb_frame : Image.Image (h, w)
            the rgb frame from the front view
        depth_frame : np.ndarray (h, w)
            the depth frame from the front view at the same time with `rgb_frame`
    Returns
    ------
        position: List[float]
            the estimated position in the form of [x, y, theta] in the global coordinate system
    """
    color_image = np.asanyarray(rgb_frame)
    _, color_encoded = cv2.imencode('.jpg', color_image)
    color_bytes = io.BytesIO(color_encoded.tobytes())

    files = {
        "image": ("color.jpg", color_bytes, "image/jpeg")
    }
    if robot_id is None:
        robot_id = 1
    data = {'robot_id': robot_id}

    if depth_frame is not None:
        depth_image = depth_frame.to(np.float32) / 1000.0
        depth_buf = io.BytesIO()
        np.save(depth_buf, depth_image)
        depth_buf.seek(0)
        files["depth_image"] = ("depth_file.npy", depth_buf, "application/octet-stream")

    start_time = time.time()      
    try:
        result = _post_model_service(vpr_endpoint, files=files, timeout=timeout, data=data)
        logger.info(f"vpr time consuming  = {time.time() - start_time}")
        result = result.json()
        result = result.get('pose')
        if not result:
            raise ModelServiceError("VPR response missing pose")
        position = [result['x'], result['y'], result['theta']]
        return position
    except (ModelServiceError, ValueError, KeyError, TypeError) as exc:
        elapsed = time.time() - start_time
        logger.warning("VPR request failed after %.3fs: %s", elapsed, exc)
        raise ModelServiceError(f"VPR request failed: {exc}") from exc

def convert_image_depth2localaction(
    action_endpoint : str,
    rgb_frame : Image.Image, 
    depth_frame : np.ndarray,
    goal: List[int],
    use_discrete_action: bool,
    timeout: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """
    Use the local planner like NavDP to predict the action sequence given the current rgb frame and depth frame.

    Args
    ------
        action_endpoint : str
            the request endpoint
        rgb_frame : Image.Image (h, w)
            the rgb frame from the front view
        depth_frame : np.ndarray (h, w)
            the depth frame from the front view at the same time with `rgb_frame`
        goal: list of [x, y]
            the relative position of the sub-goal, where x>0 represents the forward direction and y>0 represents the left direction
        use_discrete_action: bool
            whether to use discrete action like `1` means single forward 0.25m, or use continuous actions
    Returns
    ------
        action: List[Tuple[float, float]]
            the action prediction, whether in discrete or continuous, each action is like (x, y) which means rotate `x` rads and then forward `y` m
    """
    # Encode RGB PNG
    color_image = np.asarray(rgb_frame)
    success, color_encoded = cv2.imencode('.png', color_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode image")

    color_bytes = io.BytesIO(color_encoded.tobytes())

    # Encode depth as .npy
    depth_bytes = io.BytesIO()
    np.save(depth_bytes, depth_frame)   # safe binary format
    depth_bytes.seek(0)

    files = {
        "image": ("color.png", color_bytes, "image/png"),
        "depth": ("depth.npy", depth_bytes, "application/octet-stream"),
    }
    data = {
        "goal": json.dumps(goal),
        "use_discrete_action": json.dumps(bool(use_discrete_action))
        }

    start_time = time.time()
    try:
        result = _post_model_service(action_endpoint, files=files, data=data, timeout=timeout)
        logger.info(f"local planner time consuming  = {time.time() - start_time}")
        result = result.json()
        action = result['action']
        return action
    except (ModelServiceError, ValueError, KeyError, TypeError) as exc:
        elapsed = time.time() - start_time
        logger.warning("Local planner request failed after %.3fs: %s", elapsed, exc)
        raise ModelServiceError(f"local planner request failed: {exc}") from exc

def reset_model_planner(
    reset_endpoint: str,
    intrinsic: Optional[List[List[float]]] = None,
    timeout: Optional[float] = None,
) -> None:
    """
    Reset the model planner (NavDP).

    Args
    ------
        reset_endpoint : str
            the request endpoint
    """
    payload = {}
    if intrinsic is not None:
        if OmegaConf.is_config(intrinsic):
            intrinsic = OmegaConf.to_container(intrinsic, resolve=True)
        payload["intrinsic"] = json.dumps(
            [[float(value) for value in row] for row in intrinsic]
        )
    response = _post_model_service(reset_endpoint, data=payload, timeout=timeout)
    logger.info(f"Get response from reset_model_planner {response.json()['message']}")

def reset_embedding_model(
    reset_endpoint: str, 
    reference_caption: List[str],
    timeout: Optional[float] = None,
) -> None:
    """
    Reset the embedding model with reference caption library.

    Args
    ------
        reset_endpoint : str
            the request endpoint
        reference_caption : list[str]
            the ordered caption list indicating the location
    """
    response = _post_model_service(
        reset_endpoint,
        data={'document': json.dumps(reference_caption)},
        timeout=timeout,
    )
    logger.info(f"Get response from reset_embedding_model {response.json()['message']}")
