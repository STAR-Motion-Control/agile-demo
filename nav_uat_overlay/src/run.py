import subprocess
import time
from urllib import response
import numpy as np
import cv2
import math
import datetime
import logging
import os, sys
from utils.utils import convert_text2coordinate, convert_image2pose, reset_embedding_model
from utils.logger import ColorFormatter
from flask import Flask, request, jsonify, g, send_file
from io import BytesIO
from Node import NodeManager
# import open3d as o3d
import yaml
import base64, hydra
from PIL import Image
import threading
import uuid
import json

app = Flask(__name__)
# Store navigation task status
navigation_tasks = {}
logger = logging.getLogger()

def prepare4log():
    os.makedirs('./log', exist_ok=True)
    log_filename = datetime.datetime.now().strftime("./log/app_%Y-%m-%d_%H-%M-%S.log")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
    sh = logging.StreamHandler()

    formatter = ColorFormatter("[%(levelname)s] [%(asctime)s] [%(name)s]: %(message)s")
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

def prepare_reference_coordinates(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        coordinates = [[round(x, 2) for x in item["position"]] for item in data]
        caption = [item["description"] for item in data]
    return coordinates, caption


def prepare_map(cfg):
    image_path = os.path.join(cfg.map_path,'map.png')
    yaml_path = os.path.join(cfg.map_path,'config.yaml')
    sp_path = os.path.join(cfg.map_path,'prompt.txt')
    global resolution, origin, map_image, window_size, free_points
    global reference_coordinates, reference_caption
    global system_prompt
    map_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # extract all the pixel representing the feasible area
    window_size = 10
    free_points = []
    for i in range(map_image.shape[0]):
        for j in range(map_image.shape[1]):
            # bbox
            top = max(0, i - window_size)
            bottom = min(map_image.shape[0], i + window_size)
            left = max(0, j - window_size)
            right = min(map_image.shape[1], j + window_size)
            
            window = map_image[top:bottom, left:right]
            
            if np.all(window > 220):
                free_points.append((i, j))
    # YAML file
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        resolution = data['resolution']  # each pixel -> real distance
        origin = data['origin']
    if cfg.goal_recognition.enable_embedding:
        reference_coordinates, reference_caption = prepare_reference_coordinates(os.path.join(cfg.map_path,'label.json'))
        print(f"reference_coordinates = {reference_coordinates}")
        print(f"reference_caption = {reference_caption}")
        response = reset_embedding_model(cfg.goal_recognition.embedding.reset_endpoint, reference_caption)
    else:
        with open(sp_path) as f:
            system_prompt = f.read()



def localization(cfg):
    rgb_frame, depth = node.get_rgbd()
    logger.info(f"start localization!")
    x, y, theta = convert_image2pose(cfg.vpr.endpoint, rgb_frame, depth_frame=None, robot_id=cfg.vpr.robot_id)
    logger.info(f"Got vpr localization = {x, y, theta}")
    return x, y, theta

@app.route('/get_localization', methods=['POST'])
def get_localization_handler():
    x,y,theta = localization(g.cfg)
    return jsonify({"pose":[x,y,theta]})


@app.route('/stop', methods=['POST'])
def action_stop_handler():
    success_flag,info,state = node.stop()
    return jsonify({"success_flag":success_flag,"message": info,"state":state})

@app.route('/text_nav_old', methods=['POST'])
def text_nav_old_handler():
    data = request.get_json()
    goal_text = data.get('query_text')

    if not goal_text:
        return jsonify({"message": "Goal text is required"}), 400
    
    goal_recognition_kwargs = {
        'map_image': map_image,
        'origin': origin,
        'resolution': resolution,
        'window_size': window_size,
        'free_points': free_points,

    }
    if g.cfg.goal_recognition.enable_embedding:
        goal_recognition_kwargs['reference_coordinates'] = reference_coordinates
        goal_recognition_kwargs['reference_caption'] = reference_caption
    else:
        goal_recognition_kwargs['system_prompt'] = system_prompt
    x, y, theta = convert_text2coordinate(g.cfg.goal_recognition, goal_text, **goal_recognition_kwargs)
    if x is None or y is None or theta is None:
        logger.warning("Skipping navigation due to invalid pose data.")
        return False, "Skipping navigation due to invalid pose data.", -2

    success_flag,info,state = node.navigation([x, y, theta])

    return jsonify({"success_flag":success_flag,"message": info,"state":state})

def process_navigation(task_id, goal_text, cfg):
    """Process navigation in background thread"""
    try:
        navigation_tasks[task_id]["status"] = "processing"
        
        goal_recognition_kwargs = {
            'map_image': map_image,
            'origin': origin,
            'resolution': resolution,
            'window_size': window_size,
            'free_points': free_points,
        }
        if cfg.goal_recognition.enable_embedding:
            goal_recognition_kwargs['reference_coordinates'] = reference_coordinates
            goal_recognition_kwargs['reference_caption'] = reference_caption
        else:
            goal_recognition_kwargs['system_prompt'] = system_prompt
        
        x, y, theta = convert_text2coordinate(cfg.goal_recognition, goal_text, **goal_recognition_kwargs)
        
        if x is None or y is None or theta is None:
            logger.warning("Skipping navigation due to invalid pose data.")
            navigation_tasks[task_id]["status"] = "failed"
            navigation_tasks[task_id]["success_flag"] = False
            navigation_tasks[task_id]["message"] = "Skipping navigation due to invalid pose data."
            navigation_tasks[task_id]["state"] = -2
            return

        success_flag, info, state = node.navigation([x, y, theta])
        
        navigation_tasks[task_id]["status"] = "completed"
        navigation_tasks[task_id]["success_flag"] = success_flag
        navigation_tasks[task_id]["message"] = info
        navigation_tasks[task_id]["state"] = state
        
    except Exception as e:
        logger.error(f"Navigation task {task_id} failed with error: {e}")
        navigation_tasks[task_id]["status"] = "failed"
        navigation_tasks[task_id]["success_flag"] = False
        navigation_tasks[task_id]["message"] = str(e)
        navigation_tasks[task_id]["state"] = -1

@app.route('/text_nav', methods=['POST'])
def text_nav_handler():
    data = request.get_json()
    goal_text = data.get('query_text')

    if not goal_text:
        return jsonify({"message": "Goal text is required"}), 400
    
    # Create task ID and initiate async navigation
    task_id = str(uuid.uuid4())
    navigation_tasks[task_id] = {
        "status": "pending",
        "goal_text": goal_text,
        "success_flag": None,
        "message": None,
        "state": None
    }
    
    # Start navigation in background thread
    thread = threading.Thread(target=process_navigation, args=(task_id, goal_text, g.cfg))
    thread.daemon = True
    thread.start()
    
    return jsonify({"task_id": task_id})

@app.route('/nav_status/<task_id>', methods=['GET'])
def nav_status_handler(task_id):
    """Check the status of a navigation task"""
    if task_id not in navigation_tasks:
        return jsonify({"error": "Task not found"}), 404
    
    task_data = navigation_tasks[task_id]
    return jsonify({
        "status": task_data["status"],
        "success_flag": task_data["success_flag"],
        "message": task_data["message"],
        "state": task_data["state"]
    })

@app.route('/forward', methods=['POST'])
def forward_handler():
    data = request.get_json()
    direction = str(data.get('direction'))
    distance = float(data.get('distance'))

    if not distance:
        return jsonify({"message": "Distance is required"}), 400

    if direction == 'backward' or direction == 'right':
        distance = -1*distance
    if direction == 'forward' or direction == 'backward':
        success_flag,info,state = node.forward(distance)
    if direction == 'left' or direction == 'right':
        pass
        success_flag,info,state = node.shift(distance)

    return jsonify({"success_flag":success_flag,"message": info,"state":state})

@app.route('/rotate', methods=['POST'])
def rotate_handler():
    data = request.get_json()
    direction = str(data.get('direction'))
    theta = float(data.get('theta'))
    if direction == 'right':
        theta =  -1*theta

    if not theta:
        return jsonify({"message": "Theta is required"}), 400

    success_flag,info,state = node.rotate(theta)

    return jsonify({"success_flag":success_flag,"message": info,"state":state})


@app.route('/get_lidar', methods=['POST'])
def lidar_handler():
    lidar_scan, _ = node.get_lidar()
    angles = np.linspace(-2.1999948024749756, 2.1999948024749756, len(lidar_scan))
    indice = np.where( (angles >= -math.pi/4) & (angles <= math.pi/4))
    lidar_scan = lidar_scan[indice].tolist()
    return jsonify({"lidar_scan":lidar_scan})

@app.route('/get_rgbd', methods=['POST'])
def rgbd_handler():
    rgb_frame, depth = node.get_rgbd()
    rgb_frame = rgb_frame[:, :, ::-1]
    rgb_frame_pil = Image.fromarray(rgb_frame)
    img_bytes = BytesIO()
    rgb_frame_pil.save(img_bytes, format="PNG")
    rgb_base64 = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
    depth_bytes = depth.tobytes() 
    depth_base64 = base64.b64encode(depth_bytes).decode('utf-8')
    response_data = {
        'rgb': rgb_base64,
        'depth': depth_base64, 
        'depth_shape': depth.shape,
        'depth_dtype': str(depth.dtype)
    }
    return jsonify(response_data)

def prepare_node(cfg):
    global node
    node = NodeManager(cfg, map_image, origin, resolution)

@app.before_request
def before_request():
    g.cfg = app.config.get('global_cfg')

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    print(cfg)
    prepare4log()

    prepare_map(cfg)
    prepare_node(cfg)
    time.sleep(3)

    app.config['global_cfg'] = cfg
    app.run(host='::',debug=False, threaded=True)


if __name__ == "__main__":
    main()