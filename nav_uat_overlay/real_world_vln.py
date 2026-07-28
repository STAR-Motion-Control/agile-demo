import numpy as np
import cv2
import io
import requests
from flask import Flask, jsonify
from PIL import Image
from io import BytesIO
import math, base64
import time, json

# app = Flask(__name__)

action_server = 'http://192.168.112.101:5209/get_action'
reset_server = 'http://192.168.112.101:5209/reset'
rgb_server = "http://127.0.0.1:5000/get_rgbd" 
forward_server = 'http://127.0.0.1:5000/forward'
rotate_server = 'http://127.0.0.1:5000/rotate'


def fetch_image():
    response = requests.post(rgb_server)

    if response.status_code != 200:
        print('failed to connect')
        return None
    
    data = response.json()
    rgb_base64 = data['rgb']
    rgb = base64.b64decode(rgb_base64)

    rgb = Image.open(BytesIO(rgb))# bytes → PIL

    depth_base64 = data['depth']
    depth_shape = tuple(data['depth_shape'])
    depth_dtype = data['depth_dtype']  # string
    
    depth = base64.b64decode(depth_base64)# base64 → bytes

    depth = np.frombuffer(depth, dtype=np.dtype(depth_dtype))# bytes → numpy array
    depth = depth.reshape(depth_shape)

    return rgb, depth


def get_action(goal):
    rgb_frame, depth = fetch_image()

    # Encode RGB PNG
    color_image = np.asarray(rgb_frame)[:,:,::-1]
    success, color_encoded = cv2.imencode('.png', color_image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise RuntimeError("Failed to encode image")

    color_bytes = io.BytesIO(color_encoded.tobytes())

    # Encode depth as .npy
    depth_bytes = io.BytesIO()
    np.save(depth_bytes, depth)   # safe binary format
    depth_bytes.seek(0)

    files = {
        "image": ("color.png", color_bytes, "image/png"),
        "depth": ("depth.npy", depth_bytes, "application/octet-stream"),
    }
    data = {"goal": json.dumps(goal)}

    start_time = time.time()
    result = requests.post(action_server,files=files, data = data)
    print(f"time consuming  = {time.time() - start_time}")
    result = result.json()
    action = result['action']
    return action

def reset():
    response = requests.post(reset_server)


def execute_action(action_index):
    if action_index == 1:
        data = {
            'direction':'forward',
            'distance':0.4
        }
        response = requests.post(forward_server, json=data)
    elif action_index == 2:
        data = {
            'direction':'left',
            'theta':math.pi/12
        }
        response = requests.post(rotate_server, json=data)
    elif action_index == 3:
        data = {
            'direction':'right',
            'theta':math.pi/12
        }
        response = requests.post(rotate_server, json=data)

def navigation(text = 'move forward'):
    reset()
    action = 100
    count = 0
    while action!= 0:
        action = get_action(text)
        print(f'curren action = {action}')
        execute_action(action)
        count += 1
        if count == 100:
            break
    if action == 0:
        print('!!!!!STOPPED!!!!!!')


if __name__ == '__main__':
    # prompt = 'You are now facing the sofa. Turn left, and as you get close to the refrigerator, walk forward. Turn right in front of the cabinet with the microwave, then continue along the path until you stop at the doorway at the end.'
    
    # prompt = 'Turn right, and when you see the sofa, walk straight ahead, then keep close to the sofa until you stop in front of the cabinet with the microwave.'
    
    # prompt = "Turn right and Leave the bedroom, then walk straight ahead, past the glass door, and stop on the left at the first corner."
    
    # prompt = "Turn right and walk straight out of the lobby. Then turn left at the corner and walk along the corridor until you stop at the first corner."
    
    # prompt = "Turn right, walk straight ahead, enter the first room in front of you, and stop in front of the table."
    
    # prompt = "Turn left and go forward and turn right at the first intersection, then keep going forward until you enter the room at the end."
    # navigation(prompt)

    action = get_action([1,2])
    print(action)