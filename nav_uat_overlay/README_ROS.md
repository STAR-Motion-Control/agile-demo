# ROS Control README

This document explains how to start `run_ros.py`, how to use the ROS bridge, and what is available in the navigation visualization page.

## 1) Start the ROS bridge

`run_ros.py` uses Hydra with `config_path="."`, so start it from the `src` directory in this worktree:

```bash
conda activate nav
cd /home/unitree/workspace/nav_uat/src
python run_ros.py
```

Logs are written to:

- `src/log_ros.txt` (latest fixed file)
- `src/log/ros_YYYY-MM-DD_HH-MM-SS.log` (timestamped file)

### Visualization console

If `src/config.yaml` has `visualization.enable: true`, `run_ros.py` also starts the visualization backend.

Default URL:

```bash
http://<robot-ip>:8008/viz
```

The backend serves:

- Web UI: `/viz`
- WebSocket: `/viz/ws`
- Map metadata: `/viz/api/map/metadata`
- Latest image frames: `/viz/api/frame/...`

Before using the built frontend, build it once from the visualization worktree:

```bash
cd /home/unitree/workspace/nav/.worktrees/navigation-visualization/webviz
npm install
npm run build
```

### What the visualization page can do

The `/viz` page is a live control and replay console for the navigation stack. It is useful both for online monitoring and for debugging completed tasks.

In `Live` mode, the page shows:

- Connection, schema version, task id, task status, current goal text, and current mode.
- A text-navigation control bar where you can type a goal, pick a known label from the dropdown, press Enter to submit, and stop the current navigation task.
- A map panel with the static map image plus overlays for robot pose, robot heading, VPR pose, goal pose, local goal, waypoints, global path, and planner area when available.
- A vision panel with the latest RGB preview frame from the robot.
- A planner panel that shows planner mode, local goal, and the queued action sequence.
- A recent-events panel that displays the latest visualization events and payloads.

In `Replay` mode, the page lets you inspect previously recorded tasks:

- Load the saved task list from the backend.
- Select a task and switch the whole dashboard from live data to recorded snapshots.
- Scrub through the timeline with the progress slider.
- Play, pause, restart, step forward/backward, and change playback speed.
- Review the recorded map overlays, RGB frame, planner state, and event timeline at each replay step.

Notes:

- Text navigation submission is disabled while another navigation task is already running.
- The frontend talks to the backend through REST endpoints under `/viz/api/...` and a WebSocket stream on `/viz/ws`.
- If the page opens but looks empty, first check that the backend is running and that `webviz/dist` has been built successfully.

## 2) ROS interfaces

`run_ros.py` provides:

- Topic (subscribe): `/nav/text_nav` (`std_msgs/msg/String`)
- Topic (subscribe): `/nav/stop_cmd` (`std_msgs/msg/Empty`)
- Topic (subscribe): `/nav/forward_cmd` (`geometry_msgs/msg/Twist`)
- Topic (subscribe): `/nav/rotate_cmd` (`geometry_msgs/msg/Twist`)
- Topic (subscribe): `/nav/relative_cmd` (`std_msgs/msg/String`, JSON)
- Topic (subscribe): `/safety/lidar_state` (`std_msgs/msg/String`, `clear | blocked | stale`)
- Topic (publish): `/nav/status` (`std_msgs/msg/String`, JSON string)
- Service: `/nav/get_pose` (`std_srvs/srv/Trigger`)

Also, localization publishes pose to:

- Topic (publish): `/global_position` (`geometry_msgs/msg/PoseStamped`)

## 3) Text navigation

Send a natural-language destination:

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: '去玻璃大门门前'}"
```

You can also use English:

```bash
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: 'go to the kitchen'}"
```

## 4) Stop current task

Emergency stop / cancel current action:

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

## 5) LiDAR safety pause / resume

When `lidar_safety.enable` is true, the ROS bridge monitors `/safety/lidar_state`.
`blocked` and `stale` pause the current navigation executor and publish zero motion.
`clear` resumes the existing navigation task without marking it stopped.

Manual `/nav/forward_cmd` and `/nav/rotate_cmd` commands are rejected while the
LiDAR safety pause is active.

Quick simulation:

```bash
ros2 topic pub --once /safety/lidar_state std_msgs/msg/String "{data: 'blocked'}"
ros2 topic pub --once /safety/lidar_state std_msgs/msg/String "{data: 'clear'}"
```

The topic can be changed in `src/config.yaml`:

```yaml
lidar_safety:
  enable: true
  state_topic: /safety/lidar_state
```

## 6) Manual movement

### 6.1 Forward / backward

Use `linear.x` (meters). Positive = forward, negative = backward.

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist "{linear: {x: 0.5, y: 0.0, z: 0.0}}"
```

Backward example:

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist "{linear: {x: -0.3, y: 0.0, z: 0.0}}"
```

### 6.2 Shift left / right

Use `linear.y` (meters). Positive = left, negative = right.

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist "{linear: {x: 0.0, y: 0.3, z: 0.0}}"
```

Right shift example:

```bash
ros2 topic pub --once /nav/forward_cmd geometry_msgs/msg/Twist "{linear: {x: 0.0, y: -0.2, z: 0.0}}"
```

> Note: in one command, set **only one** of `linear.x` or `linear.y` to non-zero.

### 6.3 Rotate

Use `angular.z` (radians). Positive = left (CCW), negative = right (CW).

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist "{angular: {x: 0.0, y: 0.0, z: 1.57}}"
```

Right turn example:

```bash
ros2 topic pub --once /nav/rotate_cmd geometry_msgs/msg/Twist "{angular: {x: 0.0, y: 0.0, z: -1.57}}"
```

### 6.4 Relative distance + direction test

Use `/nav/relative_cmd` for the first GR00T-WBC integration test. The bridge
rotates by `direction_deg` first, then moves forward by `distance_m`.
Positive direction is left / counter-clockwise.

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String \
  "{data: '{\"distance_m\": 0.1, \"direction_deg\": 10}'}"
```

Plain text is also accepted:

```bash
ros2 topic pub --once /nav/relative_cmd std_msgs/msg/String "{data: '0.1 10'}"
```

### 6.5 Motion backend

`src/config.yaml` selects the lower-body backend:

```yaml
motion_backend:
  type: groot_http_discrete
  ipc_url: http://127.0.0.1:5001
  box_demo_module_path: /home/unitree/zihou/box_demo_1
```

G001 uses the onboard bridge started by:

```bash
cd /home/unitree/zihou/box_demo_1
bash start_g1_onboard_nav.sh
```

`groot_http_discrete` calls the local HTTP bridge and reuses
`RemoteMover/GrootMover` for distance/angle moves. Set `type:
wireless_controller` to roll back to the original `/wirelesscontroller` path.

## 7) Check robot status

Bridge status (JSON string):

```bash
ros2 topic echo /nav/status
```

Typical fields:

- `status`: `idle | processing | completed | failed | stopped | busy`
- `success_flag`: `true/false`
- `message`: text message
- `state`: numeric code (for example: `1` success, `-1` stopped/failed path)
- `task_id`: current/last task id
- `goal_text`: last text goal
- `lidar_safety`: current safety monitor state (`enabled`, `state`, `obstacle_blocked`, `paused`, `topic`)

## 8) Get current position (pose)

### Option A: Service call (`/nav/get_pose`)

```bash
ros2 service call /nav/get_pose std_srvs/srv/Trigger "{}"
```

`response.message` is a JSON string like:

```json
{"pose": [x, y, theta]}
```

Where:

- `x`, `y`: position in map frame (meters)
- `theta`: heading (radians)

### Option B: Pose topic (`/global_position`)

```bash
ros2 topic echo /global_position
```

This gives standard ROS `PoseStamped` (position + quaternion orientation).

## 9) Quick test sequence

```bash
# 1) Start bridge
bash -lic 'g1env && cd /home/unitree/workspace/nav/.worktrees/navigation-visualization/src && python run_ros.py'

# 2) In another terminal: monitor status
ros2 topic echo /nav/status

# 3) Send navigation command
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: 'go to the kitchen'}"

# 4) Query pose
ros2 service call /nav/get_pose std_srvs/srv/Trigger "{}"

# 5) Emergency stop if needed
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

## 10) Troubleshooting

- If no movement after command:
  - Confirm `run_ros.py` is running.
  - Check `/nav/status` for `failed` and message details.
- If pose is unavailable:
  - Wait a few seconds after startup for localization/odom to initialize.
  - Check `/global_position` topic.
- If Chinese text looks wrong in terminal:
  - Ensure terminal locale is UTF-8 (for example `LANG`/`LC_ALL` contains `UTF-8`).

## 11) Navigation autostart service usage

If you want navigation to be managed as a user service (`robot_nav`), use the following steps.

### 11.1 Enter the gateway script directory

```bash
cd /home/unitree/robot/gateway
```

### 11.2 Install service (only once)

```bash
cd /home/unitree/robot/gateway
./install_robot_nav_service.sh
```

> If it is already installed, you do not need to run install again.

### 11.3 Start the service

```bash
systemctl --user start robot_nav
```

### 11.4 Check service startup logs

```bash
journalctl --user-unit robot_nav -f
```

### 11.5 Check navigation runtime logs

```bash
cat /home/unitree/workspace/nav/src/log_ros.txt
```
