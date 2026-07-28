# ROS Control README

This document explains how to start `run_ros.py`, how to use the ROS bridge, and what is available in the navigation visualization page.

## 1) Start the ROS bridge

The refactored candidate refuses direct `python run_ros.py` startup. Start it
only through the outer launcher, which verifies the expected stream-only base
runtime, requires explicit human approval, and then authorizes the guarded
`run_ros.py` process:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate nav
source /opt/ros/humble/setup.bash
source ~/unitree_ros2/install/setup.bash
cd /home/unitree/zihou/agile-demo-refactor
bash nav_uat_overlay/start_nav_refactored.sh \
  --human-approved-control-start
```

Do not set the internal launcher authorization environment variable manually.
Running `run_ros.py` directly intentionally exits before Hydra, ROS, camera, or
motion-backend initialization. Use `--preflight-only` on the outer launcher for
a non-starting readiness check.

Logs are written to:

- `src/log_ros.txt` (latest fixed file)
- `src/log/ros_YYYY-MM-DD_HH-MM-SS.log` (timestamped file)

### Visualization console

If the selected config has `visualization.enable: true`, `run_ros.py` also starts the visualization backend.

Default URL:

```bash
http://<robot-ip>:8008/viz
```

The backend serves:

- Web UI: `/viz`
- WebSocket: `/viz/ws`
- Map metadata: `/viz/api/map/metadata`
- Latest image frames: `/viz/api/frame/...`

On G001 the RGBD input is published by `RGBDClient` itself. Confirm
`src/config.yaml` uses:

```yaml
rgbd_server:
  launch_rgbd_server: true
  publish_topic:
    rgb: /externel_front_image
    depth: /externel_front_depth
  subscrib_topic:
    rgb: /externel_front_image
    depth: /externel_front_depth
```

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

## 4) Stop current navigation task

This cancels the navigation task and requests zero base velocity. It is not a
hardware emergency stop and does not replace the G1 stop mechanism or the
motion-bus DAMP/LIMP latch.

```bash
ros2 topic pub --once /nav/stop_cmd std_msgs/msg/Empty "{}"
```

## 5) LiDAR safety pause / resume

When `lidar_safety.enable` is true, the ROS bridge monitors `/safety/lidar_state`.
`blocked` and `stale` pause the current navigation executor and publish zero motion.
`clear` resumes the existing navigation task without marking it stopped.

This LiDAR state is independent from the motion-bus safety latch. Clearing the
LiDAR pause cannot clear or re-arm a DAMP/LIMP epoch.

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

For the current G001 GR00T test, `run_ros.py` loads `src/config_bk.yaml` through
Hydra (`config_name="config_bk"`). `src/config.yaml` is the Unitree-controller
comparison configuration. The selected GR00T config contains:

```yaml
motion_backend:
  type: groot_motion_bus
  motion_bus_socket: /tmp/groot_motion_bus.sock
  runtime_module_path: /home/unitree/zihou/agile-demo-refactor
  box_demo_module_path: /home/unitree/zihou/agile-demo-refactor/box_demo_groot
```

The candidate runtime and navigation entry points are separate and both require
an explicit human approval flag:

```bash
cd /home/unitree/zihou/agile-demo-refactor
bash box_demo_groot/start_g1_onboard_runtime_taptap.sh \
  --human-approved-control-start
bash nav_uat_overlay/start_nav_refactored.sh \
  --human-approved-control-start
```

`groot_motion_bus` sends bounded Unix datagrams with source, sequence and lease
metadata. One broker arbitrates navigation, manipulation and operator commands,
then sends a latest-only 20 Hz Unix datagram stream directly to the adapter and
merger. Candidate B does not write or poll the legacy command JSON.
`groot_http_discrete` and the JSON path remain available only for baseline
comparisons; `wireless_controller` selects the original `/wirelesscontroller`
path.

Do not run these example commands without a person physically present and an
explicit approval for that start. `--preflight-only` performs checks without
creating a navigation or control process.

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

The first command is a real control start and may only be run by the physically
present operator after the runtime launcher has been approved and started.

```bash
# 1) Start the guarded candidate bridge
cd /home/unitree/zihou/agile-demo-refactor
bash nav_uat_overlay/start_nav_refactored.sh \
  --human-approved-control-start

# 2) In another terminal: monitor status
ros2 topic echo /nav/status

# 3) Send navigation command
ros2 topic pub --once /nav/text_nav std_msgs/msg/String "{data: 'go to the kitchen'}"

# 4) Query pose
ros2 service call /nav/get_pose std_srvs/srv/Trigger "{}"

# 5) Cancel the navigation task if needed (not the hardware emergency stop)
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

## 11) Autostart services

The refactored candidate must not be installed as an unattended autostart
service. Its launcher requires a fresh explicit human approval for each real
control start, and `run_ros.py` rejects direct service startup. Any existing
`robot_nav` user service belongs to the legacy deployment and must be disabled
or confirmed stopped before candidate preflight.
