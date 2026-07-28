# Unitree G1 robot navgation

## Use on the development borad

### Requirement
- Ubuntu-20.04
- ROS2 foxy

### Download the code
```bash
git clone -b unitree_g1 https://github.com/SpatialtemporalAI/Woosh_robot.git
cd Woosh_robot
pip install -r requirements.txt
```

### Connect to the robot
make sure that you have log to the robot's development board, and you have make the robot ready for receiving remote control.


### Getting started
This project now rely on the remote `VPR` service, please make sure you have deployed this service remotely. Then modify the `config.yaml`, especially the `vpr.enable_vpr` and `vpr.endpoint`. 

#### Run
Open a new terminal and run
```bash
cd src/
python3 run.py
```

### Send instruction
Open another terminal, then you can send instruction like
```bash
curl -X POST -H "Content-Type: application/json" -d '{"query_text": "去玻璃大门"}' http://127.0.0.1:5000/text_nav
curl -X POST -H "Content-Type: application/json" -d '{"query_text": "去沙发"}' http://127.0.0.1:5000/text_nav
curl -X POST -H "Content-Type: application/json" -d '{"query_text": "去会议室门口"}' http://127.0.0.1:5000/text_nav_old
curl -X POST -H "Content-Type: application/json" -d '{"query_text": "去冰箱"}' http://127.0.0.1:5000/text_nav_old
curl -X POST http://127.0.0.1:5000/stop
curl -X POST -H "Content-Type: application/json" -d '{"direction": "left", "theta": "0.785"}' http://127.0.0.1:5000/rotate
curl -X POST -H "Content-Type: application/json" -d '{"direction": "forward", "distance": "0.5"}' http://127.0.0.1:5000/forward
```


### TODO
1. TBD
