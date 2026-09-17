# Hawkeye Gimbal

An end-to-end ROS 2 and Isaac Sim framework for vision-based UAV gimbal tracking, integrating YOLOv8-ByteTrack perception, State-Space PD control, and multi-objective NPSO gain tuning.

## Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Usage](#-usage)
- [Repository Structure](#-repository-structure)
- [Configuration](#-configuration)
- [Data and Results](#-data-and-results)
- [Citation](#-citation)
- [Authors](#-authors)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

## Features
- **YOLOv8 + ByteTrack** for robust human target detection and tracking.
- **State-Space PD (SS-PD) controller** for yaw and pitch gimbal stabilization.
- **Multi-objective NPSO** (Novel Particle Swarm Optimization) for automatic gain tuning.
- **Image-based visual servoing** without depth estimation.
- **ROS 2 Jazzy** and **NVIDIA Isaac Sim 5.0.0** integration.
- **Real-time logging** of tracking metrics (MAE, volatility, overshoot, etc.).

## Prerequisites
- **OS:** Linux Mint 22.1 or Ubuntu 22.04 (recommended)
- **Kernel:** 6.8 or newer
- **ROS 2:** Jazzy
- **Python:** 3.12
- **NVIDIA Isaac Sim:** 5.0.0 (with ROS 2 bridge)
- **GPU:** NVIDIA RTX 5070 Ti or similar (with recent drivers)
- **Python packages:** `rclpy`, `numpy`, `opencv-python`, `ultralytics`, `cv_bridge`, etc. (see `requirements.txt`)

## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/hawkeye-gimbal.git
cd hawkeye-gimbal
```
Install Python dependencies:
```bash
pip install -r requirements.txt
```
Download YOLOv8 weights. Place yolov8m.pt in the src/sspid2/sspid2/ directory (or update the path in distance_to_center.py). You can download it from Ultralytics.

Build the ROS 2 package:
```bash
colcon build --packages-select sspid2
source install/setup.bash
```

## Usage
### Running the Tracking System
Launch the perception and controller nodes. You can use the provided launch file (if available) or run each node individually:
```bash
# Terminal 1: Perception node (YOLO + ByteTrack)
ros2 run sspid2 distance_to_center

# Terminal 2: Controller node (SS-PD)
ros2 run sspid2 controller_node
```
Alternatively, if a launch file is present:

```bash
ros2 launch sspid2 tracking.launch.py
```
The system will publish gimbal velocity commands to /cam_vel and log data to /pid_log and /control.

### Running the NPSO Optimization
The optimization script launches multiple controller instances with different parameters, collects performance data, and saves the best gains.
```bash
python optimization/run_npsoSSPID.py
```
Follow the on-screen menu to select the optimization mode (Fast, Normal, Extended, Exhaustive, or Custom). The results are saved as JSON in data/results/.

## Repository Structure
hawkeye-gimbal/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── docs/
│   └── MDPI___Hawkeye_Paper.pdf
│
├── src/
│   └── sspid2/
│       ├── package.xml
│       ├── setup.py
│       ├── setup.cfg
│       ├── resource/
│       │   └── sspid2
│       ├── sspid2/
│       │   ├── __init__.py
│       │   ├── controller.py
│       │   └── distance_to_center.py
│       ├── config/
│       │   └── custom_tracker.yaml
│       └── launch/
│           └── tracking.launch.py   # (optional)
│
├── optimization/
│   ├── npsoSSPID.py
│   └── run_npsoSSPID.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
└── notebooks/

## Configuration
Tracker configuration: custom_tracker.yaml is used by ByteTrack. You can adjust thresholds for your specific scenario. The file is loaded in distance_to_center.py; update the hardcoded path if necessary.

Controller gains: Default gains are defined in controller.py. You can also pass them as ROS 2 parameters:
```bash
ros2 run sspid2 controller_node --ros-args -p kp_yaw:=10.6 -p kd_yaw:=0.15 ...
```
Optimization bounds: Edit npsoSSPID.py to change the search ranges (KP_RANGE, KD_RANGE, OMEGA_RANGE).

## Data and Results
Raw and processed data are stored in data/raw/ and data/processed/.
Optimization reports (JSON) are saved to data/results/. These include the best particle parameters and fitness components.
If you use Git LFS, you can track large files (e.g., *.pt, *.bag). Otherwise, keep them out of the repository and document their download links.

## Citation
If you use this code in your research, please cite our paper:

Lazzarini Sola, M., Iorpenda, M.J., Willert, V. (2026). Multi-Objective NPSO-Tuned State-Space PD Control for Vision-Based UAV Gimbal Tracking. Drones. MDPI.

BibTeX:
@article{Iorpenda2026Hawkeye,
  title={Multi-Objective NPSO-Tuned State-Space PD Control for Vision-Based UAV Gimbal Tracking},
  author={Iorpenda, M.J. and Lazzarini Sola, Mario and Willert, Volker},
  journal={Drones},
  year={2026},
  publisher={MDPI}
}
## Authors
Mario Lazzarini Sola – main developer – mario.lazzarinisola@study.thws.de

M.J. Iorpenda – co-author, validation – msuega.iorpenda@thws.de

Volker Willert – supervision, funding – volker.willert@thws.de

Center for Robotics (CERI), Technical University of Applied Sciences Würzburg-Schweinfurt (THWS), Germany.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments
The authors acknowledge the support and facilities provided by the THWS Center for Robotics (CERI). They thank Ekaitz Uria Sanchez, Ferran Artero Merino, and Roque Ballesteros for their contributions during the initial phase of the work, and Julius Korch for valuable technical discussions.

This work was partially supported by the THWS Campus for Applied Research (CAF) through the Hightech Agenda Bayern (HTA) program and by the Petroleum Technology Development Fund (PTDF), Nigeria.


