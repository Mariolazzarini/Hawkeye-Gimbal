# Hawkeye Gimbal

An end-to-end ROS 2 and Isaac Sim framework for vision-based UAV gimbal tracking, integrating YOLOv8-ByteTrack perception, State-Space PD control, and multi-objective NPSO gain tuning.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Data and Results](#data-and-results)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **YOLOv8 + ByteTrack** for robust human target detection and tracking, with an exponential moving average (EMA) filter on the target centre.
- **State-Space PD (SS-PD) controller** for yaw and pitch gimbal stabilization.
- **Multi-objective NPSO** (Novel Particle Swarm Optimization) for automatic gain tuning. The four objectives are combined through a weighted scalarization (Eq. 27); no Pareto front is computed.
- **Image-based visual servoing** without depth estimation.
- **ROS 2 Jazzy** and **NVIDIA Isaac Sim 5.0.0** integration.
- **Offline evaluation** of tracking metrics (MAE, volatility, acceleration, zero-crossings, ITAE, settling time, overshoot) from the `/pid_log` topic.

> **Naming note.** The ROS 2 package is called `sspid2` and the log topic is `/pid_log` for backwards compatibility with previously recorded data. The control law reported in the paper is a **PD** law (`ki = 0` in every experiment); the integral channel of the observer is not validated and should stay disabled.

## Prerequisites
- **OS:** Ubuntu 24.04 LTS (Noble) or Linux Mint 22.1 (which is based on it)
- **Kernel:** 6.8 or newer
- **ROS 2:** Jazzy Jalisco
- **Python:** 3.12 (the version shipped with Noble and targeted by Jazzy)
- **NVIDIA Isaac Sim:** 5.0.0 (with ROS 2 bridge)
- **GPU:** NVIDIA RTX 5070 Ti or similar (driver with CUDA 12.8+ support for Blackwell)
- **Python packages:** `rclpy`, `numpy`, `opencv-python`, `ultralytics`, `cv_bridge`, `matplotlib` (see `requirements.txt`)

## Installation
Clone the repository:
```bash
git clone https://github.com/Mariolazzarini/hawkeye-gimbal.git
cd hawkeye-gimbal
```

Download the YOLOv8 weights (`yolov8m.pt`, from Ultralytics) and place them in
`src/sspid2/sspid2/config/` **before** building. They are not tracked by Git and
are installed into the package share directory by `setup.py`:
```bash
mkdir -p src/sspid2/sspid2/config
# download yolov8m.pt into src/sspid2/sspid2/config/
```

Install the Python dependencies. On Ubuntu 24.04, `pip` refuses to touch the
system interpreter (PEP 668), so either use `--break-system-packages` or a
virtual environment:
```bash
pip install -r requirements.txt --break-system-packages
```

Resolve the ROS dependencies and build the package:
```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select sspid2
source install/setup.bash
```

No manual copy of the weights into the install tree is required: they are
installed to `install/sspid2/share/sspid2/config/`, which is exactly where
`distance_to_center.py` looks for them.

## Usage
### Running the Tracking System
Start the Isaac Sim scene with the ROS 2 bridge enabled (publishing `/rgb` and
`/current_orientation`, subscribing to `/cam_vel`), then launch the two nodes:
```bash
# Terminal 1: Perception node (YOLO + ByteTrack)
ros2 run sspid2 distance_node

# Terminal 2: Controller node (SS-PD)
ros2 run sspid2 controller_node
```
The system publishes gimbal velocity commands on `/cam_vel`, numeric logs on
`/pid_log` and a human-readable status string on `/control`.

To run the perception node headless (recommended during long NPSO runs):
```bash
ros2 run sspid2 distance_node --ros-args -p show_debug:=false
```

### Running the NPSO Optimization
The optimization script launches one controller instance per candidate gain
vector, collects closed-loop data, and saves the best gains.
```bash
python3 optimization/run_npsoSSPID.py
```
The on-screen menu asks for the swarm size (Fast, Normal, Extended, Exhaustive
or Custom) and for the fitness weight preset (experiments 2-7, see
`optimize()` in `optimization/npsoSSPID.py`). The default, **Normal + experiment 6**,
reproduces the 20 particles x 11 rounds = 220 closed-loop evaluations of the paper.
Reports are written as JSON to `data/results/`.

## Repository Structure
```
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
│       │   ├── distance_to_center.py
│       │   └── config/
│       │       ├── custom_tracker.yaml
│       │       └── yolov8m.pt         # downloaded, not tracked by Git
│
├── tools/
│   └── evaluator.py
│
├── optimization/
│   ├── npsoSSPID.py
│   └── run_npsoSSPID.py
│
└── data/
    ├── raw/
    └── results/
```

## Configuration
**Tracker configuration:** `custom_tracker.yaml` is used by ByteTrack. You can
adjust thresholds for your specific scenario. It is loaded in
`distance_to_center.py` from the package share directory.

**Perception parameters:**
```bash
ros2 run sspid2 distance_node --ros-args \
  -p show_debug:=false -p ema_alpha:=0.28 -p process_every_n_frames:=1
```
`ema_alpha` controls the low-pass on the target centre. It sits inside the
control loop (roughly 86 ms of lag at 30 fps with the default value), so it
must be reported together with any tuning result.

**Controller gains:** the defaults live at the top of `controller.py`, in a
single clearly marked *active* block, with the alternative experiment gain sets
listed right below it as comments. They can also be passed as ROS 2 parameters:
```bash
ros2 run sspid2 controller_node --ros-args \
  -p kp_yaw:=8.63482313 -p kd_yaw:=0.15000789 -p omega_o_yaw:=600.29943956 \
  -p kp_pitch:=8.63482313 -p kd_pitch:=0.15000789 -p omega_o_pitch:=600.29943956
```

**Optimization bounds:** edit `KP_RANGE`, `KD_RANGE` and `OMEGA_RANGE` in
`npsoSSPID.py`. Note that the forward-Euler observer is only stable for
`omega_o * Ts < 2`, i.e. `omega_o < 2000 rad/s` at `Ts = 1 ms`.

**Metric normalizers:** `C_MAE`, `C_VOL`, `C_ACC` and `C_ZC` are defined in
`optimization/npsoSSPID.py` and mirrored in `tools/evaluator.py`. **They must be
kept identical**, otherwise the normalized values reported by the evaluator are
not the quantities that were optimized.

## Data and Results
### Offline Evaluation Tool
To evaluate tracking performance, run the following in a separate terminal
while the ROS 2 nodes are running:
```bash
python3 tools/evaluator.py --scenario:=1 --experiment:=6
python3 tools/evaluator.py --scenario:=2 --experiment:=6 --show:=false
```
- Scenario 1 (sinusoidal, 100 s): MAE, volatility, acceleration, zero-crossings.
- Scenario 2 (step response, 5 s): ITAE, settling time, overshoot.

Plots (PNG) and a decimated `plot_report.json` are written to
`data/results/Exp_<id>_scenario_<n>/`. Raw recordings are expected in `data/raw/`.
If you use Git LFS, you can track large files (e.g. `*.pt`, `*.bag`). Otherwise,
keep them out of the repository and document their download links.

## Known Limitations
These are properties of the current implementation and should be taken into
account when interpreting any reported number:

1. **Metric scope.** `/pid_log` is published only while a target is being
   tracked. All metrics are therefore computed over tracking samples; periods
   of target loss are excluded rather than penalized.
2. **Per-sample differences.** Volatility and acceleration are differences per
   *sample*, computed on a 1 kHz stream whose underlying measurement updates at
   the camera rate. They depend on the ratio between both rates, so the control
   and camera rates must be reported alongside the values.
3. **Timestamps.** The evaluator timestamps messages on arrival; `/pid_log`
   carries no header. Timing metrics (ITAE, settling time) inherit the
   subscriber's latency.
4. **Step synchronization.** The overshoot percentage is normalized by the first
   recorded sample. Start the evaluator before injecting the step.
5. **Settling time.** With the default 5 s recording and 2 s hold, no settling
   time beyond 3 s can be observed; use `--duration` for slower responses.
6. **Simulation time.** No node declares `use_sim_time`, and the NPSO evaluation
   window is measured with the wall clock. If the Isaac Sim real-time factor is
   not 1.0, different particles do not cover the same amount of simulated time.
7. **Camera geometry.** `FRAME_WIDTH`, `FRAME_HEIGHT`, `FOCAL_LENGTH` and the
   apertures are hard-coded in `controller.py` and must be kept in sync with the
   Isaac Sim camera. `VERT_APERTURE` is derived from the horizontal aperture and
   the image aspect ratio; the USD default (15.2908 mm) describes a different
   sensor and made the pitch angular error ~30 % larger than the yaw one.

## Citation
If you use this code in your research, please cite our paper:

Iorpenda, M.J., Lazzarini Sola, M., Willert, V. (2026). Multi-Objective NPSO-Tuned State-Space PD Control for Vision-Based UAV Gimbal Tracking. *Drones*, MDPI.

```bibtex
@article{Iorpenda2026Hawkeye,
  title   = {Multi-Objective NPSO-Tuned State-Space PD Control for Vision-Based UAV Gimbal Tracking},
  author  = {Iorpenda, M. J. and Lazzarini Sola, Mario and Willert, Volker},
  journal = {Drones},
  year    = {2026},
  publisher = {MDPI}
}
```

## Authors
M.J. Iorpenda – co-author, validation – msuega.iorpenda@thws.de
Mario Lazzarini Sola – main developer – mario.lazzarinisola@study.thws.de
Volker Willert – supervision, funding – volker.willert@thws.de

Center for Robotics (CERI), Technical University of Applied Sciences Würzburg-Schweinfurt (THWS), Germany.

## License
This project is licensed under the MIT License – see the LICENSE file for details.

## Acknowledgments
The authors acknowledge the support and facilities provided by the THWS Center for Robotics (CERI). They thank Ekaitz Uria Sanchez, Ferran Artero Merino, and Roque Ballesteros for their contributions during the initial phase of the work, and Julius Korch for valuable technical discussions.

This work was partially supported by the THWS Campus for Applied Research (CAF) through the Hightech Agenda Bayern (HTA) program and by the Petroleum Technology Development Fund (PTDF), Nigeria.
