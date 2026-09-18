# Hawkeye Gimbal

Code accompanying *Hawkeye: Vision-Based UAV-Gimbal Target Tracking with Multi-Objective NPSO-Tuned State-Space PD Control* (Iorpenda, Lazzarini Sola and Willert, submitted to *Drones*, MDPI, 2026).

An end-to-end ROS 2 and Isaac Sim framework for vision-based UAV gimbal tracking, integrating YOLOv8-ByteTrack perception, State-Space PD control, and multi-objective NPSO gain tuning.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [Configuration](#configuration)
- [Reproducing the Experiments](#reproducing-the-experiments)
- [Data and Results](#data-and-results)
- [Known Limitations](#known-limitations)
- [Citation](#citation)
- [Authors](#authors)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features
- **YOLOv8 + ByteTrack** for human target detection and temporal association, with spatial re-identification, a patience interval before the target is declared lost, and an exponential moving average (EMA) filter on the target centre.
- **State-Space PD (SS-PD) controller** for yaw and pitch, built on the observer-based realization of Tan et al., with the observer bandwidth `omega_o` as an additional design parameter (Eqs. 11-20).
- **Multi-objective NPSO** for joint selection of `KP`, `KD` and `omega_o`. The four objectives - MAE, volatility, acceleration-like second-order variation and zero-crossing count - are combined through **weighted scalarization** into a single fitness (Eq. 27); no Pareto front is computed.
- **Image-based visual servoing** without depth estimation, 3D reconstruction, or prior knowledge of the target geometry.
- **ROS 2 Jazzy** and **NVIDIA Isaac Sim 5.0.0** integration.
- **Offline evaluation** of the Scenario 1 and Scenario 2 metrics from the `/pid_log` topic.

> **Naming note.** The ROS 2 package is called `sspid2`, the controller class `SSPID` and the log topic `/pid_log`, all kept for backwards compatibility with previously recorded data. The control law in the paper is a **PD** law: `ki = 0` in every experiment, and the integral channel of the observer is not part of the published formulation. The controller warns at start-up if `ki != 0`.

## Prerequisites
The environment used for the published experiments (Table 1 of the paper):

| Component | Specification |
| --- | --- |
| Operating system | Linux Mint 22.1 (Ubuntu 24.04 base) |
| Kernel | Linux 6.8.0-62-generic |
| CPU | AMD Ryzen 7 9700X |
| Memory | 31 GiB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| GPU driver | 570.133.07 |
| ROS 2 distribution | Jazzy Jalisco |
| Python | 3.12.3 |
| Simulator | NVIDIA Isaac Sim 5.0.0 (with ROS 2 bridge) |

Python packages: `rclpy`, `numpy`, `opencv-python`, `ultralytics`, `cv_bridge`, `matplotlib` (see `requirements.txt`).

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

Install the Python dependencies. On Ubuntu 24.04 / Mint 22.1, `pip` refuses to
touch the system interpreter (PEP 668), so either use `--break-system-packages`
or a virtual environment:
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
The controller publishes the yaw-pitch angular-rate correction `u_t` (Eq. 8) on
`/cam_vel`; the Isaac Sim action graph integrates it into absolute joint
commands and projects them onto the admissible joint set (Eq. 9). Numeric logs
are published on `/pid_log` and a human-readable status string on `/control`.

To run the perception node headless (recommended during long NPSO runs):
```bash
ros2 run sspid2 distance_node --ros-args -p show_debug:=false
```

### Running the NPSO Optimization
The optimization script launches one controller instance per candidate
parameter vector, runs a full closed-loop simulation, and records the fitness.
```bash
python3 optimization/run_npsoSSPID.py
```
The menu asks for the swarm size and for the fitness weight preset
(Experiments 2-7 of Table 3). The default - **Normal + Experiment 6** - is the
configuration of the paper: 20 particles and 11 total rounds (1 initial
sampling + 10 update rounds), i.e. 220 closed-loop fitness evaluations.
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
control loop (roughly 86 ms of lag at 30 fps with the default value) and is not
part of the formulation in the paper, so it must be reported together with any
tuning result.

**Controller gains:** the defaults live at the top of `controller.py`, in a
single clearly marked *active* block, with every gain set of Table 3 listed
right below it as comments. They can also be passed as ROS 2 parameters:
```bash
ros2 run sspid2 controller_node --ros-args \
  -p kp_yaw:=9.6348 -p kd_yaw:=0.1508 -p omega_o_yaw:=600.2994 \
  -p kp_pitch:=9.6348 -p kd_pitch:=0.1508 -p omega_o_pitch:=600.2994
```
`observer_uses_saturated_input` (default `false`) optionally drives the observer
with the saturated rate command instead of `u[k-1]`. Enabling it prevents the
estimates from drifting while the command saturates at `MAX_VEL_TRACK`, but it
is a deviation from Eq. (16) and makes results non-comparable with the tables.

**Optimization bounds (Eq. 32):** `KP_RANGE`, `KD_RANGE` and `OMEGA_RANGE` in
`npsoSSPID.py`. `KP` and `KD` bracket the Ziegler-Nichols baseline; the
`omega_o` upper bound is half the explicit-Euler stability limit
`2/Ts = 2000 rad/s` at `Ts = 1 ms`.

**Metric normalizers (Eq. 26):** `C_MAE = 50 px`, `C_VOL = 5 px`, `C_ACC = 2 px`,
`C_ZC = 1000`. They are defined in `optimization/npsoSSPID.py` and mirrored in
`tools/evaluator.py`; **they must be kept identical**, otherwise the normalized
values reported by the evaluator are not the quantities that were optimized.

## Reproducing the Experiments
### Gain sets (Table 3)

| Exp. | Fitness | KP | KD | omega_o | w_MAE | w_vol | w_acc | w_zc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Ziegler-Nichols baseline | 7.68 | 0.48 | ~632 (ZN-SSPD) | - | - | - | - |
| 2 | MAE | 13.2213 | 0.2255 | 487.1265 | 1 | 0 | 0 | 0 |
| 3 | MAE + Volatility | 12.3629 | 0.1225 | 932.0433 | 1 | 1 | 0 | 0 |
| 4 | MAE + Zero-crossings | 12.8675 | 0.1359 | 956.2599 | 1 | 0 | 0 | 1 |
| 5 | MAE + Acceleration | 12.0321 | 0.1477 | 901.5414 | 1 | 0 | 1 | 0 |
| 6 | Combined (equal weights) | 9.6348 | 0.1508 | 600.2994 | 1 | 1 | 1 | 1 |
| 7 | Combined (reweighted) | 10.2877 | 0.1202 | 818.4926 | 1 | 0.5 | 0.5 | 1 |

### Evaluation scenarios (Table 2)

| Item | Scenario 1 | Scenario 2 |
| --- | --- | --- |
| Objective | Continuous tracking | Transient response |
| UAV carrier | Prescribed sinusoidal motion | Static |
| Target | Walking human | Static |
| Initial image-plane error | Determined by relative motion | Approximately (200, 150) px |
| Duration | 100 s | 5 s |
| Primary assessment | Accuracy and response quality | Overshoot and settling behaviour |

### Scenario 1 carrier motion profile (Eqs. 33-38)
The Isaac Sim scene is not distributed with this repository. The prescribed
carrier trajectory, applied through the Isaac Sim Action Graph and held fixed
across all Scenario 1 experiments, is:

```
x(t)      = 10 * sin(31.416 * t) - 12
y(t)      = -18.6
z(t)      = 3.6
phi_x(t)  = 0
phi_y(t)  = 30 * sin(31.416 * (t + 4))
phi_z(t)  = 0
```

No carrier or target motion is imposed in Scenario 2; the initial gimbal
orientation is chosen so as to produce the prescribed image-plane offset.

## Data and Results
### Offline Evaluation Tool
Run the following in a separate terminal while the ROS 2 nodes are running:
```bash
python3 tools/evaluator.py --scenario:=1 --experiment:=6
python3 tools/evaluator.py --scenario:=2 --experiment:=6 --show:=false
```
- **Scenario 1** reproduces Eqs. (22)-(26): MAE, volatility, acceleration-like
  variation, zero-crossing counts, and their normalized values.
- **Scenario 2** reproduces the overshoot of Eqs. (39)-(40). ITAE and settling
  time are additional diagnostics not reported in the paper.

Plots (PNG) and a decimated `plot_report.json` are written to
`data/results/Exp_<id>_scenario_<n>/`. Raw recordings are expected in `data/raw/`.
If you use Git LFS, you can track large files (e.g. `*.pt`, `*.bag`). Otherwise,
keep them out of the repository and document their download links.

## Known Limitations
Properties of the current implementation that should be taken into account when
interpreting any reported number:

1. **Not bit-reproducible by design.** As stated in Sec. 4.3, the search
   terminates after a fixed iteration budget, without a convergence criterion
   and without a fixed random seed. Re-running the optimization will not
   reproduce Table 3 exactly.
2. **Metric scope.** `/pid_log` is published only while a target is being
   tracked (`nu_t = 1`). All metrics are therefore computed over tracking
   samples; periods of target loss are excluded rather than penalized.
3. **Per-sample differences.** Volatility and acceleration are differences per
   *sample*. The controller runs at `Ts = 1 ms` while the angular feedback is
   updated at the camera rate through the sample-and-hold of Sec. 4.4, so both
   values depend on the ratio between the two rates; report them together.
4. **Timestamps.** The evaluator timestamps messages on arrival and `/pid_log`
   carries no header, so the supplementary timing diagnostics inherit the
   subscriber's latency.
5. **Step synchronization.** The overshoot percentage is normalized by the first
   recorded sample (`e_i0`). Start the evaluator before injecting the step.
6. **Simulation time.** No node declares `use_sim_time`, and the NPSO evaluation
   window is measured with the wall clock. If the Isaac Sim real-time factor is
   not 1.0, different particles do not cover the same amount of simulated time.
7. **Vertical sensor aperture.** `VERT_APERTURE` is set to the USD default
   (15.2908 mm), the value used to produce the published results. That value
   describes a ~4:3 sensor while the images are rendered at 640x360, so the
   geometrically consistent value would be `A_x * H / W = 11.787 mm`. With the
   current setting the vertical pixel pitch `s_y` is ~30 % larger than `s_x`,
   which gives the pitch axis a larger angular error than the yaw axis for the
   same pixel offset even though a single shared parameter vector is used for
   both (Eq. 11). See the comment block in `controller.py`.
8. **Shared axis gains.** As noted in Sec. 7.5, the shared parameter vector may
   not remain optimal when the yaw and pitch dynamics differ; axis-specific
   gains are left for future work.
9. **Gain discrepancy for Experiments 6 and 7.** The working tree carried
   `KP = 8.63482313 / KD = 0.15000789` (Exp. 6) and
   `KP = 10.23482313 / KD = 0.12000789 / omega_o = 818.29943956` (Exp. 7),
   which differ from Table 3 beyond rounding. The published values are used in
   `controller.py`; the original NPSO reports should be checked to determine
   which source is correct. Experiments 2-5 matched the table exactly.

## Citation
If you use this code in your research, please cite our paper:

Iorpenda, M.J.; Lazzarini Sola, M.; Willert, V. Hawkeye: Vision-Based UAV-Gimbal Target Tracking with Multi-Objective NPSO-Tuned State-Space PD Control. *Drones* **2026**. https://doi.org/10.3390/drones1010000

```bibtex
@article{Iorpenda2026Hawkeye,
  title     = {Hawkeye: Vision-Based UAV-Gimbal Target Tracking with Multi-Objective NPSO-Tuned State-Space PD Control},
  author    = {Iorpenda, Msuega Jnr and Lazzarini Sola, Mario and Willert, Volker},
  journal   = {Drones},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/drones1010000}
}
```

> The DOI above is the placeholder of the submitted version; update it, together with the volume and article number, once the paper is published.

## Authors
Msuega Jnr Iorpenda - corresponding author; validation, visualization - msuega.iorpenda@thws.de
Mario Lazzarini Sola - software, data curation - mario.lazzarinisola@study.thws.de
Volker Willert - conceptualization, supervision, funding acquisition - volker.willert@thws.de

Center for Robotics (CERI), Technical University of Applied Sciences Würzburg-Schweinfurt (THWS), Konrad-Geiger-Straße 2, 97421 Schweinfurt, Germany.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
The authors acknowledge the support and facilities provided by the THWS Center for Robotics (CERI). They thank Ekaitz Uria Sanchez, Ferran Artero Merino, and Roque Ballesteros for their contributions during the initial phase of the work as part of their bachelor's engineering project, particularly in establishing the NVIDIA Isaac Sim environment, configuring the workstation, and contributing to the initial project conceptualisation. The authors also thank Julius Korch for his valuable ideas and technical discussions with the project team.

This work was partially supported by the THWS Campus for Applied Research (CAF) through the Hightech Agenda Bayern (HTA) program and by the Petroleum Technology Development Fund (PTDF), Nigeria, through the Overseas Scholarship Scheme under Reference No. PTDF/ED/OSS/PHD/MI/1486/19 (19PHD058). Supported by the publication fund of the Technical University of Applied Sciences Würzburg-Schweinfurt.
