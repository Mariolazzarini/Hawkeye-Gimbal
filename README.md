# Hawkeye Gimbal

An end-to-end ROS 2 and Isaac Sim framework for vision-based UAV gimbal tracking, integrating YOLOv8-ByteTrack perception, State-Space PD control, and multi-objective NPSO gain tuning.

## 📋 Table of Contents
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

## 🚀 Features
- **YOLOv8 + ByteTrack** for robust human target detection and tracking.
- **State-Space PD (SS-PD) controller** for yaw and pitch gimbal stabilization.
- **Multi-objective NPSO** (Novel Particle Swarm Optimization) for automatic gain tuning.
- **Image-based visual servoing** without depth estimation.
- **ROS 2 Jazzy** and **NVIDIA Isaac Sim 5.0.0** integration.
- **Real-time logging** of tracking metrics (MAE, volatility, overshoot, etc.).

## 📦 Prerequisites
- **OS:** Linux Mint 22.1 or Ubuntu 22.04 (recommended)
- **Kernel:** 6.8 or newer
- **ROS 2:** Jazzy
- **Python:** 3.12
- **NVIDIA Isaac Sim:** 5.0.0 (with ROS 2 bridge)
- **GPU:** NVIDIA RTX 5070 Ti or similar (with recent drivers)
- **Python packages:** `rclpy`, `numpy`, `opencv-python`, `ultralytics`, `cv_bridge`, etc. (see `requirements.txt`)

## 🔧 Installation
Clone the repository:
```bash
git clone https://github.com/your-username/hawkeye-gimbal.git
cd hawkeye-gimbal
