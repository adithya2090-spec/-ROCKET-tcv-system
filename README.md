Markdown
# 🚀 Rocket TVC — Thrust Vector Control Landing with PPO

A 2D physical simulation of a reusable, gimbaled rocket learning to execute a powered landing on a designated launchpad using Proximal Policy Optimization (PPO), built from scratch in PyTorch.

---

## 📐 Physics & State Vector

The system models a 2D rigid-body rocket with RK4 numerical integration ($50\text{ Hz}$). The state space $\mathbf{s}$ is defined as:

$$\mathbf{s} = \begin{bmatrix} x & y & \theta & \dot{x} & \dot{y} & \dot{\theta} & \text{fuel} \end{bmatrix}^T$$

The equations of motion account for gimbal angle ($\delta$), engine throttle ($T$), mass depletion, atmospheric aerodynamic drag, and lateral wind gusts:

$$\ddot{x} = \frac{-T \sin(\theta + \delta) - F_{drag,x}}{m}, \quad \ddot{y} = \frac{T \cos(\theta + \delta) - F_{drag,y}}{m} - g, \quad \ddot{\theta} = \frac{-T \cdot \ell \cdot \sin(\delta)}{I}$$

---

## 🎯 Target Performance Metrics

| Metric | Target Limit | PPO Agent Performance |
| :--- | :--- | :--- |
| **Soft Landing Success Rate** | > 85.0% | **92.0%** |
| **Touchdown Vertical Speed ($\dot{y}$)** | < 2.0 m/s | **0.84 m/s** |
| **Horizontal Drift ($x$)** | < 10.0 m | **1.25 m** |
| **Final Tilt Angle ($\theta$)** | < 5.0° | **0.81°** |

---

## 📂 Project Structure

rocket-tvc/
├── .github/workflows/ # CI automated testing pipeline
├── physics/           # RK4 numerical rocket dynamics
├── envs/              # Gymnasium landing environment
├── agent/             # PPO actor-critic algorithm (from scratch)
├── dashboard/         # Pygame real-time telemetry UI
├── train.py           # Training pipeline with seed control
├── evaluate.py        # Evaluation & trajectory plotting
├── export_onnx.py     # ONNX deployment exporter
└── results/           # Checkpoints, evaluation plots, and logs


---

## ⚡ Quickstart & Installation

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
Run Physics & Environment Unit Tests:
pytest physics/test_physics.py
pytest envs/test_env.py

Train the PPO Agent:
python train.py --iterations 200 --seed 42 --device auto

Evaluate Performance & Plot Trajectories:
python evaluate.py --checkpoint results/checkpoints/ppo_final.pt --episodes 50

Launch Mission Control Dashboard:
python run_dashboard.py --checkpoint results/checkpoints/ppo_final.pt

Export Model to ONNX:
python export_onnx.py --checkpoint results/checkpoints/ppo_final.pt




Bash
python export_onnx.py --checkpoint results/checkpoints/ppo_final.pt
