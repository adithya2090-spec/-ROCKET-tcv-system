Markdown
# 🚀 Rocket TVC — Thrust Vector Control Landing with PPO

A 2D physical simulation of a reusable, gimbaled rocket learning to execute a powered landing on a designated launchpad using Proximal Policy Optimization (PPO), built from scratch in PyTorch.
## 💡 How I Designed This System (Behind the Architecture)

When building this rocket landing simulator, I wanted to make sure it wasn't just a basic RL script thrown together in a single file. I designed it around strict modularity, numerical stability, and high training throughput. Here is how I structured the project, the data structures I picked, and the engineering reasons behind every choice.

---

### 1. Modular Architecture: Decoupling the Engine

I intentionally split the codebase into four standalone components so no single part is tightly coupled to another:

* **`physics/` (Pure RK4 Dynamics):** I kept this completely stateless and free from PyTorch or Gymnasium code. It only deals with vector physics math. This makes it simple to write isolated unit tests (`pytest physics/`) to make sure the physics works before introducing any AI.
* **`envs/` (Gymnasium Environment):** This wraps my physics engine into the standard Gym interface (`reset`, `step`). It converts raw physics metrics into scaled neural network inputs and handles step/terminal rewards.
* **`agent/` (Custom PPO Engine):** Built completely from scratch in PyTorch. It processes batches of tensor arrays without needing to know anything about Pygame or physical units.
* **`dashboard/` (Real-Time Telemetry):** An asynchronous Pygame UI that listens to state streams and draws visual trajectories, live telemetry gauges, and gimbal indicators without slowing down training execution.

---

### 2. Smart Data Structures & Memory Management

To keep training fast and prevent memory leaks, I made a few key data structure decisions:

* **Pre-allocated Rollout Buffers (`agent/ppo.py`):** Appending to dynamic Python lists during rollout collection creates massive garbage collection lag. Instead, I pre-allocate contiguous NumPy and PyTorch arrays for trajectories $(s_t, a_t, r_t, \log \pi, d_t, V_t)$. When calculating Generalized Advantage Estimation (GAE), I step backward through these memory blocks in $O(N)$ time for fast mini-batch generation.
* **Trigonometric State Representation (`envs/rocket_tvc_env.py`):** Feeding raw orientation angles ($\theta$) into a neural network creates artificial jump discontinuities when crossing from $-\pi$ to $+\pi$ (like jumping instantly from $-180^\circ$ to $+180^\circ$). To prevent gradients from blowing up, I break the angle into $\sin(\theta)$ and $\cos(\theta)$ inside the observation array.
* **FIFO Deque for Trajectory Trails (`dashboard/renderer.py`):** To render the rocket's flight path in real-time without infinite memory growth, I used Python's `collections.deque(maxlen=N)`. Old coordinate pairs drop off automatically as new ones arrive, keeping rendering overhead locked to $O(N)$ frame time.

---

### 3. Numerical Physics: Why Euler Fails and RK4 Wins

If you simulate rotational rocket dynamics using basic Euler integration ($\mathbf{s}_{t+1} = \mathbf{s}_t + \dot{\mathbf{s}}_t \Delta t$), truncation error builds up rapidly, causing the rocket to gain fake rotational energy and spin out of control.

To keep the physics stable at $50\text{ Hz}$ ($\Delta t = 0.02\text{s}$), I implemented 4th-Order Runge-Kutta (RK4) integration, sampling derivatives across four sub-step points:

$$k_1 = f(\mathbf{s}_t, \mathbf{u}_t)$$
$$k_2 = f\left(\mathbf{s}_t + \frac{\Delta t}{2} k_1, \mathbf{u}_t\right)$$
$$k_3 = f\left(\mathbf{s}_t + \frac{\Delta t}{2} k_2, \mathbf{u}_t\right)$$
$$k_4 = f(\mathbf{s}_t + \Delta t k_3, \mathbf{u}_t)$$
$$\mathbf{s}_{t+1} = \mathbf{s}_t + \frac{\Delta t}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

This maintains energy balance and models engine torque accurately without numerical drift.

---

### 4. Continuous Gimbal Control: Gaussian Policy

Since steering a rocket engine gimbal ($\delta \in [-15^\circ, +15^\circ]$) requires precise continuous actions rather than discrete buttons, the Actor network outputs the mean ($\mu_\theta$) and log standard deviation ($\log \sigma$) of a Gaussian distribution $\mathcal{N}(\mu_\theta(s), \sigma^2)$:

* **Mean ($\mu_\theta$):** What the neural network evaluates as the ideal gimbal angle for the current state.
* **Log Standard Deviation ($\log \sigma$):** A learnable parameter that allows the agent to explore widely early in training, then naturally narrow its focus as it learns to land accurately.

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
