# Rocket Thrust Vector Control (TVC) with Deep Reinforcement Learning

A numerical physics simulation of a rocket with a gimbaled engine, controlled by a Deep Reinforcement Learning agent that learns to land the rocket — inspired by SpaceX Falcon 9 landing burns.

---

## Design Decisions (Locked In)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Primary task** | 🚀 Powered Landing | Descend from ~500m, land softly at origin — dramatic, portfolio-worthy |
| **RL Algorithm** | 🧠 PPO from Scratch | Maximum technical depth — we write every line of the algorithm in PyTorch |
| **Visualization** | 🎮 Pygame + Dashboard | Real-time rocket rendering + full mission control dashboard with telemetry |

---

## 1. Project Overview & Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     PROJECT ARCHITECTURE                      │
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────┐  │
│  │   Physics     │───▶│  Gymnasium   │───▶│   PPO Agent    │  │
│  │   Engine      │◀───│  Environment │◀───│  (from scratch)│  │
│  │  (NumPy/SciPy)│    │  (Gym API)   │    │   (PyTorch)    │  │
│  └──────────────┘    └──────────────┘    └────────────────┘  │
│         │                                       │             │
│         ▼                                       ▼             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │              Pygame Mission Control Dashboard         │    │
│  │  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │    │
│  │  │ Rocket  │ │ Flight   │ │ Training │ │ Gimbal & │ │    │
│  │  │ Render  │ │ Telemetry│ │ Metrics  │ │ Thrust   │ │    │
│  │  │ (live)  │ │ (gauges) │ │ (plots)  │ │ (visual) │ │    │
│  │  └─────────┘ └──────────┘ └──────────┘ └──────────┘ │    │
│  └──────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. The Physics (Summary — Unchanged from v1)

We model a 2D rigid-body rocket with a gimbaled engine. The state vector is:

$$\mathbf{s} = \begin{bmatrix} x & y & \theta & \dot{x} & \dot{y} & \dot{\theta} \end{bmatrix}^T$$

The equations of motion (derived from Newton's laws):

$$\ddot{x} = \frac{-T \sin(\theta + \delta)}{m}, \quad \ddot{y} = \frac{T \cos(\theta + \delta)}{m} - g, \quad \ddot{\theta} = \frac{-T \cdot \ell \cdot \sin(\delta)}{I}$$

Integrated numerically using **RK4** at 50 Hz ($\Delta t = 0.02s$).

The landing task: start at altitude ~500m with slight random perturbations in tilt and velocity, and land softly at the origin with $|\dot{y}| < 2$ m/s and $|\theta| < 5°$.

> [!NOTE]
> See the previous version of this plan for the full physics derivation, force diagrams, and RK4 equations. That content is unchanged.

---

## 3. Deep Dive: What is PPO and How Does It Work?

This is the heart of the AI side of the project. Let's build up the intuition from first principles.

### 3.1 The Fundamental Problem: Learning a Policy

A **policy** $\pi_\theta(a|s)$ is a function (in our case, a neural network with parameters $\theta$) that takes the current state $s$ and outputs a probability distribution over actions $a$. For our rocket:

- **Input** (state): position, velocity, angle, angular velocity → 8 numbers
- **Output** (action): a probability distribution over gimbal angles → a Gaussian $\mathcal{N}(\mu, \sigma^2)$

The goal: find the parameters $\theta$ that maximize the **expected total reward** across full episodes.

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t r_t \right]$$

where $\tau$ is a trajectory (sequence of states and actions), $\gamma = 0.99$ is a discount factor (future rewards are worth slightly less), and $r_t$ is the reward at time step $t$.

### 3.2 Starting Simple: The Policy Gradient (REINFORCE)

The most basic approach is the **Policy Gradient Theorem** (Williams, 1992). The key insight:

> You can compute the gradient of expected reward with respect to policy parameters, even though the environment is a black box.

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot R_t \right]$$

In English: **increase the probability of actions that led to high reward, decrease the probability of actions that led to low reward.**

Here's the REINFORCE algorithm (the ancestor of PPO):

```
1. Run the current policy to collect a full episode:
   s₀ → a₀ → r₀ → s₁ → a₁ → r₁ → ... → sₜ

2. For each time step, compute the return:
   Rₜ = rₜ + γrₜ₊₁ + γ²rₜ₊₂ + ...

3. Compute the gradient and update:
   θ ← θ + α · Σ ∇log π(aₜ|sₜ) · Rₜ
```

**The problem with REINFORCE**: It has extremely high **variance**. Imagine two episodes — in one, the rocket almost lands but crashes at the last second (total reward = -50). In another, it crashes immediately (total reward = -200). REINFORCE treats every action in the first episode as "equally good," even though early actions were fine and only the final ones were bad. The gradient estimates are noisy, and training is painfully slow.

### 3.3 The Baseline & Advantage: Reducing Variance

To fix the variance problem, we introduce a **baseline** — specifically, a **value function** $V(s)$ that estimates the expected return from state $s$:

$$V(s) = \mathbb{E}\left[\sum_{t'=t}^{T} \gamma^{t'-t} r_{t'} \mid s_t = s\right]$$

The **advantage** $A_t$ measures how much better (or worse) an action was compared to what we expected:

$$A_t = Q(s_t, a_t) - V(s_t)$$

- $A_t > 0$: The action was **better** than average → increase its probability
- $A_t < 0$: The action was **worse** than average → decrease its probability
- $A_t = 0$: The action was exactly as expected → no change

This is the **Actor-Critic** framework:
- **Actor** ($\pi_\theta$): The policy network. Decides what to do.
- **Critic** ($V_\phi$): The value network. Evaluates how good a state is.

```
        State s
       ╱       ╲
      ╱         ╲
  ┌──▼──┐    ┌──▼──┐
  │ACTOR │    │CRITIC│
  │ πθ   │    │ Vφ   │
  └──┬──┘    └──┬──┘
     │          │
  Action a   Value V(s)
     │          │
     └────┬─────┘
          │
    Advantage: A = R - V(s)
    → Used to update the Actor
```

### 3.4 Generalized Advantage Estimation (GAE)

Computing the advantage naively has a bias-variance tradeoff:
- **1-step TD**: $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ → Low variance, high bias (depends heavily on $V$ being accurate)
- **Monte Carlo**: $A_t = R_t - V(s_t)$ → No bias, high variance (depends on the full trajectory)

**GAE** (Schulman, 2015) gives us a smooth blend using parameter $\lambda$:

$$\hat{A}_t = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

where $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ is the TD error.

- $\lambda = 0$: Pure 1-step TD (low variance, high bias)
- $\lambda = 1$: Pure Monte Carlo (no bias, high variance)
- $\lambda = 0.95$: The sweet spot (what we'll use)

### 3.5 The Trust Region Problem (Why Vanilla Policy Gradient Fails)

Here's the critical problem that PPO solves:

With vanilla policy gradients, if you take too large a gradient step, the policy changes dramatically. A policy that was carefully learning to balance the rocket might suddenly start spinning it wildly. And unlike supervised learning, **you can't just undo a bad update** — the bad policy generates bad data, which leads to worse updates, creating a death spiral.

```
Good policy → small update → still good ✅
Good policy → BIG update  → terrible policy → terrible data → worse policy 💀
                            (can't recover from this!)
```

We need to ensure each update is "small enough" to stay in a **trust region** around the current policy.

### 3.6 PPO: The Clipped Surrogate Objective (The Key Innovation)

PPO (Schulman, 2017) solves the trust region problem with an elegantly simple idea: **clip the objective function** so the policy can't change too much.

Define the **probability ratio**:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

This measures how much more (or less) likely the new policy is to take the same action as the old policy. If $r_t = 1$, the policies are identical. If $r_t = 2$, the new policy is twice as likely to take that action.

The **PPO-Clip objective**:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

where $\epsilon = 0.2$ (the clip ratio).

**What does this do?** Let's break it down for the two cases:

#### Case 1: Advantage is positive ($\hat{A}_t > 0$ — the action was good)

We want to **increase** $r_t$ (make this action more likely). But the clip prevents $r_t$ from exceeding $1 + \epsilon = 1.2$. So the policy can become at most 20% more likely to take this action.

```
L = min(r·A, clip(r, 0.8, 1.2)·A)    where A > 0

If r < 1.2: L = r·A          (normal gradient, push r up)
If r ≥ 1.2: L = 1.2·A        (flat — no more gradient, STOP pushing)
                               ▲
Objective ─────────────────────┘────── clipped (flat)
                              1.2
```

#### Case 2: Advantage is negative ($\hat{A}_t < 0$ — the action was bad)

We want to **decrease** $r_t$ (make this action less likely). But the clip prevents $r_t$ from dropping below $1 - \epsilon = 0.8$. So the policy can become at most 20% less likely to take this action.

```
L = min(r·A, clip(r, 0.8, 1.2)·A)    where A < 0

If r > 0.8: L = r·A          (normal gradient, push r down)
If r ≤ 0.8: L = 0.8·A        (flat — no more gradient, STOP pushing)
```

**The brilliance**: This is dead simple to implement (it's literally a `torch.clamp` call), but it provides the same stability guarantees as much more complex algorithms like TRPO (Trust Region Policy Optimization).

### 3.7 The Complete PPO Training Loop

Here's the full algorithm as pseudocode, which maps directly to what we'll implement:

```
Initialize:
  - Actor network πθ (policy)
  - Critic network Vφ (value)
  - Rollout buffer (stores transitions)

FOR each training iteration (1 to num_iterations):
  
  ┌─── PHASE 1: COLLECT EXPERIENCE ───────────────────────┐
  │                                                        │
  │  FOR step = 1 to rollout_length (2048):                │
  │    1. Observe state s from environment                 │
  │    2. Sample action: a ~ πθ(·|s)                       │
  │    3. Record log_prob = log πθ(a|s)                    │
  │    4. Execute action, get: s', r, done                 │
  │    5. Store (s, a, r, log_prob, done) in buffer        │
  │    6. Query critic: V(s) = Vφ(s)                       │
  │                                                        │
  └────────────────────────────────────────────────────────┘
  
  ┌─── PHASE 2: COMPUTE ADVANTAGES (GAE) ─────────────────┐
  │                                                        │
  │  Working backwards from the last step:                 │
  │    δₜ = rₜ + γ·V(sₜ₊₁) - V(sₜ)    (TD error)        │
  │    Âₜ = δₜ + (γλ)·Âₜ₊₁              (GAE)            │
  │    Rₜ = Âₜ + V(sₜ)                  (returns)         │
  │                                                        │
  └────────────────────────────────────────────────────────┘
  
  ┌─── PHASE 3: PPO UPDATE ───────────────────────────────┐
  │                                                        │
  │  FOR epoch = 1 to K (10 epochs):                       │
  │    Shuffle data into mini-batches (size 64)            │
  │                                                        │
  │    FOR each mini-batch:                                │
  │      # --- Policy (Actor) Loss ---                     │
  │      new_log_probs = log πθ(aₜ|sₜ)                    │
  │      ratio = exp(new_log_probs - old_log_probs)        │
  │      surr1 = ratio * Â                                 │
  │      surr2 = clip(ratio, 1-ε, 1+ε) * Â                │
  │      policy_loss = -mean(min(surr1, surr2))            │
  │                                                        │
  │      # --- Value (Critic) Loss ---                     │
  │      value_pred = Vφ(s)                                │
  │      value_loss = mean((value_pred - R)²)              │
  │                                                        │
  │      # --- Entropy Bonus ---                           │
  │      entropy = mean(entropy of πθ(·|s))                │
  │                                                        │
  │      # --- Total Loss ---                              │
  │      loss = policy_loss + 0.5·value_loss - 0.01·entropy│
  │                                                        │
  │      # --- Gradient Step ---                           │
  │      optimizer.zero_grad()                             │
  │      loss.backward()                                   │
  │      clip_grad_norm_(params, max_norm=0.5)             │
  │      optimizer.step()                                  │
  │                                                        │
  └────────────────────────────────────────────────────────┘
  
  Log: mean_reward, policy_loss, value_loss, entropy
  Every N iterations: save checkpoint
```

### 3.8 The Entropy Bonus — Why We Encourage Exploration

Notice the $-0.01 \cdot \text{entropy}$ term in the loss. **Entropy** measures how "spread out" the action distribution is:

- High entropy → agent is exploring (trying different gimbal angles)
- Low entropy → agent is exploiting (always picking the same angle)

Early in training, we **want** high entropy so the agent discovers that gimbaling left corrects a rightward tilt. Without the entropy bonus, the agent might converge prematurely to a bad policy (e.g., always gimbal at 0°) and never discover better strategies.

The coefficient $0.01$ is small enough that exploitation eventually wins out as the agent finds good actions.

### 3.9 Continuous Actions: The Gaussian Policy

Since our action (gimbal angle) is **continuous**, the policy outputs a **Gaussian distribution**:

$$\pi_\theta(a|s) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(a - \mu_\theta(s))^2}{2\sigma^2}\right)$$

The **actor network** outputs:
- $\mu_\theta(s)$: The mean gimbal angle (what the network "recommends")
- $\log \sigma$: The log standard deviation (can be a learnable parameter, not state-dependent)

To sample an action:
```python
mu = actor_network(state)           # forward pass
std = exp(log_std)                   # learnable parameter
dist = Normal(mu, std)               # create Gaussian
action = dist.sample()               # sample
log_prob = dist.log_prob(action)     # needed for PPO ratio
action = torch.clamp(action, -1, 1)  # clip to valid range
```

The `log_prob` is critical — it's what we store during rollout collection and use to compute the probability ratio $r_t(\theta)$ during the PPO update.

---

## 4. The Gymnasium Environment — Landing Task

### 4.1 Initial Conditions (Randomized per Episode)

| Variable | Range | Purpose |
|----------|-------|---------|
| $x_0$ | $\mathcal{U}(-50, 50)$ m | Random horizontal offset |
| $y_0$ | $500$ m (fixed) | Starting altitude |
| $\theta_0$ | $\mathcal{U}(-10°, 10°)$ | Slight initial tilt |
| $\dot{x}_0$ | $\mathcal{U}(-10, 10)$ m/s | Slight horizontal drift |
| $\dot{y}_0$ | $\mathcal{U}(-30, -10)$ m/s | Initial descent rate |
| $\dot{\theta}_0$ | $\mathcal{U}(-5°, 5°)$/s | Slight initial spin |

### 4.2 Reward Function

```python
# Per-step reward (dense signal to guide learning)
reward = (
    -0.5 * abs(x / x_max)           # Penalize horizontal drift
    -1.0 * abs(theta)                # Penalize tilt (MOST important)
    -0.1 * abs(theta_dot)            # Penalize angular velocity
    -0.05 * abs(gimbal_angle)        # Penalize excessive gimbal
    +0.1                             # Alive bonus
)

# Terminal reward (big signal at episode end)
if landed_softly:         reward += 300   # |vy| < 2 m/s, |θ| < 5°, |x| < 10m
elif landed_hard:         reward += 50    # Touched down but too fast
elif crashed:             reward -= 200   # Hit ground at bad angle/speed
elif out_of_bounds:       reward -= 100   # Drifted too far
```

### 4.3 Observation Space (8-dimensional)

```python
observation = np.array([
    x / 500.0,            # Normalized horizontal position
    y / 500.0,            # Normalized altitude  
    vx / 50.0,            # Normalized horizontal velocity
    vy / 50.0,            # Normalized vertical velocity
    np.sin(theta),        # Angle (sin component)
    np.cos(theta),        # Angle (cos component)
    theta_dot / 5.0,      # Normalized angular velocity
    prev_gimbal / max_gimbal  # Previous gimbal command
], dtype=np.float32)
```

### 4.4 Action Space

```python
# Single continuous action: gimbal angle ∈ [-1, 1]
# Mapped to [-15°, +15°] in the physics engine
self.action_space = gymnasium.spaces.Box(low=-1, high=1, shape=(1,))
```

---

## 5. Mission Control Dashboard (Pygame)

This is what will make your portfolio pop. Instead of just a rocket on a black screen, we build a full **mission control dashboard**:

```
┌──────────────────────────────────────────────────────────────────────┐
│  🚀 ROCKET TVC — MISSION CONTROL                    [Episode: 347] │
├──────────────────────┬───────────────────────────────────────────────┤
│                      │  FLIGHT TELEMETRY            TRAINING METRICS │
│                      │  ┌──────────────────┐  ┌──────────────────┐  │
│    ╱╲                │  │ ALT:  234.5 m    │  │ Ep. Reward: 187  │  │
│   ╱  ╲               │  │ VEL:  -12.3 m/s  │  │ Policy Loss: 0.02│  │
│  │    │   ← rocket   │  │ PITCH: 3.2°      │  │ Value Loss: 0.15 │  │
│  │    │   (rotates    │  │ ω:    -0.5°/s    │  │ Entropy: 1.34    │  │
│   ╲──╱    in real     │  │ DRIFT: 15.2 m    │  │                  │  │
│    🔥      time)      │  └──────────────────┘  │ ┌──────────────┐ │  │
│    ╱                  │                         │ │  ▁▂▃▅▇█▇▅▃  │ │  │
│   ╱  (thrust vector)  │  GIMBAL & THRUST        │ │  reward curve│ │  │
│                      │  ┌──────────────────┐  │ └──────────────┘ │  │
│                      │  │    ◀──╋──▶       │  └──────────────────┘  │
│  ─ ─ landing pad ─ ─ │  │   gimbal: -3.2°  │                       │
│                      │  │   thrust: 100%    │  TRAJECTORY MAP       │
│                      │  └──────────────────┘  ┌──────────────────┐  │
│  (trajectory trail   │                         │    ·              │  │
│   shown as dots)     │  STATUS: DESCENDING     │   · ·            │  │
│                      │  REWARD: +0.73/step     │  ·   ↓ (rocket)  │  │
│                      │                         │        ╳ target  │  │
│                      │                         └──────────────────┘  │
├──────────────────────┴───────────────────────────────────────────────┤
│  [P]ause  [R]eset  [S]low-mo  [F]ast-fwd  Speed: 1x   Step: 847   │
└──────────────────────────────────────────────────────────────────────┘
```

### Dashboard Panels

| Panel | What it shows | Why it matters |
|-------|--------------|----------------|
| **Rocket Viewport** | Animated rocket with rotation, thrust flame, trajectory trail | The visual centrepiece — shows the agent controlling the rocket in real-time |
| **Flight Telemetry** | Live altitude, velocity, pitch, angular rate, drift | Shows the raw state — helps you understand what the agent is "seeing" |
| **Gimbal & Thrust** | Visual gimbal indicator + numeric readout | Shows what the agent is **doing** — the control commands it's issuing |
| **Training Metrics** | Episode reward, policy loss, value loss, entropy | Shows the AI **learning** — these numbers should improve over time |
| **Reward Curve** | Mini sparkline of recent episode rewards | At-a-glance trend — is the agent getting better? |
| **Trajectory Map** | Bird's-eye view of the descent path | Shows overall flight path quality — should converge to a clean vertical line |
| **Controls Bar** | Keyboard shortcuts for pause, reset, speed control | Makes it interactive for demos and presentations |

---

## 6. Implementation Plan — File by File

### Phase 1: Physics Engine (Days 1-2)

#### [NEW] `physics/__init__.py`
#### [NEW] `physics/rocket_dynamics.py`
- `RocketParams` dataclass (mass, inertia, thrust, gimbal limits, etc.)
- `rocket_derivatives(state, gimbal_angle, params)` → the $\dot{\mathbf{s}} = f(\mathbf{s}, \delta)$ function
- `rk4_step(state, action, params, dt)` → single RK4 integration step

#### [NEW] `physics/test_physics.py`
- Free-fall validation, vertical thrust validation, torque validation

---

### Phase 2: Gymnasium Environment (Days 2-3)

#### [NEW] `envs/__init__.py`
#### [NEW] `envs/rocket_tvc_env.py`
- `RocketTVCEnv(gymnasium.Env)` with `reset()`, `step()`, `render()`
- Reward function, termination conditions, observation normalization
- Gymnasium env checker compliance

#### [NEW] `envs/test_env.py`
- API compliance tests, reward sanity checks

---

### Phase 3: PPO Agent (Days 3-5)

#### [NEW] `agent/__init__.py`
#### [NEW] `agent/networks.py`
- `ActorNetwork(nn.Module)` → state → Gaussian(μ, σ)
- `CriticNetwork(nn.Module)` → state → scalar value V(s)
- Orthogonal weight initialization

#### [NEW] `agent/ppo.py`
- `RolloutBuffer` — stores (states, actions, rewards, log_probs, dones, values)
- `compute_gae()` — Generalized Advantage Estimation
- `PPOAgent` — the full training loop:
  - `collect_rollouts()` → run policy, fill buffer
  - `update()` → PPO clipped objective, value loss, entropy bonus
  - `save()` / `load()` → checkpoint management

---

### Phase 4: Training Script (Days 5-6)

#### [NEW] `train.py`
- CLI arguments (learning rate, epochs, rollout length, etc.)
- Training loop with logging (console + CSV)
- Periodic evaluation episodes (no exploration, deterministic policy)
- Checkpoint saving every N iterations

#### [NEW] `evaluate.py`
- Load trained model, run N episodes deterministically
- Print landing statistics (success rate, average drift, average landing speed)
- Generate trajectory plots

---

### Phase 5: Dashboard (Days 6-8)

#### [NEW] `dashboard/__init__.py`
#### [NEW] `dashboard/renderer.py`
- Pygame initialization, layout management
- Rocket drawing (body, fins, flame with gimbal direction)
- Trail rendering, landing pad

#### [NEW] `dashboard/panels.py`
- `TelemetryPanel` — flight data gauges
- `TrainingPanel` — loss/reward displays
- `GimbalPanel` — visual gimbal indicator
- `TrajectoryPanel` — mini-map of flight path
- `ControlBar` — keyboard shortcuts display

#### [NEW] `dashboard/run_dashboard.py`
- Main loop: load trained model → run episodes → render dashboard

---

### Phase 6: Polish (Days 8-9)

#### [NEW] `README.md`
- Project overview, architecture diagram, math summary
- Installation + training instructions
- Results with embedded plots/GIFs

#### [NEW] `requirements.txt`
- `torch`, `gymnasium`, `numpy`, `scipy`, `matplotlib`, `pygame`

---

## 7. Project Directory Structure

```
rocket-tvc-drl/
├── README.md
├── requirements.txt
├── physics/
│   ├── __init__.py
│   ├── rocket_dynamics.py
│   └── test_physics.py
├── envs/
│   ├── __init__.py
│   ├── rocket_tvc_env.py
│   └── test_env.py
├── agent/
│   ├── __init__.py
│   ├── networks.py
│   └── ppo.py
├── dashboard/
│   ├── __init__.py
│   ├── renderer.py
│   ├── panels.py
│   └── run_dashboard.py
├── train.py
├── evaluate.py
└── results/
    ├── checkpoints/
    └── plots/
```

---

## 8. Training Expectations

| Metric | Early Training | Mid Training | Converged |
|--------|---------------|-------------|-----------|
| Episode Reward | -200 to -100 | -50 to +50 | +150 to +250 |
| Landing Success Rate | 0% | 20-40% | 80-95% |
| Average Drift at Landing | N/A | ~30m | < 5m |
| Average Landing Speed | N/A | ~8 m/s | < 2 m/s |
| Training Steps | 0-100K | 100K-500K | 500K-2M |
| Wall-Clock Time (GPU) | — | — | ~1-3 hours |

---

## 9. Verification Plan

### Automated Tests
```bash
pytest physics/test_physics.py    # Physics engine correctness
pytest envs/test_env.py           # Gymnasium API compliance
```

### Training Validation
- Reward curve should show clear upward trend
- Policy loss should decrease and stabilize
- Entropy should decrease gradually (not collapse to 0)

### Landing Quality
- Run 100 evaluation episodes
- Report: success rate, mean landing speed, mean drift, mean final angle
- Visual inspection via dashboard — rocket should land smoothly and upright

---

## 10. Stretch Goals

1. **Throttle Control** — 2D action space (gimbal + throttle %)
2. **Wind Disturbances** — Random lateral forces during descent
3. **Fuel Depletion** — Decreasing mass shifts center of mass
4. **Curriculum Learning** — Start with easy (low altitude, no tilt) → progress to hard
5. **PPO vs. SAC Comparison** — Train both, compare learning curves
6. **Landing Legs Animation** — Deploy gear at low altitude in the dashboard
