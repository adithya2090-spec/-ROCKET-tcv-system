# Rocket TVC — Thrust Vector Control Landing with PPO

A 2D physics simulation of a reusable, gimbaled rocket learning to land on a platform using Proximal Policy Optimization (PPO), implemented from scratch in PyTorch.

The landing model includes RK4 rigid-body integration, variable mass from fuel burn, engine throttle and gimbal response limits, quadratic atmospheric drag, lateral wind, angular damping, and pad-aware touchdown rules. The agent commands both **gimbal** and **throttle**.

## Project structure

```
rocket-tvc/
├── physics/           # RK4 rocket dynamics
├── envs/              # Gymnasium landing environment
├── agent/             # PPO actor-critic (from scratch)
├── dashboard/         # Pygame mission control UI
├── train.py           # Training script
├── evaluate.py        # Evaluation + trajectory plots
└── results/           # Checkpoints and logs
```

## Setup

```bash
pip install -r requirements.txt
```

Run all commands from the project root so imports resolve correctly.

## Tests

```bash
pytest physics/test_physics.py
pytest envs/test_env.py
```

## Training

```bash
python train.py --iterations 200 --device auto
```

Checkpoints and CSV logs are written to `results/`.

## Evaluation

```bash
python evaluate.py --checkpoint results/checkpoints/ppo_final.pt --episodes 50
```

## Dashboard

Run with a trained checkpoint:

```bash
python dashboard/run_dashboard.py --checkpoint results/checkpoints/ppo_final.pt
```

Or run with random actions (no checkpoint):

```bash
python run_dashboard.py
# or
python -m dashboard
```

Controls: **P** pause, **R** reset episode, **S** change speed, **Q** quit. The mission-control UI uses a black/green palette with white outlines and shows live fuel, wind, throttle, gimbal, and trajectory data.

## Task

Land from ~500 m altitude with randomized initial tilt and velocity. Success requires:

- Vertical speed < 2 m/s
- Tilt < 5°
- Horizontal drift < 10 m
