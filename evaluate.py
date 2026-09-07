import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from agent.ppo import PPOAgent, PPOConfig
from envs.rocket_tvc_env import RocketTVCEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Rocket TVC agent")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plot-path", type=str, default="results/plots/eval_trajectories.png")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def is_soft_landing(env: RocketTVCEnv) -> bool:
    x, y, theta, vx, vy, omega = env.state
    return (
        y <= 0.0
        and abs(vy) < 2.0
        and abs(vx) < 2.0
        and abs(theta) < 0.087
        and abs(x) < 10.0
    )


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    env = RocketTVCEnv()
    config = PPOConfig(obs_dim=env.observation_space.shape[0], action_dim=env.action_space.shape[0], device=device)
    agent = PPOAgent(config)
    agent.load(args.checkpoint)

    successes = 0
    landing_speeds = []
    drifts = []
    final_angles = []
    trajectories = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=episode)
        done = False
        path = []

        while not done:
            x, y, *_ = env.state
            path.append((x, y))
            action, _, _ = agent.select_action(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        x, y, theta, _, vy, _ = env.state
        path.append((x, max(y, 0.0)))
        trajectories.append(path)

        if is_soft_landing(env):
            successes += 1
        landing_speeds.append(abs(vy))
        drifts.append(abs(x))
        final_angles.append(np.rad2deg(abs(theta)))

    success_rate = 100.0 * successes / args.episodes
    print(f"Episodes: {args.episodes}")
    print(f"Soft landing success rate: {success_rate:.1f}%")
    print(f"Mean landing speed: {np.mean(landing_speeds):.2f} m/s")
    print(f"Mean drift: {np.mean(drifts):.2f} m")
    print(f"Mean final angle: {np.mean(final_angles):.2f} deg")

    plot_path = Path(args.plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 10))
    for path in trajectories:
        xs, ys = zip(*path)
        ax.plot(xs, ys, alpha=0.6)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.scatter([0], [0], color="green", marker="x", s=80, label="Target")
    ax.set_xlabel("Horizontal position (m)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Evaluation Trajectories")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    print(f"Saved trajectory plot: {plot_path}")


if __name__ == "__main__":
    main()
