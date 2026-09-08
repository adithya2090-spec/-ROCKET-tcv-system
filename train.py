import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from agent.ppo import PPOAgent, PPOConfig
from envs.rocket_tvc_env import RocketTVCEnv


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO agent for Rocket TVC landing")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rollout-length", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    results_dir = Path(args.results_dir)
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_path = results_dir / f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    env = RocketTVCEnv()
    config = PPOConfig(
        lr=args.lr,
        rollout_length=args.rollout_length,
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device=device,
    )
    agent = PPOAgent(config)

    print(f"Training on {device} (Seed: {args.seed}) for {args.iterations} iterations")

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "iteration",
                "mean_episode_reward",
                "episodes_collected",
                "policy_loss",
                "value_loss",
                "entropy",
            ],
        )
        writer.writeheader()

        for iteration in range(1, args.iterations + 1):
            rollout_stats = agent.collect_rollouts(env)
            update_stats = agent.update(rollout_stats["last_value"])

            row = {
                "iteration": iteration,
                "mean_episode_reward": rollout_stats["mean_episode_reward"],
                "episodes_collected": rollout_stats["episodes_collected"],
                **update_stats,
            }
            writer.writerow(row)

            print(
                f"[{iteration:04d}] reward={row['mean_episode_reward']:.2f} "
                f"policy={row['policy_loss']:.4f} value={row['value_loss']:.4f} "
                f"entropy={row['entropy']:.4f}"
            )

            if iteration % args.checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"ppo_iter_{iteration:04d}.pt"
                agent.save(ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

    final_path = checkpoint_dir / "ppo_final.pt"
    agent.save(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")
    print(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
