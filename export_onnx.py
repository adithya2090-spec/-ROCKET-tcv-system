import argparse
from pathlib import Path
import torch

from agent.ppo import PPOAgent, PPOConfig
from envs.rocket_tvc_env import RocketTVCEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained PPO actor network to ONNX format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default="results/ppo_actor.onnx", help="Path for exported .onnx file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = RocketTVCEnv()
    
    config = PPOConfig(
        obs_dim=env.observation_space.shape[0],
        action_dim=env.action_space.shape[0],
        device="cpu"
    )
    agent = PPOAgent(config)
    agent.load(args.checkpoint)
    
    # Extract actor model and set to evaluation mode
    actor = agent.actor
    actor.eval()
    
    dummy_input = torch.randn(1, env.observation_space.shape[0], dtype=torch.float32)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        actor,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_mean", "action_std"],
        dynamic_axes={"observation": {0: "batch_size"}}
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")


if __name__ == "__main__":
    main()
