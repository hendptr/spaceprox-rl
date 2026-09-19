from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from final_env import FinalBlackBoxEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play SpaceProx with the final controller.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--step-seconds", type=float, default=0.080)
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model.")
    model = PPO.load(args.model, device="cuda")
    action_count = int(model.action_space.n)
    if action_count != FinalBlackBoxEnv.MOVEMENT_ACTIONS:
        raise RuntimeError(
            f"Unsupported saved action space {model.action_space}; expected current Discrete(9) model."
        )
    env = FinalBlackBoxEnv(
        step_seconds=args.step_seconds,
        auto_launch=True,
        bring_to_front=False,
        fresh_process_on_first_reset=True,
    )
    observation, _ = env.reset()
    print(
        f"Playing {Path(args.model).name} on {torch.cuda.get_device_name(0)} "
        f"with learned movement + automatic visual aim ({action_count} movement actions)"
    )
    try:
        while True:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(
                    f"Episode score={info['score']} reward={info['episode_reward_live']:.2f} "
                    f"steps={info['episode_steps']}"
                )
                observation, _ = env.reset()
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
