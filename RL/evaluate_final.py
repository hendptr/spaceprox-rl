from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from stable_baselines3 import PPO

from final_env import FinalBlackBoxEnv


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the consolidated controller.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--step-seconds", type=float, default=0.080)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this model.")
    model = PPO.load(args.model, device="cuda")
    action_count = int(model.action_space.n)
    if action_count != FinalBlackBoxEnv.MOVEMENT_ACTIONS:
        raise RuntimeError(f"Unsupported action space: {model.action_space}")
    env = FinalBlackBoxEnv(
        step_seconds=args.step_seconds,
        auto_launch=True,
        bring_to_front=False,
        fresh_process_on_first_reset=True,
    )
    scores: list[int] = []
    rewards: list[float] = []
    lengths: list[int] = []
    started = time.monotonic()
    try:
        for index in range(1, args.episodes + 1):
            observation, _ = env.reset()
            terminated = truncated = False
            info: dict = {}
            while not (terminated or truncated):
                action, _ = model.predict(
                    observation, deterministic=not args.stochastic
                )
                observation, _, terminated, truncated, info = env.step(action)
            scores.append(int(info["score"]))
            rewards.append(float(info["episode_reward_live"]))
            lengths.append(int(info["episode_steps"]))
            print(
                f"[EVAL] episode={index:02d} score={scores[-1]:5d} "
                f"reward={rewards[-1]:8.2f} steps={lengths[-1]:5d}",
                flush=True,
            )
    finally:
        env.close()

    result = {
        "model": str(Path(args.model).resolve()),
        "model_timesteps": int(model.num_timesteps),
        "action_mode": "movement_only_auto_combat",
        "action_count": action_count,
        "episodes": args.episodes,
        "deterministic": not args.stochastic,
        "scores": scores,
        "rewards": rewards,
        "lengths": lengths,
        "mean_score": statistics.fmean(scores),
        "median_score": statistics.median(scores),
        "min_score": min(scores),
        "best_score": max(scores),
        "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "mean_reward": statistics.fmean(rewards),
        "median_reward": statistics.median(rewards),
        "mean_steps": statistics.fmean(lengths),
        "median_steps": statistics.median(lengths),
        "steps_stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0.0,
        "elapsed_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT / "logs" / "final" / f"evaluation_{Path(args.model).stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved evaluation: {output}")


if __name__ == "__main__":
    main()
