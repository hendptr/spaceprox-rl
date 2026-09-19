from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from final_env import FinalBlackBoxEnv


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior probe for the consolidated policy.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--step-seconds", type=float, default=0.080)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this saved model.")
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
    observation, _ = env.reset()
    movements: Counter[int] = Counter()
    actions: Counter[int] = Counter()
    entropies: list[float] = []
    max_probabilities: list[float] = []
    player_x: list[float] = []
    player_y: list[float] = []
    corner_steps = 0
    stagnant_steps = 0
    engage_actuated = 0
    blocked_steps = 0
    low_clearance_steps = 0
    player_valid_steps = 0
    min_hp = 10**9
    last_info: dict = {}
    episodes: list[dict[str, float | int]] = []

    try:
        for _ in range(max(1, args.steps)):
            obs_tensor, _ = model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                distribution = model.policy.get_distribution(obs_tensor).distribution
                probs = distribution.probs[0].detach().cpu().numpy()
                entropy = float(distribution.entropy()[0].detach().cpu().item())
            entropies.append(entropy)
            max_probabilities.append(float(np.max(probs)))

            action, _ = model.predict(observation, deterministic=not args.stochastic)
            action_value = int(np.asarray(action).reshape(-1)[0])
            movement = env.decode_action(action_value)
            actions[action_value] += 1
            movements[movement] += 1

            observation, _, terminated, truncated, info = env.step(action_value)
            last_info = dict(info)
            min_hp = min(min_hp, int(info.get("hp", min_hp)))
            fresh_player = getattr(env, "_fresh_player", None)
            if fresh_player is not None:
                player_valid_steps += 1
                player_x.append(float(fresh_player[0]))
                player_y.append(float(fresh_player[1]))
            corner_steps += int(bool(info.get("in_corner", False)))
            stagnant_steps += int(bool(info.get("stagnant", False)))
            blocked_steps += int(bool(info.get("blocked_move", False)))
            low_clearance_steps += int(float(info.get("selected_clearance", 1.0)) < 0.30)
            engage_actuated += int(bool(info.get("engage_actuated", False)))

            if terminated or truncated:
                episodes.append(
                    {
                        "score": int(info.get("score", 0)),
                        "steps": int(info.get("episode_steps", 0)),
                        "reward": float(info.get("episode_reward_live", 0.0)),
                    }
                )
                observation, _ = env.reset()
    finally:
        env.close()

    steps = max(1, args.steps)
    max_entropy = math.log(action_count)
    result = {
        "model": str(Path(args.model).resolve()),
        "model_timesteps": int(model.num_timesteps),
        "action_mode": "movement_only_auto_combat",
        "action_count": action_count,
        "steps": steps,
        "deterministic": not args.stochastic,
        "action_counts": {str(k): v for k, v in sorted(actions.items())},
        "movement_counts": {str(k): v for k, v in sorted(movements.items())},
        "unique_actions": len(actions),
        "unique_movement_actions": len(movements),
        "target_engaged_fraction": engage_actuated / steps,
        "player_detection_fraction": player_valid_steps / steps,
        "mean_policy_entropy": float(np.mean(entropies)),
        "max_policy_entropy": max_entropy,
        "entropy_fraction_of_max": float(np.mean(entropies) / max_entropy),
        "mean_max_action_probability": float(np.mean(max_probabilities)),
        "player_x_minmax": [min(player_x), max(player_x)] if player_x else None,
        "player_y_minmax": [min(player_y), max(player_y)] if player_y else None,
        "corner_fraction": corner_steps / steps,
        "stagnant_fraction": stagnant_steps / steps,
        "blocked_fraction": blocked_steps / steps,
        "low_clearance_fraction": low_clearance_steps / steps,
        "final_score": int(last_info.get("score", 0)),
        "final_hp": int(last_info.get("hp", 0)),
        "min_hp": int(min_hp) if min_hp < 10**9 else None,
        "completed_episodes": episodes,
    }
    output = (
        Path(args.output)
        if args.output
        else ROOT / "logs" / "final" / f"probe_{Path(args.model).stem}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved behavior probe: {output}")


if __name__ == "__main__":
    main()
