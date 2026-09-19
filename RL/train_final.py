from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from collections import deque
from pathlib import Path

import torch
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from final_env import FinalBlackBoxEnv, FinalRewardConfig
from final_model import SemanticVisualExtractor
from instrumented_ppo import InstrumentedPPO


ROOT = Path(__file__).resolve().parent
CHECKPOINT_ROOT = ROOT / "checkpoints" / "final"
LOG_ROOT = ROOT / "logs" / "final"


class EpisodeStatsCallback(BaseCallback):
    def __init__(self, print_every: int = 100) -> None:
        super().__init__()
        self.print_every = int(print_every)
        self.scores: list[int] = []
        self.rewards: list[float] = []
        self.lengths: list[int] = []
        self.recent_scores: deque[int] = deque(maxlen=10)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        if not infos:
            return True
        info = infos[0]
        episode = info.get("episode")
        if episode:
            score = int(info.get("score", 0))
            reward = float(episode["r"])
            length = int(episode["l"])
            self.scores.append(score)
            self.rewards.append(reward)
            self.lengths.append(length)
            self.recent_scores.append(score)
            print(
                f"[EPISODE] #{len(self.scores):03d} score={score:5d} "
                f"reward={reward:8.2f} steps={length:5d} "
                f"mean10={statistics.fmean(self.recent_scores):7.1f}",
                flush=True,
            )
        if self.num_timesteps % self.print_every == 0:
            print(
                f"[LIVE] step={self.num_timesteps:,} HP={info.get('hp', '?')} "
                f"Ammo={info.get('ammo', '?')} Score={info.get('score', '?')} "
                f"blocked={info.get('blocked_move', False)} "
                f"stagnant={info.get('stagnant', False)} "
                f"Reward={info.get('episode_reward_live', 0.0):.2f}",
                flush=True,
            )
        return True


class WallClockStopCallback(BaseCallback):
    def __init__(self, seconds: float) -> None:
        super().__init__()
        self.seconds = float(seconds)
        self.started = 0.0
        self.triggered = False

    def _on_training_start(self) -> None:
        self.started = time.monotonic()

    def _on_step(self) -> bool:
        if self.seconds <= 0:
            return True
        if time.monotonic() - self.started >= self.seconds:
            self.triggered = True
            print(
                f"\n[STOP] Wall-clock training budget reached: {self.seconds / 60.0:.2f} min",
                flush=True,
            )
            return False
        return True


class PerRunOptimizedSnapshotCallback(BaseCallback):
    """Save exactly one reloadable model after each completed PPO train() cycle."""

    def __init__(self, run_dir: Path, prefix: str) -> None:
        super().__init__()
        self.run_dir = Path(run_dir)
        self.prefix = prefix
        self.manifest = self.run_dir / "optimized_snapshots.csv"
        self.last_saved_optimizer_epochs = 0
        self.last_saved_step = 0
        self.completed_cycles = 0
        self.last_saved_model: Path | None = None

    def _on_training_start(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.last_saved_optimizer_epochs = int(getattr(self.model, "_n_updates", 0))
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [
                    "saved_unix",
                    "ppo_cycle",
                    "model_timesteps",
                    "optimizer_epochs_total",
                    "model_path",
                ]
            )

    def _save_if_optimized(self) -> None:
        current_epochs = int(getattr(self.model, "_n_updates", 0))
        if current_epochs <= self.last_saved_optimizer_epochs:
            return
        self.completed_cycles += 1
        step = int(self.model.num_timesteps)
        stem = self.run_dir / f"{self.prefix}_cycle_{self.completed_cycles:04d}_{step}_steps"
        self.model.save(str(stem))
        model_path = Path(f"{stem}.zip")
        with self.manifest.open("a", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(
                [time.time(), self.completed_cycles, step, current_epochs, str(model_path)]
            )
        self.last_saved_optimizer_epochs = current_epochs
        self.last_saved_step = step
        self.last_saved_model = model_path
        print(
            f"[OPTIMIZED] cycle={self.completed_cycles} step={step:,} "
            f"epochs_total={current_epochs} -> {model_path.name}",
            flush=True,
        )

    def _on_rollout_start(self) -> None:
        self._save_if_optimized()

    def _on_training_end(self) -> None:
        # Covers the exact-timestep-cap case where there is no next rollout start.
        self._save_if_optimized()

    def _on_step(self) -> bool:
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the consolidated black-box SpaceProx PPO controller."
    )
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--timesteps", type=int, default=250_000)
    parser.add_argument("--step-seconds", type=float, default=0.080)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration.")

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"seed{args.seed}_{timestamp}"
    run_checkpoint_dir = CHECKPOINT_ROOT / run_name
    run_log_dir = LOG_ROOT / run_name
    if run_checkpoint_dir.exists() or run_log_dir.exists():
        raise FileExistsError(
            f"Run name already exists: {run_name}. Choose a unique --run-name."
        )
    run_checkpoint_dir.mkdir(parents=True)
    run_log_dir.mkdir(parents=True)

    reward_config = FinalRewardConfig()
    monitor_base = run_log_dir / "monitor.csv"
    epoch_metrics = run_log_dir / "ppo_epoch_metrics.csv"
    summary_path = run_log_dir / "training_summary.json"
    final_stem = run_checkpoint_dir / "final_recorded_state"

    env = Monitor(
        FinalBlackBoxEnv(
            step_seconds=args.step_seconds,
            auto_launch=True,
            bring_to_front=False,
            reward_config=reward_config,
            fresh_process_on_first_reset=True,
        ),
        filename=str(monitor_base),
        info_keywords=("score", "hp", "best"),
    )

    policy_kwargs = dict(
        features_extractor_class=SemanticVisualExtractor,
        features_extractor_kwargs=dict(features_dim=192),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
        activation_fn=torch.nn.SiLU,
        normalize_images=False,
    )
    model = InstrumentedPPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=5.0e-4,
        n_steps=128,
        batch_size=64,
        n_epochs=8,
        gamma=0.990,
        gae_lambda=0.95,
        clip_range=0.20,
        ent_coef=0.0002,
        vf_coef=0.50,
        max_grad_norm=0.50,
        target_kl=0.030,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(run_log_dir / "tensorboard"),
        device="cuda",
        verbose=1,
        seed=args.seed,
    )
    model.set_epoch_metrics_path(epoch_metrics)
    trainable = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    snapshots = PerRunOptimizedSnapshotCallback(
        run_checkpoint_dir / "optimized",
        prefix="spaceprox_ppo",
    )
    episodes = EpisodeStatsCallback(print_every=100)
    wall_stop = WallClockStopCallback(max(0.0, args.minutes * 60.0))
    callbacks = CallbackList([snapshots, episodes, wall_stop])

    print("=" * 78)
    print("SpaceProx BLACK-BOX HIERARCHICAL VISUAL PPO - CUDA")
    print(f"Run name       : {run_name}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"PyTorch        : {torch.__version__}")
    print(f"CUDA runtime   : {torch.version.cuda}")
    print(f"Observation    : {FinalBlackBoxEnv.OBS_FEATURES} screenshot-derived float features")
    print("Action         : Discrete(9) movement only")
    print("Low-level aim  : automatic nearest visible enemy + velocity intercept")
    print("PPO            : n_steps=128 batch=64 epochs=8 ent_coef=0.0002")
    print(f"Trainable pars : {trainable:,}")
    print(f"Wall budget    : {args.minutes:.2f} min")
    print("=" * 78, flush=True)

    started_unix = time.time()
    started_mono = time.monotonic()
    interrupted = False
    failure: Exception | None = None
    try:
        model.learn(
            total_timesteps=int(args.timesteps),
            callback=callbacks,
            reset_num_timesteps=True,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("\n[STOP] Ctrl+C received; preserving state.", flush=True)
    except Exception as exc:
        failure = exc
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", flush=True)
    finally:
        # This file can include metadata from a partial, unoptimized rollout. The
        # authoritative learned policy for evaluation is last_optimized_model_path.
        model.save(str(final_stem))
        elapsed = time.monotonic() - started_mono
        optimized = int(snapshots.last_saved_step)
        recorded = int(model.num_timesteps)
        summary = {
            "run_name": run_name,
            "started_unix": started_unix,
            "elapsed_seconds": elapsed,
            "recorded_interactions": recorded,
            "optimized_transitions": optimized,
            "partial_rollout_interactions": max(0, recorded - optimized),
            "completed_ppo_cycles": int(snapshots.completed_cycles),
            "optimizer_epochs_total": int(getattr(model, "_n_updates", 0)),
            "last_optimized_model_path": (
                str(snapshots.last_saved_model.resolve())
                if snapshots.last_saved_model is not None
                else None
            ),
            "final_recorded_state_path": str(Path(f"{final_stem}.zip").resolve()),
            "interrupted": interrupted,
            "failed": failure is not None,
            "failure_type": type(failure).__name__ if failure else None,
            "failure_message": str(failure) if failure else None,
            "wall_clock_stop_triggered": wall_stop.triggered,
            "target_wall_clock_seconds": max(0.0, args.minutes * 60.0),
            "seed": args.seed,
            "step_seconds": args.step_seconds,
            "observation_features": FinalBlackBoxEnv.OBS_FEATURES,
            "actions": FinalBlackBoxEnv.MOVEMENT_ACTIONS,
            "reward_config": reward_config.__dict__,
            "ppo": {
                "n_steps": 128,
                "batch_size": 64,
                "n_epochs": 8,
                "learning_rate": 5.0e-4,
                "gamma": 0.990,
                "gae_lambda": 0.95,
                "clip_range": 0.20,
                "ent_coef": 0.0002,
                "vf_coef": 0.50,
                "max_grad_norm": 0.50,
                "target_kl": 0.030,
            },
            "trainable_parameters": trainable,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "episodes": len(episodes.scores),
            "episode_scores": episodes.scores,
            "episode_rewards": episodes.rewards,
            "episode_lengths": episodes.lengths,
            "best_episode_score": max(episodes.scores, default=0),
            "mean_episode_score": (
                statistics.fmean(episodes.scores) if episodes.scores else 0.0
            ),
            "monitor_file": str(Path(f"{monitor_base}.monitor.csv").resolve()),
            "epoch_metrics_file": str(epoch_metrics.resolve()),
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        env.close()
        print(f"Saved summary : {summary_path}")
        print(f"Optimized model: {summary['last_optimized_model_path']}", flush=True)

    if failure is not None:
        raise failure


if __name__ == "__main__":
    main()
