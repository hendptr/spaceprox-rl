from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


class InstrumentedPPO(PPO):
    """Stable-Baselines3 PPO with article-grade per-optimizer-epoch logging.

    The optimization math is intentionally identical to SB3 PPO 2.9.0. The only
    addition is a CSV row after each inner PPO epoch so the experiment preserves
    more detail than the standard logger's per-update aggregate values.
    """

    EPOCH_FIELDS = [
        "wall_time",
        "model_timesteps",
        "ppo_update_cycle",
        "epoch_in_cycle",
        "optimizer_epoch_global",
        "learning_rate",
        "clip_range",
        "policy_loss_mean",
        "value_loss_mean",
        "entropy_loss_mean",
        "approx_kl_mean",
        "clip_fraction_mean",
        "total_loss_mean",
    ]

    def set_epoch_metrics_path(self, path: str | Path | None) -> None:
        self.epoch_metrics_path = Path(path) if path else None
        if self.epoch_metrics_path is None:
            return
        self.epoch_metrics_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.epoch_metrics_path.exists():
            with self.epoch_metrics_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=self.EPOCH_FIELDS).writeheader()

    def _append_epoch_metrics(self, row: dict[str, float | int]) -> None:
        path = getattr(self, "epoch_metrics_path", None)
        if path is None:
            return
        with Path(path).open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.EPOCH_FIELDS)
            writer.writerow(row)

    def train(self) -> None:
        # This follows stable_baselines3.ppo.PPO.train() and adds per-epoch CSV
        # rows. Keeping the algorithm body aligned with SB3 makes the metrics
        # instrumentation transparent to the policy update itself.
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses: list[float] = []
        pg_losses: list[float] = []
        value_losses: list[float] = []
        clip_fractions: list[float] = []
        continue_training = True
        last_loss = th.tensor(0.0, device=self.device)
        last_approx_kl_divs: list[float] = []
        # One call to train() corresponds to one completed rollout buffer.  Track
        # that directly instead of deriving a cycle index from _n_updates, because
        # SB3 increments _n_updates once per optimizer epoch and target-KL early
        # stopping can make a rollout use fewer than self.n_epochs epochs.
        train_cycle = int(getattr(self, "_instrumented_train_cycle", 0)) + 1
        self._instrumented_train_cycle = train_cycle

        for epoch in range(self.n_epochs):
            epoch_entropy: list[float] = []
            epoch_pg: list[float] = []
            epoch_value: list[float] = []
            epoch_clip: list[float] = []
            epoch_kl: list[float] = []
            epoch_total: list[float] = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_value = float(policy_loss.item())
                pg_losses.append(pg_value)
                epoch_pg.append(pg_value)

                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)
                epoch_clip.append(float(clip_fraction))

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_value = float(value_loss.item())
                value_losses.append(value_value)
                epoch_value.append(value_value)

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_value = float(entropy_loss.item())
                entropy_losses.append(entropy_value)
                epoch_entropy.append(entropy_value)

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )
                last_loss = loss
                epoch_total.append(float(loss.item()))

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().numpy()
                    approx_kl = float(approx_kl_div)
                    epoch_kl.append(approx_kl)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} due to reaching max kl: "
                            f"{approx_kl_div:.2f}"
                        )
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.policy.optimizer.step()

            self._n_updates += 1
            last_approx_kl_divs = epoch_kl
            optimizer_epoch_global = int(self._n_updates)
            current_lr = float(self.policy.optimizer.param_groups[0]["lr"])
            self._append_epoch_metrics(
                {
                    "wall_time": time.time(),
                    "model_timesteps": int(self.num_timesteps),
                    "ppo_update_cycle": train_cycle,
                    "epoch_in_cycle": epoch + 1,
                    "optimizer_epoch_global": optimizer_epoch_global,
                    "learning_rate": current_lr,
                    "clip_range": float(clip_range),
                    "policy_loss_mean": float(np.mean(epoch_pg)) if epoch_pg else np.nan,
                    "value_loss_mean": float(np.mean(epoch_value)) if epoch_value else np.nan,
                    "entropy_loss_mean": float(np.mean(epoch_entropy)) if epoch_entropy else np.nan,
                    "approx_kl_mean": float(np.mean(epoch_kl)) if epoch_kl else np.nan,
                    "clip_fraction_mean": float(np.mean(epoch_clip)) if epoch_clip else np.nan,
                    "total_loss_mean": float(np.mean(epoch_total)) if epoch_total else np.nan,
                }
            )
            if not continue_training:
                break

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten()
        )
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record(
            "train/approx_kl",
            np.mean(last_approx_kl_divs) if last_approx_kl_divs else np.nan,
        )
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", last_loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
