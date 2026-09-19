from __future__ import annotations

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SemanticVisualExtractor(BaseFeaturesExtractor):
    """Compact MLP for screenshot-derived semantic geometry."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 192) -> None:
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.SiLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.network(observations)


def count_trainable(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

