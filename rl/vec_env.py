# Adapted from rsl_rl (https://github.com/leggedrobotics/rsl_rl)
# Copyright (c) ETH Zurich, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD 3-Clause License — see https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE
# Modified by Rylan Peng, 2026

import torch
from abc import ABC, abstractmethod

class VecEnv(ABC):
    num_envs: int           # How many clones of the robot are running?
    num_actions: int        # How many motors/joints can the robot move?
    max_episode_length: int # How many steps before we force-reset the robot?
    episode_length_buf: torch.Tensor # Keeps track of how long each robot has been alive
    device: torch.device 

    @abstractmethod
    def get_observations(self) -> torch.Tensor:
        """Ask the world: 'What do the robots see right now?'"""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> torch.Tensor:
        """Restart the world from scratch."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Take an action and update the world.
        Returns: (New Observations, Rewards, Done flags, Extras Dictionary)
        """
        raise NotImplementedError
