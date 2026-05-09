# Adapted from rsl_rl (https://github.com/leggedrobotics/rsl_rl)
# Copyright (c) ETH Zurich, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD 3-Clause License — see https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE
# Modified by Rylan Peng, 2026

import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(
            self,
            num_obs,
            num_actions
    ):
        super().__init__()
        activation_fn = torch.nn.ELU()

        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512),
            activation_fn,
            nn.Linear(512, 256),
            activation_fn,
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, num_actions)
        )

        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512),
            activation_fn,
            nn.Linear(512, 256),
            activation_fn,
            nn.Linear(256, 128),
            activation_fn,
            nn.Linear(128, 1)
        )

        init_noise_std = 1.0
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # Disable validation for speed during training
        Normal.set_default_validate_args(False)

    @property
    def entropy(self):
        """A measure of how 'random' our actions are. High entropy = lots of exploration."""
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = self.std.expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """ How likely was it that we picked these specific actions? """
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Used during DEPLOYMENT: simply takes the mean action (best guess, no noise)."""
        return self.actor(observations)

    @property
    def action_std(self):
        """Returns the standard deviation of the action distribution (for logging)."""
        return self.std

    def evaluate(self, critic_observations, **kwargs):
        """Asks the Critic: 'How good is this state?' Returns the predicted Value."""
        return self.critic(critic_observations)
