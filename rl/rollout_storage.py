# Adapted from rsl_rl (https://github.com/leggedrobotics/rsl_rl)
# Copyright (c) ETH Zurich, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD 3-Clause License — see https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE
# Modified by Rylan Peng, 2026

import torch

class RolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs_shape, actions_shape, device="cpu"):
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # Initialize massive tables (Tensors) to store every step for every environment
        # Observations: What the robot saw [Time, RobotID, SensorData]
        self.observations = torch.zeros(num_transitions_per_env, num_envs, obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        # Extra info needed by the PPO math
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.step = 0 # Current page in our diary

    def add_transitions(self, transition: Transition):
        """Saves one step of experience into the tables."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow!")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.step += 1

    def clear(self):
        """Wipes the diary clean for the next training iteration."""
        self.step = 0

    def compute_returns(self, last_values, gamma):
        """
        CALCULATING THE SCORECARD
        -------------------------
        This is one of the most important parts of RL.
        We look back at our diary and figure out: 
        'Given what happened later, how good was each step actually?'
        
        It uses GAE (Generalized Advantage Estimation).
        """

        advantage = 0
        # We process the diary BACKWARDS (from the end to the start)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            
            # If the next step was a reset (died), we stop counting future rewards
            next_is_not_terminal = 1.0 - self.dones[step].float()
            
            # TD Error: reward + future_prediction - current_prediction
            # 'Did this step turn out better than I predicted?'
            # Equation: δ_t = r_t + γ*V(s_t+1) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            
            # GAE formula: blends immediate rewards with long-term predictions
            # Equation: A_t = δ_t + (γ*λ)*A_t+1
            lam = 0.95                 # GAE lambda - helps balance bias and variance in rewards
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return Equation: R_t = A_t + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Advantages: How much better was this action than average?
        self.advantages = self.returns - self.values
        # Normalization: Makes the math more stable for the neural network
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """
        Slices up the diary into random small pieces for the the brain to study.
        By shuffling the data, we help the neural network learn more general patterns.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        
        # Flatten: Change [Time, Env] into one long list of millions of steps
        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)

        for epoch in range(num_epochs):
            # Reshuffle each epoch so the network sees data in a different order every pass
            indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Hand over one small piece of data (minibatch) to the trainer
                yield observations[batch_idx], actions[batch_idx], values[batch_idx], advantages[batch_idx], returns[batch_idx], old_actions_log_prob[batch_idx]
