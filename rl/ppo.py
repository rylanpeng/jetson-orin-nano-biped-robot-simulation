# Adapted from rsl_rl (https://github.com/leggedrobotics/rsl_rl)
# Copyright (c) ETH Zurich, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD 3-Clause License — see https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE
# Modified by Rylan Peng, 2026

import torch
import torch.nn as nn
import torch.optim as optim
from .rollout_storage import RolloutStorage

class PPO:
    def __init__(
        self,
        actor_critic,
        device="cpu"
    ):
        self.num_mini_batches = 4       # How many small pieces we break the data into
        self.gamma = 0.99               # Discount factor - how much we care about future rewards      
    
        self.device = device

        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.learning_rate = 0.001  # How big of a step we take during learning
        self.desired_kl = 0.01      # Target KL divergence: how much the policy should change per update
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.storage = None # Will hold the "memories" of recent runs
        self.transition = RolloutStorage.Transition()   # A temporary container for one step

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, action_shape):
        """Prepares the 'Memory' buffer to store interactions."""
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, action_shape, device=self.device)

    def act(self, obs):
        """Takes an action and evaluates the state. Detach() means 'remember this but don't learn from it yet'."""
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, rewards, dones, extras):
        """Records what happened after the action: Reward and if we died."""
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Reward Correction: If an episode timed out (instead of failing), 
        # we add the predicted future value so the agent doesn't think 'ending' was bad.
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()

    def compute_returns(self, last_critic_obs):
        """Calculates 'Total Expected Reward' (Returns) for all stored memories."""
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma)

    def update(self):
        """THE LEARNING STEP: This is where the weights of the 'Brain' actually change."""
        num_learning_epochs = 5    # How many times we re-read the collected data
        clip_param = 0.2           # The "Safety Zone" - how much the policy can change (20%)

        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0

        # Loop through memories in small pieces (minibatches)
        generator = self.storage.mini_batch_generator(self.num_mini_batches, num_learning_epochs)
        for obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch in generator:
            # Re-run the Actor on these old observations to see what it thinks NOW
            self.actor_critic.act(obs_batch)
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(obs_batch)
            entropy_batch = self.actor_critic.entropy

            # -------------------
            # SURROGATE LOSS (The PPO Secret Sauce)
            # -------------------
            # Ratio: How much more/less likely is this action now compared to when we took it?
            # Equation: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            
            # ratio: standard policy gradient (ratio * advantage)
            surrogate = -torch.squeeze(advantages_batch) * ratio
            # surrogate_clipped: CAPPED policy gradient (keeps behavior from changing too fast)
            # Equation: L_clip = min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
            # We take the worst of the two (torch.max because losses are negative here)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # ---------------------------
            # VALUE FUNCTION LOSS (Critic)
            # ---------------------------
            # The Critic tries to predict the real Returns we actually got.
            # We apply safety clipping to the Critic's predictions to keep them stable
            # Equation: L_vf = (V_θ(s_t) - V_target)^2
            value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-clip_param, clip_param)
            value_losses = (value_batch - returns_batch).pow(2)
            value_losses_clipped = (value_clipped - returns_batch).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()

            # -------------------
            # TOTAL LOSS
            # -------------------
            # Combines:
            # 1. Be better (surrogate)
            # 2. Be more accurate at values (value_loss)
            # 3. Keep exploring (entropy)
            # Equation: L_total = L_clip + c1*L_vf - c2*L_entropy
            value_loss_coef = 1.0      # How much we care about the Critic being right
            entropy_coef = 0.01        # How much we reward "trying new things" (exploration)
            loss = surrogate_loss + value_loss_coef * value_loss - entropy_coef * entropy_batch.mean()

            # BACKPROPAGATION: The math magic that updates weights
            self.optimizer.zero_grad() # Clear old math
            loss.backward()            # Calculate how much to change each weight
            max_grad_norm = 1.0        # Prevents the "math from exploding" by capping gradients
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm) # Safety check
            self.optimizer.step()      # Actually apply the change

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()

        # Adaptive KL: measure how much the policy changed and adjust learning rate
        # If policy changed too much -> slow down; too little -> speed up
        # KL ≈ E[log(p_old) - log(p_new)], computed over the full rollout buffer
        with torch.inference_mode():
            all_obs = self.storage.observations.flatten(0, 1)
            all_actions = self.storage.actions.flatten(0, 1)
            old_log_probs = self.storage.actions_log_prob.flatten(0, 1).squeeze(-1)
            self.actor_critic.act(all_obs)
            new_log_probs = self.actor_critic.get_actions_log_prob(all_actions)
            kl = (old_log_probs - new_log_probs).mean()

        if kl > 2.0 * self.desired_kl:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
        elif kl < 0.5 * self.desired_kl:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

        num_updates = num_learning_epochs * self.num_mini_batches
        self.storage.clear() # Wipe memories for next round
        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
        }
