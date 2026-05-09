# Adapted from rsl_rl (https://github.com/leggedrobotics/rsl_rl)
# Copyright (c) ETH Zurich, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# BSD 3-Clause License — see https://github.com/leggedrobotics/rsl_rl/blob/main/LICENSE
# Modified by Rylan Peng, 2026

import torch
import time
import os
import statistics
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import record_function, ProfilerActivity
from .vec_env import VecEnv
from .actor_critic import ActorCritic
from .ppo import PPO

class OnPolicyRunner:
    def __init__(self, env: VecEnv, log_dir: str | None = None, device="cpu"):
        self.device = device
        self.env = env

        num_obs = self.env.get_observations().shape[1]
        actor_critic = ActorCritic(num_obs, self.env.num_actions).to(self.device)
        self.alg = PPO(actor_critic, device=self.device)

        self.num_steps_per_env = 32
        self.save_interval = 100

        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, num_obs, self.env.num_actions)

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, profile: bool = False):
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        # Start the robot at a random point in its life (helps learning)
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations().to(self.device)
        self.alg.actor_critic.train()

        # Buffers to track average reward over time
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration

        # Profiler setup: skip 2 (GPU warmup) -> warmup 1 -> record 5 -> done (8 iters total)
        _prof_wait, _prof_warmup, _prof_active = 2, 1, 5
        _prof_stop_iter = start_iter + _prof_wait + _prof_warmup + _prof_active
        _prof = None
        if profile:
            _prof = torch.profiler.profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=_prof_wait, warmup=_prof_warmup, active=_prof_active, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
                record_shapes=False,
                with_stack=False,
            )
            _prof.start()
            print(f"[profiler] started, will record iterations {start_iter + _prof_wait + _prof_warmup + 1} to {_prof_stop_iter}, trace → {self.log_dir}")

        tot_iter = start_iter + num_learning_iterations

        # LOOP: Each iteration is one round of 'Play then Study'
        for it in range(start_iter, tot_iter):
            start = time.time()
            # 1. THE "PLAY" PHASE (Collection)
            # We don't need to learn while playing, so we use 'inference_mode'
            ep_infos = []
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    # Brain decides what to do
                    with record_function("act"):
                        actions = self.alg.act(obs)
                    # Environment takes the action and shows us what happened
                    with record_function("env_step"):
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    # Save this experience to our diary
                    self.alg.process_env_step(rewards, dones, extras)

                    # Accumulate episodic metrics from extras
                    if "episode" in extras:
                        ep_infos.append(extras["episode"])

                    # Update stats (how many points did we get?)
                    if self.log_dir is not None:
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                
                stop = time.time()
                collection_time = stop - start

            # 2. THE "STUDY" PHASE (Update)
            # This phase includes calculating returns (GAE) and updating the Brain's weights
            start_learn = time.time()
            with torch.inference_mode():
                with record_function("compute_returns"):
                    self.alg.compute_returns(obs)

            with record_function("ppo_update"):
                loss_dict = self.alg.update()
            stop = time.time()
            learn_time = stop - start_learn
            self.current_learning_iteration = it + 1

            # Log progress and save the brain every X iterations
            if self.log_dir is not None:
                total_time = collection_time + learn_time
                self.log(
                    it=self.current_learning_iteration,
                    tot_iter=tot_iter,
                    collection_time=collection_time,
                    learn_time=learn_time,
                    loss_dict=loss_dict,
                    rewbuffer=rewbuffer,
                    lenbuffer=lenbuffer,
                    ep_infos=ep_infos
                )
                if self.current_learning_iteration % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

            if _prof is not None:
                _prof.step()
                if it + 1 == _prof_stop_iter:
                    _prof.stop()
                    print("\n[profiler] done. Top ops by CUDA time:")
                    print(_prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
                    print(f"[profiler] TensorBoard trace written to {self.log_dir}\n")
                    _prof = None  # don't step again

        # Final save when training is done
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, it, tot_iter, collection_time, learn_time, loss_dict, rewbuffer, lenbuffer, ep_infos):
        """Prints stats to the console and saves them for Tensorboard."""
        iteration_time = collection_time + learn_time
        num_envs = self.env.num_envs
        collection_size = self.num_steps_per_env * num_envs
        
        self.tot_timesteps += collection_size
        self.tot_time += iteration_time
        fps = int(collection_size / iteration_time)

        # Log Losses
        self.writer.add_scalar("Loss/value", loss_dict["value"], it)
        self.writer.add_scalar("Loss/surrogate", loss_dict["surrogate"], it)
        self.writer.add_scalar("Loss/entropy", loss_dict["entropy"], it)
        self.writer.add_scalar("Loss/learning_rate", self.alg.optimizer.param_groups[0]["lr"], it)

        # Log Policy
        self.writer.add_scalar("Policy/mean_noise_std", self.alg.actor_critic.action_std.mean().item(), it)

        # Log Performance
        self.writer.add_scalar("Perf/total_fps", fps, it)
        self.writer.add_scalar("Perf/collection_time", collection_time, it)
        self.writer.add_scalar("Perf/learning_time", learn_time, it)

        # Log Train
        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_len = statistics.mean(lenbuffer)
            self.writer.add_scalar("Train/mean_reward", mean_reward, it)
            self.writer.add_scalar("Train/mean_episode_length", mean_len, it)
            self.writer.add_scalar("Train/mean_reward/time", mean_reward, int(self.tot_time))
            self.writer.add_scalar("Train/mean_episode_length/time", mean_len, int(self.tot_time))

        # Log Episode metrics (averaged across all steps in the iteration)
        extras_string = ""
        if ep_infos:
            for key in ep_infos[0].keys():
                infotensor = torch.tensor([], device=self.device)
                for ep_info in ep_infos:
                    if key in ep_info:
                        val = ep_info[key]
                        if not isinstance(val, torch.Tensor):
                            val = torch.tensor([val], device=self.device)
                        infotensor = torch.cat((infotensor, val.flatten().to(self.device)))
                
                if infotensor.numel() > 0:
                    value = torch.mean(infotensor)
                    self.writer.add_scalar("Episode/" + key, value, it)
                    extras_string += f"{f'Mean episode {key}:':>40} {value:.4f}\n"

        # Console Output (Table format matching rsl_rl-main)
        width = 80
        pad = 40
        log_string = f"{'#' * width}\n"
        log_string += f"\033[1m{f' Learning iteration {it}/{tot_iter} '.center(width)}\033[0m \n\n"
        
        log_string += (
            f"{'Total steps:':>{pad}} {self.tot_timesteps} \n"
            f"{'Steps per second:':>{pad}} {fps:.0f} \n"
            f"{'Collection time:':>{pad}} {collection_time:.3f}s \n"
            f"{'Learning time:':>{pad}} {learn_time:.3f}s \n"
        )
        
        for key, value in loss_dict.items():
            log_string += f"{f'Mean {key} loss:':>{pad}} {value:.4f}\n"
        
        if len(rewbuffer) > 0:
            log_string += f"{'Mean reward:':>{pad}} {statistics.mean(rewbuffer):.2f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {statistics.mean(lenbuffer):.2f}\n"

        log_string += f"{'Mean action noise std:':>{pad}} {self.alg.actor_critic.action_std.mean().item():.2f}\n"
        log_string += extras_string
        
        log_string += (
            f"{'-' * width}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Time elapsed:':>{pad}} {time.strftime('%H:%M:%S', time.gmtime(self.tot_time))}\n"
        )
        print(log_string)

    def save(self, path: str):
        """Saves the current state of the brain and teacher."""
        torch.save({
            "model_state_dict": self.alg.actor_critic.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
        }, path)

    def load(self, path: str):
        """Loads a previously saved brain and teacher."""
        loaded_dict = torch.load(path, weights_only=False)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]

    def get_inference_policy(self, device=None):
        """Returns the brain in 'inference' mode (best guess, no noise)."""
        self.alg.actor_critic.eval() # Set the brain to 'evaluation' mode
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

        


        