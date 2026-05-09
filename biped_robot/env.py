# Adapted from Genesis examples/locomotion (https://github.com/Genesis-Embodied-AI/Genesis)
# Copyright (c) Genesis-Embodied-AI. All rights reserved.
# Apache License 2.0 — see LICENSE
# Modified by Rylan Peng, 2026

from dataclasses import dataclass
import genesis as gs
import math
import torch
from genesis.utils.geom import inv_quat, transform_by_quat, quat_to_xyz, transform_quat_by_quat
from rl.vec_env import VecEnv

@dataclass
class Joint:
    left_hip_yaw_joint: float = 0.0
    left_hip_pitch_joint: float = 0.4
    left_knee_joint: float = -0.9
    left_ankle_joint: float = 0.2
    right_hip_yaw_joint: float = 0.0
    right_hip_pitch_joint: float = 0.4
    right_knee_joint: float = -0.9
    right_ankle_joint: float = 0.2

def gs_rand(lower, upper, batch_shape):
    assert lower.shape == upper.shape
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower

class BipedEnv(VecEnv):
    def __init__(self, num_envs, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = 36
        self.num_actions = 8
        self.num_commands = 3
        self.device = gs.device

        # Desired episode duration in seconds "How long should one episode last in the real world"
        self.episode_length_s = 20.0
        # Simulation timestep (seconds per step) "How much real time passes when I call step() once"
        # Each step is 0.02 sec
        self.dt = 0.02
        # Number of steps per episode
        self.max_episode_length = math.ceil(self.episode_length_s / self.dt)

        self.obs_scales = {"lin_vel": 1.5, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05}
        self.reward_scales = {"tracking_lin_vel": 1.0, "tracking_ang_vel": 0.4, "lin_vel_z": -1.0, "base_height": -15.0, "action_rate": -0.01, "similar_to_default": -0.05}
        self.reward_cfg = {
            "tracking_sigma": 0.25,
            "base_height_target": 0.7,
        }

        base_init_pos = [0.0, 0.0, 0.7]
        base_init_quat = [1.0, 0.0, 0.0, 0.0]
        
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                tolerance=1e-5,
                max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        import os
        urdf_path = os.path.join(os.path.dirname(__file__), "robot.urdf")
        self.robot = self.scene.add_entity(gs.morphs.URDF(file=urdf_path, pos=base_init_pos, quat=base_init_quat))
        self.scene.build(n_envs=self.num_envs)

        self.joint = Joint()
        self.motors_dof_idx = torch.tensor([self.robot.get_joint(name).dof_start for name in self.joint.__dict__.keys()], dtype = gs.tc_int, device = gs.device)
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)

        kp, kd = 30.0, 1.0
        self.robot.set_dofs_kp([kp] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([kd] * self.num_actions, self.motors_dof_idx)

        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)

        self.init_base_pos = torch.tensor(base_init_pos, dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(base_init_quat, dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(list(self.joint.__dict__.values()), dtype = gs.tc_float, device = gs.device)
        self.init_qpos = torch.cat((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.obs_buf = torch.empty((self.num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.time_out_buf = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.episode_length_buf = torch.zeros((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)

        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )

        lin_vel_x_range = [-0.8, 0.8]
        lin_vel_y_range = [-0.3, 0.3]
        ang_vel_range = [-0.8, 0.8]
        self.commands_limits = [
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(lin_vel_x_range, lin_vel_y_range, ang_vel_range)
        ]

        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.manual_command_mask = torch.zeros((self.num_envs,), dtype=gs.tc_bool, device=gs.device)

        self.default_dof_pos = torch.tensor(list(self.joint.__dict__.values()), dtype = gs.tc_float, device = gs.device)

        self.reward_functions, self.episode_sums = dict(), dict()
        for name, value in self.reward_scales.items():
            # Scale by dt and update the dict
            # Multiplying by dt makes the reward independent of the simulation speed.
            self.reward_scales[name] = value * self.dt
            # Store reward function 
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            # Initialize episode sum
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

    def _resample_commands(self, envs_idx=None):
        if envs_idx is None:
            mask = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        else:
            mask = envs_idx.to(dtype=gs.tc_bool)
        mask = mask & (~self.manual_command_mask)
        if not mask.any():
            return

        commands = gs_rand(*self.commands_limits, (self.num_envs,))
        torch.where(mask[:, None], commands, self.commands, out=self.commands)
        self.manual_command_mask.masked_fill_(mask, False)

    def set_commands(self, commands, envs_idx=None, manual=True):
        cmd = torch.as_tensor(commands, dtype=gs.tc_float, device=gs.device)
        if cmd.ndim == 1:
            cmd = cmd.unsqueeze(0)

        if envs_idx is None:
            if cmd.shape[0] == 1 and self.num_envs > 1:
                cmd = cmd.repeat(self.num_envs, 1)
            elif cmd.shape[0] != self.num_envs:
                raise ValueError("Command batch size does not match number of environments.")
            self.commands.copy_(cmd)
            self.manual_command_mask.fill_(manual)
        else:
            mask = envs_idx.to(dtype=gs.tc_bool)
            count = int(mask.sum().item())
            if cmd.shape[0] == 1 and count > 1:
                cmd = cmd.repeat(count, 1)
            elif cmd.shape[0] != count:
                raise ValueError("Command batch size does not match selected environments.")
            self.commands[mask] = cmd
            self.manual_command_mask.masked_fill_(mask, manual)

    def step(self, actions, is_train=True):
        clip_actions, action_scale = 1.5, 0.5
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        target_dof_pos = self.actions * action_scale + self.default_dof_pos

        floating_based_dof = 6
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(floating_based_dof, floating_based_dof + self.num_actions))
        self.scene.step()

        # self.robot.get_vel() / get_ang() / get_pos(): These return values in the World Frame.
        self.episode_length_buf += 1
        self.base_pos = self.robot.get_pos()    # World
        self.base_quat = self.robot.get_quat()  # World
        # self.base_euler: robot's current orientation (Roll, Pitch, and Yaw) relative to its orientation at the very start of the episode.
        self.base_euler = quat_to_xyz(transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True)
        # self.base_lin_vel / self.base_ang_vel: These are the velocities transformed into the Body Frame.
        inv_base_quat = inv_quat(self.base_quat)    # Body
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)  # Body
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)  # Body
        # self.projected_gravity: This is the gravity vector $[0, 0, -1]$ rotated into the robot's current orientation.
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)  # Body

        # The Joint Values (self.dof_pos and self.dof_vel) are slightly different:
        # They aren't "world perspective" or "body perspective" in a Cartesian sense.
        # They are Joint Space values: the angle of the motor relative to its own zero point. These are naturally "local" to the robot's limbs.
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)    # Joint
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)    # Joint

        # Compute reward
        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            scale = self.reward_scales[name]
            rew = reward_func() * scale
            self.rew_buf += rew
            self.episode_sums[name] += rew

        resampling_time_s = 4.0
        if is_train:
            self._resample_commands(self.episode_length_buf % int(resampling_time_s / self.dt) == 0)

        termination_if_pitch_greater_than = 35.0
        termination_if_roll_greater_than = 35.0
        
        # rsl_rl expects time_outs in extras
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf = self.time_out_buf.clone()
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > termination_if_pitch_greater_than
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > termination_if_roll_greater_than

        # Prepare extras dictionary for logging
        extras = dict()
        extras["time_outs"] = self.time_out_buf
        if self.reset_buf.any():
            extras["episode"] = {}
            for name, buffer in self.episode_sums.items():
                extras["episode"]["rew_" + name] = torch.mean(buffer[self.reset_buf]) / self.episode_length_s
                self.episode_sums[name][self.reset_buf] = 0

        self._reset_idx(self.reset_buf)
        
        # These are for calculating rewards
        self.last_actions.copy_(self.actions)

        return self.get_observations(), self.rew_buf, self.reset_buf, extras
    
    def get_observations(self) -> torch.Tensor:
        # 3 + 3 + 3 + 3 + 8 + 8 + 8 = 36
        obs_parts = [
            self.base_lin_vel * self.obs_scales["lin_vel"],
            self.base_ang_vel * self.obs_scales["ang_vel"],
            self.projected_gravity,
            self.commands * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],
            self.dof_vel * self.obs_scales["dof_vel"],
            self.actions
        ]
        return torch.concatenate(obs_parts, dim=-1)

    def _reset_idx(self, envs_idx=None):
        self.robot.set_qpos(self.init_qpos, envs_idx=envs_idx, zero_velocity=True, skip_forward=True)

        if envs_idx is None:
            self.base_pos.copy_(self.init_base_pos)
            self.base_quat.copy_(self.init_base_quat)
            self.projected_gravity.copy_(self.init_projected_gravity)
            self.dof_pos.copy_(self.init_dof_pos)
            self.base_lin_vel.zero_()
            self.base_ang_vel.zero_()
            self.dof_vel.zero_()
            self.actions.zero_()
            self.last_actions.zero_()
            self.last_dof_vel.zero_()
            self.episode_length_buf.zero_()
            self.reset_buf.fill_(True)
            self.manual_command_mask.zero_()
        else:
            torch.where(envs_idx[:, None], self.init_base_pos, self.base_pos, out=self.base_pos)
            torch.where(envs_idx[:, None], self.init_base_quat, self.base_quat, out=self.base_quat)
            torch.where(
                envs_idx[:, None], self.init_projected_gravity, self.projected_gravity, out=self.projected_gravity
            )
            torch.where(envs_idx[:, None], self.init_dof_pos, self.dof_pos, out=self.dof_pos)
            self.base_lin_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.base_ang_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_actions.masked_fill_(envs_idx[:, None], 0.0)
            self.last_dof_vel.masked_fill_(envs_idx[:, None], 0.0)
            self.episode_length_buf.masked_fill_(envs_idx, 0)
            self.reset_buf.masked_fill_(envs_idx, True)
            self.manual_command_mask.masked_fill_(envs_idx, False)

        self._resample_commands(envs_idx)

    def reset(self):
        self._reset_idx()
        return self.get_observations()

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_base_height(self):
        return torch.square(self.base_pos[:, 2] - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
