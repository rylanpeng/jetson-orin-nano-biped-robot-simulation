# Adapted from Genesis examples/locomotion (https://github.com/Genesis-Embodied-AI/Genesis)
# Copyright (c) Genesis-Embodied-AI. All rights reserved.
# Apache License 2.0 — see LICENSE
# Modified by Rylan Peng, 2026

import argparse
import os
import torch
import genesis as gs
from rl.on_policy_runner import OnPolicyRunner
from biped_robot.env import BipedEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="latest")
    parser.add_argument("-m", "--model_id", type=str, default="500")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    log_dir = os.path.join("logs", args.log_dir)

    env = BipedEnv(num_envs=1, show_viewer=True)

    runner = OnPolicyRunner(env, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.model_id}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs = env.reset()
    with torch.no_grad():
        while True:
            actions = policy(obs)
            obs, rews, dones, infos = env.step(actions, is_train=False)


if __name__ == "__main__":
    main()

"""
# evaluation
python -m biped_robot.eval
"""
