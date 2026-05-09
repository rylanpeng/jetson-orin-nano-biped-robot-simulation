# Adapted from Genesis examples/locomotion (https://github.com/Genesis-Embodied-AI/Genesis)
# Copyright (c) Genesis-Embodied-AI. All rights reserved.
# Apache License 2.0 — see LICENSE
# Modified by Rylan Peng, 2026

import argparse
import os
import shutil
from datetime import datetime
import genesis as gs
from rl.on_policy_runner import OnPolicyRunner
from biped_robot.env import BipedEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="exp")
    parser.add_argument("-i", "--iterations", type=int, default=500)
    parser.add_argument("--profile", action="store_true", help="Run torch.profiler for the first few iterations and write trace to log_dir")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", performance_mode=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", f"{args.log_dir}_{timestamp}")

    os.makedirs(log_dir, exist_ok=True)

    # Create/update 'latest' symlink
    latest_link = os.path.join("logs", "latest")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(os.path.abspath(log_dir), latest_link)

    env = BipedEnv(num_envs=512)
    runner = OnPolicyRunner(env, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.iterations, profile=args.profile)

if __name__ == "__main__":
    main()
