# Copyright (c) 2026 Rylan Peng. All rights reserved.
# Apache License 2.0 — see LICENSE

import argparse
import os
import threading
import torch
import numpy as np
import genesis as gs
from pynput import keyboard
from rl.on_policy_runner import OnPolicyRunner
from biped_robot.env import BipedEnv

STEP = 0.1

BINDINGS = {
    "w": ("lin_x", +STEP),
    "s": ("lin_x", -STEP),
    "a": ("lin_y", +STEP),
    "d": ("lin_y", -STEP),
    "q": ("ang_z", +STEP),
    "e": ("ang_z", -STEP),
}

LIMITS = {
    "lin_x": (-0.8, 0.8),
    "lin_y": (-0.3, 0.3),
    "ang_z": (-0.8, 0.8),
}


class TeleopController:
    def __init__(self):
        self._lock = threading.Lock()
        self._vel = {"lin_x": 0.0, "lin_y": 0.0, "ang_z": 0.0}

    def _on_press(self, key):
        try:
            ch = key.char
        except AttributeError:
            return
        if ch in BINDINGS:
            axis, delta = BINDINGS[ch]
            lo, hi = LIMITS[axis]
            with self._lock:
                self._vel[axis] = float(np.clip(self._vel[axis] + delta, lo, hi))
            self._print()

    def _on_release(self, key):
        if key == keyboard.Key.esc:
            return False

    def _print(self):
        with self._lock:
            v = dict(self._vel)
        os.system("clear")
        print(f"lin_x: {v['lin_x']:+.1f}  lin_y: {v['lin_y']:+.1f}  ang_z: {v['ang_z']:+.1f}")
        print("w/s forward/back  a/d strafe  q/e yaw  ESC exits viewer")

    def start(self):
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=True,
        )
        self._listener.start()

    def stop(self):
        self._listener.stop()

    def command_tensor(self):
        with self._lock:
            v = self._vel
            return torch.tensor([v["lin_x"], v["lin_y"], v["ang_z"]], dtype=gs.tc_float, device=gs.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir", type=str, default="latest")
    parser.add_argument("-m", "--model_id", type=str, default="500")
    args = parser.parse_args()

    gs.init(backend=gs.gpu)

    log_dir = os.path.join("logs", args.log_dir)

    env = BipedEnv(num_envs=1, show_viewer=True)

    runner = OnPolicyRunner(env, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.model_id}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    ctrl = TeleopController()
    ctrl.start()
    print("Teleop ready. w/s forward/back  a/d strafe  q/e yaw  ESC exits viewer")

    obs = env.reset()
    try:
        with torch.no_grad():
            while True:
                env.set_commands(ctrl.command_tensor(), manual=True)
                actions = policy(obs)
                obs, _rews, dones, _infos = env.step(actions, is_train=False)
                if dones.any():
                    obs = env.reset()
    finally:
        ctrl.stop()


if __name__ == "__main__":
    main()

"""
# teleop evaluation
python -m biped_robot.eval_teleop
"""
