# Jetson Orin Nano Biped Robot Simulation

Biped locomotion training with PPO and Genesis, on the Jetson Orin Nano.

For the full walkthrough, see: [Biped Robot RL Simulation](https://rylanpeng.github.io/rylanpeng.dev/4-genesis-rl/)

## Setup

```bash
$ uv sync
$ bash patch_genesis.sh
```

## Run

### Training

```bash
# Default training (logs to logs/exp_<timestamp>)
$ uv run python -m biped_robot.train

# Custom training with base name
$ uv run python -m biped_robot.train --log_dir my_experiment --iterations 1000

# Profile training (see Profiling section below)
$ uv run python -m biped_robot.train --profile
```

### Evaluation

```bash
# Default evaluation (uses most recent run)
$ uv run python -m biped_robot.eval

# Custom evaluation
$ uv run python -m biped_robot.eval --log_dir exp_2026-04-01_23-44-00 --model_id 1000
```

### Teleoperation

```bash
# Default teleop evaluation (uses most recent run)
$ uv run python -m biped_robot.eval_teleop

# Custom teleop evaluation
$ uv run python -m biped_robot.eval_teleop -l exp_2026-04-01_23-44-00 -m 1000
```

## TensorBoard

```bash
$ uv run tensorboard --logdir logs
```

## Profiling

Use `--profile` to run `torch.profiler` for the first few training iterations and find performance bottlenecks.

```bash
$ uv run python -m biped_robot.train --profile
```

### CUPTI privileges (required for GPU timing data)

By default, CUDA profiling requires elevated privileges on Jetson. Without it, the profiler
only captures CPU time, with no CUDA time column, which makes the output useless.

**Fix (resets on reboot):**
```bash
$ sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'
```

After this, `uv run python -m biped_robot.train --profile` works as your normal user and
captures full GPU timing data.

**If `perf_event_paranoid` doesn't work**, activate the venv and use `sudo -E python` directly.
This runs as root for CUPTI access without touching `.venv` (since uv is never invoked):
```bash
$ source .venv/bin/activate
$ sudo -E python -m biped_robot.train --profile -i 10
```

The profiler skips 2 iterations (GPU warmup), then records 5 iterations, then stops. Training continues normally after.

**What you get:**

1. **Terminal table**: printed automatically when profiling finishes. Shows top ops sorted by CUDA time:
   ```
   Name               CPU total   CUDA total   # Calls
   env_step             12.1ms      298.4ms        32
   act                   3.1ms        8.2ms        32
   ppo_update            8.4ms       22.1ms         1
   ```

2. **TensorBoard trace**: written to the run's log dir alongside the model checkpoints. View it:
   ```bash
   $ uv run tensorboard --logdir logs/latest
   # Open browser -> PyTorch Profiler tab
   ```

### nsys (Nsight Systems)

**Check if nsys is installed:**
```bash
$ which nsys || which nsys-ui
# usually at /usr/local/cuda/bin/nsys on Jetson

# check if it's installed but not in PATH first:
$ find /usr/local/cuda* /opt/nvidia -name "nsys" 2>/dev/null
```

**Install:**
```bash
# if nothing find, we need to install nsys. do the following and look for whatever nsight-systems variant appears
$ sudo apt search nsight

# install nsys
$ sudo apt install nsight-systems-2024.5.4

# find it, it'll likely be at /opt/nvidia/nsight-systems/2024.5.4/bin/nsys
$ find /usr/local/cuda* /opt/nvidia -name "nsys" 2>/dev/null
/opt/nvidia/nsight-systems/2024.5.4/bin/nsys
/opt/nvidia/nsight-systems/2024.5.4/target-linux-tegra-armv8/nsys

# add it to env
echo 'export PATH=/opt/nvidia/nsight-systems/2024.5.4/bin:$PATH' >> ~/.bashrc
source ~/.bashrc  # apply to current session without reopening terminal
nsys --version    # verify
```

**Run:**
```bash
# nsys needs sudo on Jetson for full CUDA access.
# Use the full path because sudo drops your PATH.
$ sudo /opt/nvidia/nsight-systems/2024.5.4/bin/nsys profile \
    --output /tmp/train_profile \
    --force-overwrite true \
    --stats=true \
    --trace=cuda \
    --delay=120 \
    --duration=30 \
    .venv/bin/python -m biped_robot.train -i 200
```

- `--delay=120` skips the first 120s. Genesis JIT-compiles CUDA kernels lazily even
  past iter 20, so 120s (~iter 64) ensures all lazy JIT is fully done before capture starts.
- `--trace=cuda`: only trace CUDA kernels, not OS syscalls (`osrt`). Dropping osrt
  reduces nsys overhead significantly and we don't need syscall data anyway.
- `--duration=30` records 30s of steady-state training (~20 iterations) after the delay
- `-i 200`: training runs ~335s total, still going when capture ends at t=150s
- `--force-overwrite true` avoids "File exists" error on re-runs
- `--stats=true` prints a text summary to the terminal immediately after
- `--output` sets where the `.nsys-rep` trace file is written

**Terminal summary**:
```
CUDA Kernel Statistics:
Time(%)  Total Time (ns)  Instances  Name
   68.2%    1,234,567,890      1,600  gs::physics::step_kernel
   18.4%      333,000,000        160  aten::mm
   ...
```

**Visual timeline**:
Copy the `.nsys-rep` file to your laptop and open it in Nsight Systems GUI:
```bash
# On your laptop
scp rylan-nano@<jetson-ip>:/tmp/train_profile.nsys-rep ./
# Open in Nsight Systems GUI
```