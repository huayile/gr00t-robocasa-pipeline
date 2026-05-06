# GR00T N1.6 × RoboCasa: Closed-Loop VLA Evaluation Pipeline

A fully automated closed-loop Vision-Language-Action (VLA) inference pipeline connecting [RoboCasa365](https://robocasa.ai/) simulation (local WSL2) with [NVIDIA Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) policy inference (GRASP cluster).

---

## Architecture

```
┌─────────────────────────────┐         SSH Tunnel              ┌──────────────────────────────┐
│     Local (WSL2, RTX 4060)  │ ◄─────────────────────────────► │    GRASP Cluster (A40)       │
│                             │                                 │                              │
│  RoboCasa365 Simulation     │     Image Observations (→)      │  GR00T N1.6 Policy Server    │
│  - Renders RGB images       │     Action Commands  (←)        │  - VLM processing inputs     │
│  - Executes robot actions   │                                 │  - Flow matching denoising   │
│  - Franka Panda arm         │                                 │  - ROBOCASA_PANDA_OMRON tag  │
└─────────────────────────────┘                                 └──────────────────────────────┘
```

At each timestep:
1. RoboCasa renders image observations from the simulated Franka Panda arm
2. Observations are sent to the GR00T Policy Server over the SSH tunnel
3. GR00T processes the inputs and returns an action chunk using VLM and flow matching
4. RoboCasa executes the action and advances the simulation

> RoboCasa365 and the official GR00T evaluation scripts provide a fully automated rollout pipeline. The image-action loop runs internally rather than requiring manual teleoperation.

---

## System Requirements

### Local (WSL2)
- OS: Windows 10/11 with WSL2 (Ubuntu 22.04 recommended)
- GPU: NVIDIA GPU with WSL2 passthrough (tested: RTX 4060, 8GB VRAM)
- Disk: **≥ 30GB free** on the drive hosting the WSL `.vhdx` file (kitchen assets ~5GB, dependencies ~10GB)
- RAM: ≥ 8GB (tested with 8GB)
- Software: Conda, Git, `uv`

### GRASP Cluster
- Account: GRASP cluster access with SSH ed25519 key registered
- GPU: One Ampere GPU (A40 / L40 / L40S, 46GB+ VRAM). Turing GPUs (e.g. 2080Ti) can work without flash-attn but are slower

---

## Part 1: Cluster Setup — GR00T Policy Server
 
 
### Step 1 — Clone Isaac-GR00T to home
```bash
cd <your_space>
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```
 
### Step 2 — Switch to the N1.6 release tag
The `main` branch tracks N1.7, which removes `ROBOCASA_PANDA_OMRON`. Pin to N1.6:
```bash
git checkout n1.6-release
git submodule update --init --recursive
```
 
### Step 3 — Install `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```
 
### Step 4 — Set up CUDA environment
The cluster has CUDA 11.8 toolkit and a driver supporting CUDA 12.x. The driver version is what matters for runtime; flash-attn is fetched as a pre-built wheel so no `nvcc` compilation is needed.
```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-11.8
```
 
### Step 5 — Allocate a GPU and install dependencies
 
Request an interactive A40 session (or other Ampere GPUs):
```bash
srun --partition=kostas-compute --qos=kd-med --gres=gpu:a40:1 --pty bash
```
 
Once on the compute node, set CUDA env vars again and install:
```bash
cd <your_space>/Isaac-GR00T
export PATH=/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
uv sync
```
 
`uv sync` will:
- Create `.venv/` in the repo
- Install PyTorch 2.7.1+cu128, transformers, diffusers, deepspeed, etc.
- Pull a pre-built flash-attn 2.7.4 wheel

Verify:
```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
Expected: `2.7.1+cu128 True`.
 
### Step 6 — Start the GR00T policy server
 
```bash
export MPLBACKEND=Agg
export HF_HOME=/mnt/kostas-graid/datasets/<your_path>/cache/huggingface
mkdir -p $HF_HOME
.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --model-path nvidia/GR00T-N1.6-3B \
    --use-sim-policy-wrapper \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 6789
```
 
Wait for: `Server is ready and listening on tcp://0.0.0.0:6789`.
 
Note the node name (e.g. `kd-a40-0`) from the shell prompt, you'll need it for the tunnel.
 
---

## Part 2: Local Setup — RoboCasa365 Simulation (WSL2)

### Prerequisites

Install `uv` package manager (required by the setup script):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env  # or restart terminal
```

Install system-level dependencies required for MuJoCo rendering and Python compilation:
```bash
sudo apt update
sudo apt install libegl1-mesa-dev libglu1-mesa python3.10-dev
```

### Step 1 — Clone Isaac-GR00T (N1.6)

```bash
mkdir -p ~/workspace && cd ~/workspace
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git checkout n1.6-release
git submodule update --init --recursive
```

### Step 2 — Run the official RoboCasa setup script

```bash
bash gr00t/eval/sim/robocasa/setup_RoboCasa.sh
```

The script:
- Creates `.venv` at `gr00t/eval/sim/robocasa/robocasa_uv/.venv`
- Installs PyTorch 2.5.1, robosuite, robocasa, gymnasium, etc.
- Downloads kitchen textures, fixtures, Objaverse objects (~5GB)
- Runs a sanity check, prints `Env OK`

> **Do NOT install flash-attn locally.** The local side only runs simulation, so flash-attn isn't needed and would fail without a CUDA toolkit in WSL.

> The `CUDA_HOME not set` warning during install is expected and harmless.

### Step 3 — Verify installation
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -c "
import gymnasium as gym
import robocasa.utils.gym_utils.gymnasium_groot
envs = [e for e in gym.envs.registry.keys() if 'PandaOmron' in e]
print(f'Available tasks: {len(envs)}')
print(envs[:5])
"
```
Expected output: `Available tasks: 140+`


### (Optional) Visual verification
To confirm environment rendering works correctly:
```bash
cd ~/workspace/Isaac-GR00T/external_dependencies/robocasa
~/workspace/Isaac-GR00T/gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -m robocasa.demos.demo_kitchen_scenes
```
A kitchen scene should pop up. Confirms WSL2 GPU passthrough and display work.

---

## Part 3: Connecting Local ↔ Cluster (SSH Tunnel)

Open a **new local terminal**. Replace `<NODE>` with the cluster node name from Part 1 Step 6 (e.g. `kd-a40-0`):

```bash
ssh -L 6789:<NODE>:6789 <pennkey>@grasp-login1.seas.upenn.edu
```

This forwards `localhost:6789` on your laptop to `<NODE>:6789` on the cluster. Keep this terminal open.

> `client_global_hostkeys_private_confirm: server gave bad signature for RSA key`, it is harmless.

---

## Part 4: Running Evaluations

With the SSH tunnel running and the GR00T server listening, open a **third terminal** in WSL and run:

### Standard task
```bash
cd ~/workspace/Isaac-GR00T

gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 3 \
    --policy_client_host localhost \
    --policy_client_port 6789 \
    --max_episode_steps 720 \
    --env_name robocasa_panda_omron/OpenDrawer_PandaOmron_Env \
    --n_action_steps 8 \
    --n_envs 1
```

### Key parameters

| Parameter | Description | Default | Notes |
|---|---|---|---|
| `--n_episodes` | Episodes to run | 3 | Each randomly initialized |
| `--max_episode_steps` | Max steps per episode | 720 | Increase for complex tasks |
| `--n_action_steps` | Action chunk size | 8 | Lower → more correction; higher → smoother |
| `--env_name` | Task environment | — | See task list below |
| `--n_envs` | Parallel environments | 1 | Keep at 1 for remote server |
| `--render_mode` | `human` for GUI | — | Omit for fast headless rollout |

---

## Part 5: Benchmark Results

Tested with GR00T N1.6-3B on a single A40 (cluster) and Franka Panda (local WSL2), connected by SSH tunnel.

| Task | Type | n_action_steps=8 | Time/episode | n_action_steps=2 | Time/episode |
|---|---|---|---|---|---|
| `OpenDrawer_PandaOmron_Env` | Single-step, fixed target | **3/3 (100%)** | ~5 min | — | — |
| `CoffeeServeMug_PandaOmron_Env` | Pick-and-place | 0/3 (0%) | ~12 min | **3/3 (100%)** | ~16 min |
| `StackBowlsInSink_PandaOmron_Env` | Multi-step, precise | 0/3 (0%) | ~12 min | 0/3 (0%) | ~36 min |

**Key observations:**
- GR00T performs well on single-step, spatially fixed tasks (OpenDrawer: 100%)
- Pick-and-place tasks benefit significantly from smaller action chunks (n=2)
- Multi-step tasks requiring sequential grasping remain challenging zero-shot

---

## Part 6: Custom Simulation Environment (`fixed_pnp_env.py`)

A script for VLA benchmarking with configurable object types and positions. Located at `grasp_experiments/fixed_pnp_env.py`.

you can **Jump directly to [Usage](#usage)**.

### Design

The environment places object A and B on the kitchen counter next to the sink. The robot must pick object A and place it into object B. Key properties:

- **Fixed scene**: layout `[5,1]` (U_SHAPED_SMALL + SCANDANAVIAN) is enforced; `--seed 2` is the validated default.
- **Specific objects**: specify exact object names (`banana`, `bowl`) rather than random categories.
- **Precise positioning**: object XY positions are set relative to the robot base via MuJoCo joint control, overriding RoboCasa's random placement.
- **Official pipeline**: uses `GrootRoboCasaEnv` wrapper and `create_grootrobocasa_env_class` for correct `ROBOCASA_PANDA_OMRON` observation/action format.

### Coordinate System

Positions are specified relative to the robot base:
- `x`: left (−) / right (+)
- `y`: forward distance from robot (positive = away)
- `z`: automatically set from RoboCasa placement (correct surface height per object)

The default config places both objects ~0.57m from the robot base, well within reach.

### Implementation

To ensure fixed environment and overcome RoboCasa's default randomization without modifying its source code, some overrides were employed:
 
**Registration:** RoboCasa's `create_env_robosuite()` only accepts a fixed set of parameters. The solution is a dynamic subclass with parameters baked in at class definition time:
 
```python
class FixedPnPInstance(FixedPnP):
    def __init__(self, *args, **kwargs):
        super().__init__(obj_a_group="banana", obj_a_x=-0.4, ...)
 
REGISTERED_ENVS["FixedPnP"] = FixedPnPInstance
create_grootrobocasa_env_class("FixedPnP", "PandaOmron", "panda_omron")
```
 
**Layout fixing:** `create_env_robosuite()` hardcodes a list of layouts via `env_kwargs.update(...)`, so passing `layout_and_style_ids` as a kwarg has no effect. The solution is to override `self.layout_and_style_ids` directly after the parent `__init__` completes — this works because `_reset_internal` reads from `self.layout_and_style_ids` at each episode reset:
 
```python
class FixedPnP(Kitchen):
    FIXED_LAYOUT = [(5, 1)]  # U_SHAPED_SMALL + SCANDANAVIAN
 
    def __init__(self, ...):
        super().__init__(*args, **kwargs)
        self.layout_and_style_ids = self.FIXED_LAYOUT  # override after parent sets it
```
 
This cleanly enforces layout [5,1] in both preview and inference modes without modifying any library code. Verify with:
```python
inner_env = env.unwrapped.env
print(inner_env.layout_id, inner_env.style_id)  # should print: 5 1
```
 
**Position control:** After RoboCasa's standard reset, `_reset_internal` overrides XY positions using MuJoCo free joint `qpos`, while preserving each object's Z from the original placement (correct surface height per object):
 
```python
def _reset_internal(self):
    super()._reset_internal()
    base = self.robots[0].base_pos
    jnt_a = self.sim.model.body_jntadr[self.sim.model.body_name2id("obj_a_main")]
    adr_a = self.sim.model.jnt_qposadr[jnt_a]
    z_a = self.sim.data.qpos[adr_a + 2]  # preserve original surface height
    self.sim.data.qpos[adr_a:adr_a+3] = [base[0] + self.obj_a_x, base[1] + self.obj_a_y, z_a]
    self.sim.forward()
```
> Keep objects ≥ 0.15m apart in XY to avoid physics overlap. The default config (`obj_a_x=-0.4, obj_b_x=0.4`, `seed=2`) is validated stable.

### Usage
 
**Preview the scene interactively** (no policy server needed):
```bash
cd ~/workspace/Isaac-GR00T
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --preview --obj-a banana --obj-b bowl --seed 2
```
Controls: mouse drag to rotate, arrow keys to move robot, spacebar to toggle gripper, Ctrl+Q to exit.
 
**Run with GR00T policy server:**
```bash
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host localhost --port 6789 \
    --obj-a banana --obj-b bowl \
    --seed 2 --n-episodes 3 --n-action-steps 8 --save-video
```
 
**Systematic distance experiment (vary x separation):**
```bash
# Default: objects 0.8m apart (validated with layout [5,1])
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host localhost --port 6789 \
    --obj-a-x -0.4 --obj-b-x 0.4 --seed 2 --n-episodes 3
 
# far: objects 1m apart
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python grasp_experiments/fixed_pnp_env.py \
    --host localhost --port 6789 \
    --obj-a-x -0.4 --obj-b-x 0.6 --seed 2 --n-episodes 3
```
 
### Parameters
 
| Parameter | Description | Default |
|---|---|---|
| `--preview` | Launch interactive mjviewer | - |
| `--obj-a` | Object A: specific name (`banana`, `apple`, `carrot`...) or category (`fruit`, `vegetable`...) | `banana` |
| `--obj-b` | Object B: specific name (`bowl`, `plate`, `cup`...) or category (`receptacle`) | `bowl` |
| `--obj-a-x` | Object A x offset from robot base, left(-)/right(+) in meters | `-0.4` |
| `--obj-a-y` | Object A y offset from robot base, forward distance in meters | `0.4` |
| `--obj-b-x` | Object B x offset from robot base, left(-)/right(+) in meters | `0.4` |
| `--obj-b-y` | Object B y offset from robot base, forward distance in meters | `0.4` |
| `--seed` | Seed for object randomization (kitchen layout fixed to [5,1]) | `2` |
| `--n-episodes` | Number of episodes | `3` |
| `--n-action-steps` | Action chunk size | `8` |
| `--save-video` | Save rollout videos to `grasp_experiments/videos/` | - |
 
Available object names (partial list): `banana`, `apple`, `carrot`, `cucumber`, `orange`, `tomato`, `sweet_potato`, `broccoli`, `can`, `bowl`, `plate`, `cup`, `mug`, `pot`, `pan`
 
Videos are saved to `grasp_experiments/videos/`. Copy to Windows:
```bash
cp -r ~/workspace/Isaac-GR00T/grasp_experiments/videos /mnt/c/Users/Username/Desktop/
```
 
---
 
## Task List
 
```bash
# List all available Franka Panda tasks
gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python -c "
import gymnasium as gym
import robocasa.utils.gym_utils.gymnasium_groot
for e in sorted(gym.envs.registry.keys()):
    if 'PandaOmron' in e:
        print(e)
"
```
 
Examples include:
| Task | Description |
|---|---|
| `OpenDrawer_PandaOmron_Env` | Open a kitchen drawer |
| `PnPCounterToCab_PandaOmron_Env` | Pick from counter, place in cabinet |
| `CoffeeServeMug_PandaOmron_Env` | Move mug from coffee machine to counter |
| `StackBowlsInSink_PandaOmron_Env` | Pick and stack bowls in sink |
| `PrepareCoffee_PandaOmron_Env` | Multi-step coffee preparation |
| `TurnOnSinkFaucet_PandaOmron_Env` |Turn on sink faucet|
 
---

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| `Unknown embodiment tag: 'ROBOCASA_PANDA_OMRON'` | On N1.7 branch (the default `main`) | `git checkout n1.6-release && git submodule update --init --recursive` |
| `zmq.error.ZMQError: Address already in use` | Another process is using that port | Pick a different port (e.g. `6789`) and update SSH tunnel and client to match |
| `evdev build error: Python.h not found` | Missing Python dev headers in WSL | `sudo apt install python3.10-dev` |
| `ImportError: flash_attn not installed` (cluster) | Installed to system Python, not `.venv` | `uv sync` regenerates `.venv` with the correct flash-attn wheel |
| `CUDA_HOME not set` warning during local install | No CUDA toolkit in WSL | Expected — local side does not need flash-attn |
| `FlashAttention only supports Ampere GPUs or newer` | Got assigned a Turing GPU (e.g. 2080Ti) | Request `--gres=gpu:a40:1` or `--gres=gpu:l40:1` |
| Rollout hangs at `Episodes: 0%` | SSH tunnel down or server not ready | Verify the SSH tunnel terminal is alive and cluster shows `Server is ready` |
| `client_global_hostkeys_private_confirm: server gave bad signature for RSA key` | Harmless SSH host-key warning | Ignore |
| `df: ~/.triton/autotune: No such file or directory` | DeepSpeed cache dir warning on first launch | Harmless; optionally `export TRITON_CACHE_DIR=/tmp/triton_cache_$USER` |

---

## References

- [RoboCasa365](https://robocasa.ai/) — kitchen simulation framework
- [Isaac GR00T N1.6](https://github.com/NVIDIA/Isaac-GR00T) — NVIDIA foundation model for generalist robots
- [GR00T RoboCasa Eval Guide](https://github.com/NVIDIA/Isaac-GR00T/blob/main/examples/robocasa/README.md) — primary reference for this pipeline setup
- [GR00T Hardware Requirements](https://www.mintlify.com/NVIDIA/Isaac-GR00T/getting-started/hardware-requirements) — hardware guide
