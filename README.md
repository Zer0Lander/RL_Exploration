# Reinforcement Learning

A personal repo for learning reinforcement learning by building it, focused on
robot control in simulation (MuJoCo). The algorithms are written from scratch in
PyTorch rather than pulled from a library, so the goal is understanding how RL
actually works, not just calling `.train()`.

Three projects:

## `fetch_reach/` (working)

Soft Actor-Critic implemented from scratch and trained on the Fetch reach task
from [Gymnasium-Robotics](https://robotics.farama.org/). All three seeds reach
100% success. It has training curves, before/after videos, and scripts to train,
evaluate, watch, and record. See [`fetch_reach/README.md`](fetch_reach/README.md).

## `ur5e/` (working)

A custom Gymnasium environment built from scratch around the Universal Robots
UR5e arm, reusing the same SAC agent. Here the observation, action, reward,
target sampling, and success criteria are all defined by hand in `env.py`, and
SAC trains it to 100% success over 100 eval episodes (mean final error 0.8 cm).
The point is the environment-design side of RL. See
[`ur5e/README.md`](ur5e/README.md).

## `go2/` (in progress)

Teaching a Unitree Go2 quadruped to walk at a commanded velocity, with PPO
implemented from scratch and trained across parallel environments. Just started:
the robot model is vendored and stands in simulation; the environment and PPO are
next. This is the ambitious one. See [`go2/README.md`](go2/README.md).

## Setup

One shared virtual environment at the repo root:

```bash
uv venv
uv pip install -r fetch_reach/requirements.txt -r ur5e/requirements.txt
```

Then run either project from its own directory with the venv active:

```bash
source .venv/bin/activate
cd fetch_reach && python train.py
```
