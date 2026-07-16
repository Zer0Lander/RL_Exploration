# Reinforcement Learning

A personal repo for learning reinforcement learning by building it, focused on
robot control in simulation (MuJoCo). The algorithms are written from scratch in
PyTorch rather than pulled from a library, so the goal is understanding how RL
actually works, not just calling `.train()`.

It holds two projects at different stages:

## `fetch_reach/` (working)

Soft Actor-Critic implemented from scratch and trained on the Fetch reach task
from [Gymnasium-Robotics](https://robotics.farama.org/). All three seeds reach
100% success. This one is done: it has training curves, before/after videos, and
scripts to train, evaluate, watch, and record. See
[`fetch_reach/README.md`](fetch_reach/README.md).

## `ur5e/` (in progress)

A custom Gymnasium environment built from scratch around the Universal Robots
UR5e arm, reusing the same SAC agent. The point here is the environment-design
side of RL: defining observations, actions, rewards, and success ourselves. So
far this holds the vendored robot model; the environment is next. See
[`ur5e/README.md`](ur5e/README.md).

## Setup

```bash
uv venv && uv pip install -r fetch_reach/requirements.txt
```

Each project runs from its own directory (paths are relative):

```bash
cd fetch_reach && python train.py
```
