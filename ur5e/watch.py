"""Watch a UR5e policy live in the interactive MuJoCo viewer.

    python watch.py                                  # best.pt
    python watch.py runs/ur5e_reach_v3/final.pt
    python watch.py --random                         # untrained baseline

Drag to orbit the camera, scroll to zoom, close the window or Ctrl-C to quit.
Needs a display (does not work over a plain headless SSH session).
"""

import argparse
import time

import mujoco.viewer
import torch

from env import UR5eReachEnv
from sac import SACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="runs/ur5e_reach_v3/best.pt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    env = UR5eReachEnv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device)
    if not args.random:
        agent.load(args.checkpoint)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.episodes):
            obs, _ = env.reset()
            viewer.sync()
            for _ in range(env.EPISODE_STEPS):
                if not viewer.is_running():
                    return
                action = env.action_space.sample() if args.random else agent.act(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                viewer.sync()
                time.sleep(0.02)   # pace it to roughly real time so it is watchable
                if terminated or truncated:
                    break


if __name__ == "__main__":
    main()
