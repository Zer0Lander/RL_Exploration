"""Record a video of a UR5e policy (offscreen, works headless).

    MUJOCO_GL=egl python record.py                          # best.pt -> videos/ur5e_reach.mp4
    MUJOCO_GL=egl python record.py --random                 # untrained baseline

MUJOCO_GL=egl selects offscreen GPU rendering, which is what lets this run on a
headless machine (no window needed).
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from env import UR5eReachEnv
from sac import SACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="runs/ur5e_reach_v3/best.pt")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default=None, help="output path (defaults per policy)")
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    env = UR5eReachEnv(render_mode="rgb_array")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device)
    if not args.random:
        agent.load(args.checkpoint)

    frames = []
    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            success = 0.0
            for _ in range(env.EPISODE_STEPS):
                frames.append(env.render())
                action = env.action_space.sample() if args.random else agent.act(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                success = max(success, info["is_success"])
                if terminated or truncated:
                    break
            print(f"episode {ep + 1}: {'success' if success else 'miss'}")
    finally:
        env.close()

    out = args.out or ("videos/ur5e_random.mp4" if args.random else "videos/ur5e_reach.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, [np.asarray(f) for f in frames], fps=25)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
