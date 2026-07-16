"""Record a video of a trained agent.

Usage:
    python record.py runs/FetchReachDense_seed0/best.pt --env FetchReachDense --episodes 5
"""

import argparse
from pathlib import Path

import imageio
import numpy as np
import torch

from sac import SACAgent
from train import flat_obs, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--env", default="FetchReachDense")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--out", default=None)
    parser.add_argument("--random", action="store_true", help="record a random policy instead, for a before/after comparison")
    args = parser.parse_args()

    env = make_env(args.env, render_mode="rgb_array")
    obs, _ = env.reset(seed=42)
    obs_dim = flat_obs(obs).shape[0]
    act_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(obs_dim, act_dim, device)
    if not args.random:
        agent.load(args.checkpoint)

    frames = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, success = False, 0.0
        while not done:
            frames.append(env.render())
            if args.random:
                action = env.action_space.sample()
            else:
                action = agent.act(flat_obs(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            success = max(success, float(info.get("is_success", 0.0)))
            done = terminated or truncated
        print(f"episode {ep + 1}: {'success' if success else 'failure'}")

    out = args.out or ("videos/random.mp4" if args.random else "videos/trained.mp4")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out, [np.asarray(f) for f in frames], fps=25)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
