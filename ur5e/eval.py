"""Evaluate a trained UR5e reach policy.

    python eval.py                                   # best.pt, 100 episodes
    python eval.py runs/ur5e_reach_v3/final.pt --episodes 200

Reports success and mean closest approach, broken down by how far the target
starts from the arm, so you can see whether failures are near or far. Uses a
fixed seed so the number is reproducible.

Two success rates: "reached" counts an episode if the gripper came within
SUCCESS_THRESHOLD at any step; "held" is stricter and requires it to still be
within threshold on the final step.
"""

import argparse

import numpy as np
import torch

from env import UR5eReachEnv
from sac import SACAgent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="runs/ur5e_reach_v3/best.pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    env = UR5eReachEnv()
    env.reset(seed=args.seed)   # fix the eval target sequence so the number is reproducible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(env.observation_space.shape[0], env.action_space.shape[0], device)
    agent.load(args.checkpoint)

    rows = []
    for _ in range(args.episodes):
        obs, _ = env.reset()
        start = float(np.linalg.norm(env._ee_pos() - env.target))
        closest, reached, held = start, 0.0, 0.0
        for _ in range(env.EPISODE_STEPS):
            obs, _, terminated, truncated, info = env.step(agent.act(obs, deterministic=True))
            closest = min(closest, info["distance"])
            reached = max(reached, info["is_success"])
            held = info["is_success"]          # success on the final step we run
            if terminated or truncated:
                break
        rows.append((start, closest, reached, held))
    rows = np.array(rows)

    print(f"checkpoint: {args.checkpoint}   episodes: {args.episodes}   seed: {args.seed}")
    print(f"reached (any step): {rows[:, 2].mean():.0%}   held (final step): {rows[:, 3].mean():.0%}   "
          f"mean closest: {rows[:, 1].mean() * 100:.1f} cm")
    print("reached, by start distance:")
    for lo, hi in [(0.0, 0.6), (0.6, 0.8), (0.8, 1.2)]:
        m = (rows[:, 0] >= lo) & (rows[:, 0] < hi)
        if m.sum():
            print(f"  {lo:.1f}-{hi:.1f} m: {rows[m, 2].mean():3.0%}   "
                  f"closest {rows[m, 1].mean() * 100:4.1f} cm   (n={int(m.sum())})")


if __name__ == "__main__":
    main()
