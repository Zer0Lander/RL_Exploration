"""Watch a policy live in the interactive MuJoCo viewer.

    python watch.py runs/FetchReachDense_seed0/best.pt     # trained policy
    python watch.py --random                               # untrained baseline

Drag to orbit the camera, scroll to zoom, Ctrl-C to quit.
"""

import argparse

import torch

from sac import SACAgent
from train import flat_obs, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", nargs="?", default="runs/FetchReachDense_seed0/best.pt")
    parser.add_argument("--env", default="FetchReachDense")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--random", action="store_true")
    args = parser.parse_args()

    env = make_env(args.env, render_mode="human")
    obs, _ = env.reset(seed=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACAgent(flat_obs(obs).shape[0], env.action_space.shape[0], device)
    if not args.random:
        agent.load(args.checkpoint)

    wins = 0
    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done, success = False, 0.0
            while not done:
                action = env.action_space.sample() if args.random else agent.act(flat_obs(obs), deterministic=True)
                obs, _, terminated, truncated, info = env.step(action)
                success = max(success, float(info.get("is_success", 0.0)))
                done = terminated or truncated
            wins += int(success)
            print(f"episode {ep}: {'success' if success else 'failure'}  ({wins}/{ep})")
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
