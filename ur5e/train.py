"""Train SAC on the custom UR5e reach environment.

    python train.py                 # 100k steps
    python train.py --seed 1 --steps 150000

Same loop as the fetch_reach project. The only difference is that this env
returns a flat observation vector, so there is no dict to flatten.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from env import UR5eReachEnv
from sac import ReplayBuffer, SACAgent


def evaluate(env, agent, seed: int, episodes: int = 20) -> tuple[float, float]:
    """Run the deterministic policy on a fixed set of targets (re-seeded each
    call), so successive checkpoints are compared on the same task."""
    env.reset(seed=seed)
    returns, successes = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, success = False, 0.0, 0.0
        while not done:
            action = agent.act(obs, deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += rew
            success = max(success, info["is_success"])
            done = terminated or truncated
        returns.append(ep_ret)
        successes.append(success)
    return float(np.mean(returns)), float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-steps", type=int, default=2_000,
                        help="uniform-random actions before the policy takes over")
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = UR5eReachEnv()
    eval_env = UR5eReachEnv()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    agent = SACAgent(obs_dim, act_dim, device)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=args.buffer_size, device=device)

    run_name = args.run_name or f"ur5e_reach_seed{args.seed}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    eval_seed = args.seed + 10_000

    print(f"obs_dim={obs_dim}  act_dim={act_dim}  device={device}")
    obs, _ = env.reset(seed=args.seed)
    stats = {}
    best = (-1.0, -np.inf)   # (success, return), so ties break on return
    t0 = time.time()

    log_file = open(run_dir / "log.csv", "w", newline="")
    try:
        logger = csv.writer(log_file)
        logger.writerow(["step", "eval_return", "eval_success", "critic_loss", "actor_loss", "alpha"])

        for step in range(1, args.steps + 1):
            if step <= args.start_steps:
                action = env.action_space.sample()
            else:
                action = agent.act(obs)

            next_obs, rew, terminated, truncated, info = env.step(action)
            buffer.add(obs, action, rew, next_obs, float(terminated))
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()

            if step > args.update_after:
                stats = agent.update(buffer)

            if step % args.eval_every == 0:
                eval_ret, eval_success = evaluate(eval_env, agent, seed=eval_seed)
                sps = step / (time.time() - t0)
                print(f"step {step:>7}  return {eval_ret:8.2f}  success {eval_success:5.0%}  "
                      f"alpha {stats.get('alpha', float('nan')):.3f}  ({sps:.0f} steps/s)")
                logger.writerow([step, eval_ret, eval_success,
                                 stats.get("critic_loss"), stats.get("actor_loss"), stats.get("alpha")])
                log_file.flush()
                if (eval_success, eval_ret) > best:
                    best = (eval_success, eval_ret)
                    agent.save(run_dir / "best.pt")

        agent.save(run_dir / "final.pt")
    finally:
        log_file.close()
    print(f"done in {time.time() - t0:.0f}s, best success {best[0]:.0%}, saved to {run_dir}/")


if __name__ == "__main__":
    main()
