"""Train SAC on the goal-conditioned Fetch reach task.

    python train.py                          # FetchReachDense, 50k steps
    python train.py --env FetchReachDense --steps 50000 --seed 1

Fetch envs return a Dict observation with three parts: `observation`
(gripper pose/velocity + joint state), `desired_goal` (the target position),
and `achieved_goal` (the gripper position). The policy needs its own state
and the target, so we feed it concat(observation, desired_goal).
"""

import argparse
import csv
import re
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac import ReplayBuffer, SACAgent

# Fetch samples targets in a box around the initial gripper pose. The bottom of
# that box sits ~1.5cm below the tabletop (z=0.40), so a few percent of targets
# spawn inside the table and the arm clips through it reaching them. Keep the
# target a few cm clear of the surface.
TABLE_TOP_Z = 0.40
MIN_GOAL_Z = TABLE_TOP_Z + 0.05


def make_env(name: str, min_goal_z: float | None = MIN_GOAL_Z, **kwargs) -> gym.Env:
    """gym.make that resolves a bare name (e.g. 'FetchReachDense') to the newest
    registered version, and floors the target height to keep it above the table."""
    import gymnasium_robotics
    gym.register_envs(gymnasium_robotics)
    if name not in gym.registry:
        versions = [
            int(m.group(1))
            for env_id in gym.registry
            if (m := re.fullmatch(re.escape(name) + r"-v(\d+)", env_id))
        ]
        if not versions:
            raise ValueError(f"No registered env matches '{name}'")
        name = f"{name}-v{max(versions)}"

    env = gym.make(name, **kwargs)
    base = env.unwrapped
    if min_goal_z is not None and hasattr(base, "_sample_goal"):
        sample = base._sample_goal

        def sample_above_table():
            goal = sample()
            goal[2] = max(goal[2], min_goal_z)
            return goal

        base._sample_goal = sample_above_table
    return env


def flat_obs(obs_dict: dict) -> np.ndarray:
    return np.concatenate([obs_dict["observation"], obs_dict["desired_goal"]], dtype=np.float32)


def evaluate(env: gym.Env, agent: SACAgent, episodes: int = 20) -> tuple[float, float]:
    """Run the deterministic policy and return (mean return, success rate)."""
    returns, successes = [], []
    for _ in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, success = False, 0.0, 0.0
        while not done:
            action = agent.act(flat_obs(obs), deterministic=True)
            obs, rew, terminated, truncated, info = env.step(action)
            ep_ret += rew
            success = max(success, float(info.get("is_success", 0.0)))
            done = terminated or truncated
        returns.append(ep_ret)
        successes.append(success)
    return float(np.mean(returns)), float(np.mean(successes))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="FetchReachDense")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-steps", type=int, default=2_000,
                        help="uniform-random actions before the policy takes over")
    parser.add_argument("--update-after", type=int, default=1_000)
    parser.add_argument("--eval-every", type=int, default=2_000)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--render", action="store_true",
                        help="show the sim in a live viewer window while training (slower)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = make_env(args.env, render_mode="human" if args.render else None)
    eval_env = make_env(args.env)
    obs_dim = flat_obs(env.reset(seed=args.seed)[0]).shape[0]
    act_dim = env.action_space.shape[0]
    eval_env.reset(seed=args.seed + 10_000)

    agent = SACAgent(obs_dim, act_dim, device)
    buffer = ReplayBuffer(obs_dim, act_dim, capacity=args.steps, device=device)

    run_name = args.run_name or f"{args.env}_seed{args.seed}"
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(run_dir / "log.csv", "w", newline="")
    logger = csv.writer(log_file)
    logger.writerow(["step", "eval_return", "eval_success", "critic_loss", "actor_loss", "alpha"])

    print(f"env={env.spec.id}  obs_dim={obs_dim}  act_dim={act_dim}  device={device}")
    obs, _ = env.reset(seed=args.seed)
    stats = {}
    best_success = -1.0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        if step <= args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(flat_obs(obs))

        next_obs, rew, terminated, truncated, info = env.step(action)
        # Store done=terminated only. A truncation is the time limit cutting the
        # episode short, so next_obs still has value and must be bootstrapped.
        buffer.add(flat_obs(obs), action, rew, flat_obs(next_obs), float(terminated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

        if step > args.update_after:
            stats = agent.update(buffer)

        if step % args.eval_every == 0:
            eval_ret, eval_success = evaluate(eval_env, agent)
            sps = step / (time.time() - t0)
            print(f"step {step:>7}  return {eval_ret:8.2f}  success {eval_success:5.0%}  "
                  f"alpha {stats.get('alpha', float('nan')):.3f}  ({sps:.0f} steps/s)")
            logger.writerow([step, eval_ret, eval_success,
                             stats.get("critic_loss"), stats.get("actor_loss"), stats.get("alpha")])
            log_file.flush()
            if eval_success >= best_success:
                best_success = eval_success
                agent.save(run_dir / "best.pt")

    agent.save(run_dir / "final.pt")
    log_file.close()
    print(f"done in {time.time() - t0:.0f}s, best eval success {best_success:.0%}, "
          f"saved to {run_dir}/")


if __name__ == "__main__":
    main()
