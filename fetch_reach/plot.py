"""Plot training curves from one or more run logs.

Usage:
    python plot.py runs/FetchReachDense_seed0/log.csv [more log.csv ...]
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+")
    parser.add_argument("--out", default="plots/training_curves.png")
    args = parser.parse_args()

    fig, (ax_ret, ax_succ) = plt.subplots(1, 2, figsize=(11, 4))
    for log in args.logs:
        with open(log) as f:
            rows = list(csv.DictReader(f))
        steps = [int(r["step"]) for r in rows]
        label = Path(log).parent.name
        ax_ret.plot(steps, [float(r["eval_return"]) for r in rows], label=label)
        ax_succ.plot(steps, [float(r["eval_success"]) for r in rows], label=label)

    ax_ret.set(xlabel="environment steps", ylabel="eval return", title="Evaluation return")
    ax_succ.set(xlabel="environment steps", ylabel="success rate", ylim=(-0.05, 1.05),
                title="Evaluation success rate")
    for ax in (ax_ret, ax_succ):
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
