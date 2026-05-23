"""Random-action rollout of the StackCube task with the Genesis viewer open.

Usage:
    cd /home/steven/code/agentic/genesis-world
    .venv/bin/python nautilus/tasks/run_random_stackcube.py

Flags:
    --n-steps N        rollout length (default 360 = 2 episodes at horizon 180)
    --num-envs N       parallel envs to spawn; viewer renders min(4, num_envs) (default 1)
    --backend gpu|cpu  Genesis backend (default gpu — switch to cpu to avoid contention with a running GPU train)
    --seed N           seed for gs.init (default 0)
    --headless         run without viewer (smoke mode)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np

from _genesis_env import build_env  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n-steps", type=int, default=360,
                   help="Number of env.step() calls (default: %(default)s)")
    p.add_argument("--num-envs", type=int, default=1,
                   help="Parallel envs (default: %(default)s)")
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"],
                   help="Genesis backend (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0,
                   help="gs.init seed (default: %(default)s)")
    p.add_argument("--headless", action="store_true",
                   help="Disable the viewer (smoke mode).")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[run_random_stackcube] backend={args.backend} num_envs={args.num_envs} "
          f"n_steps={args.n_steps} headless={args.headless}")
    adapter = build_env(
        "stackcube",
        num_envs=args.num_envs,
        headless=args.headless,
        render_cam=False,
        backend=args.backend,
        seed=args.seed,
    )
    adapter.reset()

    bad = 0
    for i in range(args.n_steps):
        action = adapter.action_space.sample()
        obs, reward, done, trunc, info = adapter.step(action)
        if not np.isfinite(reward):
            bad += 1
            print(f"step {i}: reward={reward!r} NOT finite", file=sys.stderr)
        if i % 60 == 0:
            print(f"step {i:4d}  reward={reward:+.4f}  done={done}")

    print(f"[run_random_stackcube] done — {args.n_steps} steps, {bad} non-finite rewards")


if __name__ == "__main__":
    main()
