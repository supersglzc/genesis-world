"""Smoke test for the Genesis InsertDrawer env.

Asserts:
  - env builds with num_envs=4, headless
  - reset() returns obs of shape (4, 19)
  - action_space.shape == (4, 4)
  - 30 random-action steps run without error, every reward finite
  - extras["detailed_reward"] exposes all 8 expected reward keys
  - episode reaches time_out at step == max_episode_length (180) if no
    success fires within that horizon under random actions
  - extras["episode"] populated post-time_out
  - drawer joint pos starts at 0.30 m after reset

Run with: .venv/bin/python nautilus/tasks/smoke_insert_drawer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from _genesis_env import build_env  # type: ignore[import-not-found]


EXPECTED_REWARD_KEYS = {
    "reach_cube",
    "is_lifted",
    "lift_distance",
    "align",
    "ee_retract_to_front_face",
    "cube_inside_bonus_latch",
    "close_drawer",
    "success_bonus",
}


def main() -> int:
    print("[smoke] building InsertDrawer env (num_envs=4, headless)")
    env = build_env(task="insert_drawer", num_envs=4, headless=True)
    inner = env._inner
    assert inner.num_envs == 4, f"expected num_envs=4, got {inner.num_envs}"
    assert inner.num_actions == 4, f"expected num_actions=4, got {inner.num_actions}"
    assert inner.num_obs == 19, f"expected num_obs=19, got {inner.num_obs}"

    print(f"[smoke] action_space.shape = {env.action_space.shape}")
    assert env.action_space.shape == (4, 4), env.action_space.shape

    # Drawer joint at reset should be OPEN (0.30 m).
    drawer_jp = inner._drawer_joint_pos()
    print(f"[smoke] drawer joint pos after reset = {drawer_jp.tolist()}")
    assert torch.allclose(drawer_jp, torch.full_like(drawer_jp, 0.30), atol=2e-2), \
        f"drawer joint should be ~0.30 m at reset, got {drawer_jp.tolist()}"

    print("[smoke] reset returned obs")
    obs, info = env.reset()
    pol = obs["policy"]
    print(f"[smoke] obs[policy].shape = {tuple(pol.shape)}")
    assert pol.shape == (4, 19), pol.shape
    assert torch.isfinite(pol).all(), "non-finite obs after reset"

    print("[smoke] running 30 random-action steps")
    for i in range(30):
        a = env.action_space.sample()
        obs, rew, done, trunc, extras = env.step(a)
        assert torch.isfinite(torch.tensor(rew)).all(), f"non-finite reward at step {i}: {rew}"
        det = inner.extras.get("detailed_reward", {})
        missing = EXPECTED_REWARD_KEYS - set(det.keys())
        assert not missing, f"missing reward keys: {missing}"
        for k, v in det.items():
            assert torch.isfinite(v).all(), f"non-finite per-term reward {k}={v}"
        if i % 10 == 0:
            print(f"  step {i:3d} reward={rew:+.4f}  keys={sorted(det.keys())}")

    print(f"[smoke] running through episode horizon "
          f"(max_episode_length={inner.max_episode_length})")
    while int(inner.episode_length_buf.max().item()) < inner.max_episode_length - 1:
        a = env.action_space.sample()
        env.step(a)
    a = env.action_space.sample()
    obs, rew, done, trunc, extras = env.step(a)
    elb = inner.episode_length_buf
    print(f"[smoke] post-horizon step: episode_length_buf={elb.tolist()}")
    assert done, "expected time_out (or success) to fire at step == max_episode_length"

    epi = inner.extras.get("episode", {})
    print(f"[smoke] extras['episode'] keys = {sorted(epi.keys())}")
    expected_episode_keys = {"rew_" + k for k in EXPECTED_REWARD_KEYS}
    missing_episode = expected_episode_keys - set(epi.keys())
    assert not missing_episode, f"missing extras['episode'] keys: {missing_episode}"

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
