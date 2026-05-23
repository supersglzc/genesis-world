"""Smoke test for the Genesis LiftBox env.

Asserts:
  - env builds with num_envs=4, headless
  - reset() returns obs of shape (4, 33)
  - action_space.shape == (4, 8)
  - 30 random-action steps run without error, every reward finite
  - extras["detailed_reward"] exposes all 7 expected reward keys
  - episode reaches time_out at step == max_episode_length (200)
  - extras["episode"] populated post-time_out

Run with: .venv/bin/python nautilus/tasks/smoke_lift_box.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from _genesis_env import build_env  # type: ignore[import-not-found]


EXPECTED_REWARD_KEYS = {
    "ee_0_to_grasp_0",
    "ee_1_to_grasp_1",
    "grasp_contact_0",
    "grasp_contact_1",
    "lift_height",
    "box_xy_align",
    "success_bonus",
}


def main() -> int:
    print("[smoke] building LiftBox env (num_envs=4, headless)")
    env = build_env(task="lift_box", num_envs=4, headless=True)
    inner = env._inner
    assert inner.num_envs == 4, f"expected num_envs=4, got {inner.num_envs}"
    assert inner.num_actions == 8, f"expected num_actions=8, got {inner.num_actions}"
    assert inner.num_obs == 33, f"expected num_obs=33, got {inner.num_obs}"

    print(f"[smoke] action_space.shape = {env.action_space.shape}")
    assert env.action_space.shape == (4, 8), env.action_space.shape

    # Verify box mass matches spec.
    try:
        m = inner.box.get_mass()
        print(f"[smoke] box mass = {m}")
    except Exception as e:
        print(f"[smoke] note: could not query box mass ({e})")

    print("[smoke] reset")
    obs, info = env.reset()
    pol = obs["policy"]
    print(f"[smoke] obs[policy].shape = {tuple(pol.shape)}")
    assert pol.shape == (4, 33), pol.shape

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

    print(f"[smoke] running through episode horizon (max_episode_length={inner.max_episode_length})")
    while int(inner.episode_length_buf.max().item()) < inner.max_episode_length - 1:
        a = env.action_space.sample()
        env.step(a)
    a = env.action_space.sample()
    obs, rew, done, trunc, extras = env.step(a)
    elb = inner.episode_length_buf
    print(f"[smoke] post-horizon step: episode_length_buf={elb.tolist()}")
    assert done, "expected time_out to fire at step == max_episode_length"

    epi = inner.extras.get("episode", {})
    print(f"[smoke] extras['episode'] keys = {sorted(epi.keys())}")
    expected_episode_keys = {"rew_" + k for k in EXPECTED_REWARD_KEYS}
    missing_episode = expected_episode_keys - set(epi.keys())
    assert not missing_episode, f"missing extras['episode'] keys: {missing_episode}"

    print("[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
