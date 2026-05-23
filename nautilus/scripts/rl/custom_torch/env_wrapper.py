"""
genesis-world — env factory for custom_torch train/eval/render.

Branches at runtime on cfg.gpu_sim:
  - True  → Genesis-native GPU-batched env (scripts/_genesis_env.py::build_env).
            Genesis returns TensorDict({"policy": obs (N, D)}), reward (N,),
            reset_buf (N,), extras — we extract the "policy" key to a flat
            (N, D) tensor and pass everything else through.
  - False → gymnasium.vector.AsyncVectorEnv (legacy fallback; Genesis does NOT
            offer a CPU vec-env API, so this branch is effectively unused for
            the canonical tasks `grasp` / `go2` but kept for forward compat).

Exposes:
    create_env(cfg)         — vec/parallel env for train + eval
    create_render_env(cfg)  — single env with attached debug cam for render.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO     = Path(__file__).resolve().parents[4]   # actual repo root
NAUTILUS = Path(__file__).resolve().parents[3]   # <repo>/nautilus
sys.path.insert(0, str(NAUTILUS))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


# -----------------------------------------------------------------------------
# Adapter: Genesis-native env → the (obs_tensor, reward_tensor, done_tensor, info)
# contract that algo/ac_base.py + algo/{ppo,sac,td3}.py expect.
#
# Genesis envs (Go2Env / GraspEnv) expose:
#     reset()            -> TensorDict({"policy": (N, D)})
#     step(action_N_x_A) -> (TensorDict({"policy": (N, D)}),
#                            reward (N,), reset_buf bool (N,), extras: dict)
#     .num_envs                — batch size
#     .num_actions             — action dim
#     .max_episode_length      — step horizon (int)
#     .obs_buf                 — last computed observation buffer (N, D)
#
# We don't go through GenesisGymAdapter (which collapses reward to a scalar) —
# the algo code needs per-env reward.
# -----------------------------------------------------------------------------
class GenesisVecEnvWrapper:
    """Wrap a Genesis batched env so step/reset return torch tensors shaped for the algo code."""

    def __init__(self, inner, device):
        self.inner = inner
        self.device = torch.device(device)
        self.num_envs = int(inner.num_envs)
        self.num_actions = int(inner.num_actions)
        # Observation dim — probe obs_buf if available, else run one reset.
        obs_buf = getattr(inner, "obs_buf", None)
        if obs_buf is None:
            obs_td = inner.reset()
            obs_buf = obs_td["policy"]
        self._obs_shape = (int(obs_buf.shape[-1]),)
        self._act_shape = (self.num_actions,)
        self.observation_space = type("_Spc", (), {"shape": self._obs_shape})()
        self.action_space = type("_Spc", (), {"shape": self._act_shape})()
        self.max_episode_length = int(getattr(inner, "max_episode_length", 0)) or None

    @staticmethod
    def _unpack_obs(out):
        """Extract the (N, D) tensor from a Genesis TensorDict obs or a plain tensor."""
        if hasattr(out, "get") and not isinstance(out, dict):
            # tensordict.TensorDict — use ["policy"] key.
            return out["policy"]
        if isinstance(out, dict):
            return out["policy"]
        return out

    def reset(self, **kwargs):
        out = self.inner.reset()
        obs = self._unpack_obs(out).to(self.device).float()
        return obs, {}

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = action.to(self.device).float()
        obs_td, reward, reset_buf, extras = self.inner.step(action)
        obs = self._unpack_obs(obs_td).to(self.device).float()
        reward = reward.to(self.device).float()
        done = reset_buf.to(self.device).long()
        info = extras if isinstance(extras, dict) else {}
        return obs, reward, done, info


# -----------------------------------------------------------------------------
# Public factories
# -----------------------------------------------------------------------------
def create_env(cfg):
    """Build the training/eval env. Uses Genesis-native GPU batching for gpu_sim=True."""
    task = str(cfg.get("task"))
    n = int(cfg.get("num_envs", cfg.get("n_envs", 1)))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))

    if gpu_sim:
        # Genesis-native GPU-batched env. build_env() lives in scripts/_genesis_env.py
        # and mirrors the upstream Go2Env / GraspEnv construction.
        from _genesis_env import build_env  # type: ignore[import-not-found]
        backend = "gpu" if device.startswith("cuda") else "cpu"
        adapter = build_env(task, num_envs=n, headless=True,
                            render_cam=False, backend=backend, seed=seed)
        return GenesisVecEnvWrapper(adapter._inner, device=device)

    # CPU fallback — Genesis itself can run on CPU. Same code path with backend='cpu'.
    from _genesis_env import build_env  # type: ignore[import-not-found]
    adapter = build_env(task, num_envs=n, headless=True,
                        render_cam=False, backend="cpu", seed=seed)
    return GenesisVecEnvWrapper(adapter._inner, device=device)


def create_render_env(cfg):
    """Single env (n=1) with attached debug camera — used by render.py."""
    task = str(cfg.get("task"))
    seed = int(cfg.get("seed", 0))
    device = str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    gpu_sim = bool(cfg.get("gpu_sim", False))
    backend = "gpu" if (gpu_sim and device.startswith("cuda")) else "cpu"

    from _genesis_env import build_env  # type: ignore[import-not-found]
    adapter = build_env(task, num_envs=1, headless=True,
                        render_cam=True, backend=backend, seed=seed)
    wrapped = GenesisVecEnvWrapper(adapter._inner, device=device)
    # render.py uses _raw_env.render() to grab frames — point it at the
    # GenesisGymAdapter which knows how to read from the attached debug cam.
    wrapped._raw_env = adapter
    return wrapped
