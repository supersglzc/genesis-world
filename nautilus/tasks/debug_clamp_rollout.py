"""Rollout a trained PPO checkpoint on StackCube and verify the EE workspace clamp.

For each step the script captures:
  - the IK *target* position (post-clamp, fed into Genesis IK) in root frame
  - the *actual* EE TCP position (read after scene.step + IK + PD) in root frame
  - whether the actual position is outside the clamp bbox
  - the cube_0 z (to flag table-collision events where EE drives into the floor)

Usage:
    cd /home/steven/code/agentic/genesis-world
    .venv/bin/python nautilus/tasks/debug_clamp_rollout.py \\
        --checkpoint nautilus/outputs/ppo_stackcube_20260523-153942/checkpoint.pth \\
        [--steps 180] [--seed 0] [--backend cpu]

Outputs:
    Per-step CSV at <ckpt-dir>/clamp_audit.csv
    A summary printed to stdout: max excursions, fraction of steps outside clamp.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAUTILUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(NAUTILUS / "scripts" / "rl" / "custom_torch"))

import torch  # noqa: E402

from _genesis_env import build_env  # type: ignore[import-not-found]  # noqa: E402
from algo import alg_name_to_path  # type: ignore[import-not-found]  # noqa: E402
from utils.common import load_class_from_path  # type: ignore[import-not-found]  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper() if not name.startswith("Agent") else name
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(f"Algorithm '{name}' not found.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    args = p.parse_args()

    ckpt_path = args.checkpoint.resolve()
    print(f"[debug] loading checkpoint {ckpt_path}")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    algo = state.get("algo", "ppo")
    print(f"[debug] algo={algo}  obs_norm={state.get('obs_norm')}  value_norm={state.get('value_norm')}")

    # Build env (num_envs=1, headless).
    device = "cuda" if (args.backend == "gpu" and torch.cuda.is_available()) else "cpu"
    adapter = build_env("stackcube", num_envs=1, headless=True, render_cam=False,
                        backend=args.backend, seed=args.seed)
    env = adapter._inner

    # Capture clamp constants from the env.
    pos_lo_root = env._pos_lo_root.cpu().numpy()  # (3,)
    pos_hi_root = env._pos_hi_root.cpu().numpy()  # (3,)
    print(f"[debug] clamp ROOT  lo={pos_lo_root.tolist()}  hi={pos_hi_root.tolist()}")

    # Build a minimal PPO agent + load weights.
    # We bypass hydra and feed cfg as a dict via OmegaConf for the agent class.
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "device": device, "sim_device": device, "rl_device": device,
        "seed": args.seed, "gpu_sim": (args.backend == "gpu"),
        "num_envs": 1, "n_envs": 1,
        "hidden_dims": [512, 256, 128], "activation": "tanh",
        "init_log_std": -0.223, "learnable_std": True,
        "obs_norm": bool(state.get("obs_norm", False)),
        "value_norm": bool(state.get("value_norm", False)),
        "handle_timeout": True, "tracker_len": 1000, "reward_scale": 1.0,
        "no_tgt_actor": True, "use_gae": True,
        "act_class": "DiagGaussianMLPPolicy", "cri_class": "MLPCritic",
        "learning_rate": 5.0e-4, "actor_lr": 5.0e-4, "critic_lr": 5.0e-4,
        "gamma": 0.99, "gae_lambda": 0.95, "clip_range": 0.15, "n_steps": 16,
        "horizon_len": 16, "batch_size": 8192, "n_epochs": 8, "n_minibatches": 64,
        "update_times": 8, "ent_coef": 0.005, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "anneal_lr": False, "weight_decay": 0.0,
        "algo": {"name": algo}, "value_clip": True,
    })
    # The agent class expects cfg.algo to be a flat namespace; promote.
    cfg.algo = OmegaConf.merge(cfg, {"name": algo})

    # Wrap env to match what train.py uses.
    from env_wrapper import GenesisVecEnvWrapper  # type: ignore
    wrapped = GenesisVecEnvWrapper(env, device=device)

    agent_cls = _resolve_algo_class(algo)
    agent = agent_cls(env=wrapped, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
    agent.actor.eval()

    # Roll out one episode, capturing target vs actual EE TCP per step.
    obs, _ = wrapped.reset()

    csv_path = ckpt_path.parent / "clamp_audit.csv"
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow([
        "step", "act_dx", "act_dy", "act_dz", "act_gripper",
        "target_root_x", "target_root_y", "target_root_z",
        "actual_root_x", "actual_root_y", "actual_root_z",
        "out_lo_x", "out_lo_y", "out_lo_z",
        "out_hi_x", "out_hi_y", "out_hi_z",
        "cube0_z",
    ])

    counts_outside = [0] * 3  # per-axis counts of "actual outside clamp"
    max_excursion = [0.0] * 3
    target_outside_counts = [0] * 3
    table_crashes = 0  # steps where EE z < 0 (below table)

    with torch.no_grad():
        for step in range(args.steps):
            obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
            action = agent.actor.get_actions(obs_in, sample=False)
            if isinstance(action, tuple):
                action = action[0]
            a = action[0].cpu().numpy()

            # ACTUAL EE TCP position BEFORE step (after previous physics).
            ee_w_before, _ = env._ee_pose_w()
            actual_root_before = (ee_w_before[0] - env._robot_base_w_per_env[0]).cpu().numpy()

            # Step the env — internally clamps the IK target, applies IK, PD.
            obs, reward, done, _ = wrapped.step(action)

            # IK target (clamped) for THIS step is `_prev_applied_pos_w` after step.
            target_w = env._prev_applied_pos_w[0]
            target_root = (target_w - env._robot_base_w_per_env[0]).cpu().numpy()

            # ACTUAL EE TCP position AFTER step (physics resolved).
            ee_w_after, _ = env._ee_pose_w()
            actual_root_after = (ee_w_after[0] - env._robot_base_w_per_env[0]).cpu().numpy()

            # Excursion stats.
            out_lo = [actual_root_after[i] < pos_lo_root[i] for i in range(3)]
            out_hi = [actual_root_after[i] > pos_hi_root[i] for i in range(3)]
            tgt_lo = [target_root[i] < pos_lo_root[i] - 1e-4 for i in range(3)]
            tgt_hi = [target_root[i] > pos_hi_root[i] + 1e-4 for i in range(3)]
            for i in range(3):
                if out_lo[i] or out_hi[i]:
                    counts_outside[i] += 1
                    if out_lo[i]:
                        max_excursion[i] = max(max_excursion[i], pos_lo_root[i] - actual_root_after[i])
                    else:
                        max_excursion[i] = max(max_excursion[i], actual_root_after[i] - pos_hi_root[i])
                if tgt_lo[i] or tgt_hi[i]:
                    target_outside_counts[i] += 1

            ee_w_z = float(ee_w_after[0, 2].item())
            if ee_w_z < 0.0:
                table_crashes += 1

            cube0_z = float(env._cube_xyz_w(env.cube_0)[0, 2].item())

            writer.writerow([
                step, *a.tolist(),
                *target_root.tolist(),
                *actual_root_after.tolist(),
                *[int(x) for x in out_lo],
                *[int(x) for x in out_hi],
                cube0_z,
            ])
            if step % 20 == 0:
                print(f"step {step:3d}  act=[{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}, g={a[3]:+.2f}]  "
                      f"target_root=[{target_root[0]:+.4f}, {target_root[1]:+.4f}, {target_root[2]:+.4f}]  "
                      f"actual_root=[{actual_root_after[0]:+.4f}, {actual_root_after[1]:+.4f}, {actual_root_after[2]:+.4f}]  "
                      f"cube0_z={cube0_z:.4f}  ee_w_z={ee_w_z:.4f}")

    f.close()
    print()
    print(f"=== summary over {args.steps} steps ===")
    axes = ["x", "y", "z"]
    print(f"Clamp ROOT: lo={pos_lo_root.tolist()}  hi={pos_hi_root.tolist()}")
    for i, ax in enumerate(axes):
        print(f"  actual_{ax}: outside-clamp steps = {counts_outside[i]:3d}/{args.steps}  "
              f"max_excursion = {max_excursion[i]:+.4f} m")
    for i, ax in enumerate(axes):
        print(f"  target_{ax}: post-clamp values outside the clamp = {target_outside_counts[i]:3d}/{args.steps}  "
              f"(expected ≈ 0 since clamp is applied to target)")
    print(f"Table crashes (actual EE TCP world-z < 0): {table_crashes}/{args.steps}")
    print(f"Per-step CSV: {csv_path}")


if __name__ == "__main__":
    main()
