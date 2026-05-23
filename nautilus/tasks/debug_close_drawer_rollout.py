"""Rollout an insert_drawer checkpoint and audit close_drawer + success gates.

For each step logs:
  - latch_cube_inside (bool, sticky once True)
  - drawer_jp (joint position)
  - cube_z, cube_xy
  - ee.y vs front_face.y → gate_ee_outside
  - closeness in [0, 1]
  - close_drawer_val (the term value before weight)
  - success_fire

Usage:
    cd /home/steven/code/agentic/genesis-world
    .venv/bin/python nautilus/tasks/debug_close_drawer_rollout.py \\
        --checkpoint nautilus/outputs/ppo_insert_drawer_20260523-174853/checkpoint.pth
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NAUTILUS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(NAUTILUS / "tasks"))
sys.path.insert(0, str(NAUTILUS / "scripts" / "rl" / "custom_torch"))

import torch  # noqa: E402
from _genesis_env import build_env  # noqa: E402
from algo import alg_name_to_path  # noqa: E402
from utils.common import load_class_from_path  # noqa: E402


def _resolve_algo_class(name: str):
    target = "Agent" + name.upper()
    for cls_name, path in alg_name_to_path.items():
        if cls_name.lower() == target.lower():
            return load_class_from_path(cls_name, path)
    raise KeyError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--backend", default="gpu", choices=["gpu", "cpu"])
    args = p.parse_args()
    ckpt_path = args.checkpoint.resolve()

    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    algo = state.get("algo", "ppo")
    device = "cuda" if (args.backend == "gpu" and torch.cuda.is_available()) else "cpu"

    adapter = build_env("insert_drawer", num_envs=1, headless=True, render_cam=False,
                        backend=args.backend, seed=0)
    env = adapter._inner

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({
        "device": device, "sim_device": device, "rl_device": device, "seed": 0,
        "gpu_sim": (args.backend == "gpu"), "num_envs": 1, "n_envs": 1,
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
        "update_times": 8, "ent_coef": 0.01, "vf_coef": 0.5, "max_grad_norm": 0.5,
        "anneal_lr": False, "weight_decay": 0.0,
        "algo": {"name": algo}, "value_clip": True,
    })
    cfg.algo = OmegaConf.merge(cfg, {"name": algo})

    from env_wrapper import GenesisVecEnvWrapper
    wrapped = GenesisVecEnvWrapper(env, device=device)
    agent_cls = _resolve_algo_class(algo)
    agent = agent_cls(env=wrapped, cfg=cfg)
    agent.actor.load_state_dict(state["actor"])
    if "obs_rms" in state and getattr(agent, "obs_rms", None) is not None:
        agent.obs_rms.load_state_dict(state["obs_rms"])
    agent.actor.eval()

    obs, _ = wrapped.reset()
    csv_path = ckpt_path.parent / "close_drawer_audit.csv"
    f = open(csv_path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["step", "latch_inside", "drawer_jp", "closeness",
                "cube_z", "cube_y", "ee_y", "front_face_y",
                "gate_ee_outside", "close_drawer_val", "success_fire",
                "ee_cube_dist", "ee_far", "inside_geom"])

    from insert_drawer_env import (
        DRAWER_MAX_OPEN, DRAWER_CLOSED_THR, CUBE_INSIDE_EE_FAR_THR,
    )

    latch_was = False
    success_fired = False
    closeness_summary = []
    gate_summary = []
    drawer_jp_summary = []

    with torch.no_grad():
        for step in range(args.steps):
            obs_in = agent.obs_rms.normalize(obs) if getattr(agent, "obs_rms", None) is not None else obs
            action = agent.actor.get_actions(obs_in, sample=False)
            if isinstance(action, tuple):
                action = action[0]
            obs, reward, done, info = wrapped.step(action)

            cube_p = env.cube.get_pos()[0]
            ee_w, _ = env._ee_pose_w()
            ee_p = ee_w[0]
            front_face_w = env._drawer_front_face_w()[0]
            drawer_jp = env.drawer.get_dofs_position(env._drawer_joint_dof_idx)[0, 0].item()

            inside_geom = bool(env._cube_inside_drawer_geometric()[0].item())
            ee_cube_d = torch.norm(ee_p - cube_p).item()
            ee_far = ee_cube_d > CUBE_INSIDE_EE_FAR_THR
            latch_now = bool(env._latch_cube_inside[0].item())
            gate_ee = (ee_p[1].item() > front_face_w[1].item())
            closeness = max(0.0, min(1.0, (DRAWER_MAX_OPEN - drawer_jp) / DRAWER_MAX_OPEN))
            close_val = float(latch_now) * float(gate_ee) * closeness
            success_now = latch_now and (drawer_jp < DRAWER_CLOSED_THR)
            success_fire = success_now and not success_fired
            if success_fire:
                success_fired = True

            w.writerow([step, int(latch_now), drawer_jp, closeness,
                        cube_p[2].item(), cube_p[1].item(),
                        ee_p[1].item(), front_face_w[1].item(),
                        int(gate_ee), close_val, int(success_fire),
                        ee_cube_d, int(ee_far), int(inside_geom)])

            if step % 20 == 0:
                print(f"step {step:3d}  latch={int(latch_now)} drawer_jp={drawer_jp:+.3f} "
                      f"closeness={closeness:.2f} ee_y={ee_p[1].item():+.3f} "
                      f"front_face_y={front_face_w[1].item():+.3f} gate_ee={int(gate_ee)} "
                      f"close_val={close_val:.3f} cube_z={cube_p[2].item():.3f} "
                      f"inside={int(inside_geom)}")

            closeness_summary.append(closeness)
            gate_summary.append(int(gate_ee))
            drawer_jp_summary.append(drawer_jp)
            latch_was = latch_was or latch_now

    f.close()
    print()
    print(f"=== summary over {args.steps} steps ===")
    print(f"  latch_cube_inside fired at any point: {latch_was}")
    print(f"  drawer joint final  : {drawer_jp_summary[-1]:+.3f}")
    print(f"  drawer joint MIN    : {min(drawer_jp_summary):+.3f}  (target < {DRAWER_CLOSED_THR})")
    print(f"  closeness MAX       : {max(closeness_summary):.3f}")
    print(f"  gate_ee_outside ON  : {sum(gate_summary)}/{args.steps} steps")
    print(f"  success fired       : {success_fired}")
    print(f"  CSV                 : {csv_path}")


if __name__ == "__main__":
    main()
