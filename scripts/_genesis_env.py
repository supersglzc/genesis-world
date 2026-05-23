"""Genesis env factory + thin gym-style wrapper used by smoke scripts.

Mirrors the canonical env-build patterns from the Genesis upstream examples
(`examples/manipulation/grasp_train.py::get_task_cfgs`,
 `examples/locomotion/go2_train.py::get_cfgs`) so the smoke layer exercises
exactly what user training scripts exercise.

The thin wrapper adapts the Genesis custom-env API
(`step(action_tensor) -> (obs_dict, reward_tensor, reset_buf, extras)`,
 `reset() -> obs_dict`) to a gymnasium-style 5-tuple so the shared
`scripts/run_random.py` / `scripts/render_random.py` templates work
unmodified.

Public API
----------
build_env(task: str, *, num_envs: int = 1, headless: bool = True,
          render_cam: bool = False, cam_res=(320, 240)) -> GenesisGymAdapter

Tasks
-----
- "grasp" — Franka panda picking a random-pose box (manipulation/grasp_env.py)
- "go2"   — Unitree Go2 quadruped velocity-tracking (locomotion/go2_env.py)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

import genesis as gs

REPO = Path(__file__).resolve().parents[1]
# Add example subtrees so the GraspEnv / Go2Env modules import cleanly without
# requiring the user to set PYTHONPATH manually.
sys.path.insert(0, str(REPO / "examples" / "manipulation"))
sys.path.insert(0, str(REPO / "examples" / "locomotion"))
sys.path.insert(0, str(REPO / "nautilus" / "tasks"))


# ---------------------------------------------------------------------------
# Upstream-mirrored task configs
# ---------------------------------------------------------------------------

def _grasp_cfgs(num_envs: int) -> tuple[dict, dict, dict]:
    """Mirror examples/manipulation/grasp_train.py::get_task_cfgs."""
    env_cfg = {
        "num_envs": num_envs,
        "num_actions": 6,
        "action_scales": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        "episode_length_s": 3.0,
        "ctrl_dt": 0.01,
        "box_size": [0.08, 0.03, 0.06],
        "image_resolution": (64, 64),
        "visualize_camera": False,
    }
    reward_scales = {"keypoints": 1.0}
    robot_cfg = {
        "ee_link_name": "hand",
        "gripper_link_names": ["left_finger", "right_finger"],
        "default_arm_dof": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
        "default_gripper_dof": [0.04, 0.04],
        "ik_method": "dls_ik",
    }
    return env_cfg, reward_scales, robot_cfg


def _go2_cfgs() -> tuple[dict, dict, dict, dict]:
    """Mirror examples/locomotion/go2_train.py::get_cfgs."""
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0, "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0, "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8, "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0, "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5, "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5, "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0, "kd": 0.5,
        "termination_if_roll_greater_than": 10,
        "termination_if_pitch_greater_than": 10,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "feet_height_target": 0.075,
        "reward_scales": {
            "tracking_lin_vel": 1.0, "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0, "base_height": -50.0,
            "action_rate": -0.005, "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


# ---------------------------------------------------------------------------
# Gym-style adapter
# ---------------------------------------------------------------------------

class GenesisGymAdapter:
    """Adapt a Genesis custom env to the gym-style 5-tuple step API used by
    the shared smoke templates.

    Notes
    -----
    - Genesis envs are batched: `num_envs` envs step in lockstep. The smoke
      layer uses `num_envs=1` so we can collapse to scalar reward.
    - Reward returned by `.step()` is the mean over the batch (single env →
      scalar). This is what `run_random.py` checks with `np.isfinite`.
    - `render()` returns a single RGB (H, W, 3) uint8 ndarray from the
      attached debug camera, or None if rendering was not requested.
    """

    def __init__(self, inner: Any, num_actions: int, render_cam=None, device: str = "cpu"):
        self._inner = inner
        self._num_actions = num_actions
        self._cam = render_cam
        self._device = device
        # Smoke scripts inspect this to sample actions.
        self.action_space = _SimpleActionSpace(shape=(inner.num_envs, num_actions))
        self.num_envs = inner.num_envs

    def reset(self, *, seed: int | None = None):
        # Genesis envs don't accept a seed kw on reset(); the seed is fixed
        # at gs.init(seed=...) time. We accept and ignore it for API parity.
        obs = self._inner.reset()
        return obs, {}

    def step(self, action):
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        obs, rew, reset_buf, extras = self._inner.step(action)
        # rew is a (num_envs,) tensor; collapse to a Python float (num_envs=1).
        reward_val = float(rew.detach().mean().cpu().item())
        done = bool(reset_buf.any().item())
        return obs, reward_val, done, False, extras

    def render(self):
        if self._cam is None:
            return None
        rgb_arr, _depth, _seg, _normal = self._cam.render(rgb=True)
        if rgb_arr is None:
            return None
        arr = np.asarray(rgb_arr)
        if arr.ndim == 4:  # (num_envs, H, W, 3) when env_separate_rigid=True
            arr = arr[0]
        return arr[..., :3].astype(np.uint8)


class _SimpleActionSpace:
    """Minimal action_space exposing .sample() returning a torch.Tensor."""

    def __init__(self, shape: tuple[int, int]):
        self.shape = shape

    def sample(self) -> torch.Tensor:
        # Random actions in [-1, 1]; the env applies its own action_scale.
        return torch.empty(self.shape, dtype=torch.float32, device=gs.device).uniform_(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def _init_genesis_once(backend: str = "cpu", seed: int = 0) -> None:
    # gs.init() is idempotent within a process; calling it twice raises.
    if getattr(gs, "_initialized", False):
        return
    try:
        gs.init(backend=getattr(gs, backend), precision="32",
                logging_level="warning", seed=seed)
    except Exception:
        # Fall back to CPU if GPU is unavailable.
        gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=seed)


def build_env(task: str, *, num_envs: int = 1, headless: bool = True,
              render_cam: bool = False, cam_res: tuple[int, int] = (320, 240),
              backend: str = "cpu", seed: int = 0) -> GenesisGymAdapter:
    """Build a Genesis env wrapped in the gym-style adapter.

    Parameters
    ----------
    task : {"grasp", "go2"}
        Which canonical example env to build.
    num_envs : int
        Batched env count. Smoke uses 1.
    headless : bool
        If False, opens the on-screen viewer (not used by smoke).
    render_cam : bool
        If True, attaches a debug camera to the scene and exposes
        `.render()` returning a single RGB ndarray.
    cam_res : (W, H)
        Camera resolution when `render_cam=True`.
    backend : str
        "cpu" or "gpu" attribute name on the genesis module.
    seed : int
        Forwarded to `gs.init(seed=...)`.
    """
    _init_genesis_once(backend=backend, seed=seed)

    if task == "grasp":
        from grasp_env import GraspEnv  # type: ignore[import-not-found]
        env_cfg, reward_scales, robot_cfg = _grasp_cfgs(num_envs)
        # Inject a debug visualization camera BEFORE build() runs (GraspEnv
        # respects env_cfg["visualize_camera"]).
        if render_cam:
            env_cfg["visualize_camera"] = True
        inner = GraspEnv(
            env_cfg=env_cfg,
            reward_cfg=reward_scales,
            robot_cfg=robot_cfg,
            show_viewer=not headless,
        )
        cam = inner.vis_cam if (render_cam and hasattr(inner, "vis_cam")) else None
        num_actions = env_cfg["num_actions"]

    elif task == "stackcube":
        from stack_cube_env import StackCubeEnv  # type: ignore[import-not-found]
        inner = StackCubeEnv(
            num_envs=num_envs,
            show_viewer=not headless,
            attach_debug_camera=render_cam,
            cam_res=cam_res,
        )
        cam = inner.vis_cam if render_cam else None
        num_actions = inner.num_actions

    elif task == "lift_box":
        from lift_box_env import LiftBoxEnv  # type: ignore[import-not-found]
        inner = LiftBoxEnv(
            num_envs=num_envs,
            show_viewer=not headless,
            attach_debug_camera=render_cam,
            cam_res=cam_res,
        )
        cam = inner.vis_cam if render_cam else None
        num_actions = inner.num_actions

    elif task == "insert_drawer":
        from insert_drawer_env import InsertDrawerEnv  # type: ignore[import-not-found]
        inner = InsertDrawerEnv(
            num_envs=num_envs,
            show_viewer=not headless,
            attach_debug_camera=render_cam,
            cam_res=cam_res,
        )
        cam = inner.vis_cam if render_cam else None
        num_actions = inner.num_actions

    elif task == "go2":
        from go2_env import Go2Env  # type: ignore[import-not-found]
        env_cfg, obs_cfg, reward_cfg, command_cfg = _go2_cfgs()
        # Go2Env doesn't have a camera-attach hook, so add one to the scene
        # after construction but before build. Build is invoked inside
        # Go2Env.__init__, so we instead extend Go2Env via a post-init add
        # — Genesis allows scene.add_camera only before build, so we have to
        # patch via a subclass.
        if render_cam:
            inner = _Go2EnvWithCamera(
                num_envs=num_envs,
                env_cfg=env_cfg, obs_cfg=obs_cfg,
                reward_cfg=reward_cfg, command_cfg=command_cfg,
                show_viewer=not headless,
                cam_res=cam_res,
            )
            cam = inner._debug_cam
        else:
            inner = Go2Env(
                num_envs=num_envs,
                env_cfg=env_cfg, obs_cfg=obs_cfg,
                reward_cfg=reward_cfg, command_cfg=command_cfg,
                show_viewer=not headless,
            )
            cam = None
        num_actions = env_cfg["num_actions"]

    else:
        raise ValueError(f"Unknown task {task!r}. Choices: 'grasp', 'go2', 'stackcube', 'lift_box', 'insert_drawer'.")

    return GenesisGymAdapter(inner, num_actions=num_actions, render_cam=cam,
                             device=str(gs.device))


class _Go2EnvWithCamera:
    """Reimplement Go2Env's __init__ but attach a debug camera before build().

    Genesis requires `scene.add_camera(...)` to be called BEFORE
    `scene.build()`. Go2Env builds inside __init__, so wrapping after the
    fact is impossible — we shadow the init sequence and re-invoke build()
    only once, with the camera attached.
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg,
                 show_viewer, cam_res):
        from go2_env import Go2Env  # type: ignore[import-not-found]
        # Cooperative trick: temporarily monkey-patch scene.build on the
        # Go2Env scene attribute is too invasive. Easier: copy the body of
        # Go2Env.__init__ but interject add_camera before build. To keep
        # surgical changes, we instead patch gs.Scene.build to defer.
        import math

        self._num_envs = num_envs
        # ---- replicate Go2Env.__init__ up to before build() ----
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.cfg = env_cfg
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.simulate_action_latency = True
        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False, tolerance=1e-5, max_collision_pairs=20,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40, max_FPS=int(1.0 / self.dt),
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=env_cfg["base_init_pos"], quat=env_cfg["base_init_quat"],
            ),
        )
        # ---- INTERJECT: attach debug camera BEFORE build ----
        self._debug_cam = self.scene.add_camera(
            res=cam_res, pos=(2.5, 1.5, 1.5), lookat=(0.0, 0.0, 0.3),
            fov=45, GUI=False,
        )
        # ---- Continue Go2Env.__init__ body ----
        self.scene.build(n_envs=num_envs)

        # The remainder is buffer setup; rather than copy 60 LOC, delegate to
        # Go2Env.__init__ by calling its bound-method continuation. Since the
        # original __init__ requires building inside itself, we instead just
        # re-execute the post-build half via direct attribute mirroring.
        # The clean path: subclass Go2Env, override scene/build at class
        # level. But the simplest robust path is to invoke Go2Env's bound
        # init helpers manually:
        self._finish_init_like_go2env(Go2Env)

    def _finish_init_like_go2env(self, Go2EnvCls):
        """Run Go2Env's post-scene-build init sequence on `self`."""
        # Mirror lines 79+ of go2_env.py.
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_int, device=gs.device,
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
        from genesis.utils.geom import (
            inv_quat, transform_by_quat,
        )
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_pos = torch.tensor(self.env_cfg["base_init_pos"], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(self.env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.init_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.init_qpos = torch.concatenate((self.init_base_pos, self.init_base_quat, self.init_dof_pos))
        self.init_projected_gravity = transform_by_quat(self.global_gravity, self.inv_base_init_quat)

        # Buffers
        self.base_lin_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.empty((self.num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((self.num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.empty((self.num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.empty((self.num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device, dtype=gs.tc_float,
        )
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                self.command_cfg["lin_vel_x_range"],
                self.command_cfg["lin_vel_y_range"],
                self.command_cfg["ang_vel_range"],
            )
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.empty_like(self.actions)
        self.dof_vel = torch.empty_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.empty((self.num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.empty((self.num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            dtype=gs.tc_float, device=gs.device,
        )
        self.extras = dict()

        self.reward_functions, self.episode_sums = dict(), dict()
        for name in list(self.reward_scales.keys()):
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(Go2EnvCls, "_reward_" + name).__get__(self, type(self))
            self.episode_sums[name] = torch.zeros((self.num_envs,), dtype=gs.tc_float, device=gs.device)

        # Bind the rest of Go2Env's methods as instance methods so step/reset
        # behave identically to the upstream class.
        for method_name in ("step", "reset", "_reset_idx", "_resample_commands",
                            "_update_observation", "get_observations"):
            method = getattr(Go2EnvCls, method_name)
            setattr(self, method_name, method.__get__(self, type(self)))

        # Final reset to populate buffers (mirrors Go2Env.__init__ tail).
        self.reset()
