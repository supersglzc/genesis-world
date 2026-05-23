"""Genesis port of IsaacLab Triton-Lift-Box (dual-arm cooperative box-lift).

Two FR3 + Franka-hand robots cooperate to lift a 0.5 kg eurobox (40 x 30 x
22 cm) off a flat surface. Mirrors `stack_cube_env.py` structurally and uses
the same PD-gain / IK / EMA-delta / contact-proxy patterns; only the scene
content and reward function differ.

IsaacLab spec source-of-truth:
  IsaacLab/nautilus/create-task/triton-lift-box-implementation.md
  (§1..§7 — scene, actions, reset, termination, observation, reward, DR)

Per the spec:
- Two FR3 robots at env-local (-0.274, +0.49, 0.01) and (-0.274, -0.49, 0.01)
- Eurobox at env-local (0, 0, 0.11025) rotated 90° about +Z (long axis along
  world Y). Asset is the pre-recentered eurobox STL under
  `nautilus/assets/eurobox/eurobox_recentered.stl` (extents x=±0.20, y=±0.15,
  z=±0.11025).
- Per-robot 3-D EMA-delta xyz EE action + binary gripper (8-D total).
- Sim: ctrl_dt=1/20, substeps=6 (per-substep = 1/120 s), episode = 200 steps
  (10 s @ 20 Hz).
- Observation (33-D): per-robot ee_pose_in_root (7+7) + box_xyz (3) + box_quat
  (4) + per-robot gripper joint pos (2+2) + last_action (8).
- Reward (7 terms, composer = sum, dt-multiplier dropped per fork conv):
  ee_to_grasp_0/1, grasp_contact_0/1, lift_height, box_xy_align, success_bonus.

Genesis-only deviations from the spec (Genesis lacks the relevant APIs):
- No USD: box loaded via `gs.morphs.Mesh(file=recentered.stl)`. Density set
  via material.rho so post-build mass ≈ 0.5 kg given the box volume.
- No FrameTransformer: ee TCP computed from `fr3_hand` link pose + body-offset
  (0, 0, 0.2). Box grasp-points computed from box root_pos + body-local offsets
  (±0.20, 0, +0.11025) rotated by box quat.
- No filtered contact sensors: per-robot contact proxy mirrors stack_cube_env's
  `_contact_proxy` — EE-TCP within (BOX_GRASP_TOL) of the corresponding
  grasp-frame AND gripper command closed.
"""
from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs


# --- Constants from the IsaacLab spec ---------------------------------------

# Box (eurobox) — half-extents on the recentered asset.
BOX_HALF_X = 0.20      # m — local +x half-extent
BOX_HALF_Y = 0.15      # m — local +y half-extent
BOX_HALF_Z = 0.11025   # m — local +z half-extent (= BOX_INIT_Z)
BOX_INIT_Z = BOX_HALF_Z
BOX_MASS = 0.5         # kg

# Two FR3 robots placed symmetrically on the world y axis.
ROBOT_0_BASE_POS = (-0.274,  0.49, 0.01)
ROBOT_1_BASE_POS = (-0.274, -0.49, 0.01)

# FR3 home joint poses — robot_0 unchanged, robot_1 flips joint1 sign and zeros
# joint3/joint5 + sets joint7=-1.57 (spec §1 "Joint-pose flip rule").
ARM_HOME_0 = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3":  0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5":  0.0,
    "fr3_joint6":  1.87,
    "fr3_joint7":  0.0,
}
ARM_HOME_1 = {
    "fr3_joint1":  0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3":  0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5":  0.0,
    "fr3_joint6":  1.87,
    "fr3_joint7": -1.57,
}
FINGER_OPEN  = 0.04
FINGER_CLOSE = 0.0

# Action: per-axis scale and EMA alpha (IsaacLab spec §2).
ACTION_SCALE = (0.01, 0.01, 0.01)
EMA_ALPHA = 0.5

# Per-robot workspace clamps in EACH robot's root frame (spec §2 table).
# robot_0 (base at world +y=0.49) reaches into negative-y in its root frame.
# robot_1 (base at world -y=0.49) reaches into positive-y in its root frame.
POS_LO_ROOT_0 = (0.20, -0.65, 0.005)
POS_HI_ROOT_0 = (0.55, -0.20, 0.40)
POS_LO_ROOT_1 = (0.20,  0.20, 0.005)
POS_HI_ROOT_1 = (0.55,  0.65, 0.40)

# Box reset XY jitter (spec §3).
BOX_RESET_X = (-0.03, 0.03)
BOX_RESET_Y = (-0.03, 0.03)

# Box init quaternion (wxyz) — 90° about +Z so long axis along world Y.
BOX_INIT_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)

# IK TCP offset in the fr3_hand frame (IsaacLab `OffsetCfg(pos=[0,0,0.2])`).
TCP_LOCAL_OFFSET = (0.0, 0.0, 0.2)

# Per-robot grasp-point body-local offset on the box (spec §1).
# After the 90° Z-rotation the local +x maps to world +y (robot_0 side).
GRASP_LOCAL_0 = ( 0.20, 0.0,  0.11025)
GRASP_LOCAL_1 = (-0.20, 0.0,  0.11025)

# Sim timing (spec §1).
CTRL_DT       = 1.0 / 20.0
SUBSTEPS      = 6                # each substep = 1/120 s -> 20 Hz control
EPISODE_STEPS = 200              # 10 s @ 20 Hz

# Target (spec §4) — env-local box COM lifted by 0.25 m.
LIFT_HEIGHT       = 0.25
TARGET_XY         = (0.0, 0.0)
XY_POS_TOL        = 0.05
Z_POS_TOL         = 0.05
VEL_TOL           = 0.10

# Reward §6 — RAW per-step magnitudes (fork dropped the *dt multiplier).
REWARD_WEIGHTS = {
    "ee_0_to_grasp_0": 0.0125,
    "ee_1_to_grasp_1": 0.0125,
    "grasp_contact_0": 0.025,
    "grasp_contact_1": 0.025,
    "lift_height":     0.1875,
    "box_xy_align":    0.125,
    "success_bonus":   100.0,
}
REACH_STD     = 0.15
ALIGN_STD     = 0.15
ALIGN_LIFT_THRESHOLD = 0.05

# Contact proxy — EE TCP within (BOX_GRASP_TOL) of the per-robot grasp-frame
# AND gripper closed. Tightened to 5 cm so "contact" requires the gripper to
# actually pinch the box top edge (not just hover near it).
BOX_GRASP_TOL = 0.05


# --- URDF resolution --------------------------------------------------------

def _resolve_fr3_urdf() -> str:
    """Resolve the FR3+Franka-hand URDF (same patched copy stack_cube_env uses)."""
    here = Path(__file__).resolve().parents[2] / "nautilus" / "assets" / "fr3" / "fr3_franka_hand.urdf"
    if not here.is_file():
        raise FileNotFoundError(f"FR3 URDF not found at {here}")
    return str(here)


def _resolve_eurobox_mesh() -> str:
    """Resolve the recentered eurobox STL (origin at geometric center)."""
    here = Path(__file__).resolve().parents[2] / "nautilus" / "assets" / "eurobox" / "eurobox_recentered.stl"
    if not here.is_file():
        raise FileNotFoundError(f"Recentered eurobox mesh not found at {here}")
    return str(here)


class LiftBoxEnv:
    """Genesis-native dual-arm FR3 lift-box env, IsaacLab-spec faithful."""

    def __init__(
        self,
        num_envs: int = 1,
        show_viewer: bool = False,
        env_spacing: float = 2.5,
        attach_debug_camera: bool = False,
        cam_res: tuple[int, int] = (320, 240),
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_actions = 8            # 4 per robot (3 xyz delta + 1 gripper) * 2
        self.num_obs = 33
        self.device = gs.device

        self.ctrl_dt = CTRL_DT
        self.max_episode_length = EPISODE_STEPS

        # ---- Scene -----------------------------------------------------
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.ctrl_dt, substeps=SUBSTEPS),
            rigid_options=gs.options.RigidOptions(
                dt=self.ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(4, self.num_envs))),
                env_separate_rigid=True,
            ),
            viewer_options=gs.options.ViewerOptions(
                res=(1280, 720),
                camera_pos=(1.8, 0.0, 1.2),
                camera_lookat=(0.0, 0.0, 0.2),
                camera_fov=55,
                max_FPS=int(1.0 / self.ctrl_dt),
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # Ground plane (substitutes for the lab_table — surface at z=0).
        self.scene.add_entity(gs.morphs.Plane())

        # ---- Robots ----------------------------------------------------
        urdf_path = _resolve_fr3_urdf()
        self.robot_0 = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=ROBOT_0_BASE_POS,
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=True,
                links_to_keep=["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
            ),
        )
        self.robot_1 = self.scene.add_entity(
            gs.morphs.URDF(
                file=urdf_path,
                pos=ROBOT_1_BASE_POS,
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=True,
                links_to_keep=["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
            ),
        )

        # ---- Box -------------------------------------------------------
        # Density chosen so post-build mass ≈ BOX_MASS given the recentered
        # mesh volume ≈ 0.001398 m³ -> rho ≈ 357.7 kg/m³. We still call
        # entity.set_mass(BOX_MASS) post-build for an exact match.
        box_volume = (2 * BOX_HALF_X) * (2 * BOX_HALF_Y) * (2 * BOX_HALF_Z)  # rough
        box_rho = BOX_MASS / box_volume
        self.box = self.scene.add_entity(
            material=gs.materials.Rigid(rho=box_rho),
            morph=gs.morphs.Mesh(
                file=_resolve_eurobox_mesh(),
                pos=(0.0, 0.0, BOX_INIT_Z),
                quat=BOX_INIT_QUAT,
                fixed=False,
                convexify=True,
            ),
        )

        # Optional debug camera (must be added BEFORE scene.build()).
        self.vis_cam = None
        if attach_debug_camera:
            self.vis_cam = self.scene.add_camera(
                res=cam_res,
                pos=(1.4, 0.0, 1.0),
                lookat=(0.0, 0.0, 0.15),
                fov=50,
                GUI=False,
            )

        # Build the scene.
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))

        # ---- Force exact box mass post-build ---------------------------
        try:
            self.box.set_mass(BOX_MASS)
        except Exception as e:
            print(f"[lift_box] warn: failed to set exact box mass ({e}); using density-derived mass.")

        # ---- Robot-joint discovery + PD gains --------------------------
        # `joint.dof_start` is a GLOBAL DOF index. The entity API expects
        # `dofs_idx_local` in the LOCAL range [0, robot.n_dofs). Subtract
        # each robot's `_dof_start` to convert global -> local.
        ofs_0 = self.robot_0._dof_start
        ofs_1 = self.robot_1._dof_start
        self._arm_dof_idx_0 = torch.tensor(
            [self.robot_0.get_joint(n).dof_start - ofs_0 for n in ARM_HOME_0.keys()],
            dtype=torch.long, device=self.device,
        )
        self._arm_dof_idx_1 = torch.tensor(
            [self.robot_1.get_joint(n).dof_start - ofs_1 for n in ARM_HOME_1.keys()],
            dtype=torch.long, device=self.device,
        )
        finger_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
        self._finger_dof_idx_0 = torch.tensor(
            [self.robot_0.get_joint(n).dof_start - ofs_0 for n in finger_names],
            dtype=torch.long, device=self.device,
        )
        self._finger_dof_idx_1 = torch.tensor(
            [self.robot_1.get_joint(n).dof_start - ofs_1 for n in finger_names],
            dtype=torch.long, device=self.device,
        )
        self._all_dof_idx_0 = torch.cat([self._arm_dof_idx_0, self._finger_dof_idx_0])
        self._all_dof_idx_1 = torch.cat([self._arm_dof_idx_1, self._finger_dof_idx_1])

        # PD gains — arms use Genesis grasp_env-style (10x PhysX-equivalent stiffness),
        # fingers use stiffer kp=2000/kv=100 because the 0.5 kg eurobox needs real
        # grip force from a top-down pinch (soft kp=100 fingers can't generate enough
        # static friction to lift). For light cubes (StackCube, Insert-Drawer) the
        # softer fingers are fine.
        kp = torch.tensor(
            [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            device=self.device,
        )
        kv = torch.tensor(
            [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 100.0, 100.0],
            device=self.device,
        )
        for robot, all_idx in (
            (self.robot_0, self._all_dof_idx_0),
            (self.robot_1, self._all_dof_idx_1),
        ):
            robot.set_dofs_kp(kp, all_idx)
            robot.set_dofs_kv(kv, all_idx)
            # Force limits — spec arm 1-4 ±87, 5-7 ±12, fingers ±200.
            f_lo = torch.tensor([-87.0]*4 + [-12.0]*3 + [-200.0]*2, device=self.device)
            f_hi = torch.tensor([ 87.0]*4 + [ 12.0]*3 + [ 200.0]*2, device=self.device)
            robot.set_dofs_force_range(f_lo, f_hi, all_idx)

        # ---- IK target links -------------------------------------------
        self._ee_link_0 = self.robot_0.get_link("fr3_hand")
        self._ee_link_1 = self.robot_1.get_link("fr3_hand")
        self._tcp_local = torch.tensor(TCP_LOCAL_OFFSET, device=self.device, dtype=torch.float32)

        # Initial joint qpos buffers (7 arm + 2 finger).
        self._init_qpos_0 = torch.tensor(
            [ARM_HOME_0[n] for n in ARM_HOME_0.keys()] + [FINGER_OPEN, FINGER_OPEN],
            device=self.device, dtype=torch.float32,
        )
        self._init_qpos_1 = torch.tensor(
            [ARM_HOME_1[n] for n in ARM_HOME_1.keys()] + [FINGER_OPEN, FINGER_OPEN],
            device=self.device, dtype=torch.float32,
        )

        # Per-env robot base positions (constant — fixed-base robots).
        self._robot_0_base_w = torch.tensor(ROBOT_0_BASE_POS, device=self.device, dtype=torch.float32)
        self._robot_1_base_w = torch.tensor(ROBOT_1_BASE_POS, device=self.device, dtype=torch.float32)
        self._robot_0_base_w_per_env = self._robot_0_base_w.unsqueeze(0).expand(self.num_envs, 3).clone()
        self._robot_1_base_w_per_env = self._robot_1_base_w.unsqueeze(0).expand(self.num_envs, 3).clone()

        # Workspace clamp tensors (root frame, per robot).
        self._pos_lo_root_0 = torch.tensor(POS_LO_ROOT_0, device=self.device, dtype=torch.float32)
        self._pos_hi_root_0 = torch.tensor(POS_HI_ROOT_0, device=self.device, dtype=torch.float32)
        self._pos_lo_root_1 = torch.tensor(POS_LO_ROOT_1, device=self.device, dtype=torch.float32)
        self._pos_hi_root_1 = torch.tensor(POS_HI_ROOT_1, device=self.device, dtype=torch.float32)

        # Action scale + box init pose tensors.
        self._action_scale = torch.tensor(ACTION_SCALE, device=self.device, dtype=torch.float32)
        self._box_init_pos_local = torch.tensor((0.0, 0.0, BOX_INIT_Z), device=self.device, dtype=torch.float32)
        self._box_init_quat = torch.tensor(BOX_INIT_QUAT, device=self.device, dtype=torch.float32)
        self._grasp_local_0 = torch.tensor(GRASP_LOCAL_0, device=self.device, dtype=torch.float32)
        self._grasp_local_1 = torch.tensor(GRASP_LOCAL_1, device=self.device, dtype=torch.float32)
        self._target_xy = torch.tensor(TARGET_XY, device=self.device, dtype=torch.float32)

        # ---- Buffers ---------------------------------------------------
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Per-robot action / IK target state.
        self.del_action_0     = torch.zeros(self.num_envs, 3, device=self.device)
        self.del_action_1     = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_pos_w_0  = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_pos_w_1  = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_quat_w_0 = torch.zeros(self.num_envs, 4, device=self.device)
        self.init_ee_quat_w_1 = torch.zeros(self.num_envs, 4, device=self.device)
        self.init_ee_quat_w_0[:, 0] = 1.0
        self.init_ee_quat_w_1[:, 0] = 1.0
        self._prev_applied_pos_w_0 = torch.zeros(self.num_envs, 3, device=self.device)
        self._prev_applied_pos_w_1 = torch.zeros(self.num_envs, 3, device=self.device)
        self._needs_reanchor = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Last applied action (8-D) — used in obs.
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        # Gripper-open flags (initial = open).
        self._gripper_open_cmd_0 = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_open_cmd_1 = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Reward latches.
        self._latch_success = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Reward sums (for `extras["episode"]`).
        self.reward_keys = list(REWARD_WEIGHTS.keys())
        self.episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self.reward_keys}

        self.extras: dict = {}

        # Observation buffer.
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)

        # First reset.
        self.reset()

    # -------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------

    def _reset_idx(self, envs_idx=None) -> None:
        if envs_idx is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            mask = envs_idx.to(torch.bool)
        n = int(mask.sum().item())
        if n == 0:
            return
        envs_idx_int = torch.nonzero(mask, as_tuple=False).flatten()

        # 1. Robots -> URDF home.
        qpos_block_0 = self._init_qpos_0.unsqueeze(0).expand(n, -1).clone()
        qpos_block_1 = self._init_qpos_1.unsqueeze(0).expand(n, -1).clone()
        self.robot_0.set_qpos(
            qpos_block_0, qs_idx_local=self._all_dof_idx_0,
            envs_idx=envs_idx_int, zero_velocity=True, skip_forward=True,
        )
        self.robot_1.set_qpos(
            qpos_block_1, qs_idx_local=self._all_dof_idx_1,
            envs_idx=envs_idx_int, zero_velocity=True, skip_forward=True,
        )

        # 2. Box -> randomized pose, identity quat in env-local frame plus 90° Z.
        x = torch.empty(n, device=self.device).uniform_(BOX_RESET_X[0], BOX_RESET_X[1])
        y = torch.empty(n, device=self.device).uniform_(BOX_RESET_Y[0], BOX_RESET_Y[1])
        z = torch.full((n,), BOX_INIT_Z, device=self.device)
        pos = torch.stack([x, y, z], dim=-1)
        self.box.set_pos(pos, envs_idx=envs_idx_int, skip_forward=True)
        quat = self._box_init_quat.unsqueeze(0).expand(n, 4).clone()
        self.box.set_quat(quat, envs_idx=envs_idx_int, skip_forward=False)
        # Zero velocity so it doesn't carry over from previous episode.
        zero3 = torch.zeros(n, 3, device=self.device)
        try:
            self.box.set_velocity(lin_vel=zero3, ang_vel=zero3, envs_idx=envs_idx_int)
        except Exception:
            # Fallback: zero via dofs_velocity (free-floating root has 6 DOFs).
            pass

        # 3. Buffers.
        self.episode_length_buf[envs_idx_int] = 0
        self.reset_buf[envs_idx_int] = False
        self.del_action_0[envs_idx_int] = 0.0
        self.del_action_1[envs_idx_int] = 0.0
        self.last_action[envs_idx_int] = 0.0
        self._needs_reanchor[envs_idx_int] = True
        self._gripper_open_cmd_0[envs_idx_int] = True
        self._gripper_open_cmd_1[envs_idx_int] = True
        self._latch_success[envs_idx_int] = False

        # 4. Roll up `extras["episode"]`.
        self.extras.setdefault("episode", {})
        for k, v in self.episode_sums.items():
            mean = v[envs_idx_int].mean() if n > 0 else torch.tensor(0.0, device=self.device)
            self.extras["episode"]["rew_" + k] = mean
            v[envs_idx_int] = 0.0

    def reset(self) -> TensorDict:
        self._reset_idx(None)
        return self.get_observations()

    # -------------------------------------------------------------------
    # EE pose helpers
    # -------------------------------------------------------------------

    def _ee_pose_w(self, ee_link) -> tuple[torch.Tensor, torch.Tensor]:
        """Current EE pose in world frame (TCP, body-offset baked in)."""
        hand_pos = ee_link.get_pos()
        hand_quat = ee_link.get_quat()
        c = self._tcp_local[2]
        qw, qx, qy, qz = hand_quat[:, 0], hand_quat[:, 1], hand_quat[:, 2], hand_quat[:, 3]
        offset_w = torch.stack([
            2.0 * (qx * qz + qw * qy) * c,
            2.0 * (qy * qz - qw * qx) * c,
            (1.0 - 2.0 * (qx * qx + qy * qy)) * c,
        ], dim=-1)
        tcp_pos = hand_pos + offset_w
        return tcp_pos, hand_quat

    def _rotate_vec_by_quat(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Rotate vector(s) v (N,3) by quaternion(s) q (N,4) wxyz."""
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        # v' = q * v * q_conj. Standard expansion:
        tw = -qx*vx - qy*vy - qz*vz
        tx =  qw*vx + qy*vz - qz*vy
        ty =  qw*vy + qz*vx - qx*vz
        tz =  qw*vz + qx*vy - qy*vx
        rx = -tw*qx + tx*qw - ty*qz + tz*qy
        ry = -tw*qy + ty*qw - tz*qx + tx*qz
        rz = -tw*qz + tz*qw - tx*qy + ty*qx
        return torch.stack([rx, ry, rz], dim=-1)

    def _grasp_point_w(self, grasp_local: torch.Tensor) -> torch.Tensor:
        """World-frame position of a body-local grasp point on the box."""
        box_pos = self.box.get_pos()
        box_quat = self.box.get_quat()
        v = grasp_local.unsqueeze(0).expand(self.num_envs, 3)
        return box_pos + self._rotate_vec_by_quat(box_quat, v)

    def _world_to_root(self, pos_w: torch.Tensor, base_w_per_env: torch.Tensor) -> torch.Tensor:
        return pos_w - base_w_per_env

    def _root_to_world(self, pos_root: torch.Tensor, base_w_per_env: torch.Tensor) -> torch.Tensor:
        return pos_root + base_w_per_env

    # -------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(self.device).float()

        # Re-anchor init pose for any env that just reset.
        if self._needs_reanchor.any():
            ee_pos_w_0, ee_quat_w_0 = self._ee_pose_w(self._ee_link_0)
            ee_pos_w_1, ee_quat_w_1 = self._ee_pose_w(self._ee_link_1)
            mask = self._needs_reanchor
            self.init_ee_pos_w_0[mask]   = ee_pos_w_0[mask]
            self.init_ee_quat_w_0[mask]  = ee_quat_w_0[mask]
            self._prev_applied_pos_w_0[mask] = ee_pos_w_0[mask]
            self.init_ee_pos_w_1[mask]   = ee_pos_w_1[mask]
            self.init_ee_quat_w_1[mask]  = ee_quat_w_1[mask]
            self._prev_applied_pos_w_1[mask] = ee_pos_w_1[mask]
            self._needs_reanchor[:] = False

        # Clamp action [-1, 1].
        actions = torch.clamp(actions, -1.0, 1.0)
        self.last_action = actions.clone()

        # Split into per-robot blocks: [0:4] = robot_0 (3 delta + 1 gripper);
        # [4:8] = robot_1.
        act_0 = actions[:, 0:4]
        act_1 = actions[:, 4:8]

        # --- Robot 0 ---
        scaled_0 = act_0[:, :3] * self._action_scale
        self.del_action_0 += scaled_0
        abs_pos_w_0 = self.init_ee_pos_w_0 + self.del_action_0
        ema_pos_w_0 = EMA_ALPHA * abs_pos_w_0 + (1.0 - EMA_ALPHA) * self._prev_applied_pos_w_0
        ema_pos_root_0 = self._world_to_root(ema_pos_w_0, self._robot_0_base_w_per_env)
        ema_pos_root_0 = torch.clamp(ema_pos_root_0, self._pos_lo_root_0, self._pos_hi_root_0)
        ema_pos_w_0 = self._root_to_world(ema_pos_root_0, self._robot_0_base_w_per_env)
        self._prev_applied_pos_w_0 = ema_pos_w_0.clone()

        # --- Robot 1 ---
        scaled_1 = act_1[:, :3] * self._action_scale
        self.del_action_1 += scaled_1
        abs_pos_w_1 = self.init_ee_pos_w_1 + self.del_action_1
        ema_pos_w_1 = EMA_ALPHA * abs_pos_w_1 + (1.0 - EMA_ALPHA) * self._prev_applied_pos_w_1
        ema_pos_root_1 = self._world_to_root(ema_pos_w_1, self._robot_1_base_w_per_env)
        ema_pos_root_1 = torch.clamp(ema_pos_root_1, self._pos_lo_root_1, self._pos_hi_root_1)
        ema_pos_w_1 = self._root_to_world(ema_pos_root_1, self._robot_1_base_w_per_env)
        self._prev_applied_pos_w_1 = ema_pos_w_1.clone()

        # IK — one solve per robot. Each returns FULL qpos for that robot;
        # slice out arm DOFs (first 7 entries) since merge_fixed_links makes
        # the arm joints qpos[0:7].
        full_qpos_0 = self.robot_0.inverse_kinematics(
            link=self._ee_link_0,
            pos=ema_pos_w_0, quat=self.init_ee_quat_w_0,
            local_point=self._tcp_local,
            dofs_idx_local=self._arm_dof_idx_0,
            respect_joint_limit=True,
            max_samples=1, max_solver_iters=10,
        )
        arm_target_0 = full_qpos_0[:, :7]

        full_qpos_1 = self.robot_1.inverse_kinematics(
            link=self._ee_link_1,
            pos=ema_pos_w_1, quat=self.init_ee_quat_w_1,
            local_point=self._tcp_local,
            dofs_idx_local=self._arm_dof_idx_1,
            respect_joint_limit=True,
            max_samples=1, max_solver_iters=10,
        )
        arm_target_1 = full_qpos_1[:, :7]

        # Gripper commands.
        gripper_open_0 = act_0[:, -1] >= 0.0
        gripper_open_1 = act_1[:, -1] >= 0.0
        self._gripper_open_cmd_0 = gripper_open_0
        self._gripper_open_cmd_1 = gripper_open_1
        fq_0 = torch.where(gripper_open_0, torch.full_like(gripper_open_0, FINGER_OPEN, dtype=torch.float32),
                           torch.full_like(gripper_open_0, FINGER_CLOSE, dtype=torch.float32))
        fq_1 = torch.where(gripper_open_1, torch.full_like(gripper_open_1, FINGER_OPEN, dtype=torch.float32),
                           torch.full_like(gripper_open_1, FINGER_CLOSE, dtype=torch.float32))
        finger_block_0 = torch.stack([fq_0, fq_0], dim=-1)
        finger_block_1 = torch.stack([fq_1, fq_1], dim=-1)

        full_target_0 = torch.cat([arm_target_0, finger_block_0], dim=-1)
        full_target_1 = torch.cat([arm_target_1, finger_block_1], dim=-1)
        self.robot_0.control_dofs_position(position=full_target_0, dofs_idx_local=self._all_dof_idx_0)
        self.robot_1.control_dofs_position(position=full_target_1, dofs_idx_local=self._all_dof_idx_1)

        # Step physics.
        self.scene.step()

        # Episode time.
        self.episode_length_buf += 1

        # Termination — time_out only (spec §4 success drives reward, not done).
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        try:
            self.reset_buf = self.reset_buf | self.scene.rigid_solver.get_error_envs_mask()
        except Exception:
            pass

        # Time-out flag for value-bootstrapping.
        self.extras["time_outs"] = (self.episode_length_buf >= self.max_episode_length).float()

        # Reward BEFORE soft-reset.
        reward, per_term = self._compute_reward()
        for k in self.reward_keys:
            self.episode_sums[k] += per_term[k]

        self.extras["detailed_reward"] = per_term

        reset_mask_snapshot = self.reset_buf.clone()
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, reset_mask_snapshot, self.extras

    # -------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        # Per-robot ee pose in that robot's root frame (root has identity orient.).
        ee_pos_w_0, ee_quat_w_0 = self._ee_pose_w(self._ee_link_0)
        ee_pos_w_1, ee_quat_w_1 = self._ee_pose_w(self._ee_link_1)
        ee_pos_root_0 = self._world_to_root(ee_pos_w_0, self._robot_0_base_w_per_env)
        ee_pos_root_1 = self._world_to_root(ee_pos_w_1, self._robot_1_base_w_per_env)
        ee_pose_0 = torch.cat([ee_pos_root_0, ee_quat_w_0], dim=-1)  # (N, 7)
        ee_pose_1 = torch.cat([ee_pos_root_1, ee_quat_w_1], dim=-1)  # (N, 7)

        # Box position (env-local world frame — since `set_pos` was given
        # env-local positions and `env_separate_rigid=True`, get_pos returns
        # env-local world coords for the per-env slice).
        box_pos = self.box.get_pos()       # (N, 3)
        box_quat = self.box.get_quat()     # (N, 4)

        # Per-robot gripper joint pos (2).
        gq_0 = self.robot_0.get_dofs_position(self._finger_dof_idx_0)
        gq_1 = self.robot_1.get_dofs_position(self._finger_dof_idx_1)

        self.obs_buf = torch.cat(
            [ee_pose_0, ee_pose_1, box_pos, box_quat, gq_0, gq_1, self.last_action],
            dim=-1,
        )
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # -------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------

    def _contact_proxy(self, robot_idx: int) -> torch.Tensor:
        """Per-robot binary 'grasping the box' proxy.

        Mirror of stack_cube_env._contact_proxy but distance is measured
        between the EE TCP and the robot-specific grasp-frame on the box
        (rather than to a cube center). Gate also requires the gripper to be
        commanded closed (avg finger joint < 0.03 m).
        """
        if robot_idx == 0:
            ee_pos_w, _ = self._ee_pose_w(self._ee_link_0)
            grasp_w = self._grasp_point_w(self._grasp_local_0)
            gq = self.robot_0.get_dofs_position(self._finger_dof_idx_0)
        else:
            ee_pos_w, _ = self._ee_pose_w(self._ee_link_1)
            grasp_w = self._grasp_point_w(self._grasp_local_1)
            gq = self.robot_1.get_dofs_position(self._finger_dof_idx_1)
        d = torch.norm(ee_pos_w - grasp_w, dim=-1)
        gripper_closed = (gq.mean(dim=-1) < 0.03)
        return (d < BOX_GRASP_TOL) & gripper_closed

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Box state.
        box_pos = self.box.get_pos()                        # (N, 3) env-local world
        box_xy = box_pos[:, :2]
        box_z = box_pos[:, 2]
        try:
            box_lin_vel = self.box.get_vel()                # (N, 3)
        except Exception:
            box_lin_vel = torch.zeros_like(box_pos)
        vel_norm = torch.norm(box_lin_vel, dim=-1)

        # Per-robot grasp points + EE.
        ee_pos_w_0, _ = self._ee_pose_w(self._ee_link_0)
        ee_pos_w_1, _ = self._ee_pose_w(self._ee_link_1)
        grasp_w_0 = self._grasp_point_w(self._grasp_local_0)
        grasp_w_1 = self._grasp_point_w(self._grasp_local_1)
        d_0 = torch.norm(ee_pos_w_0 - grasp_w_0, dim=-1)
        d_1 = torch.norm(ee_pos_w_1 - grasp_w_1, dim=-1)

        # §6 term 1+2: tanh attractor (1 - tanh(d/std)).
        ee_0_to_grasp_0 = 1.0 - torch.tanh(d_0 / max(REACH_STD, 1e-6))
        ee_1_to_grasp_1 = 1.0 - torch.tanh(d_1 / max(REACH_STD, 1e-6))

        # §6 term 3+4: grasp contact proxy (0/1).
        contact_0 = self._contact_proxy(0)
        contact_1 = self._contact_proxy(1)
        grasp_contact_0 = contact_0.float()
        grasp_contact_1 = contact_1.float()

        # §6 term 5: lift_height — linear ramp gated on BOTH robots in contact.
        progress = ((box_z - BOX_INIT_Z) / max(LIFT_HEIGHT, 1e-6)).clamp(0.0, 1.0)
        dual_contact = (contact_0 & contact_1).float()
        lift_height_val = progress * dual_contact

        # §6 term 6: box_xy_align — tanh on |box_xy - target_xy|, gated on lifted.
        d_align = torch.norm(box_xy - self._target_xy.unsqueeze(0), dim=-1)
        base_align = 1.0 - torch.tanh(d_align / max(ALIGN_STD, 1e-6))
        lifted_align = (box_z > (BOX_INIT_Z + ALIGN_LIFT_THRESHOLD)).float()
        box_xy_align_val = lifted_align * base_align

        # §6 term 7: success_bonus — one-shot per episode.
        xy_err = torch.norm(box_xy - self._target_xy.unsqueeze(0), dim=-1)
        z_err = torch.abs(box_z - (BOX_INIT_Z + LIFT_HEIGHT))
        now_success = (xy_err < XY_POS_TOL) & (z_err < Z_POS_TOL) & (vel_norm < VEL_TOL)
        success_fire = now_success & (~self._latch_success)
        self._latch_success = self._latch_success | now_success

        per_term: dict[str, torch.Tensor] = {
            "ee_0_to_grasp_0": REWARD_WEIGHTS["ee_0_to_grasp_0"] * ee_0_to_grasp_0,
            "ee_1_to_grasp_1": REWARD_WEIGHTS["ee_1_to_grasp_1"] * ee_1_to_grasp_1,
            "grasp_contact_0": REWARD_WEIGHTS["grasp_contact_0"] * grasp_contact_0,
            "grasp_contact_1": REWARD_WEIGHTS["grasp_contact_1"] * grasp_contact_1,
            "lift_height":     REWARD_WEIGHTS["lift_height"]     * lift_height_val,
            "box_xy_align":    REWARD_WEIGHTS["box_xy_align"]    * box_xy_align_val,
            "success_bonus":   REWARD_WEIGHTS["success_bonus"]   * success_fire.float(),
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for v in per_term.values():
            total = total + v
        return total, per_term
