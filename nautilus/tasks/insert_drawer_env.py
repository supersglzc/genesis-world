"""Genesis port of IsaacLab Triton-Insert-Drawer (single-arm FR3 pick-cube-
place-in-drawer-then-close).

IsaacLab spec source-of-truth:
  IsaacLab/nautilus/create-task/triton-insert-drawer-implementation.md
  (§1..§7 — scene, actions, reset, termination, observation, reward, DR)

Per the spec:
- One FR3 + Franka hand at env-local (-0.274, 0.49, 0.01).
- DexCube (0.043 m edge, 55 g) — `gs.morphs.Box(size=(0.043,)*3, fixed=False)`.
- Drawer: URDF under `nautilus/assets/drawer/drawer_one_sided_handle_scaled_rotated_mesh_handle.urdf`.
  - base_link (kinematic anchor — `fixed=True` so the base is welded to world)
  - drawer (sliding part, prismatic `base_drawer_joint`, axis local +X,
    travel [0, 0.3] m). Damping 1.0, friction 1.0 — the joint slides under
    contact from the gripper.
  - Spawned at world `(0, 0, 0.10)` rotated 90° about +Z so prismatic +X-local
    maps to world +Y.
  - Initial joint pos at reset = 0.30 m (FULLY OPEN).
- Sim: ctrl_dt=1/20 Hz, substeps=6 (1/120 s each), episode = 180 steps (9 s).
- Action (4-D): 3-D EMA Cartesian EE-delta + 1-D binary gripper (same as
  stack_cube_env). Workspace clamp in robot-root frame: spec §2 says
  `[0.34, -0.8, 0.005]` to `[0.50, -0.05, 0.30]`.
- Observation (19-D): ee_pose_root (7) + cube_pos_root (3, zero-masked once
  cube_inside latch fires) + drawer_body_pos_root (3) + gripper_q (2) +
  last_action (4).
- Termination: time_out (180) | success. success = cube_inside_latch AND
  drawer joint pos < 0.10 m. extras["time_outs"] flags ONLY time_out.
- Reward (§6, sum composer, weights monotonically increasing):
  reach_cube, is_lifted, lift_distance, align, ee_retract_to_front_face,
  cube_inside_bonus_latch (one-shot), close_drawer, success_bonus (one-shot).
  Dense terms are zero-masked once the cube_inside latch fires; retract/close
  are gated ON by the latch.

Genesis-only deviations from the spec (Genesis lacks the relevant APIs):
- No USD assets: cube is `gs.morphs.Box`, drawer is `gs.morphs.URDF`.
- No FrameTransformer: ee TCP, drawer drop frame, drawer front face frame
  are computed from link poses + body-local offsets, rotated by the
  drawer link's world quat where needed.
- No filtered contact sensors: `lift_distance`'s grasp gate uses the
  stack_cube_env `_contact_proxy` (EE TCP within (CUBE_SIZE+0.02) m of the
  cube AND gripper command closed).
- The drawer URDF is loaded `fixed=True` so `base_link` is welded to world.
  The prismatic joint `base_drawer_joint` is left free (no PD); it slides
  passively under contact from the robot.

NOTE on reward weights: same convention as the other Genesis envs in this
repo — we do NOT multiply by ctrl_dt. The IsaacLab fork the spec was probed
from removed the per-weight * dt multiplier so the spec weights ARE the
per-step magnitudes.
"""
from __future__ import annotations

from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs


# --- Constants from the IsaacLab spec ---------------------------------------

CUBE_SIZE = 0.043
CUBE_INIT_Z = CUBE_SIZE / 2.0
CUBE_MASS = 0.055

# Robot base in world frame (IsaacLab `pos=(-0.274, 0.49, 0.01)`).
ROBOT_BASE_POS = (-0.274, 0.49, 0.01)

# FR3 home joint pose.
ARM_HOME = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3":  0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5":  0.0,
    "fr3_joint6":  1.87,
    "fr3_joint7":  1.57,
}
FINGER_OPEN  = 0.04
FINGER_CLOSE = 0.0

# Action scale and EMA alpha (spec §2).
ACTION_SCALE = (0.01, 0.01, 0.01)
EMA_ALPHA = 0.5

# Workspace clamp in robot-root frame (spec §2 table).
POS_LOWER_LIMIT_ROOT = (0.34, -0.8,  0.005)
POS_UPPER_LIMIT_ROOT = (0.50, -0.05, 0.30)

# IK TCP offset (spec §2 — `OffsetCfg(pos=[0,0,0.2])`).
TCP_LOCAL_OFFSET = (0.0, 0.0, 0.2)

# Sim timing (spec §1).
CTRL_DT       = 1.0 / 20.0
SUBSTEPS      = 6
EPISODE_STEPS = 180

# Drawer pose at spawn (spec §1).
DRAWER_INIT_POS = (0.0, 0.0, 0.10)
DRAWER_INIT_QUAT = (0.7071068, 0.0, 0.0, 0.7071068)
DRAWER_JOINT_OPEN = 0.30           # reset joint pos (spec §3 — fully open)
DRAWER_MAX_OPEN   = 0.30
DRAWER_CLOSED_THR = 0.10           # success threshold (spec §4)

# Drawer-local body offsets for the two reference frames (spec §1 table).
DROP_FRAME_LOCAL       = (0.0, 0.0, 0.25)   # drop above the open tray rim
FRONT_FACE_LOCAL       = (0.17, 0.0, 0.15)  # front +X-local face center

# Cube reset xy box — env-local world frame (spec §3).
CUBE_RESET_X = (0.1, 0.2)
CUBE_RESET_Y = (0.3, 0.4)

# Drawer base reset xy — env-local world frame (spec §3 `reset_drawer_pose`).
# Added to DRAWER_INIT_POS each reset. y is fixed at -0.3 so the OPEN drawer
# body (joint=+0.30 along +Y after rotation) sits at world y=0 — well separated
# from the cube's y=0.3-0.4 spawn area. Without this reset the open drawer body
# overlaps the cube spawn xy and the cube ends up "under" the drawer floor.
DRAWER_RESET_X = (0.1, 0.2)
DRAWER_RESET_Y = (-0.3, -0.3)

# Reward weights (spec §6 table — RAW per-step magnitudes, no *dt).
REWARD_WEIGHTS = {
    "reach_cube":              0.02,
    "is_lifted":               0.2,
    "lift_distance":           0.3,
    "align":                   2.0,
    "ee_retract_to_front_face": 2.0,
    "cube_inside_bonus_latch": 300.0,
    "close_drawer":            100.0,
    "success_bonus":           2000.0,
}

# Reward hyperparameters (spec §6 `params=`).
REACH_STD              = 0.1
LIFT_MINIMAL_HEIGHT    = 0.04
LIFT_DISTANCE_INIT_Z   = 0.0215
LIFT_DISTANCE_TARGET_Z = 0.25
ALIGN_STD              = 0.20
ALIGN_MIN_HEIGHT_B     = 0.25
RETRACT_STD            = 0.05
CUBE_INSIDE_XY_THR     = 0.15
CUBE_INSIDE_Z_FLOOR    = -0.02
CUBE_INSIDE_Z_CEIL     = 0.07
CUBE_INSIDE_EE_FAR_THR = 0.10
CLOSE_DRAWER_ALPHA     = 1.0

# Per-episode cap on cumulative `close_drawer` reward (weighted contribution).
# Matches the ManiSkill insert_drawer pattern (their cap is 2000; user-overridden
# to 1000 here). Prevents the policy from racking up unbounded close_drawer
# reward when the drawer stays partially closed for many steps. Once the running
# total reaches the cap, subsequent close_drawer reward is zero for the episode.
CLOSE_DRAWER_EPISODE_CAP = 1000.0


# --- Asset resolution -------------------------------------------------------

def _resolve_fr3_urdf() -> str:
    here = Path(__file__).resolve().parents[2] / "nautilus" / "assets" / "fr3" / "fr3_franka_hand.urdf"
    if not here.is_file():
        raise FileNotFoundError(f"FR3 URDF not found at {here}")
    return str(here)


def _resolve_drawer_urdf() -> str:
    here = (Path(__file__).resolve().parents[2] / "nautilus" / "assets" / "drawer"
            / "drawer_one_sided_handle_scaled_rotated_mesh_handle.urdf")
    if not here.is_file():
        raise FileNotFoundError(f"Drawer URDF not found at {here}")
    return str(here)


class InsertDrawerEnv:
    """Genesis-native FR3 insert-drawer env, IsaacLab-spec faithful."""

    def __init__(
        self,
        num_envs: int = 1,
        show_viewer: bool = False,
        env_spacing: float = 2.5,
        attach_debug_camera: bool = False,
        cam_res: tuple[int, int] = (320, 240),
    ) -> None:
        self.num_envs = int(num_envs)
        self.num_actions = 4
        self.num_obs = 19
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
                camera_pos=(1.2, -0.7, 1.0),
                camera_lookat=(0.0, 0.0, 0.1),
                camera_fov=55,
                max_FPS=int(1.0 / self.ctrl_dt),
            ),
            profiling_options=gs.options.ProfilingOptions(show_FPS=False),
            show_viewer=show_viewer,
        )

        # Ground plane (acts as the table top — surface at z=0).
        self.scene.add_entity(gs.morphs.Plane())

        # Robot — FR3 + Franka hand.
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=_resolve_fr3_urdf(),
                pos=ROBOT_BASE_POS,
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=True,
                links_to_keep=["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
            ),
        )

        # Cube — gs.morphs.Box, initial XY placeholder; actual xy set in reset.
        cube_pos_init = (0.0, 0.0, CUBE_INIT_Z)
        self.cube = self.scene.add_entity(
            gs.morphs.Box(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                pos=cube_pos_init,
                fixed=False,
                batch_fixed_verts=False,
            ),
        )

        # Drawer — URDF with `fixed=True` so base_link is welded to world.
        # The prismatic `base_drawer_joint` slides freely under robot contact.
        # `merge_fixed_links=True` collapses all the secondary walls onto
        # base_link / drawer; we keep the `drawer` link as the moving body.
        self.drawer = self.scene.add_entity(
            gs.morphs.URDF(
                file=_resolve_drawer_urdf(),
                pos=DRAWER_INIT_POS,
                quat=DRAWER_INIT_QUAT,
                fixed=True,
                merge_fixed_links=True,
                links_to_keep=["drawer"],
                # Required for per-env base position reset on a fixed entity.
                batch_fixed_verts=True,
            ),
        )

        # Optional debug camera (must be added BEFORE scene.build()).
        self.vis_cam = None
        if attach_debug_camera:
            self.vis_cam = self.scene.add_camera(
                res=cam_res,
                pos=(1.0, -0.6, 0.8),
                lookat=(0.0, 0.0, 0.15),
                fov=50,
                GUI=False,
            )

        # Build the scene.
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))

        # ---- Force exact cube mass post-build --------------------------
        try:
            self.cube.set_mass(CUBE_MASS)
        except Exception as e:
            print(f"[insert_drawer] warn: failed to set cube mass ({e}); using default mass.")

        # ---- Robot-joint discovery + PD gains --------------------------
        # `joint.dof_start` is a GLOBAL DOF index. Subtract the robot's
        # `_dof_start` to convert global -> local.
        ofs_r = self.robot._dof_start
        self._arm_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start - ofs_r for n in ARM_HOME.keys()],
            dtype=torch.long, device=self.device,
        )
        finger_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
        self._finger_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start - ofs_r for n in finger_names],
            dtype=torch.long, device=self.device,
        )
        self._all_dof_idx = torch.cat([self._arm_dof_idx, self._finger_dof_idx])

        # Genesis grasp_env-style PD gains throughout (arms AND fingers).
        kp = torch.tensor(
            [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0],
            device=self.device,
        )
        kv = torch.tensor(
            [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0],
            device=self.device,
        )
        self.robot.set_dofs_kp(kp, self._all_dof_idx)
        self.robot.set_dofs_kv(kv, self._all_dof_idx)
        f_lo = torch.tensor([-87.0]*4 + [-12.0]*3 + [-200.0]*2, device=self.device)
        f_hi = torch.tensor([ 87.0]*4 + [ 12.0]*3 + [ 200.0]*2, device=self.device)
        self.robot.set_dofs_force_range(f_lo, f_hi, self._all_dof_idx)

        # ---- Drawer DOF discovery --------------------------------------
        ofs_d = self.drawer._dof_start
        self._drawer_joint_dof_idx = torch.tensor(
            [self.drawer.get_joint("base_drawer_joint").dof_start - ofs_d],
            dtype=torch.long, device=self.device,
        )

        # ---- IK target + link handles ----------------------------------
        self._ee_link = self.robot.get_link("fr3_hand")
        self._tcp_local = torch.tensor(TCP_LOCAL_OFFSET, device=self.device, dtype=torch.float32)
        self._drawer_body_link = self.drawer.get_link("drawer")

        # Initial joint qpos buffer (7 arm + 2 finger).
        self._init_qpos = torch.tensor(
            [ARM_HOME[n] for n in ARM_HOME.keys()] + [FINGER_OPEN, FINGER_OPEN],
            device=self.device, dtype=torch.float32,
        )

        # Robot / drawer base poses per env (constant — fixed-base).
        self._robot_base_w = torch.tensor(ROBOT_BASE_POS, device=self.device, dtype=torch.float32)
        self._robot_base_w_per_env = self._robot_base_w.unsqueeze(0).expand(self.num_envs, 3).clone()

        # Workspace clamp tensors.
        self._pos_lo_root = torch.tensor(POS_LOWER_LIMIT_ROOT, device=self.device, dtype=torch.float32)
        self._pos_hi_root = torch.tensor(POS_UPPER_LIMIT_ROOT, device=self.device, dtype=torch.float32)

        # Action scale tensor.
        self._action_scale = torch.tensor(ACTION_SCALE, device=self.device, dtype=torch.float32)

        # Body-local offsets for the two frame transformers.
        self._drop_local = torch.tensor(DROP_FRAME_LOCAL, device=self.device, dtype=torch.float32)
        self._front_face_local = torch.tensor(FRONT_FACE_LOCAL, device=self.device, dtype=torch.float32)

        # ---- Buffers ---------------------------------------------------
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Action / IK target state.
        self.del_action       = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_pos_w    = torch.zeros(self.num_envs, 3, device=self.device)
        self.init_ee_quat_w   = torch.zeros(self.num_envs, 4, device=self.device)
        self.init_ee_quat_w[:, 0] = 1.0
        self._prev_applied_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._needs_reanchor = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        # Last applied action (4-D) — used in obs.
        self.last_action = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        # Gripper-open flag (initial = open).
        self._gripper_open_cmd = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)

        # Reward latches (per-env booleans).
        # `_latch_cube_inside`: set the FIRST step `cube_inside_drawer_geometric &
        # ee_far_from_cube` is true. Cleared on reset. Drives the §6 dense
        # zero-masking, the retract/close gates, and the §4 success termination.
        self._latch_cube_inside = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # `_latch_success_fired`: ensures the success_bonus only contributes +1
        # per episode (matches spec semantics).
        self._latch_success_fired = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        # Per-episode cumulative `close_drawer` contribution (weighted). Reset
        # on each episode reset; capped at CLOSE_DRAWER_EPISODE_CAP each step.
        self._close_drawer_cumulative = torch.zeros(self.num_envs, device=self.device)

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

        # 1. Robot joints -> URDF home.
        qpos_block = self._init_qpos.unsqueeze(0).expand(n, -1).clone()
        self.robot.set_qpos(
            qpos_block,
            qs_idx_local=self._all_dof_idx,
            envs_idx=envs_idx_int,
            zero_velocity=True,
            skip_forward=True,
        )

        # 2. Cube -> uniform xy in env-local frame.
        x = torch.empty(n, device=self.device).uniform_(CUBE_RESET_X[0], CUBE_RESET_X[1])
        y = torch.empty(n, device=self.device).uniform_(CUBE_RESET_Y[0], CUBE_RESET_Y[1])
        z = torch.full((n,), CUBE_INIT_Z, device=self.device)
        cube_pos = torch.stack([x, y, z], dim=-1)
        self.cube.set_pos(cube_pos, envs_idx=envs_idx_int, skip_forward=True)
        quat = torch.zeros(n, 4, device=self.device)
        quat[:, 0] = 1.0
        self.cube.set_quat(quat, envs_idx=envs_idx_int, skip_forward=True)

        # 3a. Drawer base xy reset (spec §3 `reset_drawer_pose`). Sample x in
        # [0.1, 0.2], y fixed at -0.3, z unchanged. This places the OPEN drawer
        # body (joint=+0.30 along +Y after rotation) at world y=0, well
        # separated from the cube's y=0.3-0.4 spawn region.
        dx = torch.empty(n, device=self.device).uniform_(DRAWER_RESET_X[0], DRAWER_RESET_X[1])
        dy = torch.empty(n, device=self.device).uniform_(DRAWER_RESET_Y[0], DRAWER_RESET_Y[1])
        dz = torch.full((n,), 0.0, device=self.device)
        drawer_pos = torch.stack(
            [dx + DRAWER_INIT_POS[0], dy + DRAWER_INIT_POS[1], dz + DRAWER_INIT_POS[2]],
            dim=-1,
        )
        self.drawer.set_pos(drawer_pos, envs_idx=envs_idx_int, skip_forward=True)

        # 3b. Drawer joint -> OPEN (0.30 m).
        drawer_qpos = torch.full((n, 1), DRAWER_JOINT_OPEN, device=self.device)
        self.drawer.set_qpos(
            drawer_qpos,
            qs_idx_local=self._drawer_joint_dof_idx,
            envs_idx=envs_idx_int,
            zero_velocity=True,
            skip_forward=False,
        )

        # 4. Buffers.
        self.episode_length_buf[envs_idx_int] = 0
        self.reset_buf[envs_idx_int] = False
        self.del_action[envs_idx_int] = 0.0
        self.last_action[envs_idx_int] = 0.0
        self._needs_reanchor[envs_idx_int] = True
        self._gripper_open_cmd[envs_idx_int] = True
        self._latch_cube_inside[envs_idx_int] = False
        self._latch_success_fired[envs_idx_int] = False
        self._close_drawer_cumulative[envs_idx_int] = 0.0

        # 5. Roll up `extras["episode"]`.
        self.extras.setdefault("episode", {})
        for k, v in self.episode_sums.items():
            mean = v[envs_idx_int].mean() if n > 0 else torch.tensor(0.0, device=self.device)
            self.extras["episode"]["rew_" + k] = mean
            v[envs_idx_int] = 0.0

    def reset(self) -> TensorDict:
        self._reset_idx(None)
        return self.get_observations()

    # -------------------------------------------------------------------
    # Geometry helpers
    # -------------------------------------------------------------------

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """EE TCP pose in world frame (body-offset (0,0,0.2) baked in)."""
        hand_pos = self._ee_link.get_pos()
        hand_quat = self._ee_link.get_quat()
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
        tw = -qx*vx - qy*vy - qz*vz
        tx =  qw*vx + qy*vz - qz*vy
        ty =  qw*vy + qz*vx - qx*vz
        tz =  qw*vz + qx*vy - qy*vx
        rx = -tw*qx + tx*qw - ty*qz + tz*qy
        ry = -tw*qy + ty*qw - tz*qx + tx*qz
        rz = -tw*qz + tz*qw - tx*qy + ty*qx
        return torch.stack([rx, ry, rz], dim=-1)

    def _drawer_body_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._drawer_body_link.get_pos(), self._drawer_body_link.get_quat()

    def _drawer_drop_frame_w(self) -> torch.Tensor:
        body_pos, body_quat = self._drawer_body_pose_w()
        v = self._drop_local.unsqueeze(0).expand(self.num_envs, 3)
        return body_pos + self._rotate_vec_by_quat(body_quat, v)

    def _drawer_front_face_w(self) -> torch.Tensor:
        body_pos, body_quat = self._drawer_body_pose_w()
        v = self._front_face_local.unsqueeze(0).expand(self.num_envs, 3)
        return body_pos + self._rotate_vec_by_quat(body_quat, v)

    def _drawer_joint_pos(self) -> torch.Tensor:
        """Current drawer prismatic joint position (m). Shape (N,)."""
        return self.drawer.get_dofs_position(self._drawer_joint_dof_idx)[:, 0]

    def _world_to_root(self, pos_w: torch.Tensor) -> torch.Tensor:
        return pos_w - self._robot_base_w_per_env

    def _root_to_world(self, pos_root: torch.Tensor) -> torch.Tensor:
        return pos_root + self._robot_base_w_per_env

    # -------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        actions = actions.to(self.device).float()

        # Re-anchor init pose for any env that just reset.
        if self._needs_reanchor.any():
            ee_pos_w, ee_quat_w = self._ee_pose_w()
            mask = self._needs_reanchor
            self.init_ee_pos_w[mask]    = ee_pos_w[mask]
            self.init_ee_quat_w[mask]   = ee_quat_w[mask]
            self._prev_applied_pos_w[mask] = ee_pos_w[mask]
            self._needs_reanchor[:] = False

        # Clamp action [-1, 1].
        actions = torch.clamp(actions, -1.0, 1.0)
        self.last_action = actions.clone()

        # Cartesian delta in world frame.
        scaled = actions[:, :3] * self._action_scale
        self.del_action += scaled
        abs_pos_w = self.init_ee_pos_w + self.del_action

        # EMA against previously applied target.
        ema_pos_w = EMA_ALPHA * abs_pos_w + (1.0 - EMA_ALPHA) * self._prev_applied_pos_w

        # Clamp in robot-root frame.
        ema_pos_root = self._world_to_root(ema_pos_w)
        ema_pos_root = torch.clamp(ema_pos_root, self._pos_lo_root, self._pos_hi_root)
        ema_pos_w = self._root_to_world(ema_pos_root)
        self._prev_applied_pos_w = ema_pos_w.clone()

        # Genesis IK on the 7 arm DOFs.
        full_qpos = self.robot.inverse_kinematics(
            link=self._ee_link,
            pos=ema_pos_w,
            quat=self.init_ee_quat_w,
            local_point=self._tcp_local,
            dofs_idx_local=self._arm_dof_idx,
            respect_joint_limit=True,
            max_samples=1,
            max_solver_iters=10,
        )
        arm_target = full_qpos[:, :7]

        # Gripper command from action[-1]: >= 0 -> open, < 0 -> close.
        gripper_open = actions[:, -1] >= 0.0
        self._gripper_open_cmd = gripper_open
        finger_q = torch.where(gripper_open,
                               torch.full_like(gripper_open, FINGER_OPEN, dtype=torch.float32),
                               torch.full_like(gripper_open, FINGER_CLOSE, dtype=torch.float32))
        finger_block = torch.stack([finger_q, finger_q], dim=-1)

        # PD targets on robot arm + fingers.
        full_target = torch.cat([arm_target, finger_block], dim=-1)
        self.robot.control_dofs_position(position=full_target, dofs_idx_local=self._all_dof_idx)

        # Step physics.
        self.scene.step()

        # Episode time.
        self.episode_length_buf += 1

        # Compute reward (also updates the cube_inside latch).
        reward, per_term = self._compute_reward()
        for k in self.reward_keys:
            self.episode_sums[k] += per_term[k]
        self.extras["detailed_reward"] = per_term

        # Termination: time_out | success.
        time_out = self.episode_length_buf >= self.max_episode_length
        drawer_jp_now = self._drawer_joint_pos()
        success = self._latch_cube_inside & (drawer_jp_now < DRAWER_CLOSED_THR)
        self.reset_buf = time_out | success
        try:
            self.reset_buf = self.reset_buf | self.scene.rigid_solver.get_error_envs_mask()
        except Exception:
            pass

        # extras["time_outs"] flags ONLY time_out (success is a true terminal).
        self.extras["time_outs"] = time_out.float()

        reset_mask_snapshot = self.reset_buf.clone()
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, reset_mask_snapshot, self.extras

    # -------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------

    def get_observations(self) -> TensorDict:
        # EE pose in robot root frame (robot root has identity orientation
        # here, so root-frame quat = world quat).
        ee_pos_w, ee_quat_w = self._ee_pose_w()
        ee_pos_root = self._world_to_root(ee_pos_w)
        ee_pose_root = torch.cat([ee_pos_root, ee_quat_w], dim=-1)  # (N, 7)

        # Cube pos in robot root frame, zero-masked once the cube_inside latch
        # has fired (spec §5: cube_position is the only obs gated by the latch).
        cube_pos_w = self.cube.get_pos()
        cube_pos_root = self._world_to_root(cube_pos_w)
        latch_active = self._latch_cube_inside.float().unsqueeze(-1)  # (N, 1)
        cube_pos_obs = cube_pos_root * (1.0 - latch_active)            # (N, 3)

        # Drawer body position in robot root frame (sliding `drawer` link, NOT
        # the fixed base_link).
        drawer_body_pos_w, _ = self._drawer_body_pose_w()
        drawer_body_pos_root = self._world_to_root(drawer_body_pos_w)

        # Gripper joint pos (2).
        gripper_q = self.robot.get_dofs_position(self._finger_dof_idx)  # (N, 2)

        self.obs_buf = torch.cat(
            [ee_pose_root, cube_pos_obs, drawer_body_pos_root, gripper_q, self.last_action],
            dim=-1,
        )
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # -------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------

    def _contact_proxy_cube(self) -> torch.Tensor:
        """EE TCP within (CUBE_SIZE+0.02) of cube AND gripper closed (avg
        finger joint pos < 0.03 m). Genesis substitute for IsaacLab filtered
        fingertip-contact sensors."""
        cube_pos = self.cube.get_pos()
        ee_pos_w, _ = self._ee_pose_w()
        d = torch.norm(ee_pos_w - cube_pos, dim=-1)
        gripper_q = self.robot.get_dofs_position(self._finger_dof_idx)
        gripper_closed = (gripper_q.mean(dim=-1) < 0.03)
        return (d < (CUBE_SIZE + 0.02)) & gripper_closed

    def _cube_inside_drawer_geometric(self) -> torch.Tensor:
        """Cube COM inside the drawer body's AABB in drawer-local frame."""
        cube_pos = self.cube.get_pos()
        body_pos, body_quat = self._drawer_body_pose_w()
        rel_w = cube_pos - body_pos                                   # (N, 3)
        # Express in drawer-local frame: rotate by inverse quat.
        inv_quat = body_quat.clone()
        inv_quat[:, 1:] = -inv_quat[:, 1:]                            # conjugate
        rel_local = self._rotate_vec_by_quat(inv_quat, rel_w)
        xy_in = torch.norm(rel_local[:, :2], dim=-1) < CUBE_INSIDE_XY_THR
        z_in = (rel_local[:, 2] > CUBE_INSIDE_Z_FLOOR) & (rel_local[:, 2] < CUBE_INSIDE_Z_CEIL)
        return xy_in & z_in

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # ---- Common geometry ------------------------------------------
        cube_pos_w = self.cube.get_pos()
        cube_z = cube_pos_w[:, 2]
        ee_pos_w, _ = self._ee_pose_w()
        drop_pos_w = self._drawer_drop_frame_w()
        front_face_w = self._drawer_front_face_w()
        drawer_jp = self.drawer.get_dofs_position(self._drawer_joint_dof_idx)[:, 0]

        # ---- Latch update (spec §6 helper) ----------------------------
        # cube_inside_bonus_once_per_episode fires the FIRST step that
        # cube_inside_geometric AND EE far from cube.
        inside = self._cube_inside_drawer_geometric()
        ee_cube_dist = torch.norm(ee_pos_w - cube_pos_w, dim=-1)
        ee_far = ee_cube_dist > CUBE_INSIDE_EE_FAR_THR
        now_qualifying = inside & ee_far
        latch_fire = now_qualifying & (~self._latch_cube_inside)
        self._latch_cube_inside = self._latch_cube_inside | now_qualifying
        latch_active = self._latch_cube_inside.float()                  # (N,)
        not_latch = 1.0 - latch_active

        # ---- Phase 2 dense (zero-masked once latch fires) -------------
        # reach_cube: 1 - tanh(d/std).
        d_reach = torch.norm(cube_pos_w - ee_pos_w, dim=-1)
        reach_cube_val = (1.0 - torch.tanh(d_reach / max(REACH_STD, 1e-6))) * not_latch

        # is_lifted: cube_z > minimal_height (binary).
        is_lifted_val = (cube_z > LIFT_MINIMAL_HEIGHT).float() * not_latch

        # lift_distance: ramp on cube.z, gated on EE-proximity contact proxy.
        denom = max(LIFT_DISTANCE_TARGET_Z - LIFT_DISTANCE_INIT_Z, 1e-6)
        base_lift = ((cube_z - LIFT_DISTANCE_INIT_Z) / denom).clamp(0.0, 1.0)
        grasp_gate = self._contact_proxy_cube().float()
        lift_distance_val = base_lift * grasp_gate * not_latch

        # align: cube -> drop_frame attractor, gated on cube high enough.
        d_align = torch.norm(cube_pos_w - drop_pos_w, dim=-1)
        base_align = 1.0 - torch.tanh(d_align / max(ALIGN_STD, 1e-6))
        high_enough = (cube_z > ALIGN_MIN_HEIGHT_B).float()
        align_val = high_enough * base_align * not_latch

        # ---- Phase 3 retract + latch ----------------------------------
        # ee_retract_to_front_face: y-axis attractor between EE and front-face,
        # gated ON by the latch.
        dy = torch.abs(ee_pos_w[:, 1] - front_face_w[:, 1])
        retract_val = (1.0 - torch.tanh(dy / max(RETRACT_STD, 1e-6))) * latch_active

        # cube_inside_bonus_latch: one-shot when geometric + ee_far first hit.
        cube_inside_latch_val = latch_fire.float()

        # ---- Phase 4 close ---------------------------------------------
        # latch_active * (ee.y > front_face.y) * closeness^alpha.
        closeness = ((DRAWER_MAX_OPEN - drawer_jp) / max(DRAWER_MAX_OPEN, 1e-6)).clamp(0.0, 1.0)
        shaped = closeness.pow(CLOSE_DRAWER_ALPHA)
        gate_ee_outside = (ee_pos_w[:, 1] > front_face_w[:, 1]).float()
        close_drawer_val = latch_active * gate_ee_outside * shaped

        # ---- Phase 5 success bonus -------------------------------------
        # latch_active * (drawer joint < threshold). Fire-once latched so the
        # episodic ceiling is exactly +1 reward at termination (success
        # DoneTerm also flips reset_buf simultaneously).
        drawer_closed = (drawer_jp < DRAWER_CLOSED_THR)
        success_now = self._latch_cube_inside & drawer_closed
        success_fire = success_now & (~self._latch_success_fired)
        self._latch_success_fired = self._latch_success_fired | success_now
        success_bonus_val = success_fire.float()

        # close_drawer contribution — cap per-episode cumulative at
        # CLOSE_DRAWER_EPISODE_CAP (ManiSkill insert_drawer pattern).
        contrib_close_drawer = REWARD_WEIGHTS["close_drawer"] * close_drawer_val
        remaining_cap = (CLOSE_DRAWER_EPISODE_CAP - self._close_drawer_cumulative).clamp(min=0.0)
        contrib_close_drawer = torch.minimum(contrib_close_drawer, remaining_cap)
        self._close_drawer_cumulative = self._close_drawer_cumulative + contrib_close_drawer

        per_term: dict[str, torch.Tensor] = {
            "reach_cube":              REWARD_WEIGHTS["reach_cube"]              * reach_cube_val,
            "is_lifted":               REWARD_WEIGHTS["is_lifted"]               * is_lifted_val,
            "lift_distance":           REWARD_WEIGHTS["lift_distance"]           * lift_distance_val,
            "align":                   REWARD_WEIGHTS["align"]                   * align_val,
            "ee_retract_to_front_face": REWARD_WEIGHTS["ee_retract_to_front_face"] * retract_val,
            "cube_inside_bonus_latch": REWARD_WEIGHTS["cube_inside_bonus_latch"] * cube_inside_latch_val,
            "close_drawer":            contrib_close_drawer,
            "success_bonus":           REWARD_WEIGHTS["success_bonus"]           * success_bonus_val,
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for v in per_term.values():
            total = total + v
        return total, per_term
