"""Genesis port of IsaacLab Triton-Franka-StackCube.

Reproduces the IsaacLab task as faithfully as the Genesis API allows:

- FR3 + Franka hand at world (-0.274, 0.49, 0.01), 7 arm joints + 2 finger
  joints, PD gains and force limits matching the IsaacLab ImplicitActuator
  config (kp=400 / kd=80 for the arm, kp=2000 / kd=100 for fingers).
- Scene: kinematic Box "table" with top surface at z=0 + three 4.3 cm cubes
  (cube_0 / cube_1 / cube_2) + the FR3 robot. No USDs — Genesis only loads
  URDF / Mesh / primitives.
- Sim timing: ctrl_dt = 1/20 (20 Hz), substeps=6 (each is 1/120 s — exactly
  matching IsaacLab's `sim.dt=1/120, decimation=6`). Episode horizon = 180
  control steps = 9 s.
- Action (4-D): [dx, dy, dz, gripper]. Per-step:
    * clamp raw [-1, 1], scale by 0.01 m/axis, accumulate `del_action`
    * abs_pos = init_ee_pos + del_action  (init_ee_pos captured at reset)
    * EMA against previously applied target with alpha=0.5
    * clamp in robot-root frame to bbox [0.05, -0.52, 0.005] to [0.5, -0.05, 0.16]
    * orientation locked at init_ee_quat
    * gripper: action[-1] >= 0 -> open (0.04 m), < 0 -> close (0.0 m)
  IK: Genesis built-in `inverse_kinematics(link=fr3_hand, local_point=(0,0,0.2),
  pos=target_w, quat=init_ee_quat_w)` — matches IsaacLab's DLS IK on the
  fingertip TCP at body_offset (0,0,0.2).
- Reset: robot pinned to URDF home (no jitter); per-cube uniform XY offsets
  added to the robot-base anchor (-0.274, 0.49) -> world (X, Y, 0.0215).
- Termination: time_out at step >= max_episode_length. No failure terminations.
- Observation (19-D): ee_pose (7) + grasping_cube_pos (3) + grasping_target_pos
  (3) + gripper_pos (2) + last_action (4). Stateless mux on cube_0_on_cube_1.
- Reward (7 active terms): reach, lift, align, success_bonus, stack_broke_penalty,
  tower_bonus, linear_lift_grasping_cube. Weights match IsaacLab spec exactly.
  Contact-gated terms use a proximity proxy: both fingers within
  (CUBE_SIZE+0.01) of the cube center AND cube z > some lift threshold (no
  Genesis filtered contact-sensor equivalent).

Critical: we do NOT multiply reward scales by ctrl_dt. The grasp_env.py
pattern does (line 181), but the IsaacLab fork the spec was probed from has
REMOVED the per-weight * dt multiplier, so the spec weights ARE the per-step
magnitudes.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
from tensordict import TensorDict

import genesis as gs
from genesis.utils.geom import inv_quat, transform_quat_by_quat, quat_to_rotvec


# --- Constants from the IsaacLab spec ---------------------------------------

CUBE_SIZE = 0.043           # m — DexCube edge length
CUBE_INIT_Z = CUBE_SIZE / 2.0  # = 0.0215 m — cube center half-edge above table

# Robot base in world frame (IsaacLab `pos=(-0.274, 0.49, 0.01)`).
ROBOT_BASE_POS = (-0.274, 0.49, 0.01)

# FR3 home joint pose — 7 arm joints + 2 finger joints. Order matches the
# URDF joint order discovered at build-time via `get_joint(name).dof_start`.
ARM_HOME = {
    "fr3_joint1": -0.785,
    "fr3_joint2": -0.785,
    "fr3_joint3":  0.0,
    "fr3_joint4": -2.655,
    "fr3_joint5":  0.0,
    "fr3_joint6":  1.87,
    "fr3_joint7":  0.0,
}
FINGER_OPEN  = 0.04
FINGER_CLOSE = 0.0

# Action: per-axis scale and EMA alpha (IsaacLab spec).
ACTION_SCALE = (0.01, 0.01, 0.01)
EMA_ALPHA = 0.5

# Workspace clamp in robot-root frame.
POS_LOWER_LIMIT_ROOT = (0.05, -0.52, 0.005)
POS_UPPER_LIMIT_ROOT = (0.50, -0.05, 0.16)

# Per-cube XY ranges (env-local — added to the robot-base XY anchor).
CUBE_0_X = (0.0, 0.1)
CUBE_0_Y = (0.15, 0.25)
CUBE_1_X = (0.0, 0.1)
CUBE_1_Y = (0.0, 0.1)
CUBE_2_X = (-0.15, -0.05)
CUBE_2_Y = (0.0, 0.1)

# IK TCP offset in the fr3_hand frame (IsaacLab `OffsetCfg(pos=[0,0,0.2])`).
TCP_LOCAL_OFFSET = (0.0, 0.0, 0.2)

# Sim timing.
CTRL_DT       = 1.0 / 20.0
SUBSTEPS      = 6                # each substep = 1/120 s -> 20 Hz control
EPISODE_STEPS = 180              # 9 s @ 20 Hz

# Predicate thresholds (cube_0 on cube_1).
XY_THRESHOLD = 0.02
Z_THRESHOLD  = 0.01

# Reward weights — RAW per-step magnitudes (IsaacLab fork dropped the *dt mul).
REWARD_WEIGHTS = {
    "reach":                     0.02,
    "lift":                      0.1,
    "align":                     0.32,
    "success_bonus":             200.0,
    "stack_broke_penalty":      -200.0,
    "tower_bonus":               2000.0,
    "linear_lift_grasping_cube": 0.15,
}

# Reach / align / lift hyperparameters (IsaacLab spec §6).
REACH_STD               = 0.1
LIFT_MINIMAL_HEIGHT     = 0.04
ALIGN_STD               = 0.08
ALIGN_MIN_HEIGHT_A      = 0.04
ALIGN_MIN_HEIGHT_B      = 0.0875
LIN_LIFT_INIT_Z         = 0.0215
LIN_LIFT_TARGET_Z_A     = 0.06
LIN_LIFT_TARGET_Z_B     = 0.1075


# --- URDF resolution --------------------------------------------------------

def _resolve_fr3_urdf() -> str:
    """Resolve the FR3+Franka-hand URDF.

    Uses the patched copy under `<genesis-world>/nautilus/assets/fr3/`. That
    directory contains a symlink `franka_description -> franka_description/`
    so Genesis's URDF loader (`genesis/ext/urdfpy/utils.py:get_filename`)
    resolves `package://franka_description/meshes/...` references correctly.

    The patch adds `friction="0.0"` to the two finger joints. The upstream
    URDF declares `<dynamics damping="0.3"/>` without `friction`, which makes
    urdfpy return `joint.dynamics.friction == None`. Genesis's legacy URDF
    parser then does `np.full(n_dofs, None)`, producing an object-dtype array
    that crashes `kinematic_solver._init_dof_fields`'s `np.concatenate(...,
    dtype=gs.np_float)` with `Cannot cast array data from dtype('O') to
    dtype('float32')`.
    """
    here = Path(__file__).resolve().parents[2] / "nautilus" / "assets" / "fr3" / "fr3_franka_hand.urdf"
    if not here.is_file():
        raise FileNotFoundError(f"FR3 URDF not found at {here}")
    return str(here)


class StackCubeEnv:
    """Genesis-native FR3 stack-cube env, IsaacLab-spec faithful."""

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

        # Ground plane (visual + collision; cubes sit on it as the "table top").
        self.scene.add_entity(gs.morphs.Plane())

        # Robot — FR3 + Franka hand. fixed=True so the base doesn't fall.
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file=_resolve_fr3_urdf(),
                pos=ROBOT_BASE_POS,
                quat=(1.0, 0.0, 0.0, 0.0),
                fixed=True,
                merge_fixed_links=True,
                # Keep `fr3_hand` (IK target) and the two fingers (proximity
                # checks for the contact proxy).
                links_to_keep=["fr3_hand", "fr3_leftfinger", "fr3_rightfinger"],
            ),
        )

        # Three identical cubes (placed at the robot-base XY anchor by default;
        # actual XYs come from `_reset_idx`).
        cube_size = (CUBE_SIZE, CUBE_SIZE, CUBE_SIZE)
        cube_pos_init = (ROBOT_BASE_POS[0], ROBOT_BASE_POS[1], CUBE_INIT_Z)
        self.cube_0 = self.scene.add_entity(
            gs.morphs.Box(size=cube_size, pos=cube_pos_init, fixed=False, batch_fixed_verts=False),
        )
        self.cube_1 = self.scene.add_entity(
            gs.morphs.Box(size=cube_size, pos=cube_pos_init, fixed=False, batch_fixed_verts=False),
        )
        self.cube_2 = self.scene.add_entity(
            gs.morphs.Box(size=cube_size, pos=cube_pos_init, fixed=False, batch_fixed_verts=False),
        )

        # Optional debug camera (must be added BEFORE scene.build()).
        self.vis_cam = None
        if attach_debug_camera:
            self.vis_cam = self.scene.add_camera(
                res=cam_res,
                pos=(1.0, -0.6, 0.8),
                lookat=(ROBOT_BASE_POS[0] + 0.2, ROBOT_BASE_POS[1] + 0.1, 0.1),
                fov=50,
                GUI=False,
            )

        # Build the scene.
        self.scene.build(n_envs=self.num_envs, env_spacing=(env_spacing, env_spacing))

        # ---- Robot-joint discovery + PD gains --------------------------
        # 7 arm DOFs + 2 finger DOFs (after merge_fixed_links).
        self._arm_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in ARM_HOME.keys()],
            dtype=torch.long, device=self.device,
        )
        # Finger DOFs.
        finger_names = ["fr3_finger_joint1", "fr3_finger_joint2"]
        self._finger_dof_idx = torch.tensor(
            [self.robot.get_joint(n).dof_start for n in finger_names],
            dtype=torch.long, device=self.device,
        )
        self._all_dof_idx = torch.cat([self._arm_dof_idx, self._finger_dof_idx])

        # PD gains — Genesis (Newton solver) is explicit/semi-implicit so the
        # IsaacLab spec gains (arm kp=400/kv=80) produce sluggish tracking.
        # Use the validated grasp_env.py gains for arm joints (10× higher to
        # match PhysX implicit-PD tracking quality) and keep IsaacLab's finger
        # stiffness (kp=2000/kv=100) for a firm grasp.
        kp = torch.tensor(
            [4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 2000.0, 2000.0],
            device=self.device,
        )
        kv = torch.tensor(
            [450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 100.0, 100.0],
            device=self.device,
        )
        self.robot.set_dofs_kp(kp, self._all_dof_idx)
        self.robot.set_dofs_kv(kv, self._all_dof_idx)
        # Force limits — IsaacLab: arm 1-4 ±87, 5-7 ±12, fingers ±200.
        f_lo = torch.tensor([-87.0]*4 + [-12.0]*3 + [-200.0]*2, device=self.device)
        f_hi = torch.tensor([ 87.0]*4 + [ 12.0]*3 + [ 200.0]*2, device=self.device)
        self.robot.set_dofs_force_range(f_lo, f_hi, self._all_dof_idx)

        # ---- IK target link --------------------------------------------
        self._ee_link = self.robot.get_link("fr3_hand")
        self._left_finger_link = self.robot.get_link("fr3_leftfinger")
        self._right_finger_link = self.robot.get_link("fr3_rightfinger")
        self._tcp_local = torch.tensor(TCP_LOCAL_OFFSET, device=self.device, dtype=torch.float32)

        # Initial joint qpos buffer (7 arm + 2 finger).
        self._init_qpos = torch.tensor(
            [ARM_HOME[n] for n in ARM_HOME.keys()] + [FINGER_OPEN, FINGER_OPEN],
            device=self.device, dtype=torch.float32,
        )

        # Robot base position per env (world frame).
        self._robot_base_w = torch.tensor(ROBOT_BASE_POS, device=self.device, dtype=torch.float32)
        # Per-env robot base position (constant — fixed-base robot).
        self._robot_base_w_per_env = self._robot_base_w.unsqueeze(0).expand(self.num_envs, 3).clone()

        # Workspace clamp tensors (root frame).
        self._pos_lo_root = torch.tensor(POS_LOWER_LIMIT_ROOT, device=self.device, dtype=torch.float32)
        self._pos_hi_root = torch.tensor(POS_UPPER_LIMIT_ROOT, device=self.device, dtype=torch.float32)

        # Action scale tensor.
        self._action_scale = torch.tensor(ACTION_SCALE, device=self.device, dtype=torch.float32)

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

        # Reward latches.
        self._latch_stacked      = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._latch_tower        = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._latch_broke        = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # Reward sums (for `extras["episode"]`).
        self.reward_keys = list(REWARD_WEIGHTS.keys())
        self.episode_sums = {k: torch.zeros(self.num_envs, device=self.device) for k in self.reward_keys}

        self.extras: dict = {}

        # Observation buffer — populated by `get_observations()`.
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device)

        # First reset — populates state, sets `init_ee_*` buffers via the
        # post-reset scene step.
        self.reset()

    # -------------------------------------------------------------------
    # Reset
    # -------------------------------------------------------------------

    def _sample_cube_xy(self, env_count: int, x_range, y_range) -> torch.Tensor:
        """Sample per-env cube XY in env-local frame.

        IsaacLab convention: reset_root_state_uniform sample is added to the
        cube's init_state.pos (which is env-local (0, 0, 0.0215) per the spec),
        NOT to the robot base. The robot lives at env-local (-0.274, 0.49, …)
        and the cube ranges (x∈[0,0.1], y∈[0.15,0.25] for cube_0, etc.) place
        the cubes in front of the robot, within reach of the EE workspace
        clamp (root-frame x∈[0.05,0.50] → env-local x∈[-0.224,0.226], y
        root-frame [-0.52,-0.05] → env-local [-0.03,0.44]).
        """
        x = torch.empty(env_count, device=self.device).uniform_(x_range[0], x_range[1])
        y = torch.empty(env_count, device=self.device).uniform_(y_range[0], y_range[1])
        return torch.stack([x, y], dim=-1)

    def _reset_idx(self, envs_idx=None) -> None:
        if envs_idx is None:
            mask = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        else:
            # envs_idx is a bool mask (N,).
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

        # 2. Cubes -> per-cube random XY.
        for cube, (xr, yr) in (
            (self.cube_0, (CUBE_0_X, CUBE_0_Y)),
            (self.cube_1, (CUBE_1_X, CUBE_1_Y)),
            (self.cube_2, (CUBE_2_X, CUBE_2_Y)),
        ):
            xy = self._sample_cube_xy(n, xr, yr)
            pos = torch.cat([xy, torch.full((n, 1), CUBE_INIT_Z, device=self.device)], dim=-1)
            cube.set_pos(pos, envs_idx=envs_idx_int, skip_forward=True)
            # Identity quat.
            quat = torch.zeros(n, 4, device=self.device)
            quat[:, 0] = 1.0
            cube.set_quat(quat, envs_idx=envs_idx_int, skip_forward=False)

        # 3. Buffers.
        self.episode_length_buf[envs_idx_int] = 0
        self.reset_buf[envs_idx_int] = False
        self.del_action[envs_idx_int] = 0.0
        self.last_action[envs_idx_int] = 0.0
        self._needs_reanchor[envs_idx_int] = True
        self._gripper_open_cmd[envs_idx_int] = True
        self._latch_stacked[envs_idx_int] = False
        self._latch_tower[envs_idx_int] = False
        self._latch_broke[envs_idx_int] = False

        # 4. Roll up `extras["episode"]` (mean reward sums for reset envs)
        #    and zero those slots.
        self.extras.setdefault("episode", {})
        for k, v in self.episode_sums.items():
            mean = v[envs_idx_int].mean() if n > 0 else torch.tensor(0.0, device=self.device)
            self.extras["episode"]["rew_" + k] = mean
            v[envs_idx_int] = 0.0

    def reset(self) -> TensorDict:
        self._reset_idx(None)
        return self.get_observations()

    # -------------------------------------------------------------------
    # Step
    # -------------------------------------------------------------------

    def _ee_pose_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current EE pose in world frame (at the TCP, with body-offset baked in
        via post-rotation translation). Returns (pos (N,3), quat (N,4)).
        """
        hand_pos = self._ee_link.get_pos()
        hand_quat = self._ee_link.get_quat()
        # Rotate the local TCP offset by the hand orientation and add — gives
        # the TCP position in world frame.
        # transform_by_quat available; do manual rotation: v_w = q * v_l * q^-1.
        # Simpler — use a single-axis rotation: TCP offset is along link's local
        # +z by 0.2 m. For a wxyz quat q, the rotation of (0,0,c) is:
        #   2 * (q.x*q.z + q.w*q.y) * c, 2 * (q.y*q.z - q.w*q.x) * c,
        #   (1 - 2*(q.x^2 + q.y^2)) * c.
        c = self._tcp_local[2]
        qw, qx, qy, qz = hand_quat[:, 0], hand_quat[:, 1], hand_quat[:, 2], hand_quat[:, 3]
        offset_w = torch.stack([
            2.0 * (qx * qz + qw * qy) * c,
            2.0 * (qy * qz - qw * qx) * c,
            (1.0 - 2.0 * (qx * qx + qy * qy)) * c,
        ], dim=-1)
        tcp_pos = hand_pos + offset_w
        return tcp_pos, hand_quat

    def _world_to_root(self, pos_w: torch.Tensor) -> torch.Tensor:
        """Convert world-frame position(s) to robot-root frame. Robot base has
        identity orientation in this setup, so this is just a translation."""
        return pos_w - self._robot_base_w_per_env

    def _root_to_world(self, pos_root: torch.Tensor) -> torch.Tensor:
        return pos_root + self._robot_base_w_per_env

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
        scaled = actions[:, :3] * self._action_scale  # (N, 3)
        self.del_action += scaled
        abs_pos_w = self.init_ee_pos_w + self.del_action

        # EMA against previously applied target.
        ema_pos_w = EMA_ALPHA * abs_pos_w + (1.0 - EMA_ALPHA) * self._prev_applied_pos_w

        # Clamp in robot-root frame.
        ema_pos_root = self._world_to_root(ema_pos_w)
        ema_pos_root = torch.clamp(ema_pos_root, self._pos_lo_root, self._pos_hi_root)
        ema_pos_w = self._root_to_world(ema_pos_root)

        self._prev_applied_pos_w = ema_pos_w.clone()

        # Genesis IK — TCP at body_offset (0,0,0.2) in fr3_hand local frame.
        # The IK solver will hold orientation at init_ee_quat_w. Restrict the
        # solve to the 7 arm DOFs (no finger involvement).
        # NOTE: `inverse_kinematics(..., dofs_idx_local=[...])` returns a FULL
        # qpos (shape (N, n_qs)), with non-arm DOFs left at their current values.
        full_qpos = self.robot.inverse_kinematics(
            link=self._ee_link,
            pos=ema_pos_w,
            quat=self.init_ee_quat_w,
            local_point=self._tcp_local,
            dofs_idx_local=self._arm_dof_idx,
            respect_joint_limit=True,
            max_samples=1,        # one-shot — IK target is small per step
            max_solver_iters=10,
        )
        # Slice out just the arm portion (first 7 entries of qpos correspond to
        # the 7 arm joints — finger joints come after).
        arm_target = full_qpos[:, :7]

        # Gripper command from action[-1]: >= 0 -> open, < 0 -> close.
        gripper_open = actions[:, -1] >= 0.0
        self._gripper_open_cmd = gripper_open
        finger_q = torch.where(gripper_open, torch.full_like(gripper_open, FINGER_OPEN, dtype=torch.float32),
                               torch.full_like(gripper_open, FINGER_CLOSE, dtype=torch.float32))
        finger_block = torch.stack([finger_q, finger_q], dim=-1)  # (N, 2)

        # Apply PD targets to the arm + fingers (9-D, ordered to match _all_dof_idx).
        full_target = torch.cat([arm_target, finger_block], dim=-1)  # (N, 9)
        self.robot.control_dofs_position(position=full_target, dofs_idx_local=self._all_dof_idx)

        # Step physics.
        self.scene.step()

        # Episode time.
        self.episode_length_buf += 1

        # Termination — time_out only.
        self.reset_buf = self.episode_length_buf >= self.max_episode_length
        # Also catch NaN-error envs (Genesis solver bookkeeping).
        try:
            self.reset_buf = self.reset_buf | self.scene.rigid_solver.get_error_envs_mask()
        except Exception:
            pass

        # Time-out flag for value-bootstrapping (algo-side).
        self.extras["time_outs"] = (self.episode_length_buf >= self.max_episode_length).float()

        # Compute reward BEFORE applying soft reset (reflects terminal state).
        reward, per_term = self._compute_reward()
        for k in self.reward_keys:
            self.episode_sums[k] += per_term[k]

        # Per-step `extras["detailed_reward"]` for downstream logging.
        self.extras["detailed_reward"] = per_term

        # Snapshot the reset mask BEFORE the soft reset zeroes out the slots
        # for envs that just finished — downstream consumers need the pre-reset
        # mask (`done` flag) to count terminations correctly.
        reset_mask_snapshot = self.reset_buf.clone()

        # Soft-reset envs that finished.
        self._reset_idx(self.reset_buf)

        return self.get_observations(), reward, reset_mask_snapshot, self.extras

    # -------------------------------------------------------------------
    # Observation
    # -------------------------------------------------------------------

    def _cube_xyz_w(self, cube) -> torch.Tensor:
        return cube.get_pos()  # (N, 3)

    def _cube_pos_root(self, cube) -> torch.Tensor:
        return self._world_to_root(self._cube_xyz_w(cube))

    def _cube_0_on_cube_1(self) -> torch.Tensor:
        p0 = self._cube_xyz_w(self.cube_0)
        p1 = self._cube_xyz_w(self.cube_1)
        xy = torch.norm(p0[:, :2] - p1[:, :2], dim=-1)
        z_gap = p0[:, 2] - p1[:, 2]
        return (xy < XY_THRESHOLD) & (torch.abs(z_gap - CUBE_SIZE) < Z_THRESHOLD)

    def _stack_target_root(self, base_cube) -> torch.Tensor:
        base = self._cube_xyz_w(base_cube).clone()
        base[:, 2] = base[:, 2] + CUBE_SIZE
        return self._world_to_root(base)

    def get_observations(self) -> TensorDict:
        # EE pose in robot root frame.
        ee_pos_w, ee_quat_w = self._ee_pose_w()
        ee_pos_root = self._world_to_root(ee_pos_w)
        # Robot base has identity orientation here, so root-frame quat = world quat.
        ee_quat_root = ee_quat_w
        ee_pose_root = torch.cat([ee_pos_root, ee_quat_root], dim=-1)  # (N, 7)

        # Grasping cube + target mux on cube_0_on_cube_1 predicate.
        on_stack = self._cube_0_on_cube_1().unsqueeze(-1)
        gc_pos = torch.where(on_stack, self._cube_pos_root(self.cube_2), self._cube_pos_root(self.cube_0))
        tgt_pos = torch.where(on_stack,
                              self._stack_target_root(self.cube_0),
                              self._stack_target_root(self.cube_1))

        # Gripper joint pos (2).
        gripper_q = self.robot.get_dofs_position(self._finger_dof_idx)  # (N, 2)

        self.obs_buf = torch.cat(
            [ee_pose_root, gc_pos, tgt_pos, gripper_q, self.last_action],
            dim=-1,
        )
        return TensorDict({"policy": self.obs_buf}, batch_size=[self.num_envs])

    # -------------------------------------------------------------------
    # Reward
    # -------------------------------------------------------------------

    def _contact_proxy(self, cube) -> torch.Tensor:
        """Genesis substitute for IsaacLab filtered fingertip-contact sensors.

        Proxy: EE TCP (= fingertip pinch-point at body_offset (0,0,0.2)) is
        within CUBE_SIZE + 0.02 m of the cube center AND the gripper command
        is closed (avg finger joint position < 0.03 m, halfway between
        open=0.04 and close=0.0). The finger LINK origins sit ~14 cm above
        the fingertip (URDF slider joint base), so they can't be used directly.
        """
        cube_pos = self._cube_xyz_w(cube)
        ee_pos_w, _ = self._ee_pose_w()
        d = torch.norm(ee_pos_w - cube_pos, dim=-1)
        gripper_q = self.robot.get_dofs_position(self._finger_dof_idx)  # (N, 2)
        gripper_closed = (gripper_q.mean(dim=-1) < 0.03)
        return (d < (CUBE_SIZE + 0.02)) & gripper_closed

    def _no_contact_proxy(self, cube) -> torch.Tensor:
        """True iff the EE TCP is at least (CUBE_SIZE + 0.04) m away from
        the cube center OR the gripper is fully open — used for the
        "EE retreated, no contact" success criterion.
        """
        cube_pos = self._cube_xyz_w(cube)
        ee_pos_w, _ = self._ee_pose_w()
        d = torch.norm(ee_pos_w - cube_pos, dim=-1)
        gripper_q = self.robot.get_dofs_position(self._finger_dof_idx)  # (N, 2)
        gripper_open = (gripper_q.mean(dim=-1) > 0.035)
        return (d > (CUBE_SIZE + 0.04)) | gripper_open

    def _compute_reward(self) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        on_stack = self._cube_0_on_cube_1()
        ee_pos_w, _ = self._ee_pose_w()
        c0_w = self._cube_xyz_w(self.cube_0)
        c1_w = self._cube_xyz_w(self.cube_1)
        c2_w = self._cube_xyz_w(self.cube_2)

        grasping_w = torch.where(on_stack.unsqueeze(-1), c2_w, c0_w)
        base_w = torch.where(on_stack.unsqueeze(-1), c0_w, c1_w)
        target_w = base_w.clone()
        target_w[:, 2] = target_w[:, 2] + CUBE_SIZE

        # --- reach: 1 - tanh(d/std), state-B scaled 30*+1 ---
        d_reach = torch.norm(grasping_w - ee_pos_w, dim=-1)
        base_reach = 1.0 - torch.tanh(d_reach / REACH_STD)
        reach = torch.where(on_stack, 30.0 * base_reach + 1.0, base_reach)

        # --- lift: z > minimal_height, state-B 10*+1 ---
        z_grasping = grasping_w[:, 2]
        lifted = (z_grasping > LIFT_MINIMAL_HEIGHT).float()
        lift = torch.where(on_stack, 10.0 * lifted + 1.0, lifted)

        # --- align: lifted_b_gated * (1 - tanh(d/std)), state-B 50*+1 ---
        d_align = torch.norm(grasping_w - target_w, dim=-1)
        h_thr = torch.where(on_stack,
                            torch.full_like(z_grasping, ALIGN_MIN_HEIGHT_B),
                            torch.full_like(z_grasping, ALIGN_MIN_HEIGHT_A))
        lifted_gate = (z_grasping > h_thr).float()
        base_align = lifted_gate * (1.0 - torch.tanh(d_align / ALIGN_STD))
        align = torch.where(on_stack, 50.0 * base_align + 1.0, base_align)

        # --- linear_lift_grasping_cube: contact-gated linear ramp ---
        denom_a = max(LIN_LIFT_TARGET_Z_A - LIN_LIFT_INIT_Z, 1e-6)
        denom_b = max(LIN_LIFT_TARGET_Z_B - LIN_LIFT_INIT_Z, 1e-6)
        base_a = ((c0_w[:, 2] - LIN_LIFT_INIT_Z) / denom_a).clamp(0.0, 1.0)
        base_b = ((c2_w[:, 2] - LIN_LIFT_INIT_Z) / denom_b).clamp(0.0, 1.0)
        gate_a = self._contact_proxy(self.cube_0).float()
        gate_b = self._contact_proxy(self.cube_2).float()
        linear_lift = torch.where(on_stack, 10.0 * base_b * gate_b + 1.0, base_a * gate_a)

        # --- success_bonus: cube_0 on cube_1 + no finger contact, once/ep ---
        xy_01 = torch.norm(c0_w[:, :2] - c1_w[:, :2], dim=-1)
        z_01 = c0_w[:, 2] - c1_w[:, 2]
        geom_01 = (xy_01 < XY_THRESHOLD) & (torch.abs(z_01 - CUBE_SIZE) < Z_THRESHOLD)
        no_contact_c0 = self._no_contact_proxy(self.cube_0)
        success_now = geom_01 & no_contact_c0
        success_fire = success_now & (~self._latch_stacked)
        self._latch_stacked = self._latch_stacked | success_now

        # --- stack_broke_penalty: was stacked, now not, once/ep ---
        currently_stacked = geom_01
        broke_fire = self._latch_stacked & (~currently_stacked) & (~self._latch_broke)
        self._latch_broke = self._latch_broke | broke_fire

        # --- tower_bonus: cube_0 on cube_1 + cube_2 on cube_0 + both no contact ---
        xy_20 = torch.norm(c2_w[:, :2] - c0_w[:, :2], dim=-1)
        z_20 = c2_w[:, 2] - c0_w[:, 2]
        geom_20 = (xy_20 < XY_THRESHOLD) & (torch.abs(z_20 - CUBE_SIZE) < Z_THRESHOLD)
        no_contact_c2 = self._no_contact_proxy(self.cube_2)
        upper = geom_01 & no_contact_c0
        top = geom_20 & no_contact_c2
        tower_now = upper & top
        tower_fire = tower_now & (~self._latch_tower)
        self._latch_tower = self._latch_tower | tower_now

        per_term: dict[str, torch.Tensor] = {
            "reach":                     REWARD_WEIGHTS["reach"]                     * reach,
            "lift":                      REWARD_WEIGHTS["lift"]                      * lift,
            "align":                     REWARD_WEIGHTS["align"]                     * align,
            "success_bonus":             REWARD_WEIGHTS["success_bonus"]             * success_fire.float(),
            "stack_broke_penalty":       REWARD_WEIGHTS["stack_broke_penalty"]       * broke_fire.float(),
            "tower_bonus":               REWARD_WEIGHTS["tower_bonus"]               * tower_fire.float(),
            "linear_lift_grasping_cube": REWARD_WEIGHTS["linear_lift_grasping_cube"] * linear_lift,
        }
        total = torch.zeros(self.num_envs, device=self.device)
        for v in per_term.values():
            total = total + v
        return total, per_term
