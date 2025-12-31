import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Tuple

import imageio
import numpy as np
import tqdm
import tyro
import mujoco  # fallback용으로 사용 (jacobian 계산 등)
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


# ======================================================================
# Args
# ======================================================================

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50

    # Object motion (applied every sim step)
    apply_motion: bool = True
    motion_object_name: str = ""  # If empty, use resolved target object
    motion_mode: str = "velocity"  # "position" or "velocity"
    motion_amp_xy: List[float] = dataclasses.field(default_factory=lambda: [1.0, 1.0])
    motion_freq_hz: float = 0.05

    # MPPI on/off
    use_mppi: bool = True

    # MPPI mode
    mppi_mode: str = "eef_kinematics"  # or "joint_torque"

    # MPPI params (kinematics-only defaults)
    mppi_horizon: int = 10
    mppi_num_samples: int = 32
    mppi_lambda: float = 1.0  # temperature
    mppi_noise_sigma: float = 0.02  # base sigma for xyz / orient; gripper noise scaled down
    w_ref: float = 1.0
    w_goal: float = 5.0
    w_reg: float = 0.1
    w_obs: float = 0.0  # keep 0 for now

    # Joint-torque MPPI options
    w_tau: float = 0.05  # torque regularization
    ee_body_name: str = "panda_hand"  # end-effector body/site name for FK/Jacobian
    tau_kp: float = 200.0
    tau_kd: float = 10.0

    # Phase thresholds (meters)
    d_far: float = 0.15
    d_close: float = 0.04

    video_out_path: str = "data/libero/videos"
    seed: int = 7  # Random Seed (for reproducibility)


# ======================================================================
# EEF Kinematics-only MPPI
# ======================================================================

class EEFKinematicsMPPIController:
    """Kinematics-only MPPI around pi0 action reference (no MuJoCo rollouts)."""

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dt: float,
        temperature: float,
        base_noise_sigma: float,
        weights: dict,
        d_far: float,
        d_close: float,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature
        self.base_noise_sigma = base_noise_sigma
        self.w_ref = weights["w_ref"]
        self.w_goal = weights["w_goal"]
        self.w_reg = weights["w_reg"]
        self.w_obs = weights["w_obs"]
        self.d_far = d_far
        self.d_close = d_close

    def plan(
        self,
        x0: np.ndarray,
        A_ref: np.ndarray,
        p_obj: np.ndarray,
        v_obj: np.ndarray,
        gripper_cmd_ref: float,
    ) -> np.ndarray:
        """
        Kinematics-only MPPI around pi0 reference actions.

        Args:
            x0: (7,) current state [p(3), aa(3), g(1)]
            A_ref: [H,7] reference from pi0
            p_obj: (3,) current object position
            v_obj: (3,) estimated object velocity
            gripper_cmd_ref: scalar gripper command of first ref step
        Returns:
            u_opt: (7,) to pass to env.step
        """
        assert A_ref.ndim == 2 and A_ref.shape[1] == 7
        H_ref = A_ref.shape[0]
        K = min(self.horizon, H_ref)
        u_ref = A_ref[:K]

        # Phase gating
        p_ee = x0[:3]
        d = np.linalg.norm(p_ee - p_obj)
        phase = self._compute_phase(d, gripper_cmd_ref)
        w_goal, w_ref, w_reg, noise_scale = self._phase_weights_noise(phase)

        # Grasp phase: follow pi0
        if phase == "grasp":
            return u_ref[0]

        # Noise sigma per dim (gripper noise reduced)
        sigma_vec = np.ones(7, dtype=np.float32) * self.base_noise_sigma * noise_scale
        sigma_vec[-1] *= 0.2

        eps = np.random.normal(scale=sigma_vec, size=(self.num_samples, K, 7))
        costs = np.zeros(self.num_samples, dtype=np.float32)

        for j in range(self.num_samples):
            x = x0.copy()
            J_cost = 0.0
            for k in range(K):
                u_k = u_ref[k] + eps[j, k]
                x = self._dynamics_step(x, u_k)
                t_pred = (k + 1) * self.dt
                p_obj_pred = p_obj + v_obj * t_pred
                p_k = x[:3]
                L_goal = w_goal * np.sum((p_k - p_obj_pred) ** 2)
                L_ref = w_ref * np.sum((u_k - u_ref[k]) ** 2)
                L_reg = w_reg * np.sum(u_k ** 2)
                J_cost += L_goal + L_ref + L_reg  # w_obs == 0
            costs[j] = J_cost

        J_min = np.min(costs)
        weights = np.exp(-(costs - J_min) / max(self.temperature, 1e-6))
        weights = weights / (np.sum(weights) + 1e-8)
        u0_new = u_ref[0] + np.sum(weights[:, None] * eps[:, 0, :], axis=0)
        return u0_new

    def _dynamics_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """x_{k+1} = [p + dp, aa + daa, g_cmd]"""
        out = x.copy()
        out[:3] = x[:3] + u[:3]
        out[3:6] = x[3:6] + u[3:6]
        out[6] = u[6]
        return out

    def _compute_phase(self, d: float, g_cmd: float) -> str:
        if d <= self.d_close or g_cmd < -0.2:
            return "grasp"
        if d <= self.d_far:
            return "near"
        return "approach"

    def _phase_weights_noise(self, phase: str) -> Tuple[float, float, float, float]:
        w_goal = self.w_goal
        w_ref = self.w_ref
        w_reg = self.w_reg
        noise_scale = 1.0
        if phase == "approach":
            w_goal *= 1.5
            w_ref *= 0.5
            noise_scale *= 1.5
        elif phase == "near":
            w_goal *= 1.0
            w_ref *= 2.0
            noise_scale *= 0.5
        elif phase == "grasp":
            w_goal *= 0.5
            w_ref *= 10.0
            noise_scale *= 0.1
        return w_goal, w_ref, w_reg, noise_scale


# ======================================================================
# Joint Torque MPPI (Franka 7-DOF 중심, 선형화 기반)
# ======================================================================

class JointTorqueMujocoMPPIController:
    """
    Joint-torque MPPI using a simple joint-space dynamics model:
      q_{k+1}   = q_k   + dt * qdot_k
      qdot_{k+1}= qdot_k+ dt * qddot_k,  qddot_k ≈ tau_k   (M = I 근사)

    EEF 위치는 시작 시점에서 구한 (p0, J0)를 사용해
      p(q) ≈ p0 + J0 (q - q0)
    로 선형화해서 계산.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        dt: float,
        temperature: float,
        weights: dict,
        d_far: float,
        d_close: float,
        sim,
        robot,
        ee_name: str,
        tau_kp: float,
        tau_kd: float,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.dt = dt
        self.temperature = temperature

        self.w_goal = weights["w_goal"]
        self.w_ref = weights["w_ref"]
        self.w_tau = weights["w_tau"]
        self.w_obs = weights["w_obs"]

        self.d_far = d_far
        self.d_close = d_close

        self.sim = sim
        self.robot = robot
        self.model = getattr(sim, "model", None)
        self.data = getattr(sim, "data", None)

        # Robot DOF (Franka arm = 7)
        self.ndof = int(robot.dof)

        # robosuite에서 joint index (global qpos/qvel index) 가져오기
        # (버전에 따라 _ref_joint_pos_indexes / _ref_joint_vel_indexes 가 다를 수 있음)
        self.jnt_pos_idx = np.array(getattr(robot, "_ref_joint_pos_indexes"), dtype=int)
        self.jnt_vel_idx = np.array(getattr(robot, "_ref_joint_vel_indexes"), dtype=int)

        # EEF 이름
        if ee_name is not None and len(ee_name) > 0:
            self.ee_name = ee_name
        else:
            # robosuite Robot에 보통 eef_site_name / eef_link_name 등이 있음
            self.ee_name = getattr(robot, "eef_site_name", getattr(robot, "eef_link_name", None))

        # task-space PD gains
        self.tau_kp = tau_kp
        self.tau_kd = tau_kd

        # linearization holders
        self._p0 = None          # (3,)
        self._q0_arm = None      # (7,)
        self._J0 = None          # (3,7)

    # ------------------------ public API ------------------------ #

    def plan(
        self,
        q0_full: np.ndarray,
        qdot0_full: np.ndarray,
        A_ref: np.ndarray,
        p_obj: np.ndarray,
        v_obj: np.ndarray,
        gripper_cmd_ref: float,
    ) -> np.ndarray:
        """
        Args:
            q0_full, qdot0_full: full MuJoCo state (nv-sized), from env.env.sim.data
            A_ref: [H,7] VLA reference actions (delta-pos/orient/gripper)
            p_obj, v_obj: object position / velocity
        Returns:
            delta_chunk: (K,7) delta-EEF+gripper sequence mapped from torque plan
        """
        H_ref = A_ref.shape[0]
        K = min(self.horizon, H_ref)

        # 현재 EEF 위치 (진짜 sim에서 한 번만 읽음)
        p_ee0 = self._get_ee_pos_from_sim(q0_full, qdot0_full)
        d = np.linalg.norm(p_ee0 - p_obj)
        phase = self._compute_phase(d, gripper_cmd_ref)
        w_goal, w_ref, w_tau, noise_scale = self._phase_weights_noise(phase)

        # grasp phase면 그냥 pi0 따라가기
        if phase == "grasp":
            return A_ref[:K]

        # 선형화 세팅: p0, q0_arm, J0
        self._setup_linearization(q0_full, qdot0_full)

        q0_arm = q0_full[self.jnt_pos_idx]
        qdot0_arm = qdot0_full[self.jnt_vel_idx]

        # reference trajectory in EEF space (누적된 delta pos)
        p_ref = self._build_eef_ref(p_ee0, A_ref[:K, :3])
        # reference torque sequence (feedforward용)
        tau_ref = self._build_tau_ref(q0_arm, qdot0_arm, A_ref[:K])

        # noise: joint torque space (ndof)
        sigma_vec = np.ones(self.ndof, dtype=np.float32) * self.dt * noise_scale
        eps = np.random.normal(scale=sigma_vec, size=(self.num_samples, K, self.ndof))
        costs = np.zeros(self.num_samples, dtype=np.float32)

        # MPPI sampling
        for j in range(self.num_samples):
            q = q0_arm.copy()
            qdot = qdot0_arm.copy()
            J_cost = 0.0
            for k in range(K):
                tau_k = tau_ref[k] + eps[j, k]
                q, qdot = self._dynamics_step(q, qdot, tau_k)

                t_pred = (k + 1) * self.dt
                p_obj_pred = p_obj + v_obj * t_pred
                p_ee = self._eef_pos_linear(q)

                L_goal = w_goal * np.sum((p_ee - p_obj_pred) ** 2)
                L_ref = w_ref * np.sum((p_ee - p_ref[k]) ** 2)
                L_tau = w_tau * np.sum(tau_k ** 2)

                J_cost += L_goal + L_ref + L_tau
            costs[j] = J_cost

        J_min = np.min(costs)
        weights = np.exp(-(costs - J_min) / max(self.temperature, 1e-6))
        weights = weights / (np.sum(weights) + 1e-8)

        tau_new = tau_ref + np.sum(weights[:, None, None] * eps, axis=0)  # (K, ndof)

        # deterministic rollout으로 delta-EEF chunk 생성
        delta_chunk = np.zeros((K, 7), dtype=np.float32)
        q = q0_arm.copy()
        qdot = qdot0_arm.copy()
        p_prev = p_ee0.copy()
        for k in range(K):
            q, qdot = self._dynamics_step(q, qdot, tau_new[k])
            p_curr = self._eef_pos_linear(q)
            delta_pos = p_curr - p_prev
            delta_ori = A_ref[k][3:6]  # orientation delta는 pi0 그대로 사용
            gripper_cmd = A_ref[k][-1]
            delta_chunk[k] = np.concatenate([delta_pos, delta_ori, [gripper_cmd]])
            p_prev = p_curr

        return delta_chunk

    # ------------------------ internal helpers ------------------------ #

    def _setup_linearization(self, q_full: np.ndarray, qdot_full: np.ndarray):
        """현재 상태에서 p0, q0_arm, J0(3x7) 계산."""
        self._q0_arm = q_full[self.jnt_pos_idx].copy()
        self._p0 = self._get_ee_pos_from_sim(q_full, qdot_full)
        self._J0 = self._get_task_jacobian(q_full, qdot_full)

    def _get_ee_pos_from_sim(self, q_full: np.ndarray, qdot_full: np.ndarray) -> np.ndarray:
        """
        sim의 model/data를 잠깐 덮어써서 EEF position 읽고, 바로 원래 상태 복원.
        """
        if self.model is None or self.data is None:
            # robosuite Robot API가 있으면 그걸 쓰는 것도 가능
            if hasattr(self.robot, "ee_pose"):
                pos, _ = self.robot.ee_pose()
                return np.array(pos, dtype=np.float32)
            raise RuntimeError("No model/data in sim to get ee position.")

        # snapshot
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        try:
            self.data.qpos[:] = q_full
            self.data.qvel[:] = qdot_full
            if hasattr(self.sim, "forward"):
                self.sim.forward()

            # site 우선
            pos = None
            if hasattr(self.model, "site_name2id"):
                try:
                    sid = self.model.site_name2id(self.ee_name)
                    pos = self.data.site_xpos[sid]
                except Exception:
                    pos = None
            if pos is None and hasattr(self.model, "body_name2id"):
                bid = self.model.body_name2id(self.ee_name)
                pos = self.data.body_xpos[bid]
            if pos is None:
                raise RuntimeError(f"Cannot find ee site/body with name {self.ee_name}")
            return np.array(pos, dtype=np.float32)
        finally:
            # restore
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            if hasattr(self.sim, "forward"):
                self.sim.forward()

    def _get_task_jacobian(self, q_full: np.ndarray, qdot_full: np.ndarray) -> np.ndarray:
        """
        EEF Jacobian J0 (3x7) 계산.
        1) 가능하면 robosuite Robot API 사용
        2) 안 되면 mujoco.mj_jacSite / mj_jacBody fallback
        """
        # (1) robosuite Robot method 시도
        if hasattr(self.robot, "jacobian"):
            try:
                J6 = self.robot.jacobian(self.ee_name)  # (6, ndof) 가정
                return np.array(J6[:3, :self.ndof], dtype=np.float32)
            except Exception:
                pass
        if hasattr(self.robot, "robot_jacobian"):
            try:
                J6 = self.robot.robot_jacobian(self.ee_name)
                return np.array(J6[:3, :self.ndof], dtype=np.float32)
            except Exception:
                pass

        # (2) mujoco low-level fallback
        if self.model is None or self.data is None:
            raise RuntimeError("Cannot compute Jacobian: no model/data and no robot jacobian method.")

        # snapshot
        qpos_backup = self.data.qpos.copy()
        qvel_backup = self.data.qvel.copy()

        try:
            self.data.qpos[:] = q_full
            self.data.qvel[:] = qdot_full
            if hasattr(self.sim, "forward"):
                self.sim.forward()

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)

            # site 우선 시도
            use_site = True
            sid = -1
            if hasattr(self.model, "site_name2id"):
                try:
                    sid = self.model.site_name2id(self.ee_name)
                except Exception:
                    use_site = False

            if use_site and sid >= 0:
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            else:
                # body fallback
                bid = self.model.body_name2id(self.ee_name)
                mujoco.mj_jacBody(self.model, self.data, jacp, jacr, bid)

            # joint DOF(7개)에 해당하는 column만 추출
            Jpos_arm = jacp[:, self.jnt_vel_idx]
            return np.array(Jpos_arm, dtype=np.float32)

        finally:
            self.data.qpos[:] = qpos_backup
            self.data.qvel[:] = qvel_backup
            if hasattr(self.sim, "forward"):
                self.sim.forward()

    def _eef_pos_linear(self, q_arm: np.ndarray) -> np.ndarray:
        """p(q) ≈ p0 + J0 (q - q0)."""
        return self._p0 + self._J0 @ (q_arm - self._q0_arm)

    def _build_eef_ref(self, p_start: np.ndarray, dp_seq: np.ndarray) -> np.ndarray:
        """pi0 delta-pos 누적해서 absolute EEF reference trajectory 생성."""
        K = dp_seq.shape[0]
        prefs = np.zeros((K, 3), dtype=np.float32)
        acc = p_start.copy()
        for k in range(K):
            acc = acc + dp_seq[k]
            prefs[k] = acc
        return prefs

    def _build_tau_ref(self, q0_arm: np.ndarray, qdot0_arm: np.ndarray, A_ref: np.ndarray) -> np.ndarray:
        """
        A_ref의 delta-pos를 이용해서 task-space PD + J0^T 를 거친 feedforward torque 시퀀스 생성.
        """
        K = A_ref.shape[0]
        tau_ref = np.zeros((K, self.ndof), dtype=np.float32)

        q = q0_arm.copy()
        qdot = qdot0_arm.copy()
        for k in range(K):
            dp = A_ref[k, :3]
            tau_k = self._eef_delta_to_tau(q, qdot, dp)
            tau_ref[k] = tau_k
            q, qdot = self._dynamics_step(q, qdot, tau_k)

        return tau_ref

    def _eef_delta_to_tau(self, q_arm: np.ndarray, qdot_arm: np.ndarray, dp: np.ndarray) -> np.ndarray:
        """
        Task-space PD:
          err = dp
          v_ee ≈ J0 qdot
          F = Kp * err - Kd * v_ee
          tau = J0^T F
        """
        v_ee = self._J0 @ qdot_arm
        err = dp
        F = self.tau_kp * err - self.tau_kd * v_ee
        tau = self._J0.T @ F
        return tau.astype(np.float32)

    def _dynamics_step(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        매우 단순한 joint-space dynamics 근사:
          qddot ≈ tau  (M = I, bias = 0 가정)
        """
        qddot = tau  # 필요하면 1 / inertia_scale 등으로 rescale 가능
        q_next = q + qdot * self.dt
        qdot_next = qdot + qddot * self.dt
        return q_next, qdot_next

    def _compute_phase(self, d: float, g_cmd: float) -> str:
        if d <= self.d_close or g_cmd < -0.2:
            return "grasp"
        if d <= self.d_far:
            return "near"
        return "approach"

    def _phase_weights_noise(self, phase: str) -> Tuple[float, float, float, float]:
        w_goal = self.w_goal
        w_ref = self.w_ref
        w_tau = self.w_tau
        noise_scale = 1.0
        if phase == "approach":
            w_goal *= 1.5
            w_ref *= 0.5
            noise_scale *= 1.5
        elif phase == "near":
            w_goal *= 1.0
            w_ref *= 2.0
            noise_scale *= 0.5
        elif phase == "grasp":
            w_goal *= 0.5
            w_ref *= 10.0
            noise_scale *= 0.1
        return w_goal, w_ref, w_tau, noise_scale


# ======================================================================
# eval_libero (MPPI 포함)
# ======================================================================

def eval_libero(args: Args) -> None:
    np.random.seed(args.seed)
    _validate_motion_args(args)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }.get(args.task_suite_name, 300)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    controller = None  # created lazily when we know dt

    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        target_obj_name = _resolve_target_object(env)
        motion_obj_name = args.motion_object_name or target_obj_name
        base_qpos = (
            _get_object_qpos(env, motion_obj_name)
            if args.apply_motion and motion_obj_name is not None
            else None
        )

        # build MPPI once (need dt, robot)
        if controller is None and args.use_mppi:
            dt = 1.0 / env.env.control_freq if hasattr(env.env, "control_freq") else env.env.sim.model.opt.timestep
            if args.mppi_mode == "joint_torque":
                # robosuite 기반 env라면 robots[0] 이 Panda일 것
                robot = env.env.robots[0]
                controller = JointTorqueMujocoMPPIController(
                    horizon=args.mppi_horizon,
                    num_samples=args.mppi_num_samples,
                    dt=dt,
                    temperature=args.mppi_lambda,
                    weights={
                        "w_ref": args.w_ref,
                        "w_goal": args.w_goal,
                        "w_tau": args.w_tau,
                        "w_obs": args.w_obs,
                    },
                    d_far=args.d_far,
                    d_close=args.d_close,
                    sim=env.env.sim,
                    robot=robot,
                    ee_name=args.ee_body_name,
                    tau_kp=args.tau_kp,
                    tau_kd=args.tau_kd,
                )
            else:
                controller = EEFKinematicsMPPIController(
                    horizon=args.mppi_horizon,
                    num_samples=args.mppi_num_samples,
                    dt=dt,
                    temperature=args.mppi_lambda,
                    base_noise_sigma=args.mppi_noise_sigma,
                    weights={
                        "w_ref": args.w_ref,
                        "w_goal": args.w_goal,
                        "w_reg": args.w_reg,
                        "w_obs": args.w_obs,
                    },
                    d_far=args.d_far,
                    d_close=args.d_close,
                )

        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []

            prev_obj_pos = _get_obj_pos(env, target_obj_name)
            prev_obj_time = 0.0
            dt = 1.0 / env.env.control_freq if hasattr(env.env, "control_freq") else env.env.sim.model.opt.timestep

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # let objects fall initially
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Apply external motion to target object (sine in xy)
                    if args.apply_motion and motion_obj_name is not None and base_qpos is not None:
                        _apply_xy_sine_motion(
                            env=env,
                            obj_name=motion_obj_name,
                            base_qpos=base_qpos,
                            t_sec=t * dt,
                            amp_xy=args.motion_amp_xy,
                            freq_hz=args.motion_freq_hz,
                            mode=args.motion_mode,
                        )

                    # gather images for replay
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    replay_images.append(img)

                    # replan pi0 when plan is empty
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[: args.replan_steps])
                        A_ref = np.array(action_chunk, dtype=np.float32)

                    # estimate object velocity (finite difference)
                    curr_obj_pos = _get_obj_pos(env, target_obj_name)
                    curr_time = t * dt
                    obj_vel = (curr_obj_pos - prev_obj_pos) / max(curr_time - prev_obj_time, 1e-6)
                    prev_obj_pos, prev_obj_time = curr_obj_pos, curr_time

                    # Hybrid control: MPPI around A_ref
                    if args.use_mppi and controller is not None:
                        x0 = np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                np.array([np.mean(obs["robot0_gripper_qpos"])]),
                            )
                        )
                        gripper_cmd_ref = float(A_ref[0][-1])
                        if isinstance(controller, JointTorqueMujocoMPPIController):
                            # torque MPPI: full joint state from sim
                            q0_full = env.env.sim.data.qpos.copy()
                            qdot0_full = env.env.sim.data.qvel.copy()
                            delta_chunk = controller.plan(
                                q0_full=q0_full,
                                qdot0_full=qdot0_full,
                                A_ref=A_ref,
                                p_obj=curr_obj_pos,
                                v_obj=obj_vel,
                                gripper_cmd_ref=gripper_cmd_ref,
                            )
                            action_plan.clear()
                            action_plan.extend([d.tolist() for d in delta_chunk])
                            u_opt = action_plan.popleft()
                        else:
                            u_opt = controller.plan(
                                x0=x0,
                                A_ref=A_ref,
                                p_obj=curr_obj_pos,
                                v_obj=obj_vel,
                                gripper_cmd_ref=gripper_cmd_ref,
                            )
                    else:
                        u_opt = action_plan[0]

                    obs, reward, done, info = env.step(np.asarray(u_opt).tolist())
                    action_plan.popleft()
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


# ======================================================================
# Helpers
# ======================================================================

def _get_libero_env(task, resolution, seed):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = math.sqrt(max(1.0 - quat[3] * quat[3], 0.0))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _get_obj_pos(env, obj_name: str) -> np.ndarray:
    body_id = env.env.obj_body_id[obj_name]
    return np.array(env.env.sim.data.body_xpos[body_id])


def _resolve_target_object(env) -> str:
    """Pick first non-fixture obj_of_interest; fallback to first obj_of_interest."""
    if hasattr(env, "obj_of_interest") and env.obj_of_interest:
        for name in env.obj_of_interest:
            try:
                if not env.env.is_fixture(name):
                    return name
            except Exception:
                continue
        return env.obj_of_interest[0]
    # Last resort: pick first key from objects_dict
    if hasattr(env.env, "objects_dict") and env.env.objects_dict:
        return list(env.env.objects_dict.keys())[0]
    raise ValueError("No target object could be resolved from the environment.")


def _get_object_joint_name(env, obj_name):
    if obj_name not in env.env.objects_dict:
        raise ValueError(f"Object '{obj_name}' is not a movable object.")
    obj = env.env.objects_dict[obj_name]
    if not obj.joints:
        raise ValueError(f"Object '{obj_name}' has no joints to control.")
    return obj.joints[-1]


def _get_object_qpos(env, obj_name):
    joint = _get_object_joint_name(env, obj_name)
    return env.env.sim.data.get_joint_qpos(joint).copy()


def _set_object_qpos(env, obj_name, qpos):
    joint = _get_object_joint_name(env, obj_name)
    sim = env.env.sim
    sim.data.set_joint_qpos(joint, qpos)
    # Zero the corresponding joint velocity to avoid residual drift
    try:
        qvel_addr = sim.model.get_joint_qvel_addr(joint)
        if isinstance(qvel_addr, (list, tuple, np.ndarray)):
            sim.data.qvel[qvel_addr[0]: qvel_addr[-1] + 1] = 0
        else:
            sim.data.qvel[qvel_addr] = 0
    except Exception:
        sim.data.qvel[:] = 0
    sim.forward()


def _set_object_qvel(env, obj_name, qvel):
    joint = _get_object_joint_name(env, obj_name)
    env.env.sim.data.set_joint_qvel(joint, qvel)
    env.env.sim.forward()


def _apply_xy_sine_motion(env, obj_name, base_qpos, t_sec, amp_xy, freq_hz, mode):
    # No-op if amplitude is effectively zero
    if abs(amp_xy[0]) < 1e-6 and abs(amp_xy[1]) < 1e-6:
        return

    omega = 2.0 * math.pi * freq_hz
    dx = amp_xy[0] * math.sin(omega * t_sec)
    dy = amp_xy[1] * math.cos(omega * t_sec)

    if mode == "position":
        qpos = base_qpos.copy()
        qpos[0] = base_qpos[0] + dx
        qpos[1] = base_qpos[1] + dy
        _set_object_qpos(env, obj_name, qpos)
        return

    if mode == "velocity":
        vx = amp_xy[0] * omega * math.cos(omega * t_sec)
        vy = -amp_xy[1] * omega * math.sin(omega * t_sec)
        qvel = np.zeros(6, dtype=np.float32)
        qvel[0] = vx
        qvel[1] = vy
        _set_object_qvel(env, obj_name, qvel)
        return

    raise ValueError(f"Unknown motion_mode: {mode}")


def _validate_motion_args(args: Args) -> None:
    if len(args.motion_amp_xy) != 2:
        raise ValueError("motion_amp_xy must have 2 floats (x, y amplitude).")
    if args.motion_mode not in ["position", "velocity"]:
        raise ValueError("motion_mode must be either 'position' or 'velocity'.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
