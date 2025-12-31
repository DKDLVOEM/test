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

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    # Model server parameters
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    # LIBERO environment-specific parameters
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
    mppi_lambda: float = 1.0  # temperature λ
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

    # Utils
    video_out_path: str = "data/libero/videos"
    seed: int = 7  # Random Seed (for reproducibility)


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


class JointTorqueMujocoMPPIController:
    """
    Joint-torque MPPI using MuJoCo dynamics (approximate) + FK for EEF tracking.

    - Control dimension: 7 (Franka arm DOFs)
    - Internal dynamics: full nv-state, but qddot for arm DOFs만 사용
      qddot_arm ≈ (tau_arm - bias_arm) * dof_invweight0_arm (diagonal M approximation)
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
        self.model = sim.model
        self.data = sim.data
        self.robot = robot

        # ---- Arm DOF 인덱스 설정 (7 DOF 기준) ----
        # robosuite Robot이 보통 다음 속성을 가짐:
        #   - arm_dof
        #   - _ref_joint_vel_indexes
        #   - _ref_joint_pos_indexes
        if hasattr(robot, "arm_dof"):
            self.arm_dof = robot.arm_dof
        else:
            # fallback
            self.arm_dof = 7

        if hasattr(robot, "_ref_joint_vel_indexes"):
            self.arm_vel_idx = np.array(robot._ref_joint_vel_indexes, dtype=int)
        elif hasattr(robot, "joint_indexes"):
            self.arm_vel_idx = np.array(robot.joint_indexes, dtype=int)
        else:
            self.arm_vel_idx = np.arange(self.arm_dof, dtype=int)

        if hasattr(robot, "_ref_joint_pos_indexes"):
            self.arm_pos_idx = np.array(robot._ref_joint_pos_indexes, dtype=int)
        else:
            self.arm_pos_idx = self.arm_vel_idx

        # ---- EE name → site / body index resolve (mujoco.* 호출 없이) ----
        self.ee_is_site = True
        self.ee_site_id = None
        self.ee_body_id = None
        try:
            self.ee_site_id = self.model.site_name2id(ee_name)
            self.ee_is_site = True
        except Exception:
            try:
                self.ee_body_id = self.model.body_name2id(ee_name)
                self.ee_is_site = False
            except Exception:
                raise ValueError(f"EE name '{ee_name}' not found as site or body in the model.")

        # ---- 대각 mass 근사용 inverse mass (dof_invweight0 사용) ----
        # MuJoCo에서 dof_invweight0는 대략 M^{-1}의 대각 성분에 해당.
        if hasattr(self.model, "dof_invweight0"):
            dof_inv = np.array(self.model.dof_invweight0, copy=True)
            self.arm_inv_mass = dof_inv[self.arm_vel_idx]
        else:
            # fallback: 단위 질량
            self.arm_inv_mass = np.ones(self.arm_dof, dtype=np.float64)

        self.tau_kp = tau_kp
        self.tau_kd = tau_kd

    # --------------------------------------------------------------------------
    # Public MPPI interface
    # --------------------------------------------------------------------------
    def plan(
        self,
        q0: np.ndarray,
        qdot0: np.ndarray,
        A_ref: np.ndarray,
        p_obj: np.ndarray,
        v_obj: np.ndarray,
        gripper_cmd_ref: float,
    ) -> np.ndarray:
        """
        Returns:
            delta_chunk: (K,7) delta-EEF+gripper sequence mapped from torque plan
                         (first element used for env.step).
        """
        H_ref = A_ref.shape[0]
        K = min(self.horizon, H_ref)

        # 현재 EE 포즈
        p_ee0 = self._get_ee_pos(q0, qdot0)
        d = np.linalg.norm(p_ee0 - p_obj)
        phase = self._compute_phase(d, gripper_cmd_ref)
        w_goal, w_ref, w_tau, noise_scale = self._phase_weights_noise(phase)

        # grasp phase: 그냥 pi0 따라가기
        if phase == "grasp":
            return A_ref[:K]

        # EEF reference trajectory (월드좌표)
        p_ref = self._build_eef_ref(p_ee0, A_ref[:K, :3])

        # tau_ref: EEF delta → tau_arm (7)
        tau_ref = self._build_tau_ref(q0, qdot0, A_ref[:K])

        # Noise in tau space (7 DOF)
        nu = self.arm_dof  # 7
        sigma_vec = np.ones(nu, dtype=np.float32) * self.dt * noise_scale
        eps = np.random.normal(scale=sigma_vec, size=(self.num_samples, K, nu))
        costs = np.zeros(self.num_samples, dtype=np.float32)

        # -------------------- MPPI rollouts --------------------
        for j in range(self.num_samples):
            q = q0.copy()
            qdot = qdot0.copy()
            J_cost = 0.0
            for k in range(K):
                tau_k = tau_ref[k] + eps[j, k]  # (7,)

                q, qdot = self._dynamics_step(q, qdot, tau_k)
                t_pred = (k + 1) * self.dt
                p_obj_pred = p_obj + v_obj * t_pred
                p_ee = self._get_ee_pos(q, qdot)

                L_goal = w_goal * np.sum((p_ee - p_obj_pred) ** 2)
                L_ref = w_ref * np.sum((p_ee - p_ref[k]) ** 2)
                L_tau = w_tau * np.sum(tau_k**2)
                J_cost += L_goal + L_ref + L_tau
            costs[j] = J_cost

        # -------------------- Weighting & update --------------------
        J_min = np.min(costs)
        weights = np.exp(-(costs - J_min) / max(self.temperature, 1e-6))
        weights = weights / (np.sum(weights) + 1e-8)
        tau_new = tau_ref + np.sum(weights[:, None, None] * eps, axis=0)  # (K,7)

        # -------------------- Deterministic rollout → delta-EEF chunk --------------------
        delta_chunk = np.zeros((K, 7), dtype=np.float32)
        q = q0.copy()
        qdot = qdot0.copy()
        p_prev = p_ee0.copy()
        for k in range(K):
            q, qdot = self._dynamics_step(q, qdot, tau_new[k])
            p_curr = self._get_ee_pos(q, qdot)
            delta_pos = p_curr - p_prev
            delta_ori = A_ref[k][3:6]  # orientation delta는 pi0 그대로 사용
            gripper_cmd = A_ref[k][-1]
            delta_chunk[k] = np.concatenate([delta_pos, delta_ori, [gripper_cmd]])
            p_prev = p_curr

        return delta_chunk

    # --------------------------------------------------------------------------
    # Internal dynamics & kinematics
    # --------------------------------------------------------------------------
    def _dynamics_step(self, q, qdot, tau_arm):
        """
        q (nv,), qdot (nv,), tau_arm (7,)
        - MuJoCo 전체 자유도 중 arm_vel_idx에만 torque 적용
        - qddot_arm ≈ (tau_arm - bias_arm) * dof_invweight0_arm (대각 근사)
        """
        model, data = self.model, self.data

        # 현재 상태 세팅
        data.qpos[:] = q
        data.qvel[:] = qdot
        self.sim.forward()

        # full bias (nv)
        bias_full = np.array(data.qfrc_bias, copy=True)

        # arm DOF에 대한 qddot 근사
        bias_arm = bias_full[self.arm_vel_idx]  # (7,)
        qddot_arm = (tau_arm - bias_arm) * self.arm_inv_mass  # (7,)

        qddot_full = np.zeros_like(qdot)
        qddot_full[self.arm_vel_idx] = qddot_arm

        # simple Euler integration
        q_next = q + qdot * self.dt
        qdot_next = qdot + qddot_full * self.dt
        return q_next, qdot_next

    def _get_ee_pos_from_state(self):
        if self.ee_is_site and self.ee_site_id is not None:
            return self.data.site_xpos[self.ee_site_id].copy()
        elif (not self.ee_is_site) and self.ee_body_id is not None:
            return self.data.body_xpos[self.ee_body_id].copy()
        else:
            raise RuntimeError("EE id not resolved properly.")

    def _get_ee_pos(self, q, qdot):
        data = self.data
        data.qpos[:] = q
        data.qvel[:] = qdot
        self.sim.forward()
        return self._get_ee_pos_from_state()

    def _compute_pos_jacobian_fd(self, q, qdot, eps: float = 1e-4) -> np.ndarray:
        """
        Finite-difference로 EEF position Jacobian J ∈ R^{3x7} 계산.
        - arm_pos_idx를 이용해서 각 joint qpos를 ±eps perturb.
        - mujoco.mj_jacSite / mj_jacBody 사용하지 않음.
        """
        data = self.data

        # base state
        data.qpos[:] = q
        data.qvel[:] = qdot
        self.sim.forward()
        p0 = self._get_ee_pos_from_state()

        J = np.zeros((3, self.arm_dof), dtype=np.float64)

        for j in range(self.arm_dof):
            qi = int(self.arm_pos_idx[j])

            # +eps
            data.qpos[qi] = q[qi] + eps
            self.sim.forward()
            p_plus = self._get_ee_pos_from_state()

            # -eps
            data.qpos[qi] = q[qi] - eps
            self.sim.forward()
            p_minus = self._get_ee_pos_from_state()

            # reset
            data.qpos[qi] = q[qi]

            J[:, j] = (p_plus - p_minus) / (2.0 * eps)

        # restore base
        data.qpos[:] = q
        data.qvel[:] = qdot
        self.sim.forward()
        return J

    def _build_tau_ref(self, q0: np.ndarray, qdot0: np.ndarray, A_ref: np.ndarray) -> np.ndarray:
        K = A_ref.shape[0]
        tau_ref = np.zeros((K, self.arm_dof), dtype=np.float32)
        q = q0.copy()
        qdot = qdot0.copy()
        for k in range(K):
            dp_ref = A_ref[k, :3]
            tau_k = self._eef_delta_to_tau(q, qdot, dp_ref)  # (7,)
            tau_ref[k] = tau_k
            q, qdot = self._dynamics_step(q, qdot, tau_k)
        return tau_ref

    def _eef_delta_to_tau(self, q: np.ndarray, qdot: np.ndarray, dp: np.ndarray) -> np.ndarray:
        """
        dp: 원하는 EEF position delta (world)
        → Jacobian 기반의 Cartesian PD control → joint torque (7,)
        """
        # 현재 상태 세팅
        self.data.qpos[:] = q
        self.data.qvel[:] = qdot
        self.sim.forward()

        # 현재 EE 위치
        p_cur = self._get_ee_pos_from_state()

        # J_pos ∈ R^{3x7}
        J_pos = self._compute_pos_jacobian_fd(q, qdot)  # finite-diff Jacobian
        qdot_arm = qdot[self.arm_vel_idx]  # (7,)
        v_ee = J_pos @ qdot_arm

        p_des = p_cur + dp
        err = p_des - p_cur
        F = self.tau_kp * err - self.tau_kd * v_ee  # 3D Cartesian force
        tau_arm = J_pos.T @ F  # (7,)
        return tau_arm.astype(np.float32)

    # --------------------------------------------------------------------------
    # Phase logic
    # --------------------------------------------------------------------------
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
# LIBERO eval + MPPI
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

        # build MPPI once (need dt)
        if controller is None and args.use_mppi:
            dt = 1.0 / env.env.control_freq if hasattr(env.env, "control_freq") else env.env.sim.model.opt.timestep
            if args.mppi_mode == "joint_torque":
                # robosuite Robot 객체 사용 (첫 번째 로봇)
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

                    # Hybrid control: MPPI (kinematics / joint torque) around A_ref
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
                            # Joint-torque MPPI: q, qdot from sim
                            q0 = env.env.sim.data.qpos.copy()
                            qdot0 = env.env.sim.data.qvel.copy()
                            delta_chunk = controller.plan(
                                q0=q0,
                                qdot0=qdot0,
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
# Helper functions
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
