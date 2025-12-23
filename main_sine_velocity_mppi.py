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
    motion_freq_hz: float = 0.5

    # MPPI on/off
    use_mppi: bool = True

    # MPPI params
    mppi_horizon: int = 15
    mppi_num_samples: int = 64
    mppi_temperature: float = 1.0
    mppi_noise_sigma: float = 0.02  # scaled per phase
    w_ref: float = 1.0
    w_goal: float = 5.0
    w_dyn: float = 5.0
    w_reg: float = 0.1
    w_obs: float = 0.0  # keep 0 for now

    # Phase thresholds (meters)
    d_far: float = 0.15
    d_close: float = 0.04

    video_out_path: str = "data/libero/videos"
    seed: int = 7  # Random Seed (for reproducibility)


class MPPILiberoController:
    """
    Simulator-in-the-loop MPPI around pi0 action reference.
    """

    def __init__(
        self,
        horizon: int,
        num_samples: int,
        temperature: float,
        base_noise_sigma: float,
        weights: dict,
        d_far: float,
        d_close: float,
        dt: float,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.temperature = temperature
        self.base_noise_sigma = base_noise_sigma
        self.w_ref = weights["w_ref"]
        self.w_goal = weights["w_goal"]
        self.w_dyn = weights["w_dyn"]
        self.w_reg = weights["w_reg"]
        self.w_obs = weights["w_obs"]
        self.d_far = d_far
        self.d_close = d_close
        self.dt = dt

    def plan(
        self,
        env,
        obs,
        A_ref: np.ndarray,
        obj_pos: np.ndarray,
        obj_vel: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
            env: LIBERO env (OffScreenRenderEnv)
            obs: current observation dict
            A_ref: [H, action_dim] reference from pi0
            obj_pos: current object position (3,)
            obj_vel: current object linear velocity (3,)
        Returns:
            u_opt: (action_dim,) to pass to env.step
        """
        assert A_ref.ndim == 2
        H_ref, action_dim = A_ref.shape
        K = min(self.horizon, H_ref)

        # Phase + weights/noise gating
        p_ee = obs["robot0_eef_pos"]
        d = np.linalg.norm(p_ee - obj_pos)
        phase = self._compute_phase(d, obs)
        w_ref, w_goal, w_dyn, w_reg, noise_sigma = self._phase_weights_noise(phase)

        # If in GRASP phase, just return pi0 action directly (no MPPI)
        if phase == "grasp":
            return A_ref[0]

        # Backup sim state once
        sim_state = env.get_sim_state()

        costs = np.zeros(self.num_samples, dtype=np.float32)
        actions = np.zeros((self.num_samples, K, action_dim), dtype=np.float32)

        # Sample trajectories
        for j in range(self.num_samples):
            env.set_state(sim_state)  # restore
            env.env.sim.forward()

            # Sample noise around reference
            noise = np.random.normal(scale=noise_sigma, size=(K, action_dim))
            u_seq = A_ref[:K] + noise
            actions[j] = u_seq

            J = 0.0
            for k in range(K):
                # Rollout one step
                obs_k, _, _, _ = env.step(u_seq[k].tolist())

                # Predicted object pos (linear)
                t_pred = (k + 1) * self.dt
                obj_pred = obj_pos + obj_vel * t_pred

                # Costs
                L_ref = w_ref * np.sum((u_seq[k] - A_ref[k]) ** 2)
                ee_pos = obs_k["robot0_eef_pos"]
                L_goal = w_goal * np.sum((ee_pos - obj_pred) ** 2)
                L_dyn = w_dyn * np.sum((ee_pos - obj_pred) ** 2)
                L_reg = w_reg * np.sum(u_seq[k] ** 2)
                L_obs = 0.0  # w_obs = 0 for now

                J += L_ref + L_goal + L_dyn + L_reg + L_obs

            costs[j] = J

        # Restore original state
        env.set_state(sim_state)
        env.env.sim.forward()

        # MPPI weighted update
        J_min = np.min(costs)
        weights = np.exp(-(costs - J_min) / max(self.temperature, 1e-6))
        weights = weights / (np.sum(weights) + 1e-8)
        u_opt = np.sum(actions[:, 0] * weights[:, None], axis=0)
        return u_opt

    def _compute_phase(self, d: float, obs) -> str:
        """
        Simple 3-phase gating: approach / near / grasp.
        """
        if d > self.d_far:
            return "approach"
        if d > self.d_close:
            return "near"
        return "grasp"

    def _phase_weights_noise(self, phase: str) -> Tuple[float, float, float, float, float]:
        # Start from base
        w_ref = self.w_ref
        w_goal = self.w_goal
        w_dyn = self.w_dyn
        w_reg = self.w_reg
        noise_sigma = self.base_noise_sigma

        if phase == "approach":
            # allow deviation to intercept moving target
            w_ref *= 0.5
            w_goal *= 1.5
            w_dyn *= 1.5
            noise_sigma *= 1.5
        elif phase == "near":
            w_ref *= 2.0
            w_goal *= 1.0
            w_dyn *= 1.0
            noise_sigma *= 0.5
        elif phase == "grasp":
            w_ref *= 10.0
            w_goal *= 0.5
            w_dyn *= 0.5
            noise_sigma *= 0.1
        return w_ref, w_goal, w_dyn, w_reg, noise_sigma


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
            controller = MPPILiberoController(
                horizon=args.mppi_horizon,
                num_samples=args.mppi_num_samples,
                temperature=args.mppi_temperature,
                base_noise_sigma=args.mppi_noise_sigma,
                weights={
                    "w_ref": args.w_ref,
                    "w_goal": args.w_goal,
                    "w_dyn": args.w_dyn,
                    "w_reg": args.w_reg,
                    "w_obs": args.w_obs,
                },
                d_far=args.d_far,
                d_close=args.d_close,
                dt=dt,
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
                    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
                    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size))
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

                    # Hybrid control: MPPI around A_ref[0]
                    if args.use_mppi and controller is not None:
                        u_opt = controller.plan(
                            env=env,
                            obs=obs,
                            A_ref=A_ref,
                            obj_pos=curr_obj_pos,
                            obj_vel=obj_vel,
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
            sim.data.qvel[qvel_addr[0] : qvel_addr[-1] + 1] = 0
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
