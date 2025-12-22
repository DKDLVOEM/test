import collections
import dataclasses
import logging
import math
import pathlib
from typing import List, Optional

import imageio
import mujoco
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Object motion (applied every sim step)
    #################################################################################################################
    apply_motion: bool = True
    motion_object_name: str = ""  # If empty, use first non-fixture obj_of_interest
    motion_mode: str = "position"  # "position" or "velocity"
    motion_amp_xy: List[float] = dataclasses.field(
        default_factory=lambda: [0.02, 0.02]
    )  # meters
    motion_freq_hz: float = 0.5  # Hz

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    _validate_motion_args(args)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        motion_obj_name = _resolve_motion_object_name(env, args.motion_object_name)
        if args.apply_motion and motion_obj_name is None:
            logging.warning(
                "apply_motion=True but no object of interest found; skipping motion for this task."
            )

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Base pose for motion target
            base_qpos = None
            if args.apply_motion and motion_obj_name is not None:
                base_qpos = _get_object_qpos(env, motion_obj_name)

            # Setup
            t = 0
            replay_images = []
            dt = _get_control_dt(env)
            motion_enabled = True  # disable once grasped
            grasp_contact_steps = 0
            grasp_contact_needed = 3
            grasp_force_threshold = 0.5  # Newtons in contact normal
            target_body_id = (
                env.env.obj_body_id[motion_obj_name]
                if motion_obj_name is not None
                else None
            )
            gripper_geom_ids = (
                _get_gripper_geom_ids(env.env.sim)
                if motion_obj_name is not None
                else []
            )

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Stop motion if grasped: require target contact with gripper and normal force above threshold for a few steps
                    if motion_enabled and target_body_id is not None:
                        contact_forces = _get_grasp_normal_forces(
                            env.env.sim, target_body_id, gripper_geom_ids
                        )
                        if contact_forces:
                            print(
                                f"[contact] normal forces: "
                                f"{', '.join(f'{f:.3f}' for f in contact_forces)}"
                            )
                        if any(f > grasp_force_threshold for f in contact_forces):
                            grasp_contact_steps += 1
                        else:
                            grasp_contact_steps = 0
                        if grasp_contact_steps >= grasp_contact_needed:
                            motion_enabled = False

                    # After wait: apply sine motion before stepping
                    if (
                        motion_enabled
                        and args.apply_motion
                        and motion_obj_name is not None
                    ):
                        _apply_xy_sine_motion(
                            env=env,
                            obj_name=motion_obj_name,
                            base_qpos=base_qpos,
                            t_sec=t * dt,
                            amp_xy=args.motion_amp_xy,
                            freq_hz=args.motion_freq_hz,
                            mode=args.motion_mode,
                        )

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(
                            wrist_img, args.resize_size, args.resize_size
                        )
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
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

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), (
                            f"We want to replan every {args.replan_steps} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if motion_enabled and target_body_id is not None:
                        contact_forces = _get_grasp_normal_forces(
                            env.env.sim, target_body_id, gripper_geom_ids
                        )
                        if contact_forces:
                            print(
                                f"[contact] normal forces: "
                                f"{', '.join(f'{f:.3f}' for f in contact_forces)}"
                            )
                        if any(f > grasp_force_threshold for f in contact_forces):
                            grasp_contact_steps += 1
                        else:
                            grasp_contact_steps = 0
                        if grasp_contact_steps >= grasp_contact_needed:
                            motion_enabled = False
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

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path)
                / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(
                f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)"
            )

        # Log final results
        logging.info(
            f"Current task success rate: {float(task_successes) / float(task_episodes)}"
        )
        logging.info(
            f"Current total success rate: {float(total_successes) / float(total_episodes)}"
        )

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _resolve_motion_object_name(env, requested_name):
    if requested_name:
        return requested_name
    if not env.obj_of_interest:
        return None
    for name in env.obj_of_interest:
        if not env.env.is_fixture(name):
            return name
    return env.obj_of_interest[0]


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
    env.env.sim.data.set_joint_qpos(joint, qpos)
    env.env.sim.forward()


def _set_object_qvel(env, obj_name, qvel):
    joint = _get_object_joint_name(env, obj_name)
    env.env.sim.data.set_joint_qvel(joint, qvel)
    env.env.sim.forward()


def _apply_xy_sine_motion(env, obj_name, base_qpos, t_sec, amp_xy, freq_hz, mode):
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


def _get_control_dt(env):
    if hasattr(env.env, "control_freq") and env.env.control_freq:
        return 1.0 / float(env.env.control_freq)
    return float(env.env.sim.model.opt.timestep)


def _validate_motion_args(args: Args) -> None:
    if len(args.motion_amp_xy) != 2:
        raise ValueError("motion_amp_xy must have 2 floats (x, y amplitude).")
    if args.motion_mode not in ["position", "velocity"]:
        raise ValueError("motion_mode must be either 'position' or 'velocity'.")


def _get_gripper_geom_ids(sim):
    """Collect gripper geom ids heuristically by name."""
    geom_ids = []
    for geom_id in range(sim.model.ngeom):
        name = mujoco.mj_id2name(sim.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            continue
        if any(key in name.lower() for key in ["finger", "gripper", "pad"]):
            geom_ids.append(geom_id)
    return geom_ids


def _get_grasp_normal_forces(sim, target_body_id, gripper_geom_ids):
    """Return list of normal forces for contacts between gripper geoms and target body."""
    forces = []
    force_vec = np.zeros(6, dtype=np.float64)
    ncon = sim.data.ncon
    for i in range(ncon):
        c = sim.data.contact[i]
        # Skip contacts not involving the target body
        body1 = sim.model.geom_bodyid[c.geom1]
        body2 = sim.model.geom_bodyid[c.geom2]
        if target_body_id not in (body1, body2):
            continue
        # Require the other geom to be part of the gripper
        if c.geom1 in gripper_geom_ids or c.geom2 in gripper_geom_ids:
            mujoco.mj_contactForce(sim.model, sim.data, i, force_vec)
            # Contact frame: z is normal
            normal_f = force_vec[2]
            forces.append(normal_f)
    return forces


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
