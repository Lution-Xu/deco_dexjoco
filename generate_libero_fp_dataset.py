"""
Generate a LIBERO dataset with Force Prompting (contact-force visualization).

This script REPLAYS each demonstration through the LIBERO / robosuite / MuJoCo
simulation, renders the camera images, extracts the gripper's contact forces from
MuJoCo, and overlays them as arrows on the images. Two replay strategies are
supported:

    * ``action`` (default): reset the env, settle, then re-step the recorded
      actions one by one. Matches the behaviour of LIBERO's eval rollout, so the
      produced images align with what a policy will see at inference time.
    * ``state``: teleport the sim to each saved state in turn
      (``regenerate_obs_from_state``). Cheaper and frame-exact to the original
      dataset, but the image convention / physics can differ slightly from a real
      rollout.

No-op (zero) actions are filtered out inline, exactly like the baseline
``regenerate_libero_dataset.py``, so the two concerns (replay + filtering) are no
longer split across scripts. Crucially, every image is flipped to the OpenCV
convention exactly once -- when it leaves the renderer -- and is then stored as
is. There is no second flip downstream, which removes the convention mismatch the
old two-script pipeline had. Both the training images and the eval-time images
therefore use the same OpenCV convention.

All technical knobs (resolution, force-arrow scale, settle steps, ...) live as
module-level constants below; the only command-line arguments are task-level
ones (what to read, where to write, how to replay). New force-visualization
methods can be added by editing those constants, or -- once there is more than
one -- by routing them through a yaml config.

Usage:
    python datasets/scripts/generate_libero_fp_dataset.py \\
        --input_dir  /path/to/LIBERO/original \\
        --output_dir /path/to/libero_fp \\
        --replay_mode action --suites libero_spatial

    # Resume an interrupted run (skips files already finished):
    python datasets/scripts/generate_libero_fp_dataset.py \\
        --input_dir ... --output_dir ... --resume
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, NamedTuple, Optional

import cv2
import h5py
import mujoco
import numpy as np

# Make the (vendored) libero package importable. This script depends only on
# libero / robosuite / mujoco, not on anything else in this codebase.
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(_ROOT_DIR, "LIBERO"))

from libero.libero import benchmark, get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402
from robosuite.utils.camera_utils import (  # noqa: E402
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix,
)

# ---------------------------------------------------------------------------
# Configuration (technical -- not exposed on the CLI)
# ---------------------------------------------------------------------------
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
DEMO_PER_FILE = 50  # fallback when the source hdf5 has no `num_demos` attr

IMAGE_RESOLUTION = 256          # rendered camera image side length, in pixels
FORCE_SCALE = 0.02             # current force-visualization method: arrow length scale
SETTLE_STEPS = 10             # action-replay: #dummy steps after reset, to let physics settle

CAMERA_NAMES = ["agentview", "robot0_eye_in_hand"]
OBS_KEY_TO_CAM = {
    "agentview": "agentview_image",
    "robot0_eye_in_hand": "robot0_eye_in_hand_image",
}

# gripper pad geoms whose contacts we treat as "the gripper is touching something"
GRIPPER_GEOM_NAMES = ["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"]
# geom-name substrings to ignore when reporting contacts (the world, not objects)
EXCLUDE_KEYWORDS = ["table", "floor", "ground", "wall"]
# gripper site markers rendered by default in panda_gripper.xml; we hide them
GRIPPER_SITES_TO_HIDE = [
    "gripper0_grip_site_cylinder",
    "gripper0_grip_site",
    "gripper0_ft_frame",
]


# ===========================================================================
# Section 1 -- Force visualization
# ===========================================================================

def fast_mat_inv(mat):
    """Inverse of a 4x4 homogeneous transform (rotation is orthonormal)."""
    ret = np.eye(4)
    ret[:3, :3] = mat[:3, :3].T
    ret[:3, 3] = -mat[:3, :3].T @ mat[:3, 3]
    return ret


class ForceVisualizer:
    """Project a 3D contact force onto a 2D camera image and draw it as an arrow."""

    def __init__(self, cam_intrinsic_matrix, resolution, scale_alpha=FORCE_SCALE):
        self.K = cam_intrinsic_matrix
        self.resolution = resolution
        self.scale_alpha = scale_alpha

    def _world_to_camera(self, point_world, T_cam_from_world):
        p_hom = np.append(point_world, 1.0)
        p_cam = T_cam_from_world @ p_hom
        return p_cam[:3]

    def _project_to_pixel(self, point_in_cam):
        if point_in_cam[2] <= 0:
            return None
        uv = self.K @ point_in_cam
        return (int(np.round(uv[0] / uv[2])), int(np.round(uv[1] / uv[2])))

    def draw_force_arrow(self, frame, pos_world, force_world, T_cam_from_world,
                         color=(255, 0, 0), thickness=2, tip_length=0.3):
        """Draw the force arrow onto ``frame`` in place; returns the frame."""
        start_cam = self._world_to_camera(pos_world, T_cam_from_world)
        start_px = self._project_to_pixel(start_cam)
        if start_px is None:
            return frame

        end_world = pos_world + force_world * self.scale_alpha
        end_px = self._project_to_pixel(self._world_to_camera(end_world, T_cam_from_world))
        if end_px is None:
            return frame

        h, w = self.resolution
        if not (0 <= start_px[0] < w and 0 <= start_px[1] < h):
            return frame

        cv2.arrowedLine(frame, start_px, end_px, color, thickness, tipLength=tip_length)
        return frame


# ===========================================================================
# Section 2 -- Contact-force extraction from MuJoCo
# ===========================================================================

def extract_contact_forces(sim, gripper_geom_names=None, exclude_keywords=None):
    """Return a list of griapper<->object contacts as {pos, force, magnitude, ...}."""
    gripper_geom_names = gripper_geom_names or GRIPPER_GEOM_NAMES
    exclude_keywords = exclude_keywords or EXCLUDE_KEYWORDS

    gripper_geom_ids = set()
    for name in gripper_geom_names:
        try:
            gripper_geom_ids.add(sim.model.geom_name2id(name))
        except ValueError:
            pass

    contacts = []
    for i in range(sim.data.ncon):
        contact = sim.data.contact[i]
        g1, g2 = contact.geom1, contact.geom2
        g1_is_gripper = g1 in gripper_geom_ids
        g2_is_gripper = g2 in gripper_geom_ids
        if g1_is_gripper == g2_is_gripper:
            continue  # gripper-gripper or object-object; not interesting

        gripper_gid, other_gid = (g1, g2) if g1_is_gripper else (g2, g1)
        try:
            gripper_name = sim.model.geom_id2name(gripper_gid)
            other_name = sim.model.geom_id2name(other_gid)
        except ValueError:
            continue
        if any(kw in other_name.lower() for kw in exclude_keywords):
            continue

        result = np.zeros(6)
        mujoco.mj_contactForce(sim.model._model, sim.data._data, i, result)
        normal_force = result[0]
        normal_dir = np.array(contact.frame[0:3])
        # force points from the gripper into the object: flip sign depending on geom order
        force_vec = (-normal_dir if g2_is_gripper else normal_dir) * normal_force

        contacts.append({
            "pos": np.array(contact.pos),
            "force": force_vec,
            "magnitude": abs(normal_force),
            "geom_gripper": gripper_name,
            "geom_other": other_name,
        })
    return contacts


# ===========================================================================
# Section 3 -- Environment helpers
# ===========================================================================

def make_env(task, disable_gripper_sites=True):
    """Create the OffScreenRenderEnv for ``task`` and optionally hide gripper markers."""
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=IMAGE_RESOLUTION,
        camera_widths=IMAGE_RESOLUTION,
    )
    env.seed(0)  # affects object placement even with a fixed initial state

    if disable_gripper_sites:
        for site_name in GRIPPER_SITES_TO_HIDE:
            try:
                site_id = env.sim.model.site_name2id(site_name)
                env.sim.model.site_rgba[site_id][3] = 0.0  # fully transparent
            except ValueError:
                pass
    return env


def get_camera_image_opencv(obs, camera_name):
    """Return ``camera_name``'s image from ``obs`` in OpenCV convention (upright).

    MuJoCo's offscreen renderer outputs images flipped top-to-bottom (OpenGL
    convention). This is the ONE place we flip, so every downstream consumer
    (stored raw image, stored fp image, eval-time image) stays consistent.
    """
    obs_key = OBS_KEY_TO_CAM.get(camera_name, f"{camera_name}_image")
    if obs_key not in obs:
        raise KeyError(
            f"Camera '{camera_name}' (key '{obs_key}') not found. "
            f"Available image keys: {[k for k in obs if 'image' in k]}"
        )
    return obs[obs_key][::-1].copy()


def make_visualizers(env):
    """Build one :class:`ForceVisualizer` per camera, sharing FORCE_SCALE."""
    sim = env.sim
    return {
        cam_name: ForceVisualizer(
            get_camera_intrinsic_matrix(sim, cam_name, IMAGE_RESOLUTION, IMAGE_RESOLUTION),
            (IMAGE_RESOLUTION, IMAGE_RESOLUTION),
        )
        for cam_name in CAMERA_NAMES
    }


def draw_forces_on_frames(sim, images, visualizers):
    """Overlay every detected contact force on each camera image (in place)."""
    contacts = extract_contact_forces(sim)
    for cam_name, img in images.items():
        T_world_cam = get_camera_extrinsic_matrix(sim, cam_name)
        T_cam_from_world = fast_mat_inv(T_world_cam)
        for c in contacts:
            visualizers[cam_name].draw_force_arrow(img, c["pos"], c["force"], T_cam_from_world)
    return contacts


# ===========================================================================
# Section 4 -- Replay strategies
# ===========================================================================

def is_noop(action, prev_action=None, threshold=1e-4):
    """Whether ``action`` is a no-op (robot is doing nothing).

    A no-op satisfies BOTH: (1) the non-gripper action dims are ~0, and
    (2) the gripper command is unchanged from the previous kept action. The
    second criterion stops us from dropping "stay still but open/close gripper".
    """
    if np.linalg.norm(action[:-1]) >= threshold:
        return False
    if prev_action is None:
        return True
    return action[-1] == prev_action[-1]


class EpisodeFrame(NamedTuple):
    """One kept frame of a replayed episode."""
    image_agent: np.ndarray   # OpenCV-convention agentview (raw, no arrow yet)
    image_wrist: np.ndarray   # OpenCV-convention wrist (raw, no arrow yet)
    state: np.ndarray         # MuJoCo sim state
    action: np.ndarray        # the action that produced this frame
    ee_state: np.ndarray      # (6,) ee_pos + ee_axisangle
    gripper_state: np.ndarray  # (2,) gripper qpos
    joint_state: np.ndarray   # (7,) arm joint pos
    robot_state: np.ndarray   # (9,) gripper_qpos + eef_pos + eef_quat


def _obs_to_proprio(obs):
    """Pull the proprioceptive observations out of a robosuite obs dict."""
    ee_state = np.hstack((obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])))
    gripper_state = obs["robot0_gripper_qpos"]
    joint_state = obs["robot0_joint_pos"]
    robot_state = np.concatenate(
        [obs["robot0_gripper_qpos"], obs["robot0_eef_pos"], obs["robot0_eef_quat"]]
    )
    return ee_state, gripper_state, joint_state, robot_state


def _build_frame(obs, action, state):
    """Assemble an :class:`EpisodeFrame` from a freshly rendered ``obs``."""
    ee_state, gripper_state, joint_state, robot_state = _obs_to_proprio(obs)
    return EpisodeFrame(
        image_agent=get_camera_image_opencv(obs, "agentview"),
        image_wrist=get_camera_image_opencv(obs, "robot0_eye_in_hand"),
        state=np.asarray(state),
        action=np.asarray(action),
        ee_state=ee_state,
        gripper_state=gripper_state,
        joint_state=joint_state,
        robot_state=robot_state,
    )


def _dummy_action():
    """No-op action used to roll the sim forward while the robot holds still."""
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def replay_action_mode(env, actions, init_state) -> Iterator[EpisodeFrame]:
    """Reset, settle, then re-step each action (skipping no-ops) -> kept frames.

    Mirrors ``regenerate_libero_dataset.py``: reset -> set_init_state ->
    SETTLE_STEPS dummy steps -> per action: record obs, then env.step(action).
    No-op actions are skipped (not recorded, not stepped).
    """
    env.reset()
    env.set_init_state(init_state)
    obs = None
    for _ in range(SETTLE_STEPS):
        obs, _, _, _ = env.step(_dummy_action())

    prev_action = None
    for action in actions:
        if is_noop(action, prev_action):
            continue
        # ``obs`` is the observation of the state we are about to act from: it is
        # the return value of the settle loop (for the first action) or of the
        # previous env.step. libero's wrapper exposes no obs getter, so we thread
        # it through the loop exactly like ``regenerate_libero_dataset.py`` does.
        state = env.sim.get_state().flatten()
        yield _build_frame(obs, action, state)
        obs, _, _, _ = env.step(np.asarray(action).tolist())
        prev_action = action


def replay_state_mode(env, states, actions) -> Iterator[EpisodeFrame]:
    """Teleport the sim to each saved state (skipping no-op actions) -> kept frames.

    Frame-exact to the original dataset (uses ``regenerate_obs_from_state``),
    so it is cheaper but the physics/visuals can diverge slightly from a real
    rollout. ``actions`` must align 1:1 with ``states`` by index.
    """
    prev_action = None
    for state, action in zip(states, actions):
        if is_noop(action, prev_action):
            continue
        obs = env.regenerate_obs_from_state(state)
        yield _build_frame(obs, action, state)
        prev_action = action


# ===========================================================================
# Section 5 -- Per-episode processing
# ===========================================================================

@dataclass
class EpisodeData:
    """All arrays for one replayed episode, ready to be written to hdf5."""
    image_agent: List[np.ndarray] = field(default_factory=list)   # raw (OpenCV)
    image_wrist: List[np.ndarray] = field(default_factory=list)   # raw (OpenCV)
    image_agent_fp: List[np.ndarray] = field(default_factory=list)  # raw + arrow
    image_wrist_fp: List[np.ndarray] = field(default_factory=list)  # raw + arrow
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    ee_states: List[np.ndarray] = field(default_factory=list)
    gripper_states: List[np.ndarray] = field(default_factory=list)
    joint_states: List[np.ndarray] = field(default_factory=list)
    robot_states: List[np.ndarray] = field(default_factory=list)
    success: bool = False
    num_noops: int = 0


def process_episode(env, demo_data, visualizers, replay_mode, task) -> Optional[EpisodeData]:
    """Replay one demo, draw forces, and collect the kept frames.

    Returns ``None`` if the episode is dropped (replay did not reach success).
    """
    orig_actions = demo_data["actions"][()]
    orig_states = demo_data["states"][()]

    total = len(orig_actions)
    kept = 0

    if replay_mode == "action":
        # init_state for action-replay is the first saved state of the demo
        frames = replay_action_mode(env, orig_actions, orig_states[0])
    else:
        frames = replay_state_mode(env, orig_states, orig_actions)

    ep = EpisodeData()
    for frame in frames:
        kept += 1
        # raw images are kept untouched; fp images get a copy to draw arrows on
        img_agent_fp = frame.image_agent.copy()
        img_wrist_fp = frame.image_wrist.copy()
        contacts = draw_forces_on_frames(env.sim, {
            "agentview": img_agent_fp,
            "robot0_eye_in_hand": img_wrist_fp,
        }, visualizers)

        ep.image_agent.append(frame.image_agent)
        ep.image_wrist.append(frame.image_wrist)
        ep.image_agent_fp.append(img_agent_fp)
        ep.image_wrist_fp.append(img_wrist_fp)
        ep.states.append(frame.state)
        ep.actions.append(frame.action)
        ep.ee_states.append(frame.ee_state)
        ep.gripper_states.append(frame.gripper_state)
        ep.joint_states.append(frame.joint_state)
        ep.robot_states.append(frame.robot_state)

    ep.num_noops = total - kept
    ep.success = bool(env.check_success()) if kept > 0 else False
    if not ep.success or kept == 0:
        return None
    return ep


# ===========================================================================
# Section 6 -- HDF5 I/O & batch driver
# ===========================================================================

def get_task_file_list(input_dir, suites):
    """Enumerate ``{suite, task_id, task_name, hdf5_path}`` via the benchmark API."""
    benchmark_dict = benchmark.get_benchmark_dict()
    file_list = []
    for suite_name in suites:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            hdf5_path = os.path.join(input_dir, task.problem_folder, f"{task.name}_demo.hdf5")
            file_list.append({
                "suite": suite_name,
                "task_id": task_id,
                "task_name": task.name,
                "hdf5_path": hdf5_path,
            })
    return file_list


def get_output_path(task_info, output_dir, input_dir):
    """Output path mirrors the input's relative layout under ``output_dir``."""
    rel = os.path.relpath(task_info["hdf5_path"], input_dir)
    return os.path.join(output_dir, rel)


def write_episode(out_data_grp, ep_idx, ep):
    """Write one :class:`EpisodeData` as ``demo_<ep_idx>`` under the ``data`` group."""
    def stack(lst):
        return np.stack(lst, axis=0)

    ee = stack(ep.ee_states)
    ep_data_grp = out_data_grp.create_group(f"demo_{ep_idx}")
    obs_grp = ep_data_grp.create_group("obs")
    obs_grp.create_dataset("gripper_states", data=stack(ep.gripper_states))
    obs_grp.create_dataset("joint_states", data=stack(ep.joint_states))
    obs_grp.create_dataset("ee_states", data=ee)
    obs_grp.create_dataset("ee_pos", data=ee[:, :3])
    obs_grp.create_dataset("ee_ori", data=ee[:, 3:])
    # OpenCV-convention images. ``*_rgb`` = raw, ``*_fp`` = raw + force arrows.
    obs_grp.create_dataset("agentview_rgb", data=stack(ep.image_agent))
    obs_grp.create_dataset("eye_in_hand_rgb", data=stack(ep.image_wrist))
    obs_grp.create_dataset("agentview_fp", data=stack(ep.image_agent_fp))
    obs_grp.create_dataset("eye_in_hand_fp", data=stack(ep.image_wrist_fp))
    ep_data_grp.create_dataset("actions", data=stack(ep.actions))
    ep_data_grp.create_dataset("states", data=stack(ep.states))
    ep_data_grp.create_dataset("robot_states", data=stack(ep.robot_states))
    dones = np.zeros(len(ep.actions), dtype=np.uint8)
    dones[-1] = 1
    rewards = np.zeros(len(ep.actions), dtype=np.uint8)
    rewards[-1] = 1
    ep_data_grp.create_dataset("rewards", data=rewards)
    ep_data_grp.create_dataset("dones", data=dones)


def process_file(task_info, output_dir, input_dir, replay_mode):
    """Replay every demo of one task hdf5 into a fresh output hdf5.

    Returns ``(num_episodes_written, num_noops)``. Partial output is removed on
    failure.
    """
    output_path = get_output_path(task_info, output_dir, input_dir)
    hdf5_path = task_info["hdf5_path"]
    if not os.path.exists(hdf5_path):
        raise FileNotFoundError(f"Input HDF5 not found: {hdf5_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[task_info["suite"]]()
    task = suite.get_task(task_info["task_id"])

    env = make_env(task)
    visualizers = make_visualizers(env)

    num_written = 0
    num_noops = 0
    try:
        with h5py.File(hdf5_path, "r") as src, h5py.File(output_path, "w") as dst:
            # carry over top-level attrs (there usually are none for LIBERO)
            for attr_name in src.attrs:
                dst.attrs[attr_name] = src.attrs[attr_name]
            out_data_grp = dst.create_group("data")

            data_grp = src["data"]
            # carry over the source ``data`` group attrs (bddl_file_name, env_args,
            # problem_info, total, ...) so downstream tooling still works.
            for attr_name in data_grp.attrs:
                out_data_grp.attrs[attr_name] = data_grp.attrs[attr_name]
            num_demos = int(data_grp.attrs.get("num_demos", DEMO_PER_FILE))

            for ep_idx in range(num_demos):
                ep = process_episode(
                    env, data_grp[f"demo_{ep_idx}"], visualizers, replay_mode, task
                )
                if ep is None:
                    continue
                write_episode(out_data_grp, num_written, ep)
                num_written += 1
                num_noops += ep.num_noops
                print(f"    demo_{ep_idx}: kept {len(ep.actions)} frames "
                      f"({ep.num_noops} no-op dropped), success={ep.success}")

            # update file-level metadata to reflect the regenerated dataset:
            # the actual demo count, and the OpenCV (upright) image convention we
            # store (the source was OpenGL convention).
            out_data_grp.attrs["num_demos"] = num_written
            out_data_grp.attrs["macros_image_convention"] = "opencv"
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise
    finally:
        env.close()

    return num_written, num_noops


# ---- progress (resume) ----------------------------------------------------

def _progress_file(output_dir):
    return os.path.join(output_dir, "progress.json")


def load_progress(output_dir):
    path = _progress_file(output_dir)
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(output_dir, progress):
    os.makedirs(output_dir, exist_ok=True)
    with open(_progress_file(output_dir), "w") as f:
        json.dump(progress, f, indent=2)


def is_completed(progress, hdf5_path):
    return hdf5_path in progress["completed"]


def verify_output(output_path):
    """Quick structural sanity check on a freshly written file."""
    expected_keys = ("agentview_rgb", "eye_in_hand_rgb", "agentview_fp", "eye_in_hand_fp")
    with h5py.File(output_path, "r") as f:
        demo_keys = list(f["data"].keys())
        if not demo_keys:
            return False, "no episodes written"
        obs_grp = f[f"data/{demo_keys[0]}/obs"]
        for key in expected_keys:
            if key not in obs_grp:
                return False, f"missing {key}"
            ds = obs_grp[key]
            if ds.shape[1:] != (IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3):
                return False, f"bad shape for {key}: {ds.shape}"
            if ds.dtype != np.uint8:
                return False, f"bad dtype for {key}: {ds.dtype}"
    return True, "OK"


# ---- CLI ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a LIBERO dataset with force-prompted images "
                    "(replay + no-op filter + contact-force visualization)."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing the original LIBERO hdf5 demos.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to write the force-prompted hdf5 files.")
    parser.add_argument("--replay_mode", type=str, default="action",
                        choices=["action", "state"],
                        help="How to replay: 'action' re-steps recorded actions "
                             "(matches eval, default); 'state' teleports to saved states.")
    parser.add_argument("--suites", type=str, nargs="+", default=None,
                        choices=SUITES, help="Suites to process (default: all).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip files already marked complete in progress.json.")
    args = parser.parse_args()

    suites = args.suites or SUITES
    file_list = get_task_file_list(args.input_dir, suites)
    print(f"Found {len(file_list)} task file(s) across {len(suites)} suite(s).")
    for s in suites:
        print(f"  {s}: {sum(1 for f in file_list if f['suite'] == s)}")

    progress = load_progress(args.output_dir) if args.resume else {"completed": [], "failed": []}
    if args.resume:
        remaining = [f for f in file_list if not is_completed(progress, f["hdf5_path"])]
        print(f"\nResuming: {len(progress['completed'])} done, {len(remaining)} remaining.")
        file_list = remaining

    t_start = time.time()
    for idx, task_info in enumerate(file_list):
        tag = f"[{idx + 1}/{len(file_list)}]"
        print(f"\n{tag} {task_info['suite']}/{task_info['task_name']}")
        print(f"  input:  {task_info['hdf5_path']}")
        print(f"  output: {get_output_path(task_info, args.output_dir, args.input_dir)}")

        file_t0 = time.time()
        try:
            num_written, num_noops = process_file(
                task_info, args.output_dir, args.input_dir, args.replay_mode
            )
            ok, msg = verify_output(get_output_path(task_info, args.output_dir, args.input_dir))
            if not ok:
                print(f"  WARNING: verification failed: {msg}")
            elapsed = time.time() - file_t0
            print(f"  done in {elapsed:.0f}s -- {num_written} episodes, "
                  f"{num_noops} no-op actions dropped -- {msg}")
            progress["completed"].append(task_info["hdf5_path"])
            save_progress(args.output_dir, progress)
        except Exception as e:
            elapsed = time.time() - file_t0
            print(f"  FAILED ({elapsed:.0f}s): {e}")
            traceback.print_exc()
            progress["failed"].append({
                "path": task_info["hdf5_path"],
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            save_progress(args.output_dir, progress)

    print(f"\n{'=' * 60}")
    print(f"Batch complete in {time.time() - t_start:.0f}s")
    print(f"  completed: {len(progress['completed'])}")
    print(f"  failed:    {len(progress['failed'])}")
    if progress["failed"]:
        for entry in progress["failed"]:
            print(f"  - {entry['path']}: {entry['error']}")


if __name__ == "__main__":
    main()
