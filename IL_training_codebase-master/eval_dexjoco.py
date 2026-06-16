import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import imageio
import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as R

from dexjoco_constants import (
    DEXJOCO_REPO_ROOT,
    DEXJOCO_DATASET_ROOT,
    DEXJOCO_PROMPTS,
    TASK_GROUPS,
    camera_keys_for_task,
    task_group_for_task,
)
from dexjoco_stats import load_or_compute_stats
from inference import modeling, predict_action

sys.path.insert(0, os.path.join(DEXJOCO_REPO_ROOT, "dexjoco"))
from dexjoco.tasks import CONFIG_MAPPING  # noqa: E402


@dataclass
class BufferedAction:
    action: np.ndarray
    timestamp: int


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def strip_obs_prefix(camera_key):
    return camera_key.replace("observation.images.", "")


def interp_rotvec(rotvec0, rotvec1, t):
    if t <= 0:
        return rotvec0.copy()
    if t >= 1:
        return rotvec1.copy()
    r0 = R.from_rotvec(rotvec0)
    r1 = R.from_rotvec(rotvec1)
    return (r0 * R.from_rotvec((r0.inv() * r1).as_rotvec() * t)).as_rotvec()


def blend_single(old_action, new_action, t):
    out = (1.0 - t) * old_action + t * new_action
    out[3:6] = interp_rotvec(old_action[3:6], new_action[3:6], t).astype(out.dtype, copy=False)
    return out


def blend_dual(old_action, new_action, t):
    out = (1.0 - t) * old_action + t * new_action
    out[3:6] = interp_rotvec(old_action[3:6], new_action[3:6], t).astype(out.dtype, copy=False)
    out[25:28] = interp_rotvec(old_action[25:28], new_action[25:28], t).astype(out.dtype, copy=False)
    return out


def merge_action_chunk(buffer, chunk, chunk_timestamp, now_timestamp, dual_arm):
    while buffer and buffer[0].timestamp < now_timestamp:
        buffer.popleft()
    start = now_timestamp
    end = chunk_timestamp + chunk.shape[0]
    if end <= start:
        return
    chunk_tail = chunk[start - chunk_timestamp : end - chunk_timestamp]
    interp = blend_dual if dual_arm else blend_single
    buffer_start = buffer[0].timestamp if buffer else start
    buffer_end = buffer[-1].timestamp + 1 if buffer else start
    overlap_start = max(start, buffer_start)
    overlap_end = min(end, buffer_end)
    overlap_len = max(overlap_end - overlap_start, 0)
    for ts in range(overlap_start, overlap_end):
        buffer_idx = ts - buffer_start
        action_idx = ts - start
        t = (ts - overlap_start + 1) / (overlap_len + 1)
        buffer[buffer_idx] = BufferedAction(interp(buffer[buffer_idx].action, chunk_tail[action_idx], t), ts)
    for ts in range(max(start, buffer_end), end):
        action_idx = ts - start
        buffer.append(BufferedAction(chunk_tail[action_idx], ts))


class DexJoCoDECOEnv:
    def __init__(self, task_name, regime, seed, render_mode, randomize_dynamics=False):
        self.task_name = task_name
        self.task_group = task_group_for_task(task_name)
        self.dual_arm = self.task_group == "dual"
        self.regime = regime
        self.seed = seed
        self.render_mode = render_mode
        self.randomize_dynamics = randomize_dynamics
        cam1, cam2 = camera_keys_for_task(task_name, regime)
        self.policy_cameras = (strip_obs_prefix(cam1), strip_obs_prefix(cam2))
        self.env = None
        self.obs = None
        self.raw_images = {}
        self.done = False
        self.success = False
        self.last_stay_state = None

    def start(self):
        config = CONFIG_MAPPING[self.task_name]()
        self.env = config.get_environment(
            policy_mode=True,
            render_mode=self.render_mode,
            randomize=self.regime == "rand_full",
            seed=self.seed,
            randomize_dynamics=self.randomize_dynamics,
        )

    def reset(self):
        obs, _ = self.env.reset()
        self.done = False
        self.success = False
        self._update(obs)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def _update(self, obs):
        state_dim = 46 if self.dual_arm else 23
        self.obs = {
            "img1": obs[self.policy_cameras[0]],
            "img2": obs[self.policy_cameras[1]],
            "state": obs["state"][:state_dim],
        }
        self.raw_images = {
            self.policy_cameras[0]: obs[self.policy_cameras[0]],
            self.policy_cameras[1]: obs[self.policy_cameras[1]],
        }

    def step(self, action):
        obs, _reward, terminated, _truncated, info = self.env.step(self._to_env_action(action))
        self.done = bool(terminated)
        self.success = bool(info.get("succeed", False))
        self._update(obs)

    def stay(self):
        state = self.obs["state"] if self.last_stay_state is None else self.last_stay_state
        self.last_stay_state = state
        if self.dual_arm:
            r_arm = state[:7]
            l_arm = state[7:14]
            r_hand = state[14:30]
            l_hand = state[30:46]
            action = np.concatenate([
                r_arm[:3],
                R.from_quat(r_arm[3:7], scalar_first=True).as_rotvec(),
                r_hand,
                l_arm[:3],
                R.from_quat(l_arm[3:7], scalar_first=True).as_rotvec(),
                l_hand,
            ])
        else:
            arm = state[:7]
            hand = state[7:23]
            action = np.concatenate([arm[:3], R.from_quat(arm[3:7], scalar_first=True).as_rotvec(), hand])
        self.step(action)

    def _to_env_action(self, action):
        if self.dual_arm:
            r_xyz = action[:3]
            r_rotvec = action[3:6]
            r_hand = action[6:22]
            l_xyz = action[22:25]
            l_rotvec = action[25:28]
            l_hand = action[28:44]
            r_quat = R.from_rotvec(r_rotvec).as_quat(scalar_first=True)
            l_quat = R.from_rotvec(l_rotvec).as_quat(scalar_first=True)
            return np.concatenate([r_xyz, r_quat, l_xyz, l_quat, r_hand, l_hand])
        xyz = action[:3]
        rotvec = action[3:6]
        hand = action[6:22]
        quat = R.from_rotvec(rotvec).as_quat(scalar_first=True)
        return np.concatenate([xyz, quat, hand])


def load_config_with_stats(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    data_cfg = config["data"]
    stats = load_or_compute_stats(
        data_cfg.get("data_root", DEXJOCO_DATASET_ROOT),
        data_cfg["task_group"],
        data_cfg["regime"],
        data_cfg.get("stats_path"),
    )
    data_cfg["observation_mean"] = stats["observation_mean"]
    data_cfg["observation_std"] = stats["observation_std"]
    data_cfg["observation_min"] = stats["observation_min"]
    data_cfg["observation_max"] = stats["observation_max"]
    data_cfg["action_mean"] = stats["action_mean"]
    data_cfg["action_std"] = stats["action_std"]
    data_cfg["action_min"] = stats["action_min"]
    data_cfg["action_max"] = stats["action_max"]
    return config


def load_model(config, checkpoint, device):
    model = modeling(config).to(device)
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def eval_one_task(model, config, task_name, opt, device):
    env = DexJoCoDECOEnv(task_name, config["data"]["regime"], opt.seed, opt.render_mode, opt.randomize_dynamics)
    env.start()
    output_dir = Path(opt.output) / config["data"]["regime"] / task_name
    output_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    dual_arm = task_group_for_task(task_name) == "dual"
    action_horizon = config["model"]["chunk_size"]
    task_idx = TASK_GROUPS[config["data"]["task_group"]].index(task_name)

    try:
        for episode in range(opt.episodes):
            env.reset()
            timestamp = 0
            buffer = deque()
            video_path = output_dir / f"episode_{episode:03d}.mp4"
            writer = imageio.get_writer(video_path, fps=30) if opt.save_video else None
            if writer is not None:
                writer.append_data(copy.deepcopy(env.raw_images[env.policy_cameras[0]]))

            while timestamp < opt.max_steps:
                if len(buffer) < opt.replan_ratio * action_horizon:
                    obs = env.obs
                    chunk = predict_action(
                        model,
                        device,
                        config,
                        obs["img1"],
                        obs["img2"],
                        obs=obs["state"],
                        task_idx=task_idx,
                    ).numpy()
                    merge_action_chunk(buffer, chunk, timestamp, timestamp, dual_arm)

                if buffer and buffer[0].timestamp == timestamp:
                    action = buffer.popleft().action
                    env.step(action)
                else:
                    env.stay()

                timestamp += 1
                if writer is not None:
                    writer.append_data(copy.deepcopy(env.raw_images[env.policy_cameras[0]]))
                if env.done:
                    break

            if writer is not None:
                writer.close()
            if env.success:
                successes += 1
                result_path = output_dir / f"episode_{episode:03d}_success.mp4"
            else:
                result_path = output_dir / f"episode_{episode:03d}_failure.mp4"
            if opt.save_video and video_path.exists():
                video_path.rename(result_path)
            print(f"{task_name} episode {episode + 1}/{opt.episodes}: success={env.success}")
    finally:
        env.close()

    rate = successes / max(opt.episodes, 1)
    (output_dir / f"success_rate_{successes}_{opt.episodes}.txt").touch()
    return successes, opt.episodes, rate


def main(opt):
    if opt.render_mode == "rgb_array":
        os.environ.setdefault("MUJOCO_GL", "egl")
    else:
        os.environ.setdefault("MUJOCO_GL", "glfw")
    set_seed(opt.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = load_config_with_stats(opt.config)
    model = load_model(config, opt.checkpoint, device)

    tasks = TASK_GROUPS[config["data"]["task_group"]] if opt.tasks == "all" else opt.tasks.split(",")
    results = {}
    for task_name in tasks:
        expected_group = task_group_for_task(task_name)
        if expected_group != config["data"]["task_group"]:
            raise ValueError(f"{task_name} is a {expected_group} task but config is for {config['data']['task_group']}")
        successes, episodes, rate = eval_one_task(model, config, task_name, opt, device)
        results[task_name] = {"successes": successes, "episodes": episodes, "success_rate": rate}

    total_successes = sum(item["successes"] for item in results.values())
    total_episodes = sum(item["episodes"] for item in results.values())
    results["average"] = {
        "successes": total_successes,
        "episodes": total_episodes,
        "success_rate": total_successes / max(total_episodes, 1),
    }
    output_dir = Path(opt.output) / config["data"]["regime"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"summary_{config['data']['task_group']}_{time.strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tasks", default="all", help="all or comma-separated task names")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", dest="max_steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="./outputs/dexjoco_deco_eval")
    parser.add_argument("--render-mode", dest="render_mode", choices=["rgb_array", "human"], default="rgb_array")
    parser.add_argument("--replan-ratio", dest="replan_ratio", type=float, default=0.8)
    parser.add_argument("--randomize-dynamics", dest="randomize_dynamics", action="store_true")
    parser.add_argument("--save-video", dest="save_video", action="store_true")
    main(parser.parse_args())
