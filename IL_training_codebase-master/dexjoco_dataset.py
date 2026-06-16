import os
import random
from collections import OrderedDict

import imageio
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dexjoco_constants import (
    DEXJOCO_DATASET_ROOT,
    TASK_GROUP_DIMS,
    TASK_GROUPS,
    camera_keys_for_task,
    dataset_dir,
)
from dexjoco_stats import load_or_compute_stats


class DexJoCoLeRobotDataset(Dataset):
    def __init__(
        self,
        data_root=DEXJOCO_DATASET_ROOT,
        task_group="single",
        regime="rand_obj",
        train=True,
        transform=None,
        chunk_size=30,
        norm_type="mean_std",
        stats_path=None,
        train_ratio=0.9,
        cache_videos=0,
        **_,
    ):
        self.data_root = data_root
        self.task_group = task_group
        self.regime = regime
        self.train = train
        self.transform = transform
        self.chunk_size = int(chunk_size)
        self.norm_type = norm_type
        self.base_dir = dataset_dir(data_root, regime)
        self.tasks = TASK_GROUPS[task_group]
        self.task_to_idx = {task: idx for idx, task in enumerate(self.tasks)}
        self.dims = TASK_GROUP_DIMS[task_group]
        self.cache_videos = int(cache_videos)
        self._video_cache = OrderedDict()

        stats = load_or_compute_stats(data_root, task_group, regime, stats_path)
        self.obs_mean = torch.tensor(stats["observation_mean"], dtype=torch.float32)
        self.obs_std = torch.tensor(stats["observation_std"], dtype=torch.float32).clamp_min(1e-8)
        self.obs_min = torch.tensor(stats["observation_min"], dtype=torch.float32)
        self.obs_max = torch.tensor(stats["observation_max"], dtype=torch.float32)
        self.action_mean = torch.tensor(stats["action_mean"], dtype=torch.float32)
        self.action_std = torch.tensor(stats["action_std"], dtype=torch.float32).clamp_min(1e-8)
        self.action_min = torch.tensor(stats["action_min"], dtype=torch.float32)
        self.action_max = torch.tensor(stats["action_max"], dtype=torch.float32)

        self.task_data = {}
        self.samples = []
        for task in self.tasks:
            self._load_task(task, train_ratio)

    def _load_task(self, task, train_ratio):
        task_root = os.path.join(self.base_dir, task)
        info_path = os.path.join(task_root, "meta", "info.json")
        data_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
        episodes_path = os.path.join(task_root, "meta", "episodes", "chunk-000", "file-000.parquet")
        if not os.path.exists(info_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"DexJoCo task {task} is incomplete under {task_root}")

        import json

        with open(info_path, "r") as f:
            info = json.load(f)
        data = pd.read_parquet(data_path)
        episodes = pd.read_parquet(episodes_path).set_index("episode_index")

        cam1, cam2 = camera_keys_for_task(task, self.regime)
        for cam in (cam1, cam2):
            if cam not in info["features"]:
                raise KeyError(f"{task} does not contain camera {cam}")

        episode_ids = sorted(data["episode_index"].unique().tolist())
        rng = random.Random(42)
        rng.shuffle(episode_ids)
        split = int(len(episode_ids) * train_ratio)
        selected = set(episode_ids[:split] if self.train else episode_ids[split:])

        episode_ranges = {}
        for episode_id, group in data.groupby("episode_index", sort=False):
            positions = group.index.to_numpy()
            episode_ranges[int(episode_id)] = (int(positions[0]), int(positions[-1]) + 1)

        self.task_data[task] = {
            "root": task_root,
            "info": info,
            "data": data,
            "episodes": episodes,
            "episode_ranges": episode_ranges,
            "cameras": (cam1, cam2),
            "fps": float(info.get("fps", 30)),
        }

        for pos, episode_id in enumerate(data["episode_index"].tolist()):
            if int(episode_id) in selected:
                self.samples.append((task, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        task, pos = self.samples[index]
        bundle = self.task_data[task]
        data = bundle["data"]
        row = data.iloc[pos]
        episode_id = int(row["episode_index"])
        frame_index = int(row["frame_index"])

        img1 = self._read_image(task, episode_id, frame_index, bundle["cameras"][0])
        img2 = self._read_image(task, episode_id, frame_index, bundle["cameras"][1])

        seed = random.randint(0, 1_000_000_000)
        if self.transform is not None:
            self._seed_all(seed)
            img1 = self.transform(img1)
            self._seed_all(seed)
            img2 = self.transform(img2)

        end = min(pos + self.chunk_size, bundle["episode_ranges"][episode_id][1])
        action_np = np.stack(data.iloc[pos:end]["action"].to_numpy()).astype(np.float32)
        action = torch.from_numpy(action_np)
        action_len = action.shape[0]
        if action_len < self.chunk_size:
            pad = action[-1:].repeat(self.chunk_size - action_len, 1)
            action = torch.cat([action, pad], dim=0)
        mask = torch.arange(self.chunk_size) < action_len

        obs = torch.tensor(np.asarray(row["observation.state"], dtype=np.float32))
        if self.norm_type == "min_max":
            obs = (obs - self.obs_min) / (self.obs_max - self.obs_min).clamp_min(1e-8)
            action = (action - self.action_min[None, :]) / (
                self.action_max[None, :] - self.action_min[None, :]
            ).clamp_min(1e-8)
        else:
            obs = (obs - self.obs_mean) / self.obs_std
            action = (action - self.action_mean[None, :]) / self.action_std[None, :]

        tac1 = torch.tensor([-1.0], dtype=torch.float32)
        tac2 = torch.tensor([-1.0], dtype=torch.float32)
        task_idx = torch.tensor(self.task_to_idx[task], dtype=torch.long)
        return img1, img2, tac1, tac2, obs.float(), action.float(), mask, task_idx

    def _read_image(self, task, episode_id, frame_index, camera_key):
        bundle = self.task_data[task]
        episode = bundle["episodes"].loc[episode_id]
        chunk_col = f"videos/{camera_key}/chunk_index"
        file_col = f"videos/{camera_key}/file_index"
        start_col = f"videos/{camera_key}/from_timestamp"
        chunk_idx = int(episode[chunk_col])
        file_idx = int(episode[file_col])
        from_timestamp = float(episode[start_col])
        video_frame = int(round(from_timestamp * bundle["fps"])) + int(frame_index)
        video_path = os.path.join(
            bundle["root"],
            "videos",
            camera_key,
            f"chunk-{chunk_idx:03d}",
            f"file-{file_idx:03d}.mp4",
        )
        frame = self._read_video_frame(video_path, video_frame)
        return Image.fromarray(frame)

    def _read_video_frame(self, video_path, frame_index):
        cache_key = (video_path, frame_index)
        if cache_key in self._video_cache:
            frame = self._video_cache.pop(cache_key)
            self._video_cache[cache_key] = frame
            return frame
        try:
            frame = iio.imread(video_path, index=frame_index)
        except Exception as first_error:
            try:
                reader = imageio.get_reader(video_path, "ffmpeg")
                frame = reader.get_data(frame_index)
                reader.close()
            except Exception as second_error:
                raise RuntimeError(
                    "Cannot decode DexJoCo mp4 frames. Install a video backend with "
                    "`pip install imageio[ffmpeg]` or `pip install imageio[pyav]`."
                ) from second_error or first_error

        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        frame = frame[..., :3].astype(np.uint8, copy=False)
        if self.cache_videos > 0:
            self._video_cache[cache_key] = frame
            while len(self._video_cache) > self.cache_videos:
                self._video_cache.popitem(last=False)
        return frame

    @staticmethod
    def _seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


if __name__ == "__main__":
    from torchvision.transforms import v2 as transforms
    from dataset import letterbox

    tfm = transforms.Compose([
        letterbox(256, fill=128),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ])
    ds = DexJoCoLeRobotDataset(task_group="single", regime="rand_obj", train=False, transform=tfm)
    sample = ds[0]
    print(sample[0].shape, sample[1].shape, sample[4].shape, sample[5].shape, sample[6].shape, sample[7])
