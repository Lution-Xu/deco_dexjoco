import importlib.util
import os
import random
from collections import OrderedDict

import imageio
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from dexjoco_constants_unify import (
    DEXJOCO_DATASET_ROOT,
    UNIFIED_DIMS,
    dataset_dir,
    policy_camera_keys_for_task,
    resolve_unify_task_names,
    task_group_for_task,
    unified_task_index,
)
from dexjoco_stats_unify import load_or_compute_stats


class DexJoCoUnifiedDataset(Dataset):
    def __init__(
        self,
        data_root=DEXJOCO_DATASET_ROOT,
        task_group="unify",
        regime="rand_obj",
        train=True,
        transform=None,
        chunk_size=30,
        task=None,
        tasks=None,
        norm_type="mean_std",
        stats_path=None,
        train_ratio=0.9,
        cache_videos=0,
        cache_readers=4,
        video_backend="auto",
        image_root=None,
        image_ext="jpg",
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
        self.tasks = resolve_unify_task_names(task_group, task=task, tasks=tasks)
        self.cache_videos = int(cache_videos)
        self.cache_readers = int(cache_readers)
        self.image_root = image_root
        self.image_ext = str(image_ext).lstrip(".")
        if video_backend == "auto":
            self.video_backend = "pyav" if importlib.util.find_spec("av") else "imageio"
        else:
            self.video_backend = video_backend
        self._video_cache = OrderedDict()
        self._pyav_cache = OrderedDict()
        self._reader_cache = OrderedDict()

        stats = load_or_compute_stats(
            data_root,
            task_group,
            regime,
            stats_path,
            self.chunk_size,
            task=task,
            tasks=tasks,
        )
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
        for task_name in self.tasks:
            self._load_task(task_name, train_ratio)

    def _load_task(self, task_name, train_ratio):
        task_root = os.path.join(self.base_dir, task_name)
        info_path = os.path.join(task_root, "meta", "info.json")
        data_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
        episodes_path = os.path.join(task_root, "meta", "episodes", "chunk-000", "file-000.parquet")
        if not os.path.exists(info_path) or not os.path.exists(data_path):
            raise FileNotFoundError(f"DexJoCo task {task_name} is incomplete under {task_root}")

        import json

        with open(info_path, "r") as f:
            info = json.load(f)
        data = pd.read_parquet(data_path)
        episodes = pd.read_parquet(episodes_path).set_index("episode_index")
        cameras = policy_camera_keys_for_task(task_name, self.regime)
        for camera_key in cameras:
            if camera_key not in info["features"]:
                raise KeyError(f"{task_name} does not contain camera {camera_key}")

        episode_ids = sorted(data["episode_index"].unique().tolist())
        rng = random.Random(42)
        rng.shuffle(episode_ids)
        split = int(len(episode_ids) * train_ratio)
        selected = set(episode_ids[:split] if self.train else episode_ids[split:])

        episode_ranges = {}
        for episode_id, group in data.groupby("episode_index", sort=False):
            positions = group.index.to_numpy()
            episode_ranges[int(episode_id)] = (int(positions[0]), int(positions[-1]) + 1)

        self.task_data[task_name] = {
            "root": task_root,
            "info": info,
            "data": data,
            "episodes": episodes,
            "episode_ranges": episode_ranges,
            "cameras": cameras,
            "fps": float(info.get("fps", 30)),
        }

        for pos, episode_id in enumerate(data["episode_index"].tolist()):
            if int(episode_id) in selected:
                self.samples.append((task_name, pos))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        task_name, pos = self.samples[index]
        group = task_group_for_task(task_name)
        bundle = self.task_data[task_name]
        data = bundle["data"]
        row = data.iloc[pos]
        episode_id = int(row["episode_index"])
        frame_index = int(row["frame_index"])

        if group == "dual" and not self.image_root:
            raise RuntimeError(
                "Dual-arm unified training expects preprocessed images. "
                "Run preprocess_dexjoco_videos_to_images.py and set data.image_root."
            )

        raw_images = [
            self._read_image(task_name, episode_id, frame_index, camera_key)
            for camera_key in bundle["cameras"]
        ]
        camera_mask = torch.zeros(UNIFIED_DIMS["num_cameras"], dtype=torch.bool)
        camera_mask[: len(raw_images)] = True

        seed = random.randint(0, 1_000_000_000)
        images = []
        if self.transform is not None:
            for image in raw_images:
                self._seed_all(seed)
                images.append(self.transform(image))
        else:
            images = raw_images
        if len(images) < UNIFIED_DIMS["num_cameras"]:
            if self.transform is None:
                pad_image = Image.new("RGB", raw_images[0].size)
            else:
                pad_image = torch.zeros_like(images[0])
            images.extend([pad_image] * (UNIFIED_DIMS["num_cameras"] - len(images)))
        if self.transform is None:
            raise RuntimeError("DexJoCoUnifiedDataset requires a tensor image transform for batching.")
        images = torch.stack(images, dim=0)

        end = min(pos + self.chunk_size, bundle["episode_ranges"][episode_id][1])
        action_np = np.stack(data.iloc[pos:end]["action"].to_numpy()).astype(np.float32)
        action = torch.from_numpy(action_np)
        action_len = action.shape[0]
        if action_len < self.chunk_size:
            pad = action[-1:].repeat(self.chunk_size - action_len, 1)
            action = torch.cat([action, pad], dim=0)
        time_mask = torch.arange(self.chunk_size) < action_len

        obs = torch.tensor(np.asarray(row["observation.state"], dtype=np.float32))
        obs, obs_mask = self._unify_obs(obs, group)
        action, action_mask = self._unify_action(action, group)
        obs = self._normalize(obs, obs_mask, self.obs_mean, self.obs_std, self.obs_min, self.obs_max)
        action = self._normalize(
            action,
            action_mask,
            self.action_mean,
            self.action_std,
            self.action_min,
            self.action_max,
        )

        tac1 = torch.tensor([-1.0], dtype=torch.float32)
        tac2 = torch.tensor([-1.0], dtype=torch.float32)
        task_idx = torch.tensor(unified_task_index(task_name), dtype=torch.long)
        return (
            images.float(),
            camera_mask,
            tac1,
            tac2,
            obs.float(),
            obs_mask,
            action.float(),
            action_mask,
            time_mask,
            task_idx,
        )

    def _unify_action(self, action, group):
        out = torch.zeros(self.chunk_size, UNIFIED_DIMS["action_dim"], dtype=torch.float32)
        mask = torch.zeros(UNIFIED_DIMS["action_dim"], dtype=torch.bool)
        if group == "single":
            out[:, :22] = action[:, :22]
            mask[:22] = True
        else:
            out[:, :] = action[:, :44]
            mask[:] = True
        return out, mask

    def _unify_obs(self, obs, group):
        out = torch.zeros(UNIFIED_DIMS["obs_dim"], dtype=torch.float32)
        mask = torch.zeros(UNIFIED_DIMS["obs_dim"], dtype=torch.bool)
        if group == "single":
            out[:7] = obs[:7]
            out[14:30] = obs[7:23]
            mask[:7] = True
            mask[14:30] = True
        else:
            out[:] = obs[:46]
            mask[:] = True
        return out, mask

    def _normalize(self, value, mask, mean, std, min_value, max_value):
        out = torch.zeros_like(value)
        if value.ndim == 1:
            valid = mask
            if self.norm_type == "min_max":
                denom = (max_value - min_value).clamp_min(1e-8)
                out[valid] = ((value - min_value) / denom)[valid].clamp(0, 1.0)
            else:
                out[valid] = ((value - mean) / std)[valid]
            return out

        valid = mask[None, :].expand_as(value)
        if self.norm_type == "min_max":
            denom = (max_value - min_value).clamp_min(1e-8)
            normalized = ((value - min_value[None, :]) / denom[None, :]).clamp(0, 1.0)
        else:
            normalized = (value - mean[None, :]) / std[None, :]
        out[valid] = normalized[valid]
        return out

    def _read_image(self, task_name, episode_id, frame_index, camera_key, metadata_camera_key=None):
        bundle = self.task_data[task_name]
        if self.image_root:
            image_index = bundle["cameras"].index(camera_key) + 1
            image_path = self._image_path(task_name, episode_id, image_index, frame_index)
            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"DexJoCo image frame is missing: {image_path}. "
                    "Run preprocess_dexjoco_videos_to_images.py before training."
                )
            with Image.open(image_path) as image:
                return image.convert("RGB")

        episode = bundle["episodes"].loc[episode_id]
        metadata_camera_key = metadata_camera_key or camera_key
        chunk_col = f"videos/{metadata_camera_key}/chunk_index"
        file_col = f"videos/{metadata_camera_key}/file_index"
        start_col = f"videos/{metadata_camera_key}/from_timestamp"
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

    def _image_path(self, task_name, episode_id, image_index, frame_index):
        return os.path.join(
            self.image_root,
            task_name,
            f"episode_{int(episode_id):06d}",
            f"image{int(image_index)}",
            f"{int(frame_index):06d}.{self.image_ext}",
        )

    def _read_video_frame(self, video_path, frame_index):
        cache_key = (video_path, frame_index)
        if cache_key in self._video_cache:
            frame = self._video_cache.pop(cache_key)
            self._video_cache[cache_key] = frame
            return frame

        frame = None
        if self.video_backend == "pyav":
            try:
                frame = self._read_video_frame_pyav(video_path, frame_index)
            except ImportError:
                raise RuntimeError("PyAV is not installed. Install it with `pip install av`.")

        if frame is None:
            frame = self._read_video_frame_imageio(video_path, frame_index)

        frame = np.asarray(frame)
        if frame.ndim == 2:
            frame = np.repeat(frame[..., None], 3, axis=-1)
        frame = frame[..., :3].astype(np.uint8, copy=False)
        if self.cache_videos > 0:
            self._video_cache[cache_key] = frame
            while len(self._video_cache) > self.cache_videos:
                self._video_cache.popitem(last=False)
        return frame

    def _read_video_frame_pyav(self, video_path, frame_index):
        container, stream, fps = self._get_pyav_container(video_path)
        try:
            timestamp = int((frame_index / fps) / stream.time_base)
            container.seek(max(timestamp, 0), stream=stream, backward=True)
            last_frame = None
            decoded_index = -1
            for decoded in container.decode(stream):
                decoded_index += 1
                last_frame = decoded
                if decoded.pts is None:
                    if decoded_index >= frame_index:
                        return decoded.to_ndarray(format="rgb24")
                    continue
                current = int(round(float(decoded.pts * stream.time_base) * fps))
                if current >= frame_index:
                    return decoded.to_ndarray(format="rgb24")
            if last_frame is not None:
                return last_frame.to_ndarray(format="rgb24")
        except Exception:
            self._drop_pyav_container(video_path)
            raise
        raise IndexError(f"Frame {frame_index} is outside {video_path}")

    def _get_pyav_container(self, video_path):
        if video_path in self._pyav_cache:
            bundle = self._pyav_cache.pop(video_path)
            self._pyav_cache[video_path] = bundle
            return bundle

        import av

        container = av.open(video_path)
        stream = container.streams.video[0]
        if stream.average_rate:
            fps = float(stream.average_rate)
        elif stream.base_rate:
            fps = float(stream.base_rate)
        else:
            fps = 30.0
        bundle = (container, stream, fps)
        if self.cache_readers > 0:
            self._pyav_cache[video_path] = bundle
            while len(self._pyav_cache) > self.cache_readers:
                old_container, _, _ = self._pyav_cache.popitem(last=False)[1]
                old_container.close()
        return bundle

    def _drop_pyav_container(self, video_path):
        bundle = self._pyav_cache.pop(video_path, None)
        if bundle is not None:
            bundle[0].close()

    def _read_video_frame_imageio(self, video_path, frame_index):
        reader = self._get_imageio_reader(video_path)
        try:
            return reader.get_data(frame_index)
        except Exception as first_error:
            self._drop_imageio_reader(video_path)
            try:
                reader = self._get_imageio_reader(video_path)
                return reader.get_data(frame_index)
            except Exception as second_error:
                raise RuntimeError(
                    "Cannot decode DexJoCo mp4 frames. Install a video backend with "
                    "`pip install imageio[ffmpeg]` or `pip install imageio[pyav]`."
                ) from second_error or first_error

    def _get_imageio_reader(self, video_path):
        if video_path in self._reader_cache:
            reader = self._reader_cache.pop(video_path)
            self._reader_cache[video_path] = reader
            return reader
        reader = imageio.get_reader(video_path, "ffmpeg")
        if self.cache_readers > 0:
            self._reader_cache[video_path] = reader
            while len(self._reader_cache) > self.cache_readers:
                _, old_reader = self._reader_cache.popitem(last=False)
                old_reader.close()
        return reader

    def _drop_imageio_reader(self, video_path):
        reader = self._reader_cache.pop(video_path, None)
        if reader is not None:
            reader.close()

    def __del__(self):
        for container, _, _ in getattr(self, "_pyav_cache", {}).values():
            try:
                container.close()
            except Exception:
                pass
        for reader in getattr(self, "_reader_cache", {}).values():
            try:
                reader.close()
            except Exception:
                pass

    @staticmethod
    def _seed_all(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
