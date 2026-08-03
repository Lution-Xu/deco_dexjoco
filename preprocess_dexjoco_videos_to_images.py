import argparse
import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import yaml
from tqdm import tqdm

from dexjoco_constants import (
    TASK_GROUPS,
    dataset_dir,
    policy_camera_keys_for_task,
)


def _normalize_ext(ext):
    ext = str(ext).lower().lstrip(".")
    if ext == "jpeg":
        ext = "jpg"
    if ext not in {"jpg", "png"}:
        raise ValueError("--ext must be jpg or png")
    return ext


def _jpeg_qscale(quality):
    if quality >= 95:
        return 2
    if quality >= 90:
        return 3
    if quality >= 80:
        return 5
    return 8


def _load_data_config(config_path):
    if not config_path:
        return {}
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config.get("data", {})


def _required_episode_value(episode, column, task, episode_id):
    if column not in episode.index:
        raise KeyError(f"{task} episode {episode_id} is missing metadata column {column}")
    return episode[column]


def _collect_episode_records(data_root, task_group, regime, image_root, ext):
    base_dir = dataset_dir(data_root, regime)
    records = []
    for task in TASK_GROUPS[task_group]:
        task_root = os.path.join(base_dir, task)
        episodes_path = os.path.join(task_root, "meta", "episodes", "chunk-000", "file-000.parquet")
        if not os.path.exists(episodes_path):
            raise FileNotFoundError(f"DexJoCo episodes metadata is missing: {episodes_path}")

        episodes = pd.read_parquet(episodes_path).sort_values("episode_index")
        camera_keys = policy_camera_keys_for_task(task, regime)
        for _, episode in episodes.iterrows():
            episode_id = int(episode["episode_index"])
            length = int(_required_episode_value(episode, "length", task, episode_id))
            if length <= 0:
                raise ValueError(f"{task} episode {episode_id} has invalid length {length}")

            episode_dir = os.path.join(image_root, task, f"episode_{episode_id:06d}")
            jobs = []
            sources = {}
            for image_index, camera_key in enumerate(camera_keys, start=1):
                prefix = f"videos/{camera_key}"
                chunk_idx = int(_required_episode_value(episode, f"{prefix}/chunk_index", task, episode_id))
                file_idx = int(_required_episode_value(episode, f"{prefix}/file_index", task, episode_id))
                start = float(_required_episode_value(episode, f"{prefix}/from_timestamp", task, episode_id))
                video_path = os.path.join(
                    task_root,
                    "videos",
                    camera_key,
                    f"chunk-{chunk_idx:03d}",
                    f"file-{file_idx:03d}.mp4",
                )
                if not os.path.exists(video_path):
                    raise FileNotFoundError(f"DexJoCo source video is missing: {video_path}")

                slot = f"image{image_index}"
                image_dir = os.path.join(episode_dir, slot)
                pattern = os.path.join(image_dir, f"%06d.{ext}")
                jobs.append(("episode", video_path, image_dir, pattern, start, length))
                sources[slot] = {"camera_key": camera_key, "from_timestamp": start}

            records.append({
                "task": task,
                "episode_index": episode_id,
                "episode_dir": episode_dir,
                "jobs": jobs,
                "manifest": {
                    "task": task,
                    "episode_index": episode_id,
                    "length": length,
                    "images": sources,
                },
            })
    return records


def _done_path(image_dir):
    return os.path.join(image_dir, "_images.done")


def _count_images(image_dir, ext):
    if not os.path.isdir(image_dir):
        return 0
    return sum(1 for name in os.listdir(image_dir) if name.endswith(f".{ext}"))


def _process_one(job):
    kind = job[0]
    if kind != "episode":
        raise ValueError(f"Unknown preprocess job kind: {kind}")
    _, video_path, image_dir, pattern, start, expected_frames, ext, quality, overwrite = job

    done_path = _done_path(image_dir)
    if os.path.exists(done_path) and not overwrite:
        with open(done_path, "r") as f:
            meta = json.load(f)
        frames = _count_images(image_dir, ext)
        if (
            frames == expected_frames
            and int(meta.get("frames", -1)) == expected_frames
            and meta.get("source") == video_path
            and abs(float(meta.get("from_timestamp", -1.0)) - start) < 1e-6
        ):
            return kind, frames, True

    os.makedirs(image_dir, exist_ok=True)
    for name in os.listdir(image_dir):
        if name.endswith(f".{ext}") or name == "_images.done":
            os.remove(os.path.join(image_dir, name))

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.9f}",
        "-i",
        video_path,
        "-frames:v",
        str(expected_frames),
    ]
    cmd.extend(["-start_number", "0"])
    if ext == "jpg":
        cmd.extend(["-q:v", str(_jpeg_qscale(quality))])
    cmd.append(pattern)
    subprocess.run(cmd, check=True)

    frames = _count_images(image_dir, ext)
    if frames != expected_frames:
        raise RuntimeError(
            f"Decoded {frames} frames but expected {expected_frames}: "
            f"source={video_path}, start={start}, output={image_dir}"
        )
    with open(done_path, "w") as f:
        json.dump(
            {
                "kind": kind,
                "source": video_path,
                "from_timestamp": start,
                "frames": frames,
            },
            f,
            indent=2,
        )
    return kind, frames, False


def _write_episode_manifest(record):
    os.makedirs(record["episode_dir"], exist_ok=True)
    path = os.path.join(record["episode_dir"], "episode.json")
    with open(path, "w") as f:
        json.dump(record["manifest"], f, indent=2)


def _write_task_manifests(records):
    by_task = {}
    for record in records:
        by_task.setdefault(record["task"], []).append(record)
    for task, task_records in by_task.items():
        task_records.sort(key=lambda item: item["episode_index"])
        task_dir = os.path.dirname(task_records[0]["episode_dir"])
        with open(os.path.join(task_dir, "episodes.json"), "w") as f:
            json.dump(
                {
                    "task": task,
                    "num_episodes": len(task_records),
                    "episode_indices": [item["episode_index"] for item in task_records],
                },
                f,
                indent=2,
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Convert DexJoCo videos to image frames for training.")
    parser.add_argument("--config", type=str, default=None, help="Training YAML to read data defaults from.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--task-group", choices=["single", "dual"], default=None)
    parser.add_argument("--regime", choices=["rand_obj", "rand_full"], default=None)
    parser.add_argument("--output", type=str, default=None, help="Image root written into data.image_root.")
    parser.add_argument("--ext", choices=["jpg", "jpeg", "png"], default=None)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality, mapped to ffmpeg q:v.")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N episodes for testing.")
    return parser.parse_args()


def main():
    args = parse_args()
    defaults = _load_data_config(args.config)
    data_root = args.data_root or defaults.get("data_root")
    task_group = args.task_group or defaults.get("task_group")
    regime = args.regime or defaults.get("regime")
    image_root = args.output or defaults.get("image_root")
    ext = _normalize_ext(args.ext or defaults.get("image_ext", "jpg"))

    missing = [
        name
        for name, value in {
            "data_root": data_root,
            "task_group": task_group,
            "regime": regime,
            "output/image_root": image_root,
        }.items()
        if not value
    ]
    if missing:
        raise ValueError(f"Missing required values: {', '.join(missing)}")

    records = _collect_episode_records(data_root, task_group, regime, image_root, ext)
    if args.limit is not None:
        records = records[: args.limit]
    base_jobs = [job for record in records for job in record["jobs"]]
    jobs = [
        (*job, ext, args.quality, args.overwrite)
        for job in base_jobs
    ]

    print(f"Found {len(records)} episodes and {len(jobs)} camera streams")
    print(f"Image root: {image_root}")

    decoded = 0
    skipped = 0
    total_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_process_one, job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures)):
            _, frames, was_skipped = future.result()
            total_frames += frames
            if was_skipped:
                skipped += 1
            else:
                decoded += 1

    for record in records:
        _write_episode_manifest(record)
    _write_task_manifests(records)

    print(
        f"Done. episodes={len(records)}, decoded_streams={decoded}, "
        f"skipped_streams={skipped}, frames={total_frames}"
    )


if __name__ == "__main__":
    main()
