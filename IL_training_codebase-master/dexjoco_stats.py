import json
import os

import numpy as np
import pandas as pd

from dexjoco_constants import (
    DEXJOCO_DATASET_ROOT,
    TASK_GROUPS,
    dataset_dir,
    resolve_task_names,
)


def _as_array(value):
    return np.asarray(value, dtype=np.float64)


def _read_feature_stats(task_root, feature):
    with open(os.path.join(task_root, "meta", "stats.json"), "r") as f:
        stats = json.load(f)[feature]
    return {
        "count": int(np.asarray(stats["count"]).reshape(-1)[0]),
        "mean": _as_array(stats["mean"]),
        "std": _as_array(stats["std"]),
        "min": _as_array(stats["min"]),
        "max": _as_array(stats["max"]),
    }


def _merge_feature_stats(task_roots, feature):
    parts = [_read_feature_stats(task_root, feature) for task_root in task_roots]
    total = sum(part["count"] for part in parts)
    if total <= 0:
        raise ValueError(f"No samples found while merging {feature}")

    mean = sum(part["count"] * part["mean"] for part in parts) / total
    var = sum(
        part["count"] * (part["std"] ** 2 + (part["mean"] - mean) ** 2)
        for part in parts
    ) / total

    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(np.maximum(var, 1e-12)).astype(float).tolist(),
        "min": np.minimum.reduce([part["min"] for part in parts]).astype(float).tolist(),
        "max": np.maximum.reduce([part["max"] for part in parts]).astype(float).tolist(),
        "count": total,
    }


def _read_task_actions(task_root):
    data_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing DexJoCo data parquet: {data_path}")
    data = pd.read_parquet(data_path, columns=["action", "episode_index"])
    return data


def _compute_task_chunk_action_stats(task_root, chunk_size):
    """Compute stats over the action chunks used by DECO training.

    This mirrors `DexJoCoLeRobotDataset.__getitem__`: for each frame, take the
    next `chunk_size` actions and pad the episode tail by repeating the last
    action. The stats are then computed over all chunk positions flattened as
    individual action vectors.
    """

    data = _read_task_actions(task_root)
    count = 0
    total = None
    total_sq = None
    min_value = None
    max_value = None

    for _, group in data.groupby("episode_index", sort=False):
        actions = np.stack(group["action"].to_numpy()).astype(np.float64, copy=False)
        episode_len, action_dim = actions.shape
        if episode_len <= 0:
            continue

        base_indices = np.arange(episode_len)
        for offset in range(chunk_size):
            indices = np.minimum(base_indices + offset, episode_len - 1)
            chunk_step = actions[indices]
            if total is None:
                total = np.zeros(action_dim, dtype=np.float64)
                total_sq = np.zeros(action_dim, dtype=np.float64)
                min_value = chunk_step.min(axis=0)
                max_value = chunk_step.max(axis=0)
            else:
                min_value = np.minimum(min_value, chunk_step.min(axis=0))
                max_value = np.maximum(max_value, chunk_step.max(axis=0))

            total += chunk_step.sum(axis=0)
            total_sq += np.square(chunk_step).sum(axis=0)
            count += episode_len

    if count <= 0 or total is None:
        raise ValueError(f"No action chunks found under {task_root}")

    mean = total / count
    var = total_sq / count - np.square(mean)
    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(np.maximum(var, 1e-12)).astype(float).tolist(),
        "min": min_value.astype(float).tolist(),
        "max": max_value.astype(float).tolist(),
        "count": count,
    }


def _merge_chunk_action_stats(task_roots, chunk_size):
    parts = [_compute_task_chunk_action_stats(task_root, chunk_size) for task_root in task_roots]
    total = sum(part["count"] for part in parts)
    if total <= 0:
        raise ValueError("No action chunks found while merging DexJoCo tasks")

    means = [_as_array(part["mean"]) for part in parts]
    stds = [_as_array(part["std"]) for part in parts]
    counts = [part["count"] for part in parts]
    mean = sum(count * part_mean for count, part_mean in zip(counts, means)) / total
    var = sum(
        count * (part_std ** 2 + (part_mean - mean) ** 2)
        for count, part_mean, part_std in zip(counts, means, stds)
    ) / total

    return {
        "mean": mean.astype(float).tolist(),
        "std": np.sqrt(np.maximum(var, 1e-12)).astype(float).tolist(),
        "min": np.minimum.reduce([_as_array(part["min"]) for part in parts]).astype(float).tolist(),
        "max": np.maximum.reduce([_as_array(part["max"]) for part in parts]).astype(float).tolist(),
        "count": total,
    }


def _normalize_regimes(regime):
    if regime == "both":
        return ["rand_obj", "rand_full"]
    if isinstance(regime, (list, tuple)):
        return list(regime)
    return [regime]


def compute_dexjoco_stats(
    data_root,
    task_group,
    regime,
    chunk_size=30,
    task=None,
    tasks=None,
):
    selected_tasks = resolve_task_names(task_group, task=task, tasks=tasks)
    regimes = _normalize_regimes(regime)
    task_roots = []
    for regime_name in regimes:
        base_dir = dataset_dir(data_root, regime_name)
        task_roots.extend(os.path.join(base_dir, task_name) for task_name in selected_tasks)
    missing = [path for path in task_roots if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing DexJoCo task directories: {missing}")

    action = _merge_chunk_action_stats(task_roots, int(chunk_size))
    state = _merge_feature_stats(task_roots, "observation.state")
    return {
        "task_group": task_group,
        "regime": regimes[0] if len(regimes) == 1 else regimes,
        "tasks": selected_tasks,
        "chunk_size": int(chunk_size),
        "action_stat_mode": "chunk_flattened",
        "action_mean": action["mean"],
        "action_std": action["std"],
        "action_min": action["min"],
        "action_max": action["max"],
        "observation_mean": state["mean"],
        "observation_std": state["std"],
        "observation_min": state["min"],
        "observation_max": state["max"],
        "action_count": action["count"],
        "observation_count": state["count"],
    }


def load_or_compute_stats(
    data_root,
    task_group,
    regime,
    stats_path=None,
    chunk_size=30,
    task=None,
    tasks=None,
):
    selected_tasks = resolve_task_names(task_group, task=task, tasks=tasks)
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        stats_tasks = stats.get("tasks")
        if stats_tasks is not None and list(stats_tasks) != selected_tasks:
            raise ValueError(
                f"Stats file {stats_path} was computed for tasks {stats_tasks}, "
                f"but this run selected {selected_tasks}. Use a task-specific "
                "stats_path or remove the stale stats file."
            )
        return stats
    return compute_dexjoco_stats(
        data_root,
        task_group,
        regime,
        chunk_size,
        tasks=selected_tasks,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEXJOCO_DATASET_ROOT)
    parser.add_argument("--task-group", choices=sorted(TASK_GROUPS), required=True)
    parser.add_argument("--task", default=None)
    parser.add_argument(
        "--regime", choices=["rand_obj", "rand_full", "both"], required=True
    )
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stats = compute_dexjoco_stats(
        args.data_root,
        args.task_group,
        args.regime,
        args.chunk_size,
        task=args.task,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote DexJoCo stats to {args.output}")


if __name__ == "__main__":
    main()
