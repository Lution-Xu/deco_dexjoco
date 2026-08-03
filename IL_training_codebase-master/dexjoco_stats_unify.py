import json
import os

import numpy as np
import pandas as pd

from dexjoco_constants_unify import (
    DEXJOCO_DATASET_ROOT,
    UNIFIED_DIMS,
    dataset_dir,
    resolve_unify_task_names,
    task_group_for_task,
)


def _normalize_regimes(regime):
    if regime == "both":
        return ["rand_obj", "rand_full"]
    if isinstance(regime, (list, tuple)):
        return list(regime)
    return [regime]


def _empty_acc(dim):
    return {
        "count": np.zeros(dim, dtype=np.int64),
        "sum": np.zeros(dim, dtype=np.float64),
        "sum_sq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
    }


def _update_acc(acc, values, mask):
    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if values.ndim == 1:
        values = values[None, :]
    if mask.ndim == 1:
        mask = np.broadcast_to(mask[None, :], values.shape)

    valid_counts = mask.sum(axis=0)
    if not valid_counts.any():
        return
    masked_values = np.where(mask, values, 0.0)
    acc["count"] += valid_counts
    acc["sum"] += masked_values.sum(axis=0)
    acc["sum_sq"] += np.square(masked_values).sum(axis=0)
    for dim in np.where(valid_counts > 0)[0]:
        dim_values = values[mask[:, dim], dim]
        acc["min"][dim] = min(acc["min"][dim], dim_values.min())
        acc["max"][dim] = max(acc["max"][dim], dim_values.max())


def _finalize_acc(acc):
    count = acc["count"]
    valid = count > 0
    mean = np.zeros_like(acc["sum"], dtype=np.float64)
    std = np.ones_like(acc["sum"], dtype=np.float64)
    min_value = np.zeros_like(acc["sum"], dtype=np.float64)
    max_value = np.zeros_like(acc["sum"], dtype=np.float64)

    mean[valid] = acc["sum"][valid] / count[valid]
    var = np.zeros_like(mean)
    var[valid] = acc["sum_sq"][valid] / count[valid] - np.square(mean[valid])
    std[valid] = np.sqrt(np.maximum(var[valid], 1e-12))
    min_value[valid] = acc["min"][valid]
    max_value[valid] = acc["max"][valid]
    return {
        "mean": mean.astype(float).tolist(),
        "std": std.astype(float).tolist(),
        "min": min_value.astype(float).tolist(),
        "max": max_value.astype(float).tolist(),
        "count": count.astype(int).tolist(),
        "total_count": int(count.sum()),
    }


def _unify_action(action, task_name):
    action = np.asarray(action, dtype=np.float64)
    out = np.zeros(UNIFIED_DIMS["action_dim"], dtype=np.float64)
    mask = np.zeros(UNIFIED_DIMS["action_dim"], dtype=bool)
    if task_group_for_task(task_name) == "single":
        out[:22] = action[:22]
        mask[:22] = True
    else:
        out[:] = action[:44]
        mask[:] = True
    return out, mask


def _unify_observation(obs, task_name):
    obs = np.asarray(obs, dtype=np.float64)
    out = np.zeros(UNIFIED_DIMS["obs_dim"], dtype=np.float64)
    mask = np.zeros(UNIFIED_DIMS["obs_dim"], dtype=bool)
    if task_group_for_task(task_name) == "single":
        out[:7] = obs[:7]
        out[14:30] = obs[7:23]
        mask[:7] = True
        mask[14:30] = True
    else:
        out[:] = obs[:46]
        mask[:] = True
    return out, mask


def _read_task_data(task_root):
    data_path = os.path.join(task_root, "data", "chunk-000", "file-000.parquet")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing DexJoCo data parquet: {data_path}")
    return pd.read_parquet(data_path, columns=["action", "observation.state", "episode_index"])


def compute_dexjoco_unify_stats(
    data_root,
    task_group,
    regime,
    chunk_size=30,
    task=None,
    tasks=None,
):
    selected_tasks = resolve_unify_task_names(task_group, task=task, tasks=tasks)
    regimes = _normalize_regimes(regime)
    action_acc = _empty_acc(UNIFIED_DIMS["action_dim"])
    obs_acc = _empty_acc(UNIFIED_DIMS["obs_dim"])

    task_roots = []
    for regime_name in regimes:
        base_dir = dataset_dir(data_root, regime_name)
        task_roots.extend((task_name, os.path.join(base_dir, task_name)) for task_name in selected_tasks)
    missing = [path for _, path in task_roots if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing DexJoCo task directories: {missing}")

    for task_name, task_root in task_roots:
        data = _read_task_data(task_root)
        obs_values = []
        obs_masks = []
        for obs in data["observation.state"].to_numpy():
            unified_obs, obs_mask = _unify_observation(obs, task_name)
            obs_values.append(unified_obs)
            obs_masks.append(obs_mask)
        _update_acc(obs_acc, np.stack(obs_values), np.stack(obs_masks))

        for _, group in data.groupby("episode_index", sort=False):
            raw_actions = np.stack(group["action"].to_numpy())
            unified_actions = []
            action_masks = []
            for action in raw_actions:
                unified_action, action_mask = _unify_action(action, task_name)
                unified_actions.append(unified_action)
                action_masks.append(action_mask)
            unified_actions = np.stack(unified_actions)
            action_masks = np.stack(action_masks)
            episode_len = unified_actions.shape[0]
            base_indices = np.arange(episode_len)
            for offset in range(int(chunk_size)):
                indices = np.minimum(base_indices + offset, episode_len - 1)
                _update_acc(action_acc, unified_actions[indices], action_masks[indices])

    action = _finalize_acc(action_acc)
    obs = _finalize_acc(obs_acc)
    return {
        "task_group": task_group,
        "regime": regimes[0] if len(regimes) == 1 else regimes,
        "tasks": selected_tasks,
        "chunk_size": int(chunk_size),
        "action_stat_mode": "unified_chunk_flattened_masked",
        "action_mean": action["mean"],
        "action_std": action["std"],
        "action_min": action["min"],
        "action_max": action["max"],
        "observation_mean": obs["mean"],
        "observation_std": obs["std"],
        "observation_min": obs["min"],
        "observation_max": obs["max"],
        "action_count": action["count"],
        "observation_count": obs["count"],
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
    selected_tasks = resolve_unify_task_names(task_group, task=task, tasks=tasks)
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
    return compute_dexjoco_unify_stats(
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
    parser.add_argument("--task-group", choices=sorted(["single", "dual", "unify"]), default="unify")
    parser.add_argument("--task", default=None)
    parser.add_argument("--tasks", default=None)
    parser.add_argument("--regime", choices=["rand_obj", "rand_full", "both"], required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tasks = args.tasks.split(",") if args.tasks else None
    stats = compute_dexjoco_unify_stats(
        args.data_root,
        args.task_group,
        args.regime,
        args.chunk_size,
        task=args.task,
        tasks=tasks,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote unified DexJoCo stats to {args.output}")


if __name__ == "__main__":
    main()
