import json
import os

import numpy as np

from dexjoco_constants import DEXJOCO_DATASET_ROOT, TASK_GROUPS, dataset_dir


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


def compute_dexjoco_stats(data_root, task_group, regime):
    tasks = TASK_GROUPS[task_group]
    base_dir = dataset_dir(data_root, regime)
    task_roots = [os.path.join(base_dir, task) for task in tasks]
    missing = [path for path in task_roots if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"Missing DexJoCo task directories: {missing}")

    action = _merge_feature_stats(task_roots, "action")
    state = _merge_feature_stats(task_roots, "observation.state")
    return {
        "task_group": task_group,
        "regime": regime,
        "tasks": tasks,
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


def load_or_compute_stats(data_root, task_group, regime, stats_path=None):
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            return json.load(f)
    return compute_dexjoco_stats(data_root, task_group, regime)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEXJOCO_DATASET_ROOT)
    parser.add_argument("--task-group", choices=sorted(TASK_GROUPS), required=True)
    parser.add_argument("--regime", choices=["rand_obj", "rand_full"], required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    stats = compute_dexjoco_stats(args.data_root, args.task_group, args.regime)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote DexJoCo stats to {args.output}")


if __name__ == "__main__":
    main()
