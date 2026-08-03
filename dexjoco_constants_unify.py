from dexjoco_constants import (
    DEXJOCO_DATASET_ROOT,
    DEXJOCO_PROMPTS,
    DEXJOCO_REPO_ROOT,
    DUAL_ARM_TASKS,
    REGIME_DIRS,
    SINGLE_ARM_TASKS,
    camera_keys_for_task,
    dataset_dir,
    policy_camera_keys_for_task,
    task_group_for_task,
)


UNIFIED_TASKS = list(SINGLE_ARM_TASKS) + list(DUAL_ARM_TASKS)
UNIFIED_TASK_GROUPS = {
    "unify": UNIFIED_TASKS,
    "single": SINGLE_ARM_TASKS,
    "dual": DUAL_ARM_TASKS,
}
UNIFIED_DIMS = {
    "action_dim": 44,
    "obs_dim": 46,
    "num_cameras": 3,
}

SINGLE_ACTION_SLICE = slice(0, 22)
SINGLE_OBS_ARM_SLICE = slice(0, 7)
SINGLE_OBS_HAND_SRC_SLICE = slice(7, 23)
SINGLE_OBS_HAND_DST_SLICE = slice(14, 30)


def resolve_unify_task_names(task_group="unify", task=None, tasks=None):
    if task_group not in UNIFIED_TASK_GROUPS:
        raise ValueError(
            f"Unknown unified DexJoCo task group: {task_group!r}; "
            f"expected one of {sorted(UNIFIED_TASK_GROUPS)}"
        )
    if task is not None and tasks is not None:
        raise ValueError("Set only one of data.task or data.tasks")

    if task is not None:
        selected = [task]
    elif tasks is None:
        selected = list(UNIFIED_TASK_GROUPS[task_group])
    elif isinstance(tasks, str):
        selected = [tasks]
    else:
        selected = list(tasks)

    if not selected:
        raise ValueError("DexJoCo task selection cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"DexJoCo task selection contains duplicates: {selected}")

    allowed = UNIFIED_TASK_GROUPS[task_group]
    invalid = [name for name in selected if name not in allowed]
    if invalid:
        raise ValueError(
            f"Tasks {invalid} do not belong to task group {task_group!r}; "
            f"expected a subset of {allowed}"
        )
    return selected


def unified_task_index(task_name):
    try:
        return UNIFIED_TASKS.index(task_name)
    except ValueError as exc:
        raise ValueError(f"Unknown DexJoCo unified task: {task_name}") from exc
