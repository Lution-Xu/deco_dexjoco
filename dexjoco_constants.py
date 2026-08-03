DEXJOCO_DATASET_ROOT = "/home/xz/deco/datasets/dexjoco_dataset"
DEXJOCO_REPO_ROOT = "/home/xz/deco/dexjoco-main"

REGIME_DIRS = {
    "rand_obj": "dexjoco_lerobot_datasets",
    "rand_full": "dexjoco_lerobot_datasets_rand_full",
}

SINGLE_ARM_TASKS = [
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
]

DUAL_ARM_TASKS = [
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
]

TASK_GROUPS = {
    "single": SINGLE_ARM_TASKS,
    "dual": DUAL_ARM_TASKS,
}

TASK_GROUP_DIMS = {
    "single": {"action_dim": 22, "obs_dim": 23},
    "dual": {"action_dim": 44, "obs_dim": 46},
}


def resolve_task_names(task_group, task=None, tasks=None):
    """Resolve an optional task selection while preserving group-level task IDs."""
    if task_group not in TASK_GROUPS:
        raise ValueError(f"Unknown DexJoCo task group: {task_group!r}")
    if task is not None and tasks is not None:
        raise ValueError("Set only one of data.task or data.tasks")

    if task is not None:
        selected = [task]
    elif tasks is None:
        selected = list(TASK_GROUPS[task_group])
    elif isinstance(tasks, str):
        selected = [tasks]
    else:
        selected = list(tasks)

    if not selected:
        raise ValueError("DexJoCo task selection cannot be empty")
    if len(selected) != len(set(selected)):
        raise ValueError(f"DexJoCo task selection contains duplicates: {selected}")

    allowed = TASK_GROUPS[task_group]
    invalid = [name for name in selected if name not in allowed]
    if invalid:
        raise ValueError(
            f"Tasks {invalid} do not belong to task group {task_group!r}; "
            f"expected a subset of {allowed}"
        )
    return selected

SINGLE_CAMERA1_BY_TASK = {
    "click_mouse": "observation.images.ego_right",
    "fold_glasses": "observation.images.front",
    "hammer_nail": "observation.images.front",
    "pick_bucket": "observation.images.front",
    "pinch_tongs": "observation.images.front",
    "water_plant": "observation.images.front",
}

SINGLE_RAND_FULL_CAMERA1 = "observation.images.random_camera"
SINGLE_CAMERA2 = "observation.images.wrist"
DUAL_RAND_OBJ_CAMERA1 = "observation.images.ego"
DUAL_CAMERA1 = "observation.images.wrist_left"
DUAL_CAMERA2 = "observation.images.wrist_right"

DEXJOCO_PROMPTS = {
    "bimanual_assembly": "Place the blue part onto the red base with both hands.",
    "bimanual_hanoi": (
        "Execute the final two moves of the three-level Tower of Hanoi: move the "
        "medium disk from the middle peg to the right peg with the right hand, "
        "then move the small disk from the left peg to the right peg with the left hand."
    ),
    "bimanual_microwave_cook": "Open the microwave, put the food inside, and close the microwave.",
    "bimanual_photograph": "Pick up the camera with both hands and take a photograph.",
    "bimanual_unlock_ipad": "Unlock the iPad by pressing the correct password digits.",
    "click_mouse": "Move the mouse to the purple mouse pad and click the left mouse button.",
    "fold_glasses": "Fold the glasses and place them on the table.",
    "hammer_nail": "Pick up the hammer and hammer the nail into the block.",
    "pick_bucket": "Pick up the bucket by its handle.",
    "pinch_tongs": "Pick up the tongs and pinch the target object.",
    "water_plant": "Grasp the watering can and apply water to the plant.",
}


def task_group_for_task(task_name):
    if task_name in SINGLE_ARM_TASKS:
        return "single"
    if task_name in DUAL_ARM_TASKS:
        return "dual"
    raise ValueError(f"Unknown DexJoCo task: {task_name}")


def dataset_dir(data_root, regime):
    if regime not in REGIME_DIRS:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {sorted(REGIME_DIRS)}")
    return f"{data_root.rstrip('/')}/{REGIME_DIRS[regime]}"


def camera_keys_for_task(task_name, regime):
    group = task_group_for_task(task_name)
    if group == "dual":
        return DUAL_CAMERA1, DUAL_CAMERA2
    cam1 = SINGLE_RAND_FULL_CAMERA1 if regime == "rand_full" else SINGLE_CAMERA1_BY_TASK[task_name]
    return cam1, SINGLE_CAMERA2


def policy_camera_keys_for_task(task_name, regime):
    """Return camera keys in model input order (image1, image2[, image3])."""
    if task_group_for_task(task_name) == "dual":
        centric = SINGLE_RAND_FULL_CAMERA1 if regime == "rand_full" else DUAL_RAND_OBJ_CAMERA1
        return centric, DUAL_CAMERA1, DUAL_CAMERA2
    return camera_keys_for_task(task_name, regime)
