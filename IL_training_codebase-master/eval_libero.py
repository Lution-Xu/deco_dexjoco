import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'LIBERO'))
import time
import copy
import math
import tqdm
import yaml
import json
import torch
import imageio
import numpy as np
from PIL import Image
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from inference import modeling, predict_action, ACTTemporalEnsembler
from collections import deque

DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DATE = time.strftime("%Y_%m_%d")


class Config:
    yaml_path: str = "/root/yusun/train_libero_dino/config/deco_libero.yaml"
    task_config: str = "/root/yusun/dataset/libero/tasks.json"
    # LIBERO 场景，可选: libero_spatial, libero_object, libero_goal, libero_10
    task_suite_name: str = "libero_10"
    # 每个任务重复执行的次数
    num_trials_per_task: int = 50
    local_log_dir: str = "/root/yusun/eval_libero/DECO_dinoB1/libero_long"        # Local directory for eval logs
    open_loop: bool = True   # True achieve better performance
    select_action: int = 16  # total_chunk_size / 2
    # ACTTemporalEnsembler 相关配置
    use_temporal_ensembler: bool = False  # open_loop=False 时是否使用 temporal ensembler
    temporal_ensemble_coeff: float = 0.1  # temporal ensemble 系数，越大越依赖近期预测


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def quat2axisangle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle [ax, ay, az]."""
    w = float(np.clip(quat[3], -1.0, 1.0))
    den = np.sqrt(1.0 - w * w)
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.acos(w)
    return (quat[:3] * angle) / den


def save_rollout_video(rollout_images, idx, success, task_description, log_file=None, rollout_dir="./"):
    """Saves an MP4 replay of an episode."""
    rollout_dir = os.path.join(rollout_dir, DATE)
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


config = Config()
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def eval_libero():
    # 加载模型
    yaml_config = yaml.safe_load(open(config.yaml_path, 'r'))
    model = modeling(yaml_config)
    model = model.to(DEVICE)

    # 获取 chunk_size 用于初始化 temporal ensembler
    chunk_size = yaml_config['model']['chunk_size']

    # 初始化 temporal ensembler (仅在 open_loop=False 时使用)
    if config.use_temporal_ensembler:
        temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, chunk_size)
        print(f"Initialized ACTTemporalEnsembler with coeff={config.temporal_ensemble_coeff}, chunk_size={chunk_size}")

    with open(config.task_config, 'r') as f:
        tasks_config = json.load(f)

    # reverse key and values in task
    tasks_config = {v: k for k, v in tasks_config.items()}

    # 初始化日志
    run_id = f"EVAL-{config.task_suite_name}-{DATE_TIME}"
    os.makedirs(config.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(config.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    print(f"Logging to local log file: {local_log_filepath}")

    # 初始化LIBERO任务
    benchmark_dict = benchmark.get_benchmark_dict()

    # 获取指定的任务场景
    task_suite = benchmark_dict[config.task_suite_name]()

    # 指定场景中的任务数量
    num_tasks_in_suite = task_suite.n_tasks

    print(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}")
    log_file.write(f"Task suite: {config.task_suite_name}, task_num: {num_tasks_in_suite}\n")

    # test
    total_episodes, total_successes = 0, 0

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):

        task = task_suite.get_task(task_id)
        # 从指定的场景中，获取第 i 个任务的初始状态
        initial_states = task_suite.get_task_init_states(task_id)
        # 任务每次执行时，物品位置略有差异，因此 initial_states 是个数组，表示每次初试状态的轻微差异

        # 初始化LIBERO仿真环境，并获得任务描述
        env, task_description = get_libero_env(task, resolution=256)

        # 将上述任务执行 num_trials_per_task 次
        task_episodes, task_successes = 0, 0

        # num_trials_per_task: int = 2
        for episode_idx in tqdm.tqdm(range(config.num_trials_per_task)):
            print(f"\nTask: {task_description}")
            log_file.write(f"\nTask: {task_description}\n")
            task_id = int(tasks_config[task_description])
            env.reset()

            # 设置初始状态
            obs = env.set_init_state(initial_states[episode_idx])

            t = 0
            replay_images = []
            action_queue = deque()  # 初始化 action queue

            # 重置 temporal ensembler (每个 episode 开始时)
            if config.use_temporal_ensembler:
                temporal_ensembler.reset()

            if config.task_suite_name == "libero_spatial":
                max_steps = 220  # longest training demo has 193 steps
            elif config.task_suite_name == "libero_object":
                max_steps = 280  # longest training demo has 254 steps
            elif config.task_suite_name == "libero_goal":
                max_steps = 300  # longest training demo has 270 steps
            elif config.task_suite_name == "libero_10":
                max_steps = 520  # 520  # longest training demo has 505 steps
            elif config.task_suite_name == "libero_90":
                max_steps = 400  # longest training demo has 373 steps
            else:
                raise NotImplementedError

            print(f"Starting episode {task_episodes+1}...")
            log_file.write(f"Starting episode {task_episodes+1}...\n")

            while t < max_steps + 10:
                # 前 10 个时间步不进行操作，因为物体可能正在掉落
                if t < 10:
                    # step 传入一个 action，action 是一个长度为 7 的数组，分别表示机械臂的七个关节的角度
                    obs, reward, done, info = env.step([0, 0, 0, 0, 0, 0, -1])
                    t += 1
                    continue

                # 获取环境的观测
                img1 = obs["agentview_image"][::-1, ::-1]
                img2 = obs["robot0_eye_in_hand_image"][::-1, ::-1]
                replay_images.append(copy.deepcopy(img1))
                state = np.concatenate([
                            obs["robot0_eef_pos"],                       # 3D position
                            quat2axisangle(obs["robot0_eef_quat"]),      # quaternion → axis-angle (3D)
                            obs["robot0_gripper_qpos"],                   # 2D gripper
                        ])

                # 基于 queue 的 action 管理
                if len(action_queue) == 0:
                    # queue 为空时，预测新的 actions 并放入 queue
                    actions = predict_action(model, DEVICE, yaml_config, img1, img2, obs=state, task_idx=task_id)
                    actions[..., -1] = np.sign(actions[..., -1])

                    if config.open_loop:
                        # open_loop=True: 直接使用预测的 actions
                        action_queue.extend(actions[:config.select_action].tolist())
                    else:
                        # open_loop=False: 使用 ACTTemporalEnsembler 进行平滑
                        if config.use_temporal_ensembler:
                            # temporal_ensembler.update 需要 (batch, chunk_size, dim) 格式
                            action = temporal_ensembler.update(actions.unsqueeze(0))
                            action = action.squeeze(0).numpy()
                            action_queue.append(action.tolist())
                        else:
                            action_queue.append(actions[0].tolist())

                # 从 queue 中取出一个 action 执行
                action = action_queue.popleft()
                # 更新环境
                obs, reward, done, info = env.step(action)
                t += 1

                # 是否完成
                if done:
                    task_successes += 1
                    total_successes += 1
                    break

            task_episodes += 1
            total_episodes += 1

            # 保存视频
            save_rollout_video(
                replay_images, total_episodes, success=done, task_description=task_description, log_file=log_file, rollout_dir=config.local_log_dir
            )

            # 添加日志
            print(f"Success: {done}")
            print(f"# episodes completed so far: {total_episodes}")
            print(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            log_file.write(f"Success: {done}\n")
            log_file.write(f"# episodes completed so far: {total_episodes}\n")
            log_file.write(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)\n")
            log_file.flush()

        # 添加总结日志
        print(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        print(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        log_file.write(f"Current task success rate: {float(task_successes) / float(task_episodes)}\n")
        log_file.write(f"Current total success rate: {float(total_successes) / float(total_episodes)}\n")
        log_file.flush()

    log_file.close()


if __name__ == "__main__":
    eval_libero()
