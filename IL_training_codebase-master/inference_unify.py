import importlib

import torch
from PIL import Image
from torchvision.transforms import v2 as transforms

from dexjoco_constants_unify import task_group_for_task, unified_task_index
from inference import ACTTemporalEnsembler, letterbox


def unified_action_mask(task_name, device=None):
    mask = torch.zeros(44, dtype=torch.bool, device=device)
    if task_group_for_task(task_name) == "single":
        mask[:22] = True
    else:
        mask[:] = True
    return mask


def unified_obs(obs, task_name, device=None):
    obs = torch.tensor(obs, dtype=torch.float32, device=device)
    out = torch.zeros(46, dtype=torch.float32, device=device)
    mask = torch.zeros(46, dtype=torch.bool, device=device)
    if task_group_for_task(task_name) == "single":
        out[:7] = obs[:7]
        out[14:30] = obs[7:23]
        mask[:7] = True
        mask[14:30] = True
    else:
        out[:] = obs[:46]
        mask[:] = True
    return out, mask


def normalize_masked(value, mask, yaml_config, prefix):
    norm_type = yaml_config['data']['norm_type']
    out = torch.zeros_like(value)
    if norm_type == 'mean_std':
        mean = torch.tensor(yaml_config['data'][f'{prefix}_mean'], dtype=value.dtype, device=value.device)
        std = torch.tensor(yaml_config['data'][f'{prefix}_std'], dtype=value.dtype, device=value.device).clamp_min(1e-8)
        normalized = (value - mean) / std
    else:
        min_value = torch.tensor(yaml_config['data'][f'{prefix}_min'], dtype=value.dtype, device=value.device)
        max_value = torch.tensor(yaml_config['data'][f'{prefix}_max'], dtype=value.dtype, device=value.device)
        normalized = ((value - min_value) / (max_value - min_value).clamp_min(1e-8)).clamp(0, 1.0)
    out[mask] = normalized[mask]
    return out


def denormalize_masked(action, action_mask, yaml_config):
    norm_type = yaml_config['data']['norm_type']
    out = torch.zeros_like(action)
    if norm_type == 'mean_std':
        mean = torch.tensor(yaml_config['data']['action_mean'], dtype=action.dtype, device=action.device)
        std = torch.tensor(yaml_config['data']['action_std'], dtype=action.dtype, device=action.device).clamp_min(1e-8)
        denorm = action * std[None, :] + mean[None, :]
    else:
        min_value = torch.tensor(yaml_config['data']['action_min'], dtype=action.dtype, device=action.device)
        max_value = torch.tensor(yaml_config['data']['action_max'], dtype=action.dtype, device=action.device)
        denorm = action * (max_value - min_value)[None, :] + min_value[None, :]
    out[:, action_mask] = denorm[:, action_mask]
    return out


def preprocess(img1, img2, obs, yaml_config, task_name, device=None, img3=None):
    obs, obs_mask = unified_obs(obs, task_name, device=device)
    obs = normalize_masked(obs, obs_mask, yaml_config, "observation").unsqueeze(0)
    obs_mask = obs_mask.unsqueeze(0)

    img_config = yaml_config['img']
    if yaml_config['img']['img_size'] == [256, 256]:
        resize = letterbox(img_config['img_size'][0])
    else:
        resize = transforms.Resize(img_config['img_size'])
    test_transform = transforms.Compose([
        resize,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_config['img_mean'], std=img_config['img_std']),
    ])

    image_tensors = [
        test_transform(Image.fromarray(img1)),
        test_transform(Image.fromarray(img2)),
    ]
    camera_mask = torch.tensor([True, True, False], dtype=torch.bool, device=device)
    if img3 is not None:
        image_tensors.append(test_transform(Image.fromarray(img3)))
        camera_mask[2] = True
    else:
        image_tensors.append(torch.zeros_like(image_tensors[0]))
    images = torch.stack(image_tensors, dim=0).unsqueeze(0).to(device)
    camera_mask = camera_mask.unsqueeze(0)
    return images, camera_mask, obs, obs_mask


def predict_action(model, device, yaml_config, task_name, img1, img2, obs, tac1=None, tac2=None, img3=None):
    del tac1, tac2
    task_idx = torch.tensor(unified_task_index(task_name), dtype=torch.long, device=device).unsqueeze(0)
    action_mask = unified_action_mask(task_name, device=device).unsqueeze(0)
    with torch.no_grad():
        images, camera_mask, obs, obs_mask = preprocess(
            img1,
            img2,
            obs,
            yaml_config,
            task_name,
            device=device,
            img3=img3,
        )
        action = model(
            images=images,
            camera_mask=camera_mask,
            obs=obs,
            obs_mask=obs_mask,
            act=None,
            task_idx=task_idx,
            action_mask=action_mask,
            training=False,
        )
        action = action.cpu().squeeze(0)
        action_mask_cpu = action_mask.cpu().squeeze(0)
        action = denormalize_masked(action, action_mask_cpu, yaml_config)
    if task_group_for_task(task_name) == "single":
        return action[:, :22]
    return action


def modeling(yaml_config):
    model_name = yaml_config['model_name']
    importmodule = importlib.import_module(f"models.{model_name}")
    return importmodule.modeling(**yaml_config['model'])
