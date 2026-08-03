import os

import torch
from torch import nn

from dexjoco_constants_unify import UNIFIED_TASKS
from models.deco.deco_unify import DECOUnify


class OfflineT5TaskEncoder(nn.Module):
    def __init__(self, text_feature_path, num_tasks, dim):
        super().__init__()
        text_features = self._load_text_features(text_feature_path, num_tasks)
        self.register_buffer("task_text_features", text_features, persistent=True)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_features.shape[-1]),
            nn.Linear(text_features.shape[-1], dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )
        self._initialize_weights()

    def forward(self, task_idx):
        task_idx = task_idx.to(device=self.task_text_features.device, dtype=torch.long)
        text_features = self.task_text_features.index_select(0, task_idx)
        text_features = text_features.to(dtype=self.text_proj[1].weight.dtype)
        return self.text_proj(text_features)

    def _initialize_weights(self):
        for module in self.text_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _load_text_features(self, text_feature_path, num_tasks):
        path = self._resolve_path(text_feature_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Offline T5 text feature file not found: {text_feature_path}. "
                "Run precompute_deco_text_features.py first."
            )
        payload = torch.load(path, map_location="cpu")
        features = payload["features"] if isinstance(payload, dict) else payload
        features = features.float()
        if features.ndim != 2:
            raise ValueError(f"Expected text features [num_tasks, dim], got {tuple(features.shape)}")
        if features.shape[0] != num_tasks:
            raise ValueError(f"Expected {num_tasks} text features, got {features.shape[0]}")
        if isinstance(payload, dict) and "task_names" in payload:
            task_names = list(payload["task_names"])
            expected_task_names = list(UNIFIED_TASKS)
            if task_names != expected_task_names:
                raise ValueError(
                    "Offline T5 text feature task_names must match "
                    "dexjoco_constants_unify.UNIFIED_TASKS. "
                    f"Expected {expected_task_names}, got {task_names}."
                )
        return features

    @staticmethod
    def _resolve_path(path):
        if os.path.isabs(path):
            return path
        cwd_path = os.path.abspath(path)
        if os.path.exists(cwd_path):
            return cwd_path
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.abspath(os.path.join(repo_root, path))


def modeling(
    action_dim,
    chunk_size,
    obs_state,
    obs_dim=None,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_task_condition=True,
    num_tasks=11,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    pretrain_model_path=False,
    adapter_model_path=False,
    num_cameras=3,
    text_feature_path=None,
    **_,
):
    if not use_task_condition:
        raise ValueError("deco_text_resnet requires use_task_condition=True for offline T5 features")
    if text_feature_path is None:
        raise ValueError("deco_text_resnet requires model.text_feature_path")

    model = DECOUnify(
        act_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        obs_dim=obs_dim,
        use_tactile=use_tactile,
        plugin=plugin,
        plugin_rank=plugin_rank,
        use_task_condition=True,
        num_tasks=num_tasks,
        inf_step=inf_step,
        num_attn_blocks=num_attn_blocks,
        heads=heads,
        dim=dim,
        rope_axes_dim=rope_axes_dim,
        num_cameras=num_cameras,
    )
    model.task_encoder = OfflineT5TaskEncoder(text_feature_path, num_tasks, dim)

    if pretrain_model_path:
        model_path = adapter_model_path or pretrain_model_path
        model_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model_dict, dict) and "model" in model_dict:
            model_dict = model_dict["model"]
        if adapter_model_path:
            model.load_state_dict(model_dict, strict=True)
        else:
            model_state = model.state_dict()
            pretrain_dict = {
                key: value
                for key, value in model_dict.items()
                if key in model_state and value.shape == model_state[key].shape
            }
            model.load_state_dict(pretrain_dict, strict=False)
    return model
