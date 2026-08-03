import torch
from torch import nn

from models.deco.deco import modeling as deco_modeling


def modeling(
    action_dim,
    chunk_size,
    obs_state,
    obs_dim=None,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_task_condition=False,
    num_tasks=10,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    num_cameras=2,
    pretrain_model_path=False,
    adapter_model_path=False,
    **_,
):
    model = deco_modeling(
        action_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        use_tactile=use_tactile,
        plugin=plugin,
        plugin_rank=plugin_rank,
        use_task_condition=use_task_condition,
        num_tasks=num_tasks,
        inf_step=inf_step,
        num_attn_blocks=num_attn_blocks,
        heads=heads,
        dim=dim,
        rope_axes_dim=rope_axes_dim,
        num_cameras=num_cameras,
        pretrain_model_path=pretrain_model_path,
        adapter_model_path=adapter_model_path,
    )
    if obs_state:
        obs_dim = action_dim if obs_dim is None else obs_dim
        if obs_dim != action_dim:
            model.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
    return model
