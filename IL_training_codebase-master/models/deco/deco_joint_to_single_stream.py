"""DECO ablation: joint two-stream blocks followed by shared single-stream blocks."""

import torch
import einops
from torch.nn import functional as F

from models.deco.deco_unify import (
    DECOUnify,
    MMAttentionUnify,
    modeling as base_modeling,
)
from models.deco.rope import apply_rotary_emb


class SingleStreamAttention(MMAttentionUnify):
    def share_stream_parameters(self):
        """Alias action modules to image modules while preserving checkpoint keys."""
        self.act_bais = self.img_bais
        self.act_norm1 = self.img_norm1
        self.act_qkv = self.img_qkv
        self.act_qknorm = self.img_qknorm
        self.act_proj = self.img_proj
        self.act_norm2 = self.img_norm2
        self.act_mlp = self.img_mlp
        if self.use_tactile and self.plugin:
            self.act_qkv_pi = self.img_qkv_pi
            self.act_proj_pi = self.img_proj_pi
            self.act_mlp_pi = self.img_mlp_pi

    def forward(
        self,
        img,
        act,
        t,
        image_rotary_emb,
        tactile=None,
        image_token_mask=None,
    ):
        total_img_len = img.shape[1]
        tokens = torch.cat([img, act], dim=1)

        scale1, shift1, gate1, scale2, shift2, gate2 = self.img_bais(t)
        tokens_norm = (1 + scale1) * self.img_norm1(tokens) + shift1
        qkv = self.img_qkv(tokens_norm)
        if self.use_tactile and self.plugin:
            qkv += self.img_qkv_pi(tokens_norm)

        q, k, v = einops.rearrange(
            qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim
        )
        q, k = self.img_qknorm(q, k, v)

        feat_len = image_rotary_emb[0].shape[0]
        if total_img_len % feat_len != 0:
            raise ValueError(
                f"Image token length {total_img_len} is not divisible by per-camera length {feat_len}"
            )
        for start in range(0, total_img_len, feat_len):
            end = start + feat_len
            q[:, :, start:end, :] = apply_rotary_emb(
                q[:, :, start:end, :], image_rotary_emb
            )
            k[:, :, start:end, :] = apply_rotary_emb(
                k[:, :, start:end, :], image_rotary_emb
            )

        attn_mask = None
        if image_token_mask is not None:
            act_key_mask = torch.ones(
                act.shape[:2], dtype=torch.bool, device=act.device
            )
            key_mask = torch.cat(
                [image_token_mask.to(device=act.device), act_key_mask], dim=1
            )
            attn_mask = key_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        if self.use_tactile and tactile is not None:
            tactile_k = einops.rearrange(
                self.tactile_key(tactile),
                "B L (H D) -> B H L D",
                H=self.head,
                D=self.head_dim,
            )
            tactile_v = einops.rearrange(
                self.tactile_value(tactile),
                "B L (H D) -> B H L D",
                H=self.head,
                D=self.head_dim,
            )
            attn = attn + F.scaled_dot_product_attention(q, tactile_k, tactile_v)

        attn = einops.rearrange(attn, "B H L D -> B L (H D)")
        tokens = tokens + gate1 * self.img_proj(attn)
        if self.use_tactile and self.plugin:
            tokens = tokens + gate1 * self.img_proj_pi(attn)

        mlp_input = (1 + scale2) * self.img_norm2(tokens) + shift2
        tokens = tokens + gate2 * self.img_mlp(mlp_input)
        if self.use_tactile and self.plugin:
            tokens = tokens + gate2 * self.img_mlp_pi(mlp_input)

        return tokens[:, :total_img_len, :], tokens[:, total_img_len:, :]


class DECOJointToSingleStream(DECOUnify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_attention()

    def _configure_attention(self):
        midpoint = len(self.mmattn) // 2
        for block in self.mmattn[midpoint:]:
            block.__class__ = SingleStreamAttention
            block.share_stream_parameters()


def modeling(
    action_dim,
    chunk_size,
    obs_state,
    obs_dim=None,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_task_condition=False,
    num_tasks=11,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    pretrain_model_path=False,
    adapter_model_path=False,
    num_cameras=3,
    **_,
):
    model = base_modeling(
        action_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        obs_dim=obs_dim,
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
        pretrain_model_path=pretrain_model_path,
        adapter_model_path=adapter_model_path,
        num_cameras=num_cameras,
    )
    model.__class__ = DECOJointToSingleStream
    model._configure_attention()
    return model
