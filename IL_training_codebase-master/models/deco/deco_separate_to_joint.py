"""DECO ablation: separate streams in the first half, joint attention later."""

import torch
import einops
from torch.nn import functional as F

from models.deco.deco_unify import (
    DECOUnify,
    MMAttentionUnify,
    modeling as base_modeling,
)
from models.deco.rope import apply_rotary_emb


class SeparateAttention(MMAttentionUnify):
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
        scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)
        img_norm = (1 + scale1_feat) * self.img_norm1(img) + shift1_feat
        img_qkv = self.img_qkv(img_norm)
        if self.use_tactile and self.plugin:
            img_qkv += self.img_qkv_pi(img_norm)

        img_q, img_k, img_v = einops.rearrange(
            img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim
        )
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)
        feat_len = image_rotary_emb[0].shape[0]
        if total_img_len % feat_len != 0:
            raise ValueError(
                f"Image token length {total_img_len} is not divisible by per-camera length {feat_len}"
            )
        for start in range(0, total_img_len, feat_len):
            end = start + feat_len
            img_q[:, :, start:end, :] = apply_rotary_emb(
                img_q[:, :, start:end, :], image_rotary_emb
            )
            img_k[:, :, start:end, :] = apply_rotary_emb(
                img_k[:, :, start:end, :], image_rotary_emb
            )

        scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t)
        act_norm = (1 + scale1_act) * self.act_norm1(act) + shift1_act
        act_qkv = self.act_qkv(act_norm)
        if self.use_tactile and self.plugin:
            act_qkv += self.act_qkv_pi(act_norm)

        act_q, act_k, act_v = einops.rearrange(
            act_qkv, "B L (K H D) -> K B H L D", K=3, H=self.head, D=self.head_dim
        )
        act_q, act_k = self.act_qknorm(act_q, act_k, act_v)

        # The two calls prevent all image/action interaction in this block.
        img_attn_mask = None
        if image_token_mask is not None:
            img_attn_mask = image_token_mask[:, None, None, :].to(device=img.device)
        img_attn = F.scaled_dot_product_attention(
            img_q, img_k, img_v, attn_mask=img_attn_mask
        )
        act_attn = F.scaled_dot_product_attention(act_q, act_k, act_v)

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
            img_attn = img_attn + F.scaled_dot_product_attention(img_q, tactile_k, tactile_v)
            act_attn = act_attn + F.scaled_dot_product_attention(act_q, tactile_k, tactile_v)

        img_attn = einops.rearrange(img_attn, "B H L D -> B L (H D)")
        act_attn = einops.rearrange(act_attn, "B H L D -> B L (H D)")

        img = img + gate1_feat * self.img_proj(img_attn)
        if self.use_tactile and self.plugin:
            img = img + gate1_feat * self.img_proj_pi(img_attn)
        img_mlp_input = (1 + scale2_feat) * self.img_norm2(img) + shift2_feat
        img = img + gate2_feat * self.img_mlp(img_mlp_input)
        if self.use_tactile and self.plugin:
            img = img + gate2_feat * self.img_mlp_pi(img_mlp_input)

        act = act + gate1_act * self.act_proj(act_attn)
        if self.use_tactile and self.plugin:
            act = act + gate1_act * self.act_proj_pi(act_attn)
        act_mlp_input = (1 + scale2_act) * self.act_norm2(act) + shift2_act
        act = act + gate2_act * self.act_mlp(act_mlp_input)
        if self.use_tactile and self.plugin:
            act = act + gate2_act * self.act_mlp_pi(act_mlp_input)

        return img, act


class DECOSeparateToJoint(DECOUnify):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._configure_attention()

    def _configure_attention(self):
        midpoint = len(self.mmattn) // 2
        for block in self.mmattn[:midpoint]:
            block.__class__ = SeparateAttention


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
    model.__class__ = DECOSeparateToJoint
    model._configure_attention()
    return model
