import math
import os
from contextlib import nullcontext

import einops
import torch
from torch import nn

from models.deco_dino.deco_dino_padding import DECOTextUnify as DECODINOTextUnify


class DECOTextUnify(DECODINOTextUnify):
    def __init__(
        self,
        *args,
        siglip_model_name="google/siglip-base-patch16-256",
        local_files_only=False,
        hf_cache_dir=None,
        freeze_vision_encoder=True,
        pretrain_model_path=False,
        adapter_model_path=False,
        plugin=False,
        **kwargs,
    ):
        from transformers import SiglipImageProcessor, SiglipVisionModel

        super().__init__(
            *args,
            local_files_only=local_files_only,
            hf_cache_dir=hf_cache_dir,
            freeze_vision_encoder=freeze_vision_encoder,
            pretrain_model_path=False,
            adapter_model_path=False,
            plugin=plugin,
            **kwargs,
        )

        local_files_only = bool(local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1")
        self.vision_encoder_siglip = SiglipVisionModel.from_pretrained(
            siglip_model_name,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        siglip_processor = SiglipImageProcessor.from_pretrained(
            siglip_model_name,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        siglip_dim = self.vision_encoder_siglip.config.hidden_size
        self.vision_proj_siglip = nn.Linear(siglip_dim, self.vision_proj.out_features)

        dino_config = self.vision_encoder.pretrained_cfg
        self.dino_image_size = tuple(dino_config["input_size"][-2:])
        siglip_image_size = self.vision_encoder_siglip.config.image_size
        if isinstance(siglip_image_size, int):
            siglip_image_size = (siglip_image_size, siglip_image_size)
        self.siglip_image_size = tuple(siglip_image_size)
        if self.dino_image_size != self.siglip_image_size:
            raise ValueError(
                "DINO and SigLIP input sizes must match for token addition, "
                f"got DINO {self.dino_image_size} and SigLIP {self.siglip_image_size}"
            )

        self.register_buffer(
            "dino_image_mean",
            torch.tensor(dino_config["mean"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "dino_image_std",
            torch.tensor(dino_config["std"]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_image_mean",
            torch.tensor(siglip_processor.image_mean).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "siglip_image_std",
            torch.tensor(siglip_processor.image_std).view(1, 3, 1, 1),
            persistent=False,
        )

        if not plugin:
            nn.init.xavier_uniform_(self.vision_proj_siglip.weight)
            if self.vision_proj_siglip.bias is not None:
                nn.init.constant_(self.vision_proj_siglip.bias, 0)

        if freeze_vision_encoder:
            self.freeze()

        if pretrain_model_path:
            self._load_checkpoint(pretrain_model_path, adapter_model_path)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_encoder_siglip.eval()
        return self

    def freeze(self):
        for encoder in (
            getattr(self, "vision_encoder", None),
            getattr(self, "vision_encoder_siglip", None),
        ):
            if encoder is None:
                continue
            for param in encoder.parameters():
                param.requires_grad = False
            encoder.eval()

    def img_encoding(self, images, camera_mask):
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, N, C, H, W], got shape {tuple(images.shape)}")
        batch, num_cameras = images.shape[:2]
        if num_cameras > self.num_cameras:
            raise ValueError(f"Model supports {self.num_cameras} cameras, but received {num_cameras}")
        if tuple(images.shape[-2:]) != self.dino_image_size:
            raise ValueError(
                f"Expected {self.dino_image_size} images for both vision encoders, "
                f"got {tuple(images.shape[-2:])}"
            )

        flat_images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        dino_dtype = next(self.vision_encoder.parameters()).dtype
        siglip_dtype = next(self.vision_encoder_siglip.parameters()).dtype
        dino_images = flat_images.to(dtype=dino_dtype)
        dino_mean = self.dino_image_mean.to(dtype=dino_dtype)
        dino_std = self.dino_image_std.to(dtype=dino_dtype)
        dino_images = (dino_images - dino_mean) / dino_std
        siglip_images = flat_images.to(dtype=siglip_dtype)
        siglip_mean = self.siglip_image_mean.to(dtype=siglip_dtype)
        siglip_std = self.siglip_image_std.to(dtype=siglip_dtype)
        siglip_images = (siglip_images - siglip_mean) / siglip_std
        vision_context = torch.no_grad() if self.freeze_vision_encoder else nullcontext()
        with vision_context:
            dino_feature_map = self.vision_encoder(dino_images)[0]
            siglip_outputs = self.vision_encoder_siglip(
                pixel_values=siglip_images,
                return_dict=True,
            )

        feat_h, feat_w = dino_feature_map.shape[-2:]
        dino_feat = einops.rearrange(
            dino_feature_map,
            "(b n) c h w -> (b n) (h w) c",
            b=batch,
            n=num_cameras,
        )
        siglip_feat = self._drop_non_spatial_token_if_needed(siglip_outputs.last_hidden_state)
        siglip_seq_len = siglip_feat.shape[1]
        siglip_grid_size = int(math.sqrt(siglip_seq_len))
        if siglip_grid_size * siglip_grid_size != siglip_seq_len:
            raise ValueError(f"SigLIP token count {siglip_seq_len} cannot be reshaped to a square grid")
        if (feat_h, feat_w) != (siglip_grid_size, siglip_grid_size):
            raise ValueError(
                "DINO and SigLIP token grids must match before addition, "
                f"got DINO {feat_h}x{feat_w} and SigLIP {siglip_grid_size}x{siglip_grid_size}"
            )

        dino_feat = self.vision_proj(dino_feat.to(dtype=self.vision_proj.weight.dtype))
        siglip_feat = self.vision_proj_siglip(
            siglip_feat.to(dtype=self.vision_proj_siglip.weight.dtype)
        )
        feat = dino_feat + siglip_feat.to(dtype=dino_feat.dtype)
        seq_len = feat.shape[1]
        image_rotary_emb = self.rope(feat_h, feat_w)

        feat = einops.rearrange(feat, "(b n) l c -> b n l c", b=batch, n=num_cameras)
        slot_ids = torch.arange(num_cameras, device=images.device)
        slot_emb = self.pos_idx_embedd(slot_ids)[None, :, None, :]
        feat = feat + slot_emb
        feat = einops.rearrange(feat, "b n l c -> b (n l) c")
        image_token_mask = camera_mask[:, :, None].expand(batch, num_cameras, seq_len)
        image_token_mask = image_token_mask.reshape(batch, num_cameras * seq_len)
        return feat, image_rotary_emb, image_token_mask


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
    vision_model_name="vit_small_patch16_dinov3.lvd1689m",
    siglip_model_name="google/siglip-base-patch16-256",
    local_files_only=False,
    hf_cache_dir=None,
    freeze_vision_encoder=True,
    text_feature_path=None,
    **_,
):
    return DECOTextUnify(
        act_dim=action_dim,
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
        vision_model_name=vision_model_name,
        siglip_model_name=siglip_model_name,
        local_files_only=local_files_only,
        hf_cache_dir=hf_cache_dir,
        freeze_vision_encoder=freeze_vision_encoder,
        text_feature_path=text_feature_path,
    )
