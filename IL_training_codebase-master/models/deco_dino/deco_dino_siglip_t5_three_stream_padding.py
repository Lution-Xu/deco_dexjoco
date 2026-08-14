import math
import os
from contextlib import nullcontext

import einops
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from dexjoco_constants_unify import UNIFIED_TASKS
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import RotaryPosEmbed, apply_rotary_emb


class QKNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_norm = nn.RMSNorm(dim, eps=1e-6)
        self.key_norm = nn.RMSNorm(dim, eps=1e-6)

    def forward(self, query, key, value):
        query = self.query_norm(query)
        key = self.key_norm(key)
        return query.to(dtype=value.dtype), key.to(dtype=value.dtype)


class ThreeStreamAttentionBlock(nn.Module):
    def __init__(self, heads=8, dim=512):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got dim={dim}, heads={heads}")

        self.heads = heads
        self.head_dim = dim // heads

        self.image_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.image_qkv = nn.Linear(dim, dim * 3)
        self.image_qknorm = QKNorm(self.head_dim)
        self.image_proj = nn.Linear(dim, dim)
        self.image_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.image_ffn = self._build_ffn(dim)

        self.action_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.action_qkv = nn.Linear(dim, dim * 3)
        self.action_qknorm = QKNorm(self.head_dim)
        self.action_proj = nn.Linear(dim, dim)
        self.action_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.action_ffn = self._build_ffn(dim)

        self.text_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.text_qkv = nn.Linear(dim, dim * 3)
        self.text_qknorm = QKNorm(self.head_dim)
        self.text_proj = nn.Linear(dim, dim)
        self.text_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.text_ffn = self._build_ffn(dim)

    @staticmethod
    def _build_ffn(dim):
        return nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dim, dim),
        )

    def _qkv(self, stream, projection, qk_norm):
        qkv = projection(stream)
        query, key, value = einops.rearrange(
            qkv,
            "b l (k h d) -> k b h l d",
            k=3,
            h=self.heads,
            d=self.head_dim,
        )
        query, key = qk_norm(query, key, value)
        return query, key, value

    @staticmethod
    def _apply_image_rope(query, key, image_rotary_emb):
        if image_rotary_emb is None:
            return query, key

        tokens_per_camera = image_rotary_emb[0].shape[0]
        image_len = query.shape[2]
        if image_len % tokens_per_camera != 0:
            raise ValueError(
                f"Image token length {image_len} is not divisible by "
                f"per-camera length {tokens_per_camera}"
            )

        query_chunks = []
        key_chunks = []
        for start in range(0, image_len, tokens_per_camera):
            end = start + tokens_per_camera
            query_chunks.append(apply_rotary_emb(query[:, :, start:end, :], image_rotary_emb))
            key_chunks.append(apply_rotary_emb(key[:, :, start:end, :], image_rotary_emb))
        return torch.cat(query_chunks, dim=2), torch.cat(key_chunks, dim=2)

    def forward(
        self,
        image,
        action,
        text,
        image_rotary_emb,
        image_token_mask,
        text_attention_mask,
    ):
        image_norm = self.image_norm1(image)
        action_norm = self.action_norm1(action)
        text_norm = self.text_norm1(text)

        image_q, image_k, image_v = self._qkv(
            image_norm, self.image_qkv, self.image_qknorm
        )
        action_q, action_k, action_v = self._qkv(
            action_norm, self.action_qkv, self.action_qknorm
        )
        text_q, text_k, text_v = self._qkv(text_norm, self.text_qkv, self.text_qknorm)
        image_q, image_k = self._apply_image_rope(
            image_q, image_k, image_rotary_emb
        )

        query = torch.cat([image_q, action_q, text_q], dim=2)
        key = torch.cat([image_k, action_k, text_k], dim=2)
        value = torch.cat([image_v, action_v, text_v], dim=2)

        action_token_mask = torch.ones(
            action.shape[0],
            action.shape[1],
            dtype=torch.bool,
            device=action.device,
        )
        key_mask = torch.cat(
            [
                image_token_mask.to(device=action.device, dtype=torch.bool),
                action_token_mask,
                text_attention_mask.to(device=action.device, dtype=torch.bool),
            ],
            dim=1,
        )
        joint_attention = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=key_mask[:, None, None, :],
        )
        joint_attention = einops.rearrange(
            joint_attention, "b h l d -> b l (h d)"
        )

        image_len = image.shape[1]
        action_len = action.shape[1]
        text_len = text.shape[1]
        image_start = 0
        image_end = image_start + image_len
        action_start = image_end
        action_end = action_start + action_len
        text_start = action_end
        text_end = text_start + text_len
        if text_end != joint_attention.shape[1]:
            raise RuntimeError(
                f"Unexpected joint attention length {joint_attention.shape[1]}, "
                f"expected {text_end}"
            )

        image_attn = joint_attention[:, image_start:image_end, :]
        action_attn = joint_attention[:, action_start:action_end, :]
        text_attn = joint_attention[:, text_start:text_end, :]

        image = image + self.image_proj(image_attn)
        image = image + self.image_ffn(self.image_norm2(image))
        action = action + self.action_proj(action_attn)
        action = action + self.action_ffn(self.action_norm2(action))
        text = text + self.text_proj(text_attn)
        text = text + self.text_ffn(self.text_norm2(text))
        return image, action, text


class DECOTextUnify(nn.Module):
    def __init__(
        self,
        act_dim,
        chunk_size,
        obs_state=False,
        use_tactile=False,
        plugin=False,
        use_task_condition=True,
        num_tasks=11,
        inf_step=10,
        num_attn_blocks=6,
        heads=8,
        dim=512,
        rope_axes_dim=(256, 256),
        num_cameras=3,
        vision_model_name="vit_small_patch16_dinov3.lvd1689m",
        siglip_model_name="google/siglip2-base-patch16-256",
        text_model_name="google-t5/t5-base",
        text_token_num=64,
        prompt_config_path="config2/deco_text_dexjoco_prompts.yaml",
        local_files_only=False,
        hf_cache_dir=None,
        freeze_vision_encoder=True,
        pretrain_model_path=False,
        adapter_model_path=False,
    ):
        super().__init__()
        if use_tactile:
            raise ValueError("Three-stream DECO currently requires use_tactile=False")
        if plugin:
            raise ValueError("Three-stream DECO currently requires plugin=False")
        if not use_task_condition:
            raise ValueError("Three-stream DECO requires use_task_condition=True")
        if text_token_num != 64:
            raise ValueError(f"text_token_num must be 64, got {text_token_num}")
        if dim % heads != 0:
            raise ValueError(f"dim must be divisible by heads, got dim={dim}, heads={heads}")

        import timm
        from transformers import AutoTokenizer, SiglipImageProcessor, SiglipVisionModel
        from transformers import T5EncoderModel

        local_files_only = bool(local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1")
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step
        self.num_cameras = num_cameras
        self.freeze_vision_encoder = freeze_vision_encoder
        self.text_token_num = text_token_num
        self.rope = RotaryPosEmbed(dim // heads, rope_axes_dim)

        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),
        )
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
        tokenizer = AutoTokenizer.from_pretrained(
            text_model_name,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        self.text_encoder = T5EncoderModel.from_pretrained(
            text_model_name,
            cache_dir=hf_cache_dir,
            local_files_only=local_files_only,
        )
        for parameter in self.text_encoder.parameters():
            parameter.requires_grad = False
        self.text_encoder.eval()

        task_order, prompts = self._load_prompts(prompt_config_path, num_tasks)
        tokenized = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        self.task_order = tuple(task_order)
        self.register_buffer("text_input_ids", tokenized["input_ids"], persistent=False)
        self.register_buffer(
            "text_attention_masks", tokenized["attention_mask"], persistent=False
        )

        vision_dim = self.vision_encoder.feature_info.channels()[0]
        siglip_dim = self.vision_encoder_siglip.config.hidden_size
        text_dim = getattr(self.text_encoder.config, "d_model", None)
        if text_dim is None:
            text_dim = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_dim, dim)
        self.vision_proj_siglip = nn.Linear(siglip_dim, dim)
        self.text_proj = nn.Linear(text_dim, dim)

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

        self.pos_idx_embedd = nn.Embedding(num_cameras, dim)
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )
        self.transformer_blocks = nn.ModuleList(
            [ThreeStreamAttentionBlock(heads, dim) for _ in range(num_attn_blocks)]
        )
        self.action_output = nn.Linear(dim, act_dim)

        self.initialize_weights()
        if freeze_vision_encoder:
            self.freeze()
        if pretrain_model_path:
            self._load_checkpoint(pretrain_model_path, adapter_model_path)

    def forward(
        self,
        images=None,
        camera_mask=None,
        obs=None,
        obs_mask=None,
        act=None,
        task_idx=None,
        tac1=None,
        tac2=None,
        action_mask=None,
        training=True,
        img1=None,
        img2=None,
        img3=None,
    ):
        images, camera_mask = self._prepare_images(images, camera_mask, img1, img2, img3)
        image, image_rotary_emb, image_token_mask = self.img_encoding(images, camera_mask)
        text, text_attention_mask = self.encode_text(task_idx)
        action_mask = self._prepare_action_mask(action_mask, images)

        if training:
            if act is None:
                raise ValueError("Training forward requires act")
            act = act * action_mask
            timestep = torch.sigmoid(
                torch.randn((act.shape[0],), device=act.device, dtype=act.dtype)
            )
            noisy_action, noise = self.add_noise(act, timestep, action_mask)
            predicted_velocity = self.atten_forward(
                image,
                noisy_action,
                text,
                image_rotary_emb,
                image_token_mask,
                text_attention_mask,
            )
            return predicted_velocity * action_mask, noise

        sample = torch.randn(
            images.shape[0],
            self.chunk_size,
            self.act_dim,
            dtype=images.dtype,
            device=images.device,
        ) * action_mask
        schedule = get_schedule(self.inference_step, self.chunk_size)
        for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
            predicted_velocity = self.atten_forward(
                image,
                sample,
                text,
                image_rotary_emb,
                image_token_mask,
                text_attention_mask,
            )
            predicted_velocity = predicted_velocity * action_mask
            sample = (sample + (t_prev - t_curr) * predicted_velocity) * action_mask
        return sample

    def train(self, mode=True):
        super().train(mode)
        self.text_encoder.eval()
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
            self.vision_encoder_siglip.eval()
        return self

    def encode_text(self, task_idx):
        if task_idx is None:
            raise ValueError("Three-stream DECO requires task_idx")
        task_idx = task_idx.to(device=self.text_input_ids.device, dtype=torch.long).reshape(-1)
        input_ids = self.text_input_ids.index_select(0, task_idx)
        attention_mask = self.text_attention_masks.index_select(0, task_idx)
        with torch.no_grad():
            hidden_states = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            ).last_hidden_state
        hidden_states = hidden_states.to(
            device=self.text_proj.weight.device,
            dtype=self.text_proj.weight.dtype,
        )
        text = self.text_proj(hidden_states)
        return text, attention_mask.to(device=text.device, dtype=torch.bool)

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
        dino_images = (
            dino_images - self.dino_image_mean.to(dtype=dino_dtype)
        ) / self.dino_image_std.to(dtype=dino_dtype)
        siglip_images = flat_images.to(dtype=siglip_dtype)
        siglip_images = (
            siglip_images - self.siglip_image_mean.to(dtype=siglip_dtype)
        ) / self.siglip_image_std.to(dtype=siglip_dtype)

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
        siglip_feat = self._drop_non_spatial_token_if_needed(
            siglip_outputs.last_hidden_state
        )
        siglip_seq_len = siglip_feat.shape[1]
        siglip_grid_size = int(math.sqrt(siglip_seq_len))
        if siglip_grid_size * siglip_grid_size != siglip_seq_len:
            raise ValueError(
                f"SigLIP token count {siglip_seq_len} cannot be reshaped to a square grid"
            )
        if (feat_h, feat_w) != (siglip_grid_size, siglip_grid_size):
            raise ValueError(
                "DINO and SigLIP token grids must match before addition, "
                f"got DINO {feat_h}x{feat_w} and SigLIP "
                f"{siglip_grid_size}x{siglip_grid_size}"
            )

        dino_feat = self.vision_proj(
            dino_feat.to(
                device=self.vision_proj.weight.device,
                dtype=self.vision_proj.weight.dtype,
            )
        )
        siglip_feat = self.vision_proj_siglip(
            siglip_feat.to(
                device=self.vision_proj_siglip.weight.device,
                dtype=self.vision_proj_siglip.weight.dtype,
            )
        )
        image = dino_feat + siglip_feat.to(device=dino_feat.device, dtype=dino_feat.dtype)
        tokens_per_camera = image.shape[1]
        image_rotary_emb = self.rope(feat_h, feat_w)

        image = einops.rearrange(
            image, "(b n) l c -> b n l c", b=batch, n=num_cameras
        )
        slot_ids = torch.arange(num_cameras, device=image.device)
        image = image + self.pos_idx_embedd(slot_ids)[None, :, None, :]
        image = einops.rearrange(image, "b n l c -> b (n l) c")
        image_token_mask = camera_mask[:, :, None].expand(
            batch, num_cameras, tokens_per_camera
        )
        image_token_mask = image_token_mask.reshape(
            batch, num_cameras * tokens_per_camera
        )
        return image, image_rotary_emb, image_token_mask

    def atten_forward(
        self,
        image,
        noisy_action,
        text,
        image_rotary_emb,
        image_token_mask,
        text_attention_mask,
    ):
        action = self.action_encoder(noisy_action)
        action = action + self.action_embedd
        for block in self.transformer_blocks:
            image, action, text = block(
                image,
                action,
                text,
                image_rotary_emb,
                image_token_mask,
                text_attention_mask,
            )
        return self.action_output(action)

    def add_noise(self, action: Tensor, timestep: Tensor, action_mask=None):
        noise = torch.randn_like(action)
        if action_mask is not None:
            noise = noise * action_mask
        timestep = timestep.view(action.shape[0], 1, 1)
        noisy_action = (1 - timestep) * action + timestep * noise
        if action_mask is not None:
            noisy_action = noisy_action * action_mask
        return noisy_action, noise

    def freeze(self):
        for encoder in (self.vision_encoder, self.vision_encoder_siglip):
            for parameter in encoder.parameters():
                parameter.requires_grad = False
            encoder.eval()

    def initialize_weights(self):
        modules = [
            self.vision_proj,
            self.vision_proj_siglip,
            self.text_proj,
            self.action_encoder,
            self.transformer_blocks,
            self.action_output,
        ]
        for root_module in modules:
            for module in root_module.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
        nn.init.constant_(self.action_output.weight, 0)
        nn.init.constant_(self.action_output.bias, 0)

    def _prepare_action_mask(self, action_mask, images):
        if action_mask is None:
            return torch.ones(
                images.shape[0],
                1,
                self.act_dim,
                dtype=images.dtype,
                device=images.device,
            )
        if action_mask.ndim == 2:
            action_mask = action_mask[:, None, :]
        return action_mask.to(device=images.device, dtype=images.dtype)

    def _prepare_images(self, images, camera_mask, img1, img2, img3):
        inferred_camera_mask = None
        if images is None:
            if img1 is None or img2 is None:
                raise ValueError("DECOTextUnify.forward expects images or img1/img2 inputs")
            image_list = [img1, img2]
            if img3 is not None:
                image_list.append(img3)
            valid_cameras = len(image_list)
            while len(image_list) < self.num_cameras:
                image_list.append(torch.zeros_like(image_list[0]))
            images = torch.stack(image_list[: self.num_cameras], dim=1)
            inferred_camera_mask = torch.zeros(
                images.shape[:2], dtype=torch.bool, device=images.device
            )
            inferred_camera_mask[:, : min(valid_cameras, self.num_cameras)] = True
        if camera_mask is None:
            camera_mask = inferred_camera_mask
            if camera_mask is None:
                camera_mask = torch.ones(
                    images.shape[:2], dtype=torch.bool, device=images.device
                )
        camera_mask = camera_mask.to(device=images.device, dtype=torch.bool)
        return images, camera_mask

    @staticmethod
    def _drop_non_spatial_token_if_needed(feature):
        seq_len = feature.shape[1]
        grid_size = int(math.sqrt(seq_len))
        if grid_size * grid_size == seq_len:
            return feature
        if seq_len > 1:
            grid_size = int(math.sqrt(seq_len - 1))
            if grid_size * grid_size == seq_len - 1:
                return feature[:, 1:, :]
        return feature

    @classmethod
    def _load_prompts(cls, prompt_config_path, num_tasks):
        path = cls._resolve_path(prompt_config_path)
        with open(path, "r") as file:
            prompt_config = yaml.safe_load(file)

        task_order = prompt_config.get("task_order")
        prompts = prompt_config.get("prompts", {})
        expected_task_order = list(UNIFIED_TASKS)
        if task_order != expected_task_order:
            raise ValueError(
                "Prompt task_order must exactly match "
                "dexjoco_constants_unify.UNIFIED_TASKS. "
                f"Expected {expected_task_order}, got {task_order}"
            )
        missing_tasks = [task for task in task_order if task not in prompts]
        if missing_tasks:
            raise ValueError(f"Prompt config is missing prompts for: {missing_tasks}")
        if len(task_order) != num_tasks:
            raise ValueError(
                f"Prompt config contains {len(task_order)} tasks, expected {num_tasks}"
            )
        return task_order, [prompts[task] for task in task_order]

    def _load_checkpoint(self, pretrain_model_path, adapter_model_path=False):
        model_path = adapter_model_path or pretrain_model_path
        model_dict = torch.load(model_path, map_location="cpu")
        if isinstance(model_dict, dict) and "model" in model_dict:
            model_dict = model_dict["model"]
        if adapter_model_path:
            self.load_state_dict(model_dict, strict=True)
            return
        model_state = self.state_dict()
        pretrain_dict = {
            key: value
            for key, value in model_dict.items()
            if key in model_state and value.shape == model_state[key].shape
        }
        self.load_state_dict(pretrain_dict, strict=False)

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
    obs_state=False,
    use_tactile=False,
    plugin=False,
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
    siglip_model_name="google/siglip2-base-patch16-256",
    text_model_name="google-t5/t5-base",
    text_token_num=64,
    prompt_config_path="config2/deco_text_dexjoco_prompts.yaml",
    local_files_only=False,
    hf_cache_dir=None,
    freeze_vision_encoder=True,
    **_,
):
    return DECOTextUnify(
        act_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
        use_tactile=use_tactile,
        plugin=plugin,
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
        text_model_name=text_model_name,
        text_token_num=text_token_num,
        prompt_config_path=prompt_config_path,
        local_files_only=local_files_only,
        hf_cache_dir=hf_cache_dir,
        freeze_vision_encoder=freeze_vision_encoder,
    )
