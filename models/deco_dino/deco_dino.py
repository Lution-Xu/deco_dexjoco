import os
from contextlib import nullcontext

import einops
import torch
from torch import Tensor, nn

from dexjoco_constants_unify import UNIFIED_TASKS
from models.deco.deco import timeEmb
from models.deco.deco_unify import MMAttentionUnify
from models.deco.denoise_schedular import get_schedule


def build_2d_sincos_position_embedding(height, width, dim, device, dtype):
    """Build a fixed 2D position embedding for flattened image patches."""
    if dim % 4 != 0:
        raise ValueError(f"Position embedding dimension must be divisible by 4, got {dim}")

    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=torch.float32),
        torch.arange(width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / (dim // 4)))
    x = x.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    position = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=-1)
    return position.to(dtype=dtype)


class QFormerBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, queries, image_tokens, image_token_mask):
        norm_queries = self.self_norm(queries)
        self_attn, _ = self.self_attn(
            norm_queries,
            norm_queries,
            norm_queries,
            need_weights=False,
        )
        queries = queries + self_attn

        cross_queries = self.cross_norm(queries)
        cross_attn, _ = self.cross_attn(
            cross_queries,
            image_tokens,
            image_tokens,
            key_padding_mask=~image_token_mask,
            need_weights=False,
        )
        queries = queries + cross_attn
        return queries + self.mlp(self.mlp_norm(queries))


class VisualQFormer(nn.Module):
    def __init__(self, dim, heads, num_queries=128, num_layers=2):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.empty(1, num_queries, dim))
        self.blocks = nn.ModuleList([QFormerBlock(dim, heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        nn.init.normal_(self.query_tokens, std=0.02)

    def forward(self, image_tokens, image_token_mask):
        queries = self.query_tokens.expand(image_tokens.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, image_tokens, image_token_mask)
        return self.norm(queries)


class DECOTextUnify(nn.Module):
    def __init__(
        self,
        act_dim,
        chunk_size,
        obs_state=True,
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
        num_cameras=3,
        vision_model_name="vit_small_patch16_dinov3.lvd1689m",
        freeze_vision_encoder=True,
        text_feature_path=None,
        pretrain_model_path=False,
        adapter_model_path=False,
        num_visual_queries=128,
        qformer_layers=2,
    ):
        super().__init__()
        if not use_task_condition:
            raise ValueError("DECOTextUnify requires use_task_condition=True for offline T5 features")
        if text_feature_path is None:
            raise ValueError("DECOTextUnify requires model.text_feature_path")

        import timm

        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step
        self.num_cameras = num_cameras
        self.freeze_vision_encoder = freeze_vision_encoder

        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),
        )
        vision_dim = self.vision_encoder.feature_info.channels()[0]
        self.vision_proj = nn.Linear(vision_dim, dim)
        self.qformer = VisualQFormer(
            dim=dim,
            heads=heads,
            num_queries=num_visual_queries,
            num_layers=qformer_layers,
        )
        if freeze_vision_encoder:
            self.freeze()

        text_features = self._load_text_features(text_feature_path, num_tasks)
        self.register_buffer("task_text_features", text_features, persistent=True)
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_features.shape[-1]),
            nn.Linear(text_features.shape[-1], dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        self.pos_idx_embedd = nn.Embedding(num_cameras, dim)
        if self.obs_state:
            obs_input_dim = act_dim if obs_dim is None else obs_dim
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_input_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )

        if self.use_tactile:
            self.init_tac_regions()
            self.gated = nn.Linear(68, 68, bias=False)
            self.pos_tac_embedd = nn.Sequential(
                timeEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim),
            )
            self.tactile_encoder = nn.Sequential(
                nn.Linear(1062 * 2, 512),
                nn.Mish(),
                nn.Linear(512, 34),
            )

        self.time_embedd = nn.Sequential(
            timeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )
        self.mmattn = nn.ModuleList(
            [MMAttentionUnify(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)]
        )
        self.linear = nn.Linear(dim, act_dim)

        if not plugin:
            self.initialize_weights()

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
        feat = self.img_encoding(images, camera_mask)
        tactile = self._encode_tactile(tac1, tac2)

        if self.obs_state:
            if obs is None:
                raise ValueError("obs_state=True requires obs")
            if obs_mask is not None:
                obs = obs * obs_mask.to(device=obs.device, dtype=obs.dtype)
            obs = self.obs_encoder(obs)

        if task_idx is None:
            raise ValueError("DECOTextUnify requires task_idx to select offline T5 features")
        task_emb = self.encode_task(task_idx)

        action_mask = self._prepare_action_mask(action_mask, images)
        if training:
            if act is None:
                raise ValueError("Training forward requires act")
            act = act * action_mask
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))
            act, noise = self.add_noise(act, t, action_mask=action_mask)
            t = self.time_embedd(t)
            if self.obs_state:
                t = t + obs
            t = t + task_emb
            _, act = self.atten_forward(
                feat,
                act,
                t=t,
                tactile=tactile,
            )
            return act * action_mask, noise

        sample = torch.randn(
            images.shape[0],
            self.chunk_size,
            self.act_dim,
            dtype=images.dtype,
            device=images.device,
        ) * action_mask
        schedule = get_schedule(self.inference_step, self.chunk_size)
        for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
            t_vec = torch.full((images.shape[0],), t_curr, dtype=images.dtype, device=images.device)
            t_vec = self.time_embedd(t_vec)
            if self.obs_state:
                t_vec = t_vec + obs
            t_vec = t_vec + task_emb
            _, denoise_act = self.atten_forward(
                feat,
                sample,
                t=t_vec,
                tactile=tactile,
            )
            denoise_act = denoise_act * action_mask
            sample = (sample + (t_prev - t_curr) * denoise_act) * action_mask
        return sample

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_vision_encoder:
            self.vision_encoder.eval()
        return self

    def encode_task(self, task_idx):
        task_idx = task_idx.to(device=self.task_text_features.device, dtype=torch.long)
        text_features = self.task_text_features.index_select(0, task_idx)
        text_features = text_features.to(dtype=self.text_proj[1].weight.dtype)
        return self.text_proj(text_features)

    def img_encoding(self, images, camera_mask):
        """Encode valid cameras and resample them into fixed visual tokens."""
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, N, C, H, W], got shape {tuple(images.shape)}")
        batch, num_cameras = images.shape[:2]
        if num_cameras > self.num_cameras:
            raise ValueError(f"Model supports {self.num_cameras} cameras, but received {num_cameras}")
        if camera_mask.shape != images.shape[:2]:
            raise ValueError(
                f"Expected camera_mask shape {tuple(images.shape[:2])}, got {tuple(camera_mask.shape)}"
            )

        valid_counts = camera_mask.sum(dim=1)
        if torch.any(valid_counts == 0):
            raise ValueError("Each sample must contain at least one valid camera")

        valid_indices = camera_mask.nonzero(as_tuple=False)
        valid_images = images[valid_indices[:, 0], valid_indices[:, 1]]
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        vision_context = torch.no_grad() if self.freeze_vision_encoder else nullcontext()
        with vision_context:
            feature_map = self.vision_encoder(valid_images.to(dtype=vision_dtype))[0]
        feature_map = feature_map.to(dtype=images.dtype)

        feat_h, feat_w = feature_map.shape[-2:]
        tokens = einops.rearrange(feature_map, "v c h w -> v (h w) c")
        tokens = self.vision_proj(tokens)
        spatial_emb = build_2d_sincos_position_embedding(
            feat_h,
            feat_w,
            tokens.shape[-1],
            tokens.device,
            tokens.dtype,
        )
        slot_emb = self.pos_idx_embedd(valid_indices[:, 1])[:, None, :]
        tokens = tokens + spatial_emb[None, :, :] + slot_emb

        tokens_per_camera = feat_h * feat_w
        max_tokens = int(valid_counts.max().item()) * tokens_per_camera
        padded_tokens = tokens.new_zeros(batch, max_tokens, tokens.shape[-1])
        image_token_mask = torch.zeros(batch, max_tokens, dtype=torch.bool, device=tokens.device)

        camera_rank = camera_mask.long().cumsum(dim=1) - 1
        token_starts = camera_rank[camera_mask] * tokens_per_camera
        patch_offsets = torch.arange(tokens_per_camera, device=tokens.device)
        token_positions = token_starts[:, None] + patch_offsets[None, :]
        batch_indices = valid_indices[:, 0, None].expand_as(token_positions)
        padded_tokens[batch_indices, token_positions] = tokens
        image_token_mask[batch_indices, token_positions] = True

        return self.qformer(padded_tokens, image_token_mask)

    def atten_forward(self, img, act, t, tactile=None):
        act = self.action_encoder(act)
        act = act + self.action_embedd
        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb=None, tactile=tactile)
        act = self.linear(act)
        return img, act

    def add_noise(self, act: Tensor, t: Tensor, action_mask=None):
        noise = torch.randn_like(act).to(act.device)
        if action_mask is not None:
            noise = noise * action_mask
        t = t.view(act.shape[0], 1, 1)
        act = (1 - t) * act + t * noise
        if action_mask is not None:
            act = act * action_mask
        return act, noise

    def freeze(self):
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.vision_encoder.eval()

    def initialize_weights(self):
        for name, module in self.named_modules():
            if name.startswith("vision_encoder"):
                continue
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def init_tac_regions(self):
        touch_regions = [
            ("fingerone_tip_touch", 3 * 3),
            ("fingerone_top_touch", 12 * 8),
            ("fingerone_palm_touch", 10 * 8),
            ("fingertwo_tip_touch", 3 * 3),
            ("fingertwo_top_touch", 12 * 8),
            ("fingertwo_palm_touch", 10 * 8),
            ("fingerthree_tip_touch", 3 * 3),
            ("fingerthree_top_touch", 12 * 8),
            ("fingerthree_palm_touch", 10 * 8),
            ("fingerfour_tip_touch", 3 * 3),
            ("fingerfour_top_touch", 12 * 8),
            ("fingerfour_palm_touch", 10 * 8),
            ("fingerfive_tip_touch", 3 * 3),
            ("fingerfive_top_touch", 12 * 8),
            ("fingerfive_middle_touch", 3 * 3),
            ("fingerfive_palm_touch", 12 * 8),
            ("palm_touch", 8 * 14),
        ]
        touch_start = 0
        self.tactile_data_index = {}
        for region_name, region_size in touch_regions:
            self.tactile_data_index[region_name] = [touch_start, touch_start + region_size]
            touch_start += region_size
        assert touch_start == 1062, "Total tactile data length should be 1062"

    def _encode_tactile(self, tac1, tac2):
        if not self.use_tactile:
            return None
        if tac1 is None or tac2 is None:
            raise ValueError("use_tactile=True requires tac1 and tac2")
        tac1_avg = torch.stack([tac1[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)
        tac2_avg = torch.stack([tac2[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)
        tactile_emb = self.tactile_encoder(torch.cat([tac1, tac2], dim=-1))
        tactile = torch.cat([tac1_avg, tac2_avg, tactile_emb], dim=-1)
        tactile = tactile * torch.sigmoid(self.gated(tactile))
        return self.pos_tac_embedd(tactile)

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
            inferred_camera_mask = torch.zeros(images.shape[:2], dtype=torch.bool, device=images.device)
            inferred_camera_mask[:, : min(valid_cameras, self.num_cameras)] = True
        if camera_mask is None:
            camera_mask = inferred_camera_mask
            if camera_mask is None:
                camera_mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
        camera_mask = camera_mask.to(device=images.device, dtype=torch.bool)
        return images, camera_mask

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
    freeze_vision_encoder=True,
    text_feature_path=None,
    num_visual_queries=128,
    qformer_layers=2,
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
        freeze_vision_encoder=freeze_vision_encoder,
        text_feature_path=text_feature_path,
        num_visual_queries=num_visual_queries,
        qformer_layers=qformer_layers,
    )
