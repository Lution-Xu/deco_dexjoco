import math
import os
from contextlib import nullcontext

import einops
import torch
from torch import Tensor, nn

from dexjoco_constants_unify import UNIFIED_TASKS
from models.deco.deco import timeEmb
from models.deco.deco_unify import MMAttentionUnify
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import RotaryPosEmbed


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
        local_files_only=False,
        hf_cache_dir=None,
        freeze_vision_encoder=True,
        text_feature_path=None,
        pretrain_model_path=False,
        adapter_model_path=False,
    ):
        super().__init__()
        if not use_task_condition:
            raise ValueError("DECOTextUnify requires use_task_condition=True for offline T5 features")
        if text_feature_path is None:
            raise ValueError("DECOTextUnify requires model.text_feature_path")

        import timm

        local_files_only = bool(local_files_only or os.environ.get("HF_HUB_OFFLINE") == "1")
        head_dim = dim // heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step
        self.num_cameras = num_cameras
        self.freeze_vision_encoder = freeze_vision_encoder
        self.rope = RotaryPosEmbed(head_dim, rope_axes_dim)

        self.vision_encoder = timm.create_model(
            vision_model_name,
            pretrained=True,
            features_only=True,
            out_indices=(-1,),
        )
        vision_dim = self.vision_encoder.feature_info.channels()[0]
        self.vision_proj = nn.Linear(vision_dim, dim)
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
        feat, image_rotary_emb, image_token_mask = self.img_encoding(images, camera_mask)
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
                image_rotary_emb=image_rotary_emb,
                t=t,
                tactile=tactile,
                image_token_mask=image_token_mask,
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
                image_rotary_emb=image_rotary_emb,
                t=t_vec,
                tactile=tactile,
                image_token_mask=image_token_mask,
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
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, N, C, H, W], got shape {tuple(images.shape)}")
        batch, num_cameras = images.shape[:2]
        if num_cameras > self.num_cameras:
            raise ValueError(f"Model supports {self.num_cameras} cameras, but received {num_cameras}")

        flat_images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        vision_dtype = next(self.vision_encoder.parameters()).dtype
        vision_context = torch.no_grad() if self.freeze_vision_encoder else nullcontext()
        with vision_context:
            feature_map = self.vision_encoder(flat_images.to(dtype=vision_dtype))[0]
        feat_h, feat_w = feature_map.shape[-2:]
        feat = einops.rearrange(feature_map, "(b n) c h w -> (b n) (h w) c", b=batch, n=num_cameras)
        feat = feat.to(dtype=images.dtype)
        seq_len = feat.shape[1]
        image_rotary_emb = self.rope(feat_h, feat_w)

        feat = self.vision_proj(feat)
        feat = einops.rearrange(feat, "(b n) l c -> b n l c", b=batch, n=num_cameras)
        slot_ids = torch.arange(num_cameras, device=images.device)
        slot_emb = self.pos_idx_embedd(slot_ids)[None, :, None, :]
        feat = feat + slot_emb
        feat = einops.rearrange(feat, "b n l c -> b (n l) c")
        image_token_mask = camera_mask[:, :, None].expand(batch, num_cameras, seq_len)
        image_token_mask = image_token_mask.reshape(batch, num_cameras * seq_len)
        return feat, image_rotary_emb, image_token_mask

    def atten_forward(self, img, act, image_rotary_emb, t, tactile=None, image_token_mask=None):
        act = self.action_encoder(act)
        act = act + self.action_embedd
        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb, tactile, image_token_mask=image_token_mask)
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

    def _drop_non_spatial_token_if_needed(self, feat):
        seq_len = feat.shape[1]
        grid = int(math.sqrt(seq_len))
        if grid * grid == seq_len:
            return feat
        if seq_len > 1:
            grid = int(math.sqrt(seq_len - 1))
            if grid * grid == seq_len - 1:
                return feat[:, 1:, :]
        return feat

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
        local_files_only=local_files_only,
        hf_cache_dir=hf_cache_dir,
        freeze_vision_encoder=freeze_vision_encoder,
        text_feature_path=text_feature_path,
    )
