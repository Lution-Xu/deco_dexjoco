import torch
import einops
from torch import Tensor, nn
from torch.nn import functional as F

from models.deco.deco import DECO, PI_Adapter, QKNorm, adaLN
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import apply_rotary_emb


class DECOUnify(DECO):
    def __init__(
        self,
        act_dim,
        chunk_size,
        obs_state=True,
        obs_dim=None,
        use_tactile=False,
        plugin=False,
        plugin_rank=32,
        use_task_condition=False,
        num_tasks=11,
        inf_step=10,
        img_pretrain=False,
        num_attn_blocks=6,
        heads=8,
        dim=512,
        rope_axes_dim=(256, 256),
        freeze_backbone=True,
        num_cameras=3,
    ):
        super().__init__(
            act_dim=act_dim,
            chunk_size=chunk_size,
            obs_state=obs_state,
            use_tactile=use_tactile,
            plugin=plugin,
            plugin_rank=plugin_rank,
            use_task_condition=use_task_condition,
            num_tasks=num_tasks,
            inf_step=inf_step,
            img_pretrain=img_pretrain,
            num_attn_blocks=num_attn_blocks,
            heads=heads,
            dim=dim,
            rope_axes_dim=rope_axes_dim,
            freeze_backbone=freeze_backbone,
            num_cameras=num_cameras,
        )
        self.num_cameras = num_cameras
        self.mmattn = nn.ModuleList(
            [MMAttentionUnify(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)]
        )
        if obs_state and obs_dim is not None and obs_dim != act_dim:
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
        if not plugin:
            self.initialize_weights()

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
        inferred_camera_mask = None
        if images is None:
            if img1 is None or img2 is None:
                raise ValueError("DECOUnify.forward expects images or img1/img2 inputs")
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

        feat, image_rotary_emb, image_token_mask = self.img_encoding(images, camera_mask)
        tactile = self._encode_tactile(tac1, tac2)

        if self.obs_state:
            if obs_mask is not None:
                obs = obs * obs_mask.to(device=obs.device, dtype=obs.dtype)
            obs = self.obs_encoder(obs)
        if self.use_task_condition:
            task_emb = self.task_encoder(task_idx)

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
            if self.use_task_condition:
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
            if self.use_task_condition:
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

    def _encode_tactile(self, tac1, tac2):
        if not self.use_tactile:
            return None
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

    def img_encoding(self, images, camera_mask):
        if images.ndim != 5:
            raise ValueError(f"Expected images [B, N, C, H, W], got shape {tuple(images.shape)}")
        batch, num_cameras = images.shape[:2]
        if num_cameras > self.num_cameras:
            raise ValueError(f"Model supports {self.num_cameras} cameras, but received {num_cameras}")

        flat_images = einops.rearrange(images, "b n c h w -> (b n) c h w")
        feat = self.img_encoder(flat_images)
        feat = self.img_head(feat)
        feat_h, feat_w = feat.shape[-2:]
        image_rotary_emb = self.rope(feat_h, feat_w)

        feat = einops.rearrange(feat, "(b n) c h w -> b n (h w) c", b=batch, n=num_cameras)
        seq_len = feat.shape[2]
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

    def add_noise(self, act: torch.Tensor, t: torch.Tensor, action_mask=None):
        noise = torch.randn_like(act).to(act.device)
        if action_mask is not None:
            noise = noise * action_mask
        t = t.view(act.shape[0], 1, 1)
        act = (1 - t) * act + t * noise
        if action_mask is not None:
            act = act * action_mask
        return act, noise


class MMAttentionUnify(nn.Module):
    def __init__(self, heads=8, dim=512, use_tactile=False, plugin=False, plugin_rank=32):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = dim // heads
        self.head = heads
        self.use_tactile = use_tactile
        self.plugin = plugin

        self.img_bais = adaLN(dim)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.img_qknorm = QKNorm(head_dim)
        self.img_proj = nn.Linear(dim, dim)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, bias=True),
        )

        self.act_bais = adaLN(dim)
        self.act_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_qkv = nn.Linear(dim, dim * 3)
        self.act_qknorm = QKNorm(head_dim)
        self.act_proj = nn.Linear(dim, dim)
        self.act_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, bias=True),
        )

        if self.use_tactile:
            self.tactile_key = nn.Linear(dim, dim)
            self.tactile_value = nn.Linear(dim, dim)
            if plugin:
                self.img_qkv_pi = PI_Adapter(dim, dim * 3, plugin_rank)
                self.img_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.img_mlp_pi = PI_Adapter(dim, dim, plugin_rank)
                self.act_qkv_pi = PI_Adapter(dim, dim * 3, plugin_rank)
                self.act_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.act_mlp_pi = PI_Adapter(dim, dim, plugin_rank)

    def forward(self, img, act, t, image_rotary_emb, tactile=None, image_token_mask=None):
        total_img_len = img.shape[1]
        scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)
        img_norm = self.img_norm1(img)
        img_norm = (1 + scale1_feat) * img_norm + shift1_feat
        img_qkv = self.img_qkv(img_norm)
        if self.use_tactile and self.plugin:
            img_qkv += self.img_qkv_pi(img_norm)
        img_q, img_k, img_v = einops.rearrange(
            img_qkv,
            "B L (K H D) -> K B H L D",
            K=3,
            H=self.head,
            D=self.head_dim,
        )
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)

        if image_rotary_emb is not None:
            feat_len = image_rotary_emb[0].shape[0]
            if total_img_len % feat_len != 0:
                raise ValueError(f"Image token length {total_img_len} is not divisible by per-camera length {feat_len}")
            for start in range(0, total_img_len, feat_len):
                end = start + feat_len
                img_q[:, :, start:end, :] = apply_rotary_emb(img_q[:, :, start:end, :], image_rotary_emb)
                img_k[:, :, start:end, :] = apply_rotary_emb(img_k[:, :, start:end, :], image_rotary_emb)

        scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t)
        act_norm = self.act_norm1(act)
        act_norm = (1 + scale1_act) * act_norm + shift1_act
        act_qkv = self.act_qkv(act_norm)
        if self.use_tactile and self.plugin:
            act_qkv += self.act_qkv_pi(act_norm)
        act_q, act_k, act_v = einops.rearrange(
            act_qkv,
            "B L (K H D) -> K B H L D",
            K=3,
            H=self.head,
            D=self.head_dim,
        )
        act_q, act_k = self.act_qknorm(act_q, act_k, act_v)

        q = torch.cat([img_q, act_q], dim=2)
        k = torch.cat([img_k, act_k], dim=2)
        v = torch.cat([img_v, act_v], dim=2)
        attn_mask = None
        if image_token_mask is not None:
            act_key_mask = torch.ones(
                act.shape[0],
                act.shape[1],
                dtype=torch.bool,
                device=act.device,
            )
            key_mask = torch.cat([image_token_mask.to(device=act.device), act_key_mask], dim=1)
            attn_mask = key_mask[:, None, None, :]
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        if self.use_tactile and tactile is not None:
            tactile_k = self.tactile_key(tactile)
            tactile_v = self.tactile_value(tactile)
            tactile_k = einops.rearrange(tactile_k, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)
            tactile_v = einops.rearrange(tactile_v, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)
            attn = attn + F.scaled_dot_product_attention(q, tactile_k, tactile_v)

        attn = einops.rearrange(attn, "B H L D -> B L (H D)")
        img_attn, act_attn = attn[:, :total_img_len, :], attn[:, total_img_len:, :]

        img = img + gate1_feat * self.img_proj(img_attn)
        if self.use_tactile and self.plugin:
            img = img + gate1_feat * self.img_proj_pi(img_attn)
        img = img + gate2_feat * self.img_mlp((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)
        if self.use_tactile and self.plugin:
            img = img + gate2_feat * self.img_mlp_pi((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)

        act = act + gate1_act * self.act_proj(act_attn)
        if self.use_tactile and self.plugin:
            act = act + gate1_act * self.act_proj_pi(act_attn)
        act = act + gate2_act * self.act_mlp((1 + scale2_act) * self.act_norm2(act) + shift2_act)
        if self.use_tactile and self.plugin:
            act = act + gate2_act * self.act_mlp_pi((1 + scale2_act) * self.act_norm2(act) + shift2_act)
        return img, act


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
    model = DECOUnify(
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
        num_cameras=num_cameras,
    )

    if pretrain_model_path:
        model_dict = torch.load(pretrain_model_path, map_location="cpu")
        if adapter_model_path:
            model_dict = torch.load(adapter_model_path, map_location="cpu")
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
