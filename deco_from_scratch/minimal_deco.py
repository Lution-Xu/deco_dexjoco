"""Minimal DECO reproduction for learning and smoke tests.

This file intentionally keeps the first reproduction path small:
- no tactile branch
- no plugin adapter
- no task condition

It still preserves the core DECO contract:
    image/context tokens + noisy action chunk + obs/time condition
        -> prediction of (noise - clean_action)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet34_Weights, resnet34


def time_shift(mu: float, sigma: float, t: Tensor) -> Tensor:
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 4,
    y1: float = 0.5,
    x2: float = 128,
    y2: float = 1.15,
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1)
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


class TimeEmbedding(nn.Module):
    """1D sinusoidal embedding used for diffusion/flow time values."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=x.device) * -emb_scale)
        emb = x.unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class AdaLN(nn.Module):
    """Adaptive LayerNorm parameters generated from obs/time condition."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(dim, dim * 6)
        self.silu = nn.SiLU()

    def forward(self, vec: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        out = self.linear(self.silu(vec))
        return out[:, None, :].chunk(6, dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=dtype) * self.scale


class QKNorm(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class RotaryPosEmbed(nn.Module):
    """2D rotary position embedding for image q/k tokens."""

    def __init__(
        self,
        dim: int,
        rope_axes_dim: tuple[int, int] = (256, 256),
        theta: float = 10000.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.rope_axes_dim = rope_axes_dim
        self.theta = theta

    def forward(self, height: int, width: int) -> tuple[Tensor, Tensor]:
        dim_h, dim_w = self.dim // 2, self.dim // 2
        h_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_h, 2).float()[: dim_h // 2] / dim_h)
        )
        w_inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, dim_w, 2).float()[: dim_w // 2] / dim_w)
        )

        h_seq = torch.arange(self.rope_axes_dim[0])
        w_seq = torch.arange(self.rope_axes_dim[1])
        freqs_h = torch.outer(h_seq, h_inv_freq)
        freqs_w = torch.outer(w_seq, w_inv_freq)

        inner_h_idx = torch.arange(height) * self.rope_axes_dim[0] // height
        inner_w_idx = torch.arange(width) * self.rope_axes_dim[1] // width
        freqs_h = freqs_h[inner_h_idx].unsqueeze(1).expand(height, width, -1)
        freqs_w = freqs_w[inner_w_idx].unsqueeze(0).expand(height, width, -1)

        freqs = torch.cat([freqs_h, freqs_w], dim=-1)
        freqs = torch.cat([freqs, freqs], dim=-1).reshape(height * width, -1)
        return freqs.cos(), freqs.sin()


def apply_rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    cos, sin = freqs_cis
    cos = cos[None, None, :, :].to(x.device)
    sin = sin[None, None, :, :].to(x.device)
    x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)
    x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


class MMAttention(nn.Module):
    """Joint image/action attention block from the DECO core."""

    def __init__(self, heads: int = 8, dim: int = 512) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by heads ({heads}).")

        self.heads = heads
        self.head_dim = dim // heads

        self.img_bias = AdaLN(dim)
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_qkv = nn.Linear(dim, dim * 3)
        self.img_qknorm = QKNorm(self.head_dim)
        self.img_proj = nn.Linear(dim, dim)
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

        self.act_bias = AdaLN(dim)
        self.act_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_qkv = nn.Linear(dim, dim * 3)
        self.act_qknorm = QKNorm(self.head_dim)
        self.act_proj = nn.Linear(dim, dim)
        self.act_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.act_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim),
        )

    def forward(
        self,
        img: Tensor,
        act: Tensor,
        cond: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        total_img_len = img.shape[1]

        scale1_i, shift1_i, gate1_i, scale2_i, shift2_i, gate2_i = self.img_bias(cond)
        img_norm = self.img_norm1(img)
        img_norm = (1 + scale1_i) * img_norm + shift1_i
        img_qkv = self.img_qkv(img_norm)
        img_q, img_k, img_v = self._split_heads(img_qkv)
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)

        if image_rotary_emb is not None:
            feat_len = total_img_len // 2
            img_q[:, :, :feat_len] = apply_rotary_emb(img_q[:, :, :feat_len], image_rotary_emb)
            img_k[:, :, :feat_len] = apply_rotary_emb(img_k[:, :, :feat_len], image_rotary_emb)
            img_q[:, :, feat_len:] = apply_rotary_emb(img_q[:, :, feat_len:], image_rotary_emb)
            img_k[:, :, feat_len:] = apply_rotary_emb(img_k[:, :, feat_len:], image_rotary_emb)

        scale1_a, shift1_a, gate1_a, scale2_a, shift2_a, gate2_a = self.act_bias(cond)
        act_norm = self.act_norm1(act)
        act_norm = (1 + scale1_a) * act_norm + shift1_a
        act_qkv = self.act_qkv(act_norm)
        act_q, act_k, act_v = self._split_heads(act_qkv)
        act_q, act_k = self.act_qknorm(act_q, act_k, act_v)

        q = torch.cat([img_q, act_q], dim=2)
        k = torch.cat([img_k, act_k], dim=2)
        v = torch.cat([img_v, act_v], dim=2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(img.shape[0], -1, self.heads * self.head_dim)

        img_attn, act_attn = attn[:, :total_img_len], attn[:, total_img_len:]
        img = img + gate1_i * self.img_proj(img_attn)
        img = img + gate2_i * self.img_mlp((1 + scale2_i) * self.img_norm2(img) + shift2_i)
        act = act + gate1_a * self.act_proj(act_attn)
        act = act + gate2_a * self.act_mlp((1 + scale2_a) * self.act_norm2(act) + shift2_a)
        return img, act

    def _split_heads(self, qkv: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        bsz, seq_len, _ = qkv.shape
        qkv = qkv.view(bsz, seq_len, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()
        return qkv[0], qkv[1], qkv[2]


class MinimalDECO(nn.Module):
    """A compact DECO implementation suitable for from-scratch reproduction."""

    def __init__(
        self,
        action_dim: int = 28,
        chunk_size: int = 32,
        obs_state: bool = True,
        inference_steps: int = 5,
        num_attn_blocks: int = 2,
        heads: int = 4,
        dim: int = 128,
        use_image_encoder: bool = False,
        image_token_len: int = 128,
        rope_axes_dim: tuple[int, int] = (256, 256),
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.obs_state = obs_state
        self.inference_steps = inference_steps
        self.use_image_encoder = use_image_encoder
        self.image_token_len = image_token_len
        self.dim = dim

        if use_image_encoder:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
            resnet = resnet34(weights=weights)
            self.img_encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)
            self.camera_id = nn.Embedding(2, dim)
            self.rope = RotaryPosEmbed(dim // heads, rope_axes_dim)
        else:
            self.img_encoder = None
            self.img_head = None
            self.camera_id = None
            self.rope = None

        if obs_state:
            self.obs_encoder = nn.Sequential(
                nn.Linear(action_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
        else:
            self.obs_encoder = None

        self.time_encoder = nn.Sequential(
            TimeEmbedding(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        self.action_pos = nn.Parameter(torch.zeros(1, chunk_size, dim))
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([MMAttention(heads=heads, dim=dim) for _ in range(num_attn_blocks)])
        self.action_head = nn.Linear(dim, action_dim)
        self.initialize_weights()

    def forward(
        self,
        img1: Tensor | None = None,
        img2: Tensor | None = None,
        obs: Tensor | None = None,
        act: Tensor | None = None,
        image_tokens: Tensor | None = None,
        training: bool = True,
    ) -> Tensor | tuple[Tensor, Tensor]:
        img, image_rotary_emb = self.encode_image(img1=img1, img2=img2, image_tokens=image_tokens)
        obs_emb = self.obs_encoder(obs) if self.obs_state else None

        if training:
            if act is None:
                raise ValueError("act is required when training=True.")
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))
            noisy_act, noise = self.add_noise(act, t)
            cond = self.time_encoder(t)
            if obs_emb is not None:
                cond = cond + obs_emb
            _, pred = self.attend_forward(img, noisy_act, cond, image_rotary_emb)
            return pred, noise

        if img1 is not None:
            bsz = img1.shape[0]
            device = img1.device
            dtype = img1.dtype
        else:
            bsz = image_tokens.shape[0]  # type: ignore[union-attr]
            device = image_tokens.device  # type: ignore[union-attr]
            dtype = image_tokens.dtype  # type: ignore[union-attr]

        sample = torch.randn(bsz, self.chunk_size, self.action_dim, device=device, dtype=dtype)
        schedule = get_schedule(self.inference_steps, self.chunk_size)
        for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
            t_vec = torch.full((bsz,), t_curr, device=device, dtype=dtype)
            cond = self.time_encoder(t_vec)
            if obs_emb is not None:
                cond = cond + obs_emb
            _, denoise_act = self.attend_forward(img, sample, cond, image_rotary_emb)
            sample = sample + (t_prev - t_curr) * denoise_act
        return sample

    def encode_image(
        self,
        img1: Tensor | None,
        img2: Tensor | None,
        image_tokens: Tensor | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        if not self.use_image_encoder:
            if image_tokens is None:
                if img1 is None:
                    raise ValueError("Provide image_tokens or enable/use image encoder inputs.")
                image_tokens = torch.randn(
                    img1.shape[0],
                    self.image_token_len,
                    self.dim,
                    device=img1.device,
                    dtype=img1.dtype,
                )
            return image_tokens, None

        if img1 is None or img2 is None:
            raise ValueError("img1 and img2 are required when use_image_encoder=True.")
        if img1.shape != img2.shape:
            raise ValueError(f"img1 and img2 must share shape, got {img1.shape} and {img2.shape}.")

        img = torch.cat([img1, img2], dim=0)
        feat = self.img_head(self.img_encoder(img))  # type: ignore[arg-type,operator]
        feat1, feat2 = feat.chunk(2, dim=0)
        _, _, height, width = feat1.shape
        image_rotary_emb = self.rope(height, width)  # type: ignore[operator]

        feat1 = feat1.flatten(2).transpose(1, 2).contiguous()
        feat2 = feat2.flatten(2).transpose(1, 2).contiguous()
        tokens = torch.cat([feat1, feat2], dim=1)

        ids = torch.cat(
            [
                torch.zeros(feat1.shape[1], dtype=torch.long, device=img1.device),
                torch.ones(feat2.shape[1], dtype=torch.long, device=img1.device),
            ]
        )
        tokens = tokens + self.camera_id(ids)[None, :, :]  # type: ignore[operator]
        return tokens, image_rotary_emb

    def attend_forward(
        self,
        img: Tensor,
        act: Tensor,
        cond: Tensor,
        image_rotary_emb: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, Tensor]:
        act = self.action_encoder(act) + self.action_pos
        for block in self.blocks:
            img, act = block(img, act, cond, image_rotary_emb)
        return img, self.action_head(act)

    @staticmethod
    def add_noise(act: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        noise = torch.randn_like(act)
        t = t.view(act.shape[0], 1, 1)
        noisy_act = (1 - t) * act + t * noise
        return noisy_act, noise

    def initialize_weights(self) -> None:
        def basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(basic_init)
        nn.init.constant_(self.action_head.weight, 0)
        nn.init.constant_(self.action_head.bias, 0)
