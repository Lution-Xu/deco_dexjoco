"""Python -> PyTorch 最小补课：结合 DECO 的可运行教学脚本。

运行方式：
    python deco_from_scratch/python_to_pytorch_lesson.py

建议学习方式：
1. 先完整运行一遍，看每一节打印的 shape。
2. 再从上到下逐节阅读中文注释。
3. 修改 B、action_dim、chunk_size、hidden_dim，观察 shape 如何变化。

这份脚本只依赖 torch，不依赖真实 DECO 数据集。
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def title(name: str) -> None:
    print(f"\n{'=' * 80}\n{name}\n{'=' * 80}")


def section_1_python_basics() -> None:
    title("1. Python 基础：变量、列表、函数、类")

    # Python 变量没有固定类型，赋值时自动决定。
    action_dim = 28
    chunk_size = 32
    camera_names = ["camera_0", "camera_1"]

    print("action_dim:", action_dim)
    print("chunk_size:", chunk_size)
    print("camera_names:", camera_names)

    # 函数：把一段可复用逻辑封装起来。
    def total_action_values(batch_size: int, horizon: int, dim: int) -> int:
        return batch_size * horizon * dim

    print("B=4 的 action tensor 元素个数:", total_action_values(4, chunk_size, action_dim))

    # 类：把数据和行为放在一起。nn.Module 本质上也是 Python 类。
    class ShapeNote:
        def __init__(self, name: str, shape: tuple[int, ...]) -> None:
            self.name = name
            self.shape = shape

        def describe(self) -> str:
            return f"{self.name}: shape={self.shape}"

    note = ShapeNote("DECO action chunk", (4, chunk_size, action_dim))
    print(note.describe())


def section_2_tensor_shapes() -> None:
    title("2. Tensor 基础：shape、dtype、device")

    # 在 DECO 中，最常见的 batch action 形状是 [B, chunk_size, action_dim]。
    # B 是 batch size，chunk_size 是一次预测未来多少步，action_dim 是每步动作维度。
    B = 4
    chunk_size = 32
    action_dim = 28

    action = torch.randn(B, chunk_size, action_dim)
    obs = torch.randn(B, action_dim)

    print("action.shape:", action.shape)
    print("obs.shape:", obs.shape)
    print("action.dtype:", action.dtype)
    print("action.device:", action.device)

    # 索引：取第 0 个样本、前 3 个时间步、前 5 个动作维度。
    small = action[0, :3, :5]
    print("action[0, :3, :5].shape:", small.shape)

    # view/reshape：改变张量形状，但元素总数必须一致。
    flat = action.reshape(B, chunk_size * action_dim)
    print("flatten action:", flat.shape)

    # unsqueeze：增加一个长度为 1 的维度。
    # DECO 的 add_noise 里会把 t: [B] 变成 [B, 1, 1]，方便广播到 action。
    t = torch.rand(B)
    print("t before:", t.shape)
    print("t after view:", t.view(B, 1, 1).shape)

    # 广播 broadcasting：形状 [B,1,1] 可以自动扩展到 [B,32,28]。
    noisy_weight = t.view(B, 1, 1)
    mixed = (1 - noisy_weight) * action
    print("broadcast result:", mixed.shape)


def section_3_linear_and_mlp() -> None:
    title("3. nn.Linear 与 MLP：从 obs 预测 action chunk")

    B = 8
    action_dim = 28
    chunk_size = 32
    hidden_dim = 128

    obs = torch.randn(B, action_dim)

    # nn.Linear(in_features, out_features)
    # 输入最后一维必须是 in_features，输出最后一维会变成 out_features。
    linear = nn.Linear(action_dim, hidden_dim)
    hidden = linear(obs)
    print("obs -> hidden:", obs.shape, "->", hidden.shape)

    # nn.Sequential 可以把多个层串起来。
    # 这里把 obs: [B, 28] 映射成 action chunk: [B, 32, 28]。
    mlp = nn.Sequential(
        nn.Linear(action_dim, hidden_dim),
        nn.Mish(),  # 激活函数，让模型能拟合非线性关系。
        nn.Linear(hidden_dim, chunk_size * action_dim),
    )

    pred_flat = mlp(obs)
    pred_action = pred_flat.reshape(B, chunk_size, action_dim)

    print("pred_flat.shape:", pred_flat.shape)
    print("pred_action.shape:", pred_action.shape)


def section_4_module_forward() -> None:
    title("4. nn.Module：像 DECO 一样写模型类")

    class ObsToActionChunk(nn.Module):
        """最小策略网络：obs -> 未来 action chunk。

        DECO 更复杂，但 nn.Module 的基本写法完全一样：
        - __init__ 里定义层
        - forward 里描述数据怎么流动
        """

        def __init__(self, action_dim: int = 28, chunk_size: int = 32, hidden_dim: int = 128) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.net = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, chunk_size * action_dim),
            )

        def forward(self, obs: Tensor) -> Tensor:
            # obs: [B, 28]
            B = obs.shape[0]
            out = self.net(obs)  # [B, chunk_size * action_dim]
            out = out.reshape(B, self.chunk_size, self.action_dim)  # [B, 32, 28]
            return out

    model = ObsToActionChunk()
    obs = torch.randn(4, 28)
    action_pred = model(obs)  # 调用 model(x) 会自动执行 forward(x)。
    print("action_pred.shape:", action_pred.shape)

    # named_parameters 可以查看模型里有哪些可训练参数。
    for name, param in list(model.named_parameters())[:3]:
        print(name, tuple(param.shape))


def section_5_loss_backward_optimizer() -> None:
    title("5. loss.backward() 与 optimizer.step()：训练到底做了什么")

    torch.manual_seed(0)

    class ObsToActionChunk(nn.Module):
        def __init__(self, action_dim: int = 28, chunk_size: int = 32, hidden_dim: int = 128) -> None:
            super().__init__()
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.net = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, chunk_size * action_dim),
            )

        def forward(self, obs: Tensor) -> Tensor:
            B = obs.shape[0]
            return self.net(obs).reshape(B, self.chunk_size, self.action_dim)

    model = ObsToActionChunk()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    obs = torch.randn(16, 28)
    target_action = torch.randn(16, 32, 28)

    # 一次标准训练 step：
    pred = model(obs)
    loss = F.mse_loss(pred, target_action)

    # 1. 清空旧梯度。PyTorch 默认会累积梯度，所以每步训练前要清掉。
    optimizer.zero_grad()

    # 2. 反向传播，计算每个参数对 loss 的梯度。
    loss.backward()

    # 3. 根据梯度更新参数。
    optimizer.step()

    # loss.item() 会把只有一个元素的 Tensor 转成普通 Python 数字，适合打印日志。
    print("loss:", loss.item())
    print("第一个参数的梯度 shape:", next(model.parameters()).grad.shape)


def section_6_overfit_100_samples() -> None:
    title("6. 必做练习：MLP overfit 100 个随机样本")

    torch.manual_seed(1)
    B = 100
    action_dim = 28
    chunk_size = 32
    hidden_dim = 256

    # 构造一个固定 toy dataset。
    # 这里故意让 target_action 由 obs 决定，这样小 MLP 有机会学会它。
    obs = torch.randn(B, action_dim)
    time_ramp = torch.linspace(-0.5, 0.5, chunk_size)[None, :, None]
    target_action = torch.tanh(obs[:, None, :]) + time_ramp

    class MLPPolicy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Mish(),
                nn.Linear(hidden_dim, chunk_size * action_dim),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x).reshape(x.shape[0], chunk_size, action_dim)

    model = MLPPolicy()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)

    for step in range(101):
        pred = model(obs)
        loss = F.mse_loss(pred, target_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"step={step:03d}, loss={loss.item():.6f}")

    print("如果 loss 明显下降，说明你已经掌握了最小训练闭环。")


def deco_add_noise(action: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """和 DECO.add_noise 等价的最小实现。

    action: [B, chunk_size, action_dim]
    t:      [B]
    noise:  [B, chunk_size, action_dim]
    """

    noise = torch.randn_like(action)
    t = t.view(action.shape[0], 1, 1)
    noisy_action = (1 - t) * action + t * noise
    return noisy_action, noise


def section_7_deco_training_target() -> None:
    title("7. DECO 训练目标：为什么是 pred(noise - action)")

    B = 4
    chunk_size = 32
    action_dim = 28

    clean_action = torch.randn(B, chunk_size, action_dim)

    # 原始 DECO 里用 sigmoid(randn) 采样 t，使 t 落在 0 到 1 之间。
    # t 越接近 0，noisy_action 越像 clean_action；
    # t 越接近 1，noisy_action 越像纯噪声。
    t = torch.sigmoid(torch.randn(B))
    noisy_action, noise = deco_add_noise(clean_action, t)

    # 模型输入 noisy_action，应该输出这个方向：noise - clean_action。
    target_velocity = noise - clean_action

    print("clean_action.shape:", clean_action.shape)
    print("t.shape:", t.shape)
    print("noisy_action.shape:", noisy_action.shape)
    print("target_velocity.shape:", target_velocity.shape)
    print("训练 loss: mse(pred_velocity, noise - clean_action)")


def section_8_attention_minimum() -> None:
    title("8. 注意力最小理解：image tokens 与 action tokens 交互")

    B = 2
    image_len = 128
    chunk_size = 32
    dim = 64

    image_tokens = torch.randn(B, image_len, dim)
    action_tokens = torch.randn(B, chunk_size, dim)

    # DECO 的 MMAttention 会分别对 image/action 做 qkv，再拼成同一个序列做 joint attention。
    # 这里用 PyTorch 内置 MultiheadAttention 做概念演示。
    all_tokens = torch.cat([image_tokens, action_tokens], dim=1)
    attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
    out, attn_weights = attn(all_tokens, all_tokens, all_tokens)

    image_out = out[:, :image_len]
    action_out = out[:, image_len:]

    print("all_tokens.shape:", all_tokens.shape)
    print("image_out.shape:", image_out.shape)
    print("action_out.shape:", action_out.shape)
    print("attn_weights.shape:", attn_weights.shape)
    print("含义：action token 可以关注 image token，image token 也可以关注 action token。")


def section_9_conv2d_for_image_features() -> None:
    title("9. Conv2d：DECO 图像特征为什么会变成 tokens")

    B = 2
    C = 3
    H = 64
    W = 64
    dim = 128

    img = torch.randn(B, C, H, W)

    # 真实 DECO 使用 ResNet34 backbone 输出 [B, 512, Hf, Wf]，
    # 然后用 img_head = Conv2d(512, dim, kernel_size=3, padding=1) 映射到 dim。
    # 这里用一个小 Conv2d 演示通道数变化。
    conv = nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, padding=1)
    feat = conv(img)  # [B, dim, H, W]

    # flatten(2) 把 H,W 合并成序列长度 H*W；
    # transpose(1,2) 把 [B, dim, H*W] 变成 [B, H*W, dim]。
    tokens = feat.flatten(2).transpose(1, 2)

    print("img.shape:", img.shape)
    print("feat.shape:", feat.shape)
    print("tokens.shape:", tokens.shape)
    print("这就是图像从 feature map 变成 image tokens 的核心操作。")


def section_10_layernorm_and_adaln() -> None:
    title("10. LayerNorm 与 AdaLN：obs/time 如何调制 Transformer")

    B = 2
    seq_len = 32
    dim = 64

    tokens = torch.randn(B, seq_len, dim)
    cond = torch.randn(B, dim)  # 在 DECO 里，cond = time_embedding + obs_embedding。

    norm = nn.LayerNorm(dim, elementwise_affine=False)
    linear = nn.Linear(dim, dim * 6)

    normalized = norm(tokens)
    scale1, shift1, gate1, scale2, shift2, gate2 = linear(torch.nn.functional.silu(cond))[
        :, None, :
    ].chunk(6, dim=-1)

    modulated = (1 + scale1) * normalized + shift1

    print("tokens.shape:", tokens.shape)
    print("cond.shape:", cond.shape)
    print("scale1.shape:", scale1.shape)
    print("modulated.shape:", modulated.shape)
    print("含义：obs/time 不作为普通 token，而是改变每层归一化后的特征分布。")


def section_11_time_embedding() -> None:
    title("11. TimeEmbedding：把标量 t 变成向量")

    B = 4
    dim = 64
    t = torch.sigmoid(torch.randn(B))

    # 这是 DECO 中 timeEmb 的核心公式：用不同频率的 sin/cos 表示一个标量。
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    freqs = torch.exp(torch.arange(half_dim) * -emb_scale)
    emb = t.unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

    print("t:", t)
    print("time embedding shape:", emb.shape)
    print("DECO 后续会用 MLP 把这个 embedding 变成 AdaLN 条件。")


def main() -> None:
    section_1_python_basics()
    section_2_tensor_shapes()
    section_3_linear_and_mlp()
    section_4_module_forward()
    section_5_loss_backward_optimizer()
    section_6_overfit_100_samples()
    section_7_deco_training_target()
    section_8_attention_minimum()
    section_9_conv2d_for_image_features()
    section_10_layernorm_and_adaln()
    section_11_time_embedding()


if __name__ == "__main__":
    main()
