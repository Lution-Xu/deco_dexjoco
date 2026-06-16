# DECO 从 0 复现小工程

这个目录用于把 `IL_training_codebase-master/models/deco/deco.py` 的核心架构拆成一个适合学习和验收的最小复现版本。目标读者是掌握基础 Python、刚开始系统学习 PyTorch 和机器人策略模型的研究生。

第一轮只复现视觉-本体版本：

```text
双相机图像 / 随机 image tokens + obs + noisy action
  -> joint image/action attention
  -> pred(noise - clean_action)
  -> 推理时从随机 action sample 逐步去噪
```

暂不包含触觉、plugin adapter、task condition，也不包含前景监督改造。

## 文件说明

- `minimal_deco.py`：自包含的最小 DECO 实现，包含 `TimeEmbedding`、`AdaLN`、`RMSNorm/QKNorm`、`MMAttention`、`RotaryPosEmbed`、`MinimalDECO`。
- `checks.py`：可执行验收脚本，用于 shape、梯度、toy overfit 和真实数据 smoke test。
- `python_to_pytorch_lesson.py`：结合 DECO 的 Python 到 PyTorch 最小补课脚本，用代码和中文注释讲基本函数、shape 和训练闭环。
- 原始参考代码：`../IL_training_codebase-master/models/deco/deco.py`。

## 环境

最小模型检查需要：

```text
torch
torchvision
```

真实数据 `dataset-smoke` 还会调用原始工程的 `dataset.py`，因此额外需要：

```text
pandas
numpy
Pillow
PyYAML
```

如果你使用本工作区已有环境，可以这样运行：

```bash
.venvs/sam2/bin/python3 deco_from_scratch/checks.py shape-v0
```

如果使用新环境，建议直接安装原始工程依赖：

```bash
pip install -r IL_training_codebase-master/requirements.txt
```

## 学习路线

### 0. Python 到 PyTorch 最小补课

先运行这一份带中文注释的教学脚本：

```bash
python deco_from_scratch/python_to_pytorch_lesson.py
```

如果当前默认 Python 没有安装 PyTorch，可使用本工作区已有环境：

```bash
.venvs/sam2/bin/python3 deco_from_scratch/python_to_pytorch_lesson.py
```

这一节会覆盖：

```text
Python 变量/函数/类
Tensor shape、reshape、unsqueeze、broadcast
nn.Linear、nn.Sequential、nn.Module.forward
loss.backward、optimizer.step
MLP overfit 100 个随机样本
DECO 的 add_noise 和 MSE(pred, noise - action)
image/action tokens 的 joint attention
Conv2d feature map -> image tokens
LayerNorm / AdaLN / time embedding
```

### 1. 先跑最小 Transformer diffusion 闭环

不接 ResNet，不接 RoPE，用随机 `image_tokens` 模拟视觉上下文：

```bash
python deco_from_scratch/checks.py inspect-block
python deco_from_scratch/checks.py shape-v0
python deco_from_scratch/checks.py gradient
python deco_from_scratch/checks.py toy-overfit --steps 80
```

你需要能解释这些 shape：

```text
image_tokens: [B, 128, dim]
obs:          [B, 28]
action:       [B, 32, 28]
pred/noise:   [B, 32, 28]
sample:       [B, 32, 28]
```

训练目标是：

```python
pred, noise = model(image_tokens=image_tokens, obs=obs, act=action, training=True)
loss = mse(pred, noise - action)
```

关键理解点：模型不是直接预测动作，而是预测从干净动作指向噪声动作的方向 `noise - action`。

### 2. 再接入视觉编码和 2D RoPE

这个检查会使用轻量设置跑 `ResNet34[:-2] + Conv2d(512, dim)`，并给两路相机 token 加 camera id embedding：

```bash
python deco_from_scratch/checks.py shape-vision
```

你需要能解释：

```text
img1/img2
  -> batch 维拼接
  -> ResNet34 backbone
  -> img_head
  -> feat1/feat2
  -> flatten 成 tokens
  -> camera id embedding
  -> image q/k 分别应用 2D RoPE
```

RoPE 只加在 image q/k 上，不加在 action q/k 上，因为图像 token 有二维空间位置，action token 的位置由 `action_pos` 学习参数表示。

### 3. 接入真实 dataset 做 smoke test

如果已经有 DECO 格式数据：

```bash
python deco_from_scratch/checks.py dataset-smoke \
  --data ./Deco-50/task1_merged \
  --config IL_training_codebase-master/config/deco.yaml
```

数据目录需要符合原始工程约定：

```text
episode_xxxxxx/
  colors/
    000_color_0.jpg
    000_color_1.jpg
  tactiles/
    000_left_ee_tactile.npy
    000_right_ee_tactile.npy
  data.pkl
```

第一轮模型不使用触觉，但原始 `dataset.py` 仍会读取 `tac1/tac2`，因此数据文件需要存在；如果没有触觉数据，可以按原始 README 的建议在 dataset 层返回 dummy tactile。

## 推荐复现顺序

1. 手写一个 MLP，完成 `[B, 28] -> [B, 32, 28]` 的随机数据 overfit。
2. 阅读 `MinimalDECO.add_noise()`，确认 `noisy_act = (1 - t) * action + t * noise`。
3. 阅读 `MMAttention.forward()`，只追踪一次 `q/k/v` 的 shape。
4. 跑 `shape-v0`，确认最小模型训练和推理接口。
5. 跑 `gradient`，确认 loss 能反向传播。
6. 跑 `toy-overfit`，观察 loss 是否稳定下降或至少没有 NaN/Inf。
7. 跑 `shape-vision`，理解双相机图像如何变成 tokens。
8. 跑 `dataset-smoke`，把 toy batch 换成真实 batch。
9. 回到原始工程，用 `IL_training_codebase-master/train.py` 做小数据单 GPU 训练。
10. 基础版本稳定后，再阅读和复现触觉分支与 plugin adapter。

## 与原始 DECO 的差异

- 默认 `dim=128` 或更小，方便 CPU/GPU 快速检查；原始配置默认 `dim=512`。
- 默认 `num_attn_blocks=1/2`，原始配置默认 `6`。
- 默认 `use_image_encoder=False`，先用随机 image tokens 降低入门难度。
- `pretrained_backbone=False`，避免第一次运行时下载 ImageNet 权重。
- 不实现 tactile、plugin、task condition；这些属于第二轮扩展。

## 验收标准

完成第一轮复现后，学生应该能独立说明：

- 为什么训练 loss 是 `MSE(pred, noise - action)`。
- `training=True` 和 `training=False` 的 forward 区别。
- `obs + time embedding` 为什么作为 AdaLN 条件，而不是普通 token。
- image token 和 action token 如何在 joint attention 中交互。
- 两路相机为什么共享视觉 backbone，但通过 camera id embedding 区分。
- RoPE 为什么只用于 image q/k。
