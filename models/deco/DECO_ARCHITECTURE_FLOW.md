# DeCO 架构流程梳理

本文档基于 `models/deco/deco.py` 的实际代码路径整理，重点说明整体数据流、各函数/类的输入输出、作用，以及函数之间的引用关系。

## 1. 模型一句话概览

`DECO` 是一个用于模仿学习动作序列预测的多模态去噪模型。它接收两张图像、可选机器人状态、可选任务编号、可选触觉数据，在训练时学习从“加噪动作”预测去噪方向，在推理时从随机动作噪声开始迭代生成一段动作 chunk。

核心结构：

```text
img1/img2 -> ResNet34 + Conv2d -> image tokens
obs/task/timestep -> condition vector
tac1/tac2 -> tactile tokens, optional
noisy actions -> action tokens

image tokens + action tokens (+ tactile tokens)
    -> stacked MMAttention blocks
    -> predicted action velocity / denoise direction
```

常用维度记号：

| 记号 | 含义 |
|---|---|
| `B` | batch size |
| `C` | 图像通道数，通常为 3 |
| `H, W` | 输入图像高宽 |
| `h, w` | ResNet 输出特征图高宽 |
| `L = h * w` | 单张图像 token 数 |
| `chunk` | 一次预测的动作步数，即 `chunk_size` |
| `act_dim` | 每步动作维度 |
| `dim` | 模型隐藏维度，默认 512 |
| `heads` | 注意力头数，默认 8 |
| `head_dim = dim // heads` | 每个注意力头维度 |

## 2. 顶层调用入口

训练/评估脚本通常通过 `modeling(...)` 创建模型，而不是直接实例化 `DECO(...)`。

```text
外部训练脚本
  -> models.deco.modeling(...)
      -> DECO(...)
      -> 可选加载 pretrain / adapter 权重
      -> 返回模型

训练时:
  net(..., training=True)
      -> DECO.forward(...)
      -> 返回 pred, noise
      -> train_one_epoch.py 中使用 MSE(pred, noise - action)

验证/推理时:
  net(..., training=False)
      -> DECO.forward(...)
      -> 返回预测动作序列 sample
```

注意：`DECO.forward(..., training=True/False)` 里的 `training` 是这个模型自定义的分支开关，不等同于 PyTorch 的 `model.train()` / `model.eval()`。训练脚本会同时调用 `net.train()`，并传入 `training=True`；验证脚本会调用 `net.eval()`，并传入 `training=False`。

## 3. DECO.__init__：搭建网络模块

### 输入参数

| 参数 | 作用 |
|---|---|
| `act_dim` | 每一步动作维度 |
| `chunk_size` | 一次预测多少步动作 |
| `obs_state` | 是否使用机器人状态 `obs` |
| `use_tactile` | 是否使用触觉 `tac1/tac2` |
| `plugin` | 是否启用低秩 `PI_Adapter` |
| `plugin_rank` | adapter 中间秩 |
| `use_task_condition` | 是否使用任务编号条件 |
| `num_tasks` | 任务 embedding 表大小 |
| `inf_step` | 推理时去噪迭代步数 |
| `num_attn_blocks` | `MMAttention` 堆叠层数 |
| `heads` | 注意力头数 |
| `dim` | token 隐藏维度 |
| `rope_axes_dim` | RoPE 参考二维网格大小 |
| `freeze_backbone` | 当前构造函数里没有实际使用 |

### 创建的主要子模块

| 成员 | 类型/结构 | 作用 |
|---|---|---|
| `self.rope` | `RotaryPosEmbed` | 给图像 q/k 生成二维 RoPE |
| `self.img_encoder` | ResNet34 去掉池化和分类头 | 提取图像特征 `[B,3,H,W] -> [B,512,h,w]` |
| `self.img_head` | `Conv2d(512, dim, 3x3)` | 把 ResNet 通道映射到 `dim` |
| `self.pos_idx_embedd` | `Embedding(2, dim)` | 区分 token 来自第 1 张还是第 2 张图 |
| `self.obs_encoder` | MLP | `obs: [B,act_dim] -> [B,dim]` |
| `self.task_encoder` | `Embedding(num_tasks, dim)` | `task_idx: [B] -> [B,dim]` |
| `self.time_embedd` | `timeEmb + MLP` | `t: [B] -> [B,dim]` |
| `self.action_embedd` | learnable parameter | 动作序列位置编码 `[1,chunk,dim]` |
| `self.action_encoder` | MLP | `act: [B,chunk,act_dim] -> [B,chunk,dim]` |
| `self.mmattn` | `ModuleList[MMAttention]` | 多层视觉-动作联合注意力 |
| `self.linear` | `Linear(dim, act_dim)` | 动作 token 输出头 |

当 `use_tactile=True` 时，还会创建：

| 成员 | 作用 |
|---|---|
| `init_tac_regions()` | 建立每只手 1062 维触觉向量的 17 个区域索引 |
| `self.gated` | 对 68 维触觉描述做门控 |
| `self.pos_tac_embedd` | 把每个触觉标量编码成 `dim` 维 token |
| `self.tactile_encoder` | 从完整左右手触觉向量学习 34 维补充特征 |

如果 `plugin=False`，构造函数最后调用 `initialize_weights()` 初始化线性层，并把最终输出头 `self.linear` 置零。

## 4. forward：总流程

函数签名：

```python
forward(
    img1,
    img2,
    obs=None,
    act=None,
    task_idx=None,
    tac1=None,
    tac2=None,
    action_mask=None,
    training=True,
)
```

### 输入输出

| 输入 | 形状 | 是否必需 | 说明 |
|---|---:|---|---|
| `img1` | `[B,C,H,W]` | 是 | 第一张图像 |
| `img2` | `[B,C,H,W]` | 是 | 第二张图像，形状必须和 `img1` 相同 |
| `obs` | `[B,act_dim]` | 当 `obs_state=True` | 当前机器人状态 |
| `act` | `[B,chunk,act_dim]` | 训练时必需 | 真实动作序列 |
| `task_idx` | `[B]` | 当 `use_task_condition=True` | 任务编号 |
| `tac1` | `[B,1062]` | 当 `use_tactile=True` | 左手/第一路触觉 |
| `tac2` | `[B,1062]` | 当 `use_tactile=True` | 右手/第二路触觉 |
| `action_mask` | 任意 | 否 | 当前 `deco.py` 中未使用 |
| `training` | bool | 是 | 选择训练或推理分支 |

| 分支 | 返回 |
|---|---|
| `training=True` | `(pred, noise)`，两者形状都是 `[B,chunk,act_dim]` |
| `training=False` | `sample`，形状 `[B,chunk,act_dim]` |

### forward 公共前处理

```text
1. feat, image_rotary_emb = img_encoding(img1, img2)
2. 如果 use_tactile:
      tac1/tac2 -> 17 区域均值
      cat(tac1_avg, tac2_avg, tactile_encoder([tac1,tac2])) -> [B,68]
      gated -> [B,68]
      pos_tac_embedd -> tactile tokens [B,68,dim]
   否则 tactile = None
3. 如果 obs_state:
      obs_encoder(obs) -> [B,dim]
4. 如果 use_task_condition:
      task_encoder(task_idx) -> [B,dim]
```

## 5. 训练流程：training=True

训练分支的目标是让模型预测 `noise - action`。

代码路径：

```text
DECO.forward(training=True)
  -> img_encoding(img1, img2)
  -> 可选 tactile 编码
  -> 可选 obs/task 编码
  -> t = sigmoid(randn([B]))
  -> noisy_act, noise = add_noise(action, t)
  -> t_cond = time_embedd(t)
  -> t_cond += obs_emb, optional
  -> t_cond += task_emb, optional
  -> _, pred = atten_forward(feat, noisy_act, image_rotary_emb, t_cond, tactile)
  -> return pred, noise

train_one_epoch.py:
  loss = mse_loss(pred, noise - action)
```

关键公式：

```text
noisy_action = (1 - t) * action + t * noise
target       = noise - action
```

因此 `pred` 表示从真实动作指向随机噪声的方向，也可理解为流匹配/去噪速度场。推理时会沿相反方向把随机噪声拉回动作。

## 6. 推理流程：training=False

推理时没有真实动作输入，模型从随机动作噪声开始迭代。

代码路径：

```text
DECO.forward(training=False)
  -> img_encoding(img1, img2)
  -> 可选 tactile 编码
  -> 可选 obs/task 编码
  -> sample = randn([B,chunk,act_dim])
  -> schedule = get_schedule(inference_step, chunk_size)
  -> for t_curr, t_prev in zip(schedule[:-1], schedule[1:]):
         t_cond = time_embedd(full([B], t_curr))
         t_cond += obs_emb, optional
         t_cond += task_emb, optional
         _, denoise_act = atten_forward(feat, sample, image_rotary_emb, t_cond, tactile)
         sample = sample + (t_prev - t_curr) * denoise_act
  -> return sample
```

`get_schedule` 返回从接近 `1` 递减到 `0` 的时间表。由于 `t_prev < t_curr`，更新项 `(t_prev - t_curr)` 为负数，而模型预测的是 `noise - action` 方向，所以负步长会把 `sample` 从噪声方向推回动作方向。

## 7. img_encoding：图像编码

函数签名：

```python
img_encoding(self, img1, img2)
```

### 输入输出

| 输入 | 形状 | 说明 |
|---|---:|---|
| `img1` | `[B,C,H,W]` | 第一张图像 |
| `img2` | `[B,C,H,W]` | 第二张图像，必须和 `img1` 同形状 |

| 输出 | 形状 | 说明 |
|---|---:|---|
| `feat` | `[B,2*L,dim]` | 两张图像 token 拼接后的视觉 token |
| `image_rotary_emb` | `(cos, sin)` | RoPE 位置编码，单张图像 token 使用 |

### 内部流程

```text
img = cat([img1, img2], dim=0)              # [2B,C,H,W]
feat = ResNet34(img)                        # [2B,512,h,w]
feat = img_head(feat)                       # [2B,dim,h,w]
feat1, feat2 = feat.chunk(2, dim=0)         # 各 [B,dim,h,w]
image_rotary_emb = rope(h, w)
feat1/feat2 flatten                         # 各 [B,L,dim]
img_id = pos_idx_embedd([0...0,1...1])       # [B,2L,dim]
feat = cat([feat1, feat2], dim=1) + img_id  # [B,2L,dim]
```

## 8. atten_forward：堆叠多模态注意力

函数签名：

```python
atten_forward(self, img, act, image_rotary_emb, t, tactile=None)
```

### 输入输出

| 输入 | 形状 | 说明 |
|---|---:|---|
| `img` | `[B,2L,dim]` | 图像 token |
| `act` | `[B,chunk,act_dim]` | 当前动作或加噪动作 |
| `image_rotary_emb` | `(cos, sin)` | 图像 RoPE |
| `t` | `[B,dim]` | 条件向量，包含时间/状态/任务 |
| `tactile` | `[B,68,dim]` 或 `None` | 触觉 token |

| 输出 | 形状 | 说明 |
|---|---:|---|
| `img` | `[B,2L,dim]` | 多层注意力更新后的图像 token |
| `act` | `[B,chunk,act_dim]` | 输出头投影后的动作方向预测 |

### 内部流程

```text
act = action_encoder(act)       # [B,chunk,dim]
act = act + action_embedd       # 加动作位置编码

for mma in self.mmattn:
    img, act = mma(img, act, t, image_rotary_emb, tactile)

act = linear(act)               # [B,chunk,act_dim]
return img, act
```

## 9. add_noise：动作加噪

函数签名：

```python
add_noise(self, act, t)
```

### 输入输出

| 输入 | 形状 | 说明 |
|---|---:|---|
| `act` | `[B,chunk,act_dim]` | 真实动作 |
| `t` | `[B]` | 每个样本的噪声强度 |

| 输出 | 形状 | 说明 |
|---|---:|---|
| `act` | `[B,chunk,act_dim]` | 加噪后的动作 |
| `noise` | `[B,chunk,act_dim]` | 采样的标准正态噪声 |

公式：

```text
noise = randn_like(action)
noisy_action = (1 - t) * action + t * noise
```

## 10. MMAttention：多模态联合注意力块

`MMAttention` 是 DeCO 的核心计算块。每一层同时更新图像 token 和动作 token。

### 输入输出

函数签名：

```python
MMAttention.forward(self, img, act, t, image_rotary_emb, tactile=None)
```

| 输入 | 形状 | 说明 |
|---|---:|---|
| `img` | `[B,2L,dim]` | 图像 token |
| `act` | `[B,chunk,dim]` | 动作 token |
| `t` | `[B,dim]` | 条件向量 |
| `image_rotary_emb` | `(cos, sin)` | 图像 RoPE |
| `tactile` | `[B,68,dim]` 或 `None` | 触觉 token |

| 输出 | 形状 | 说明 |
|---|---:|---|
| `img` | `[B,2L,dim]` | 更新后的图像 token |
| `act` | `[B,chunk,dim]` | 更新后的动作 token |

### 内部主要步骤

```text
1. img_bais(t) -> 图像分支 adaLN 参数
2. img_norm1(img) -> scale/shift 调制 -> img_qkv
3. img_qkv -> img_q, img_k, img_v
4. img_qknorm(img_q, img_k)
5. 对 img1/img2 的 q/k 分别应用同一套 image_rotary_emb

6. act_bais(t) -> 动作分支 adaLN 参数
7. act_norm1(act) -> scale/shift 调制 -> act_qkv
8. act_qkv -> act_q, act_k, act_v
9. act_qknorm(act_q, act_k)

10. q = cat([img_q, act_q], dim=token)
    k = cat([img_k, act_k], dim=token)
    v = cat([img_v, act_v], dim=token)
    attn = scaled_dot_product_attention(q, k, v)

11. 如果 use_tactile:
      tactile -> tactile_k/tactile_v
      cross_attn = attention(q, tactile_k, tactile_v)
      attn = attn + cross_attn

12. attn 拆回 img_attn / act_attn

13. img = img + gate1_feat * img_proj(img_attn)
    img = img + gate2_feat * img_mlp(norm2(img) with adaLN)

14. act = act + gate1_act * act_proj(act_attn)
    act = act + gate2_act * act_mlp(norm2(act) with adaLN)
```

### plugin / PI_Adapter 路径

当 `use_tactile=True` 且 `plugin=True` 时，`MMAttention` 会给以下分支添加低秩增量：

| 分支 | adapter |
|---|---|
| 图像 qkv | `img_qkv_pi` |
| 图像 attention 输出投影 | `img_proj_pi` |
| 图像 MLP | `img_mlp_pi` |
| 动作 qkv | `act_qkv_pi` |
| 动作 attention 输出投影 | `act_proj_pi` |
| 动作 MLP | `act_mlp_pi` |

计算形式是主干输出加 adapter 输出，例如：

```text
img_qkv = img_qkv(img_norm) + img_qkv_pi(img_norm)
```

adapter 的 `up` 层初始化为 0，因此刚开始不会明显扰动已加载的预训练主干。

## 11. 触觉编码路径

触觉只在 `use_tactile=True` 时启用。

输入：

```text
tac1: [B,1062]
tac2: [B,1062]
```

流程：

```text
init_tac_regions() 建立 17 个区域索引

tac1_avg = stack(mean(tac1[:, start:end]) for each region)  # [B,17]
tac2_avg = stack(mean(tac2[:, start:end]) for each region)  # [B,17]

tactile_emb = tactile_encoder(cat([tac1,tac2], -1))         # [B,34]

tactile = cat([tac1_avg, tac2_avg, tactile_emb], -1)        # [B,68]
tactile = tactile * sigmoid(gated(tactile))                 # [B,68]
tactile = pos_tac_embedd(tactile)                           # [B,68,dim]
```

后续 `MMAttention` 中，图像+动作 token 作为 query，触觉 token 作为 key/value 做 cross-attention。

## 12. 辅助类和函数

### PI_Adapter

| 项 | 内容 |
|---|---|
| 输入 | `x: [..., dim]` |
| 输出 | `[..., out_dim]` |
| 作用 | 低秩增量模块，结构为 `Linear(dim, rank) -> Linear(rank, out_dim)` |
| 引用位置 | `MMAttention` 的 plugin 分支 |

### adaLN

| 项 | 内容 |
|---|---|
| 输入 | `vec: [B,dim]` |
| 输出 | 6 个 `[B,1,dim]`：`scale1, shift1, gate1, scale2, shift2, gate2` |
| 作用 | 根据条件向量动态生成 LayerNorm 调制参数和残差门控 |
| 引用位置 | `MMAttention.img_bais`、`MMAttention.act_bais` |

### RMSNorm

| 项 | 内容 |
|---|---|
| 输入 | `x: [..., dim]` |
| 输出 | 同形状 |
| 作用 | 对最后一维做 RMS 归一化 |
| 引用位置 | `QKNorm` |

### QKNorm

| 项 | 内容 |
|---|---|
| 输入 | `q, k, v`，通常为 `[B,heads,L,head_dim]` |
| 输出 | 归一化后的 `q, k` |
| 作用 | 稳定 attention score |
| 引用位置 | `MMAttention` 图像和动作 q/k 分支 |

### timeEmb

| 项 | 内容 |
|---|---|
| 输入 | `[B]` 或 `[B,L]` 标量 |
| 输出 | `[B,dim]` 或 `[B,L,dim]` |
| 作用 | 正弦/余弦编码，用于 timestep，也复用于触觉标量编码 |
| 引用位置 | `DECO.time_embedd`、`DECO.pos_tac_embedd` |

### RotaryPosEmbed / apply_rotary_emb

来自 `models/deco/rope.py`。

| 函数/类 | 输入 | 输出 | 作用 |
|---|---|---|---|
| `RotaryPosEmbed.forward(height, width)` | ResNet 特征图高宽 | `(cos, sin)` | 根据二维网格生成 RoPE |
| `apply_rotary_emb(x, freqs)` | `x: [B,heads,L,head_dim]` | 同形状 | 把 RoPE 应用于图像 q/k |

### get_schedule

来自 `models/deco/denoise_schedular.py`。

| 输入 | 输出 | 作用 |
|---|---|---|
| `num_steps, seq_len, base_shift, max_shift, shift` | 长度为 `num_steps + 1` 的 list | 生成推理去噪时间表 |

默认会先生成 `torch.linspace(1, 0, num_steps + 1)`，然后根据 `seq_len` 做 time shift，使时间步分布更偏向高噪声区域。

## 13. modeling：模型工厂与权重加载

函数签名：

```python
modeling(
    action_dim,
    chunk_size,
    obs_state,
    use_tactile=False,
    plugin=False,
    plugin_rank=32,
    use_task_condition=False,
    num_tasks=10,
    inf_step=10,
    num_attn_blocks=6,
    heads=8,
    dim=512,
    rope_axes_dim=(256, 256),
    pretrain_model_path=False,
    adapter_model_path=False,
)
```

输出：`DECO` 模型实例。

权重加载逻辑：

| 条件 | 行为 |
|---|---|
| `pretrain_model_path=False` | 只创建新模型，不加载权重 |
| `use_tactile=False` 且有 `pretrain_model_path` | strict 加载纯视觉模型权重 |
| `use_tactile=True, plugin=False` 且有 `pretrain_model_path` | strict 加载视觉-触觉完整权重 |
| `use_tactile=True, plugin=True, adapter_model_path` | strict 加载 adapter 推理/续训权重 |
| `use_tactile=True, plugin=True, pretrain_model_path` 且无 adapter | 只加载名称和形状匹配的视觉预训练权重，冻结已有参数，只训练新增参数 |

## 14. 函数/类引用关系总表

```text
modeling
  -> DECO.__init__
      -> RotaryPosEmbed
      -> torchvision.models.resnet34
      -> timeEmb
      -> init_tac_regions, when use_tactile=True
      -> MMAttention * num_attn_blocks
          -> adaLN
          -> QKNorm
              -> RMSNorm
          -> PI_Adapter, when use_tactile=True and plugin=True
      -> initialize_weights, when plugin=False

DECO.forward
  -> img_encoding
      -> self.img_encoder
      -> self.img_head
      -> self.rope.forward
      -> self.pos_idx_embedd
  -> tactile branch, when use_tactile=True
      -> self.tactile_encoder
      -> self.gated
      -> self.pos_tac_embedd
          -> timeEmb
  -> self.obs_encoder, when obs_state=True
  -> self.task_encoder, when use_task_condition=True
  -> training=True:
      -> add_noise
      -> self.time_embedd
          -> timeEmb
      -> atten_forward
          -> self.action_encoder
          -> MMAttention.forward repeated
              -> adaLN.forward
              -> QKNorm.forward
              -> apply_rotary_emb
              -> scaled_dot_product_attention
              -> optional tactile cross-attention
          -> self.linear
      -> returns pred, noise
  -> training=False:
      -> get_schedule
      -> loop:
          -> self.time_embedd
          -> atten_forward
          -> Euler update sample
      -> returns sample
```

## 15. 端到端形状示例

假设：

```text
B = 8
img1/img2 = [8,3,224,224]
ResNet 输出 h=w=7，因此 L=49
chunk_size = 32
act_dim = 28
dim = 512
heads = 8
head_dim = 64
```

训练时主要张量形状：

| 阶段 | 张量 | 形状 |
|---|---|---:|
| 输入图像 | `img1`, `img2` | `[8,3,224,224]` |
| ResNet 后 | `feat1`, `feat2` | `[8,512,7,7]` |
| Conv 后 | `feat1`, `feat2` | `[8,512,7,7]`，这里 `dim=512` |
| flatten 后 | `feat1`, `feat2` | `[8,49,512]` |
| 拼接图像 token | `feat` | `[8,98,512]` |
| 动作输入 | `act` | `[8,32,28]` |
| 加噪动作 | `noisy_act` | `[8,32,28]` |
| action encoder 后 | `act` | `[8,32,512]` |
| joint attention q/k/v token 长度 | `2L + chunk` | `98 + 32 = 130` |
| 输出头后 | `pred` | `[8,32,28]` |
| 训练返回 | `(pred, noise)` | `[8,32,28]`, `[8,32,28]` |

## 16. 阅读代码时的主线

如果顺着代码读，建议顺序如下：

1. `modeling(...)`：看模型如何被创建、如何加载权重。
2. `DECO.__init__(...)`：看模块组成。
3. `DECO.forward(...)`：看训练/推理分支。
4. `img_encoding(...)`：看图像如何变成 token。
5. `add_noise(...)`：看训练监督信号从哪里来。
6. `atten_forward(...)`：看动作 token 如何进入注意力堆栈。
7. `MMAttention.forward(...)`：看核心多模态注意力。
8. `adaLN / QKNorm / RMSNorm / timeEmb / PI_Adapter`：看辅助机制。
9. `rope.py` 和 `denoise_schedular.py`：看位置编码和推理时间表。

