# DeCO 多模态数据流梳理

本文按“流入模型的信息模态”梳理 `models/deco/deco.py` 中所有输入的走向：每个模态的输入 shape、经过的模块、输出 shape、融合位置和作用。这里不展开模块内部细节，只把完整流程串清楚。

## 0. 当前配置下的默认维度

以 `config/deco.yaml` 为例：

| 配置项 | 值 |
|---|---:|
| `action_dim` | 28 |
| `chunk_size` | 32 |
| `obs_state` | `True` |
| `use_task_condition` | `False` |
| `use_tactile` | `False` |
| `inf_step` | 5 |
| `heads` | 8 |
| `dim` | 512 |
| `img_size` | `[256, 256]` |

通用记号：

| 记号 | 含义 |
|---|---|
| `B` | batch size |
| `C` | 图像通道数，通常为 3 |
| `H, W` | 输入图像高宽 |
| `h, w` | ResNet 输出特征图高宽 |
| `L = h * w` | 单张图像 token 数 |
| `chunk` | 动作序列长度，即 `chunk_size` |
| `act_dim` | 每一步动作维度 |
| `dim` | 模型内部 token 维度 |

如果输入图像是 `[B, 3, 256, 256]`，ResNet34 去掉池化和分类头后通常得到 `[B, 512, 8, 8]`，所以单张图像 `L=64`，两张图像合计 `2L=128` 个视觉 token。

## 1. 总体多模态融合图

```text
视觉:
  img1/img2
    -> img_encoding
    -> image tokens [B, 2L, dim]

动作:
  training=True: 真实动作 act [B, chunk, act_dim]
      -> add_noise
      -> noisy action [B, chunk, act_dim]
  training=False: 随机初始化 sample [B, chunk, act_dim]
      -> 作为当前待去噪动作
  noisy action / sample
      -> action_encoder + action_embedd
      -> action tokens [B, chunk, dim]

触觉, optional:
  tac1/tac2
    -> tactile branch
    -> tactile tokens [B, 68, dim]

本体状态, optional:
  obs
    -> obs_encoder
    -> obs condition [B, dim]

任务编号, optional:
  task_idx
    -> task_encoder
    -> task condition [B, dim]

时间步:
  t
    -> time_embedd
    -> time condition [B, dim]

条件融合:
  cond = time condition + obs condition + task condition

主体融合:
  image tokens + action tokens + cond (+ tactile tokens)
    -> repeated MMAttention
    -> action prediction [B, chunk, act_dim]
```

## 2. 视觉模态数据流

视觉输入是两张图：`img1` 和 `img2`。代码中两张图共用同一个 ResNet34 图像编码器。

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 原始输入 | `forward(img1, img2, ...)` | `img1: [B,C,H,W]`, `img2: [B,C,H,W]` | 同输入 | 两路视觉观测 |
| batch 维拼接 | `img_encoding` | 两个 `[B,C,H,W]` | `[2B,C,H,W]` | 一次性送入共享视觉骨干 |
| 图像骨干 | `self.img_encoder` | `[2B,C,H,W]` | `[2B,512,h,w]` | 提取空间视觉特征 |
| 通道投影 | `self.img_head` | `[2B,512,h,w]` | `[2B,dim,h,w]` | 映射到 Transformer token 维度 |
| 拆回两张图 | `feat.chunk(2, dim=0)` | `[2B,dim,h,w]` | 两个 `[B,dim,h,w]` | 恢复 img1/img2 两路特征 |
| 生成位置编码 | `self.rope(h, w)` | `h,w` | `(cos, sin)` | 给图像 token 的 q/k 提供二维位置信息 |
| 空间展平 | `einops.rearrange` | `[B,dim,h,w]` | `[B,L,dim]` | 把每个空间位置变成一个 token |
| 图像来源编码 | `self.pos_idx_embedd` | `[2L]` 的 0/1 id | `[B,2L,dim]` | 标记 token 来自第 1 张或第 2 张图 |
| 拼接两张图 token | `torch.cat([feat1, feat2], dim=1)` | 两个 `[B,L,dim]` | `[B,2L,dim]` | 得到统一视觉 token 序列 |
| 进入主干融合 | `atten_forward -> MMAttention` | `[B,2L,dim]` | 每层仍为 `[B,2L,dim]` | 与动作 token 做联合注意力 |

视觉模态最终不会直接输出动作，而是作为上下文 token 与动作 token 在 `MMAttention` 中融合，帮助模型根据场景预测动作方向。

## 3. 动作模态数据流

动作模态在训练和推理时入口不同，但进入 `atten_forward` 后流程一致。

### 3.1 训练时动作流

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 真实动作输入 | `forward(..., act=action, training=True)` | `[B,chunk,act_dim]` | 同输入 | 训练标签动作序列 |
| 随机时间步 | `t = sigmoid(randn([B]))` | `[B]` | `[B]` | 为每个样本采样噪声强度 |
| 加噪 | `add_noise(act, t)` | `act: [B,chunk,act_dim]`, `t: [B]` | `noisy_act: [B,chunk,act_dim]`, `noise: [B,chunk,act_dim]` | 构造模型输入动作和监督目标相关噪声 |
| 进入动作编码 | `atten_forward(..., act=noisy_act, ...)` | `[B,chunk,act_dim]` | 传入 `atten_forward` | 把加噪动作作为当前待去噪状态 |

训练分支最终返回：

```text
pred:  [B, chunk, act_dim]
noise: [B, chunk, act_dim]
```

训练脚本中使用：

```text
loss = MSE(pred, noise - action)
```

所以 `pred` 学的是动作去噪方向，而不是直接回归原始动作。

### 3.2 推理时动作流

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 随机初始化 | `sample = randn(...)` | 无动作输入 | `[B,chunk,act_dim]` | 从纯随机动作开始 |
| 时间表 | `get_schedule(inf_step, chunk_size)` | 标量配置 | 长度 `inf_step+1` 的 list | 生成从高噪声到低噪声的迭代时间 |
| 单步去噪输入 | 每次循环 `atten_forward(..., act=sample, ...)` | `[B,chunk,act_dim]` | `denoise_act: [B,chunk,act_dim]` | 预测当前 sample 的更新方向 |
| 更新动作 | `sample = sample + dt * denoise_act` | `[B,chunk,act_dim]` | `[B,chunk,act_dim]` | 逐步从噪声变成动作 |

推理分支最终返回：

```text
sample: [B, chunk, act_dim]
```

### 3.3 进入注意力前的动作编码

无论训练还是推理，动作进入 `atten_forward` 后都会经过同一段：

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 动作值编码 | `self.action_encoder` | `[B,chunk,act_dim]` | `[B,chunk,dim]` | 把动作向量映射为 token |
| 动作位置编码 | `+ self.action_embedd` | `[B,chunk,dim]` + `[1,chunk,dim]` | `[B,chunk,dim]` | 标记动作序列中的第几步 |
| 多层融合 | `self.mmattn` | `[B,chunk,dim]` | `[B,chunk,dim]` | 与视觉 token 交互，并受条件向量调制 |
| 输出投影 | `self.linear` | `[B,chunk,dim]` | `[B,chunk,act_dim]` | 输出动作方向预测 |

动作 token 是最终被投影成动作输出的主路径。

## 4. 触觉模态数据流

触觉只有在 `use_tactile=True` 时启用。当前 `config/deco.yaml` 中默认是 `False`，因此默认配置下这一路不会参与计算。

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 原始触觉输入 | `forward(..., tac1, tac2, ...)` | `tac1: [B,1062]`, `tac2: [B,1062]` | 同输入 | 左右两路触觉观测 |
| 区域索引 | `init_tac_regions` | 固定区域定义 | 17 个 `(start,end)` | 定义每只手触觉区域 |
| 区域均值 | list comprehension | `[B,1062]` | `tac1_avg: [B,17]`, `tac2_avg: [B,17]` | 每个触觉区域压成一个值 |
| 学习补充特征 | `self.tactile_encoder(cat([tac1,tac2]))` | `[B,2124]` | `[B,34]` | 从完整触觉向量得到额外触觉描述 |
| 拼接触觉描述 | `torch.cat([...], dim=-1)` | `[B,17] + [B,17] + [B,34]` | `[B,68]` | 得到 68 维触觉描述 |
| 门控 | `tactile * sigmoid(self.gated(tactile))` | `[B,68]` | `[B,68]` | 调整触觉通道权重 |
| token 化 | `self.pos_tac_embedd` | `[B,68]` | `[B,68,dim]` | 把每个触觉通道变为一个 token |
| 融入主干 | `MMAttention` tactile cross-attention | `tactile: [B,68,dim]` | 融入 attention 输出 | 作为图像+动作 token 可读取的额外上下文 |

触觉不是和视觉/动作 token 直接拼接到同一个 self-attention 序列里，而是在每个 `MMAttention` 中作为 key/value，被视觉+动作 query 通过 cross-attention 读取。

## 5. 本体状态 obs 数据流

本体状态只有在 `obs_state=True` 时启用。当前 `config/deco.yaml` 中默认启用。

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 原始状态输入 | `forward(..., obs=obs, ...)` | `[B,act_dim]` | 同输入 | 当前机器人本体状态 |
| 状态编码 | `self.obs_encoder` | `[B,act_dim]` | `[B,dim]` | 映射为条件向量 |
| 条件相加 | `t = t + obs` | `time_emb: [B,dim]`, `obs: [B,dim]` | `[B,dim]` | 把本体状态注入条件向量 |
| 调制主干 | `MMAttention(..., t=cond, ...)` | `[B,dim]` | 影响每层 token 更新 | 控制视觉/动作分支的归一化调制和残差门控 |

本体状态不作为 token 进入 attention 序列，而是被加到时间条件向量中，作为每层 `MMAttention` 的全局条件。

## 6. 任务编号 task_idx 数据流

任务编号只有在 `use_task_condition=True` 时启用。当前 `config/deco.yaml` 中默认是 `False`。

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 原始任务编号 | `forward(..., task_idx=task_idx, ...)` | `[B]` | 同输入 | 每个样本的任务 id |
| 任务编码 | `self.task_encoder` | `[B]` | `[B,dim]` | 查表得到任务条件向量 |
| 条件相加 | `t = t + task_emb` | `time_emb/obs_emb: [B,dim]`, `task_emb: [B,dim]` | `[B,dim]` | 把任务信息注入全局条件 |
| 调制主干 | `MMAttention(..., t=cond, ...)` | `[B,dim]` | 影响每层 token 更新 | 让同一视觉/动作输入可按任务产生不同动作 |

任务编号同样不作为 token 进入 attention，而是作为全局条件参与调制。

## 7. 时间步 t 数据流

时间步是去噪模型的核心条件。训练时随机采样，推理时来自去噪 schedule。

### 7.1 训练时

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 随机采样 | `torch.sigmoid(torch.randn((B,)))` | 无 | `[B]` | 每个样本一个噪声强度 |
| 用于加噪 | `add_noise(act, t)` | `t: [B]` | 加噪动作 `[B,chunk,act_dim]` | 控制动作和噪声的混合比例 |
| 时间编码 | `self.time_embedd(t)` | `[B]` | `[B,dim]` | 变成主干可用的条件向量 |
| 融合其他条件 | `+ obs`, `+ task_emb` | 多个 `[B,dim]` | `[B,dim]` | 得到最终条件向量 |

### 7.2 推理时

| 阶段 | 代码位置/模块 | 输入 shape | 输出 shape | 作用 |
|---|---|---:|---:|---|
| 时间表生成 | `get_schedule(inference_step, chunk_size)` | 标量配置 | list，长度 `inference_step+1` | 生成迭代去噪时间 |
| 当前时间步 | `torch.full((B,), t_curr)` | 标量 `t_curr` | `[B]` | 当前迭代步的时间条件 |
| 时间编码 | `self.time_embedd(t_vec)` | `[B]` | `[B,dim]` | 变成主干条件向量 |
| 融合其他条件 | `+ obs`, `+ task_emb` | 多个 `[B,dim]` | `[B,dim]` | 得到当前迭代步条件 |

时间步最终也不作为 token 进入 attention，而是和 obs/task 合成全局条件 `cond`，传给每个 `MMAttention`。

## 8. action_mask 数据流

`forward` 接口中有 `action_mask` 参数：

```python
action_mask=None
```

但在 `deco.py` 的模型内部没有使用它。训练时 `train_one_epoch.py` 也明确注释 diffusion loss 不使用 mask；验证时 mask 在模型外部用于计算验证 loss。

| 位置 | 使用情况 |
|---|---|
| `DECO.forward` | 参数存在，但未参与计算 |
| 训练 loss | 未使用 |
| 验证 loss | 在 `val(...)` 中用于 mask 掉无效动作位置 |

## 9. 多模态融合位置汇总

| 模态 | 是否变成 token | 进入位置 | 与其他模态如何融合 | 最终影响 |
|---|---|---|---|---|
| 视觉 `img1/img2` | 是，`[B,2L,dim]` | `atten_forward` | 与动作 token 做 joint attention | 提供场景上下文 |
| 动作 `act/sample` | 是，`[B,chunk,dim]` | `atten_forward` | 与视觉 token 做 joint attention | 是最终输出动作的主路径 |
| 触觉 `tac1/tac2` | 是，`[B,68,dim]` | `MMAttention` 内部 | 作为 key/value 被视觉+动作 query 读取 | 提供接触上下文 |
| 本体状态 `obs` | 否，`[B,dim]` 条件向量 | `forward` 中加到 `t` | 作为 `MMAttention` 的条件调制 | 按当前机器人状态调整更新 |
| 任务 `task_idx` | 否，`[B,dim]` 条件向量 | `forward` 中加到 `t` | 作为 `MMAttention` 的条件调制 | 按任务调整策略 |
| 时间步 `t` | 否，`[B,dim]` 条件向量 | 每次去噪前生成 | 与 obs/task 相加成条件向量 | 告诉模型当前噪声阶段 |
| `action_mask` | 否 | 模型外部 | 不进入模型 | 仅验证 loss 使用 |

## 10. 一次训练 forward 的完整信息流

```text
输入:
  img1:     [B, C, H, W]
  img2:     [B, C, H, W]
  obs:      [B, act_dim]              if obs_state=True
  act:      [B, chunk, act_dim]
  task_idx: [B]                       if use_task_condition=True
  tac1:     [B, 1062]                 if use_tactile=True
  tac2:     [B, 1062]                 if use_tactile=True

视觉流:
  img1/img2 -> img_encoding -> feat [B, 2L, dim], image_rotary_emb

动作流:
  act + random t -> add_noise -> noisy_act [B, chunk, act_dim], noise [B, chunk, act_dim]
  noisy_act -> action_encoder + action_embedd -> action tokens [B, chunk, dim]

条件流:
  t -> time_embedd -> [B, dim]
  obs -> obs_encoder -> [B, dim]                       optional
  task_idx -> task_encoder -> [B, dim]                 optional
  cond = time_emb + obs_emb + task_emb                 [B, dim]

触觉流:
  tac1/tac2 -> tactile branch -> tactile [B, 68, dim]  optional

融合:
  feat [B, 2L, dim]
  action tokens [B, chunk, dim]
  cond [B, dim]
  tactile [B, 68, dim] optional
    -> MMAttention * num_attn_blocks
    -> updated action tokens [B, chunk, dim]

输出:
  linear(action tokens) -> pred [B, chunk, act_dim]
  return pred, noise
```

## 11. 一次推理 forward 的完整信息流

```text
输入:
  img1:     [B, C, H, W]
  img2:     [B, C, H, W]
  obs:      [B, act_dim]              if obs_state=True
  task_idx: [B]                       if use_task_condition=True
  tac1:     [B, 1062]                 if use_tactile=True
  tac2:     [B, 1062]                 if use_tactile=True

视觉流:
  img1/img2 -> img_encoding -> feat [B, 2L, dim], image_rotary_emb

初始化动作:
  sample = randn([B, chunk, act_dim])

固定条件:
  obs -> obs_encoder -> [B, dim]                       optional
  task_idx -> task_encoder -> [B, dim]                 optional
  tac1/tac2 -> tactile branch -> [B, 68, dim]          optional

循环去噪, 共 inference_step 次:
  t_curr -> time_embedd -> [B, dim]
  cond = time_emb + obs_emb + task_emb                 [B, dim]
  sample -> action_encoder + action_embedd             [B, chunk, dim]
  feat + action tokens + cond (+ tactile)
      -> MMAttention * num_attn_blocks
      -> denoise_act [B, chunk, act_dim]
  sample = sample + (t_prev - t_curr) * denoise_act

输出:
  sample [B, chunk, act_dim]
```

## 12. 按默认配置的具体 shape 示例

使用 `config/deco.yaml` 默认值：

```text
action_dim = 28
chunk_size = 32
dim = 512
img_size = [256, 256]
```

假设 batch size 为 `B`：

| 模态 | 输入 shape | 中间 shape | 融合处 shape | 输出/影响 |
|---|---:|---:|---:|---|
| 视觉 | `img1/img2: [B,3,256,256]` | ResNet 后约 `[B,512,8,8]` | `feat: [B,128,512]` | 作为视觉上下文 |
| 动作训练 | `act: [B,32,28]` | `noisy_act: [B,32,28]` -> token `[B,32,512]` | `[B,32,512]` | 输出 `pred: [B,32,28]` |
| 动作推理 | 无真实动作输入 | `sample: [B,32,28]` -> token `[B,32,512]` | `[B,32,512]` | 输出 `sample: [B,32,28]` |
| 本体状态 | `obs: [B,28]` | `obs_emb: [B,512]` | `cond: [B,512]` | 调制每层注意力块 |
| 时间步 | `t: [B]` | `time_emb: [B,512]` | `cond: [B,512]` | 指示当前噪声阶段 |
| 任务编号 | 默认关闭 | 若启用：`task_idx [B] -> [B,512]` | `cond: [B,512]` | 调制策略 |
| 触觉 | 默认关闭 | 若启用：`tac1/tac2 [B,1062] -> [B,68,512]` | tactile cross-attention | 提供接触上下文 |

