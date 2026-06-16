# DECO 方法流程与代码解读

本文档用于单独理解 DECO 方法本身：它不是项目部署手册，而是把论文中的方法流程和当前仓库代码一一对应起来，帮助你知道“这个方法为什么这样设计、数据如何变成动作、每个模块在代码里到底做了什么”。

参考材料：

- 本地论文：`2602.05513v2.pdf`
- 前置导读：`deco_paper_prerequisites.md`
- 核心代码：`IL_training_codebase-master/models/deco/deco.py`
- 训练代码：`IL_training_codebase-master/models/deco/train_one_epoch.py`
- 数据代码：`IL_training_codebase-master/dataset.py`
- 推理代码：`IL_training_codebase-master/inference.py`

---

## 1. 方法一句话概括

**输入：** 论文方法层面的所有观测来源，包括双目图像、本体状态、可选触觉、可选任务条件，以及动作生成过程中的噪声动作序列。

**输出：** 对 DECO 方法的全局理解：它最终输出的是未来一段机器人动作 `action chunk`，并且不同模态通过不同路径进入模型。

DECO 是一个用于双臂灵巧操作的多模态动作生成策略。它把 **视觉、本体状态、触觉、任务条件** 分开注入模型，而不是简单拼接：

```text
双目图像
  → 视觉 token

噪声动作序列
  → 动作 token

本体状态 + 任务条件 + 扩散/flow 时间步
  → AdaLN 条件调制

触觉信号
  → 触觉 token / adapter / cross-attention

最终输出
  → 未来一段机器人动作 action chunk
```

它的核心思想可以拆成三句话：

1. 视觉和动作 token 通过 joint self-attention 互相交互。
2. 本体状态和任务条件不作为普通 token，而是通过 AdaLN 调制 Transformer。
3. 触觉作为接触相关的稀疏信号，通过专门的 tactile adapter 和 cross-attention 插入视觉策略。

---

## 2. 从训练到推理的整体流程

**输入：** 训练阶段输入示教数据中的当前观测和未来真实动作；推理阶段输入真实机器人或仿真环境的当前观测，以及随机初始化的噪声动作。

**输出：** 训练阶段输出模型参数和训练损失；推理阶段输出一个已经去噪的动作 chunk，供仿真环境或真实机器人执行。

### 2.1 训练阶段

训练时，DECO 并不是直接学习：

```text
观测 → 动作
```

而是学习：

```text
观测 + 被加噪的动作 → 如何把噪声动作修正回真实动作
```

代码流程：

```text
dataset.py
→ 读取当前图像、状态、触觉、未来动作 chunk
→ train.py
→ models.deco.train_one_epoch.train()
→ DECO.forward(training=True)
→ add_noise()
→ atten_forward()
→ 输出修正方向
→ MSE(out, noise - action)
```

对应关键代码：

```python
# models/deco/deco.py
t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))
act, noise = self.add_noise(act, t)
...
feat, act = self.atten_forward(...)
return act, noise
```

```python
# models/deco/train_one_epoch.py
out, noise = net(..., act=action, training=True)
loss = F.mse_loss(out, noise - action)
```

这里 `action` 是真实动作 chunk，`noise` 是高斯噪声。模型训练目标是 `noise - action`，可以理解为从真实动作指向噪声动作的“速度/方向”。推理时会沿相反方向把噪声逐步拉回动作。

### 2.2 推理阶段

推理时没有真实动作，所以 DECO 从随机噪声动作开始：

```text
随机动作 sample
→ 输入当前观测
→ 模型预测修正方向 denoise_act
→ sample 更新
→ 重复 inf_step 次
→ 得到动作 chunk
```

对应代码：

```python
# models/deco/deco.py
sample = torch.randn(img1.shape[0], self.chunk_size, self.act_dim).to(img1.device)
t = get_schedule(self.inference_step, self.chunk_size)
for t_curr, t_prev in zip(t[:-1], t[1:]):
    ...
    _, denoise_act = self.atten_forward(...)
    sample = sample + (t_prev - t_curr) * denoise_act
return sample
```

因为 `t` 从接近 1 走向 0，`t_prev - t_curr` 通常是负数；模型预测的是 `noise - action`，乘上负方向后，sample 会从噪声向动作靠近。

---

## 3. 数据如何进入 DECO

**输入：** `Dataset/episode_xxxxxx` 下的双图像、触觉 `.npy`、`data.pkl` 中的状态/动作/任务字段，以及 YAML 中的数据统计量。

**输出：** DataLoader batch：`img1, img2, tact1, tact2, obs_state, action_padd, mask, task_idx`，供 `train.py` 和 `DECO.forward()` 使用。

### 3.1 数据集样本

真实数据由 `dataset.py` 加载。每个样本来自一个 episode 的某一帧，取当前观测和未来 `chunk_size` 步动作。

代码入口：

```python
class my_Dataset(Dataset):
    def __getitem__(self, index):
        ...
        return img1, img2, tact1, tact2, obs_state, action_padd, mask, task_idx
```

一个 batch 的结构如下：

| 字段 | 代码变量 | 典型形状 | 含义 |
|---|---|---:|---|
| 左/第一视角图像 | `img1` | `[B, 3, H, W]` | `*_color_0.jpg` |
| 右/第二视角图像 | `img2` | `[B, 3, H, W]` | `*_color_1.jpg` |
| 左手触觉 | `tact1` | `[B, 1062]` | 左 Inspire Hand 触觉 |
| 右手触觉 | `tact2` | `[B, 1062]` | 右 Inspire Hand 触觉 |
| 本体状态 | `obs_state` | `[B, 28]` | 左臂/手 + 右臂/手 + 头部 |
| 未来动作 | `action_padd` | `[B, chunk_size, 28]` | 未来动作序列 |
| padding mask | `mask` | `[B, chunk_size]` | episode 尾部补齐标记 |
| 任务编号 | `task_idx` | `[B]` | 可选任务条件 |

### 3.2 28 维状态和动作含义

当前代码中 DECO 默认 `action_dim=28`，其语义来自数据拼接和部署拆分。

训练侧：

```python
obs_state = left_obs + right_obs + head_obs
action = left_action + right_action + head_action
```

部署侧 `deploy_h1.py` 的切片逻辑更明确：

```text
action[0:7]    → left arm
action[7:13]   → left hand
action[13:20]  → right arm
action[20:26]  → right hand
action[26:28]  → active camera head
```

所以理解 DECO 的动作空间时，不要只看 `action_dim=28`，要把它理解成：

```text
7 左臂关节 + 6 左手关节 + 7 右臂关节 + 6 右手关节 + 2 主动相机关节
```

### 3.3 归一化

训练时 `dataset.py` 对状态和动作做归一化：

```python
obs_state = (obs_state - self.obs_mean) / self.obs_std
action_padd = (action_padd - self.action_mean[None, :]) / self.action_std[None, :]
```

推理时 `inference.py` 做同样的输入归一化，并在输出后反归一化：

```python
obs = (obs - obs_mean) / obs_std
...
action = action * action_std[None, :] + action_mean[None, :]
```

这说明：训练和部署必须使用同一份 `observation_mean/std`、`action_mean/std`。如果统计量错了，模型即使权重正确，实机动作幅度也会错。

---

## 4. DECO 网络结构总览

**输入：** 已归一化的图像张量、本体状态、动作 chunk、可选触觉、可选任务编号，以及扩散/flow 时间步。

**输出：** 训练时输出动作修正方向和噪声；推理时输出归一化动作空间中的动作 chunk，随后由 `inference.py` 反归一化。

代码中的主类：

```python
class DECO(nn.Module):
```

核心模块初始化如下：

```python
self.img_encoder = ResNet34 without final layers
self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)
self.obs_encoder = Linear(action_dim → dim)
self.task_encoder = Embedding(num_tasks, dim)
self.time_embedd = sinusoidal time embedding + MLP
self.action_encoder = Linear(action_dim → dim)
self.mmattn = ModuleList([MMAttention(...)])
self.linear = Linear(dim → action_dim)
```

整体结构可以画成：

```text
img1, img2
  → ResNet34
  → Conv2d projection
  → flatten 成视觉 token
  → 加相机 ID embedding
  → RoPE 位置编码
              \
               → MMAttention blocks → action tokens → Linear → 动作修正方向
              /
noisy action
  → Linear projection
  → action positional embedding

timestep + obs + task
  → time / state / task embedding 相加
  → AdaLN 调制每个 attention block

tactile
  → tactile encoder
  → tactile cross-attention
```

---

## 5. 视觉分支：双图像如何变成 token

**输入：** 两路图像 `img1`、`img2`，形状通常为 `[B, 3, 256, 256]`，已经经过 resize、tensor 化和 ImageNet 归一化。

**输出：** 带有相机 ID embedding 和二维位置信息的视觉 token 序列，形状近似为 `[B, 2 * image_token_num, dim]`。

视觉处理在 `DECO.img_encoding()` 中：

```python
img = torch.cat([img1, img2], dim=0)
feat = self.img_encoder(img)
feat = self.img_head(feat)
feat1, feat2 = feat.chunk(2, dim=0)
```

解释：

1. `img1` 和 `img2` 先沿 batch 维拼起来，这样可以共用同一个 ResNet34 编码器。
2. ResNet34 输出 feature map。
3. `img_head` 把通道从 512 映射到模型隐藏维度 `dim`，默认 512。
4. 再拆回两路图像。

随后把 feature map 拉平成 token：

```python
feat1 = einops.rearrange(feat1, 'b c h w -> b (h w) c')
feat2 = einops.rearrange(feat2, 'b c h w -> b (h w) c')
feat = torch.cat([feat1, feat2], dim=1)
```

如果输入图像是 `[B, 3, 256, 256]`，ResNet34 最后特征通常约为 `[B, 512, 8, 8]`，那么每路图像约有 `64` 个 token，两路合起来约 `128` 个视觉 token。

代码还加了相机 ID embedding：

```python
img_id = torch.tensor([0]*feat1.shape[1] + [1]*feat2.shape[1]).to(img2.device)
img_id = self.pos_idx_embedd(img_id).repeat(img1.shape[0], 1, 1)
feat = feat + img_id
```

这一步的作用是告诉模型：前一半 token 来自第一路图像，后一半 token 来自第二路图像。否则两路图像 flatten 后只是一串 token，模型难以区分来源。

---

## 6. 动作分支：动作 chunk 如何变成 token

**输入：** 训练阶段输入加噪前的真实动作 chunk；进入 action encoder 前，代码会先把真实动作与高斯噪声按时间步 `t` 混合成 noisy action。

**输出：** 动作 token 序列 `[B, chunk_size, dim]`；经过多层 attention 后，再由 `self.linear` 映射回 `[B, chunk_size, action_dim]` 的动作修正方向。

训练时输入的 `act` 是真实动作加噪后的动作序列，形状：

```text
[B, chunk_size, action_dim]
```

在代码中：

```python
act = self.action_encoder(act)
act = act + self.action_embedd
```

解释：

1. 每个时间步的 28 维动作通过 MLP 映射到 `dim` 维。
2. `self.action_embedd` 是可学习的位置 embedding，用于区分 chunk 内第 0、1、2... 个未来动作。

因此动作 token 的含义是：

```text
第 i 个动作 token = 第 i 个未来时间步的 noisy action 表示
```

DECO 最终要更新的主要对象就是这些动作 token。

---

## 7. 本体状态、任务条件和时间步：为什么通过 AdaLN

**输入：** 本体状态 `obs`、任务编号 `task_idx`、flow/diffusion 时间步 `t` 或 `t_vec`。

**输出：** 一个全局条件向量 `[B, dim]`，用于生成 AdaLN 的 `scale / shift / gate`，进而调制视觉分支和动作分支。

DECO 没有把本体状态 `obs` 直接拼到 token 序列里，而是编码成一个全局条件向量：

```python
if self.obs_state:
    obs = self.obs_encoder(obs)
if self.use_task_condition:
    task_emb = self.task_encoder(task_idx)
```

训练时：

```python
t = self.time_embedd(t)
if self.obs_state:
    t = t + obs
if self.use_task_condition:
    t = t + task_emb
```

推理时也类似：

```python
t_vec = self.time_embedd(t_vec)
t_vec = t_vec + obs + task_emb
```

这个合成后的向量 `t` 会传入每个 `MMAttention` block，用于 AdaLN：

```python
scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)
img_norm = self.img_norm1(img)
img_norm = (1 + scale1_feat) * img_norm + shift1_feat
```

同样动作分支也有：

```python
scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t)
act_norm = self.act_norm1(act)
act_norm = (1 + scale1_act) * act_norm + shift1_act
```

直观理解：

```text
本体状态/任务/时间步不是一个被注意力读取的普通 token，
而是控制整个 Transformer block 如何归一化、放大、偏移和门控的条件。
```

这样做的好处是，本体状态描述的是全局机器人姿态，不一定需要像图像一样形成空间 token；通过 AdaLN 调制整个 block，更像是在告诉模型“当前机器人处在什么状态、现在去噪到第几步、任务是什么”。

---

## 8. MMAttention：视觉 token 和动作 token 如何交互

**输入：** 视觉 token `img`、动作 token `act`、全局条件向量 `t`、视觉 RoPE 位置编码，以及可选触觉 token。

**输出：** 更新后的视觉 token 和动作 token；其中动作 token 会继续进入下一层 block，最终被映射为动作修正方向。

`MMAttention` 是 DECO 的核心 block。

输入：

```text
img: [B, image_tokens, dim]
act: [B, chunk_size, dim]
t:   [B, dim]
```

### 8.1 视觉和动作各自做 QKV

视觉分支：

```python
img_qkv = self.img_qkv(img_norm)
img_q, img_k, img_v = rearrange(...)
```

动作分支：

```python
act_qkv = self.act_qkv(act_norm)
act_q, act_k, act_v = rearrange(...)
```

注意：视觉和动作不是共用同一个 QKV projection，而是各自有独立 projection。这符合“decoupled multimodal”的思想：不同模态先用自己的投影方式进入 attention。

### 8.2 视觉 RoPE

视觉 Q/K 会加旋转位置编码：

```python
img_q[:, :, :feat_len, :] = apply_rotary_emb(...)
img_q[:, :, feat_len:, :] = apply_rotary_emb(...)
```

两路图像分别应用 RoPE，帮助模型保留图像 token 的二维空间位置信息。

### 8.3 joint self-attention

最关键的一步：

```python
q = torch.cat([img_q, act_q], dim=2)
k = torch.cat([img_k, act_k], dim=2)
v = torch.cat([img_v, act_v], dim=2)
attn = F.scaled_dot_product_attention(q, k, v)
```

这里视觉 token 和动作 token 被放进同一个 attention 空间。效果是：

```text
动作 token 可以看图像 token：知道场景和物体在哪里。
图像 token 也可以看动作 token：根据当前动作生成过程调整视觉表征。
动作 token 之间也能互相看：保证 chunk 内动作连贯。
```

### 8.4 split 回各自模态

attention 之后再拆开：

```python
img_attn, act_attn = attn[:, :total_img_len, :], attn[:, total_img_len:, :]
```

然后分别走 residual、projection、MLP：

```python
img = img + gate1_feat * self.img_proj(img_attn)
img = img + gate2_feat * self.img_mlp(...)

act = act + gate1_act * self.act_proj(act_attn)
act = act + gate2_act * self.act_mlp(...)
```

最终 `DECO.atten_forward()` 只把动作 token 通过 `self.linear` 映射回动作维度：

```python
act = self.linear(act)
return img, act
```

---

## 9. 触觉分支：tactile adapter 如何工作

**输入：** 左右手原始触觉向量 `tac1`、`tac2`，每只手 `1062` 维，且只有在 `use_tactile=True` 时进入模型。

**输出：** 触觉 token `[B, 68, dim]`，作为 cross-attention 的 key/value，为视觉 token 和动作 token 提供接触信息。

触觉只在 `use_tactile=True` 时启用。

### 9.1 原始触觉维度

每只 Inspire Hand 的触觉输入是 `1062` 维。代码中把它分成 17 个 tactile region：

```python
assert _touch_start == 1062
```

每个 region 对应手指尖、手指腹、掌部等区域。

### 9.2 两种触觉特征

代码中触觉特征有两部分。

第一部分：区域均值特征。

```python
tac1_avg = torch.stack([tac1[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)
tac2_avg = torch.stack([tac2[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()], dim=1)
```

这会得到：

```text
左手 17 维 + 右手 17 维
```

第二部分：可学习 tactile encoder。

```python
tactile_emb = self.tactile_encoder(torch.cat([tac1, tac2], dim=-1))
```

输出是 34 维，对应左右手各 17 个区域的学习特征。

然后拼接：

```python
tactile = torch.cat([tac1_avg, tac2_avg, tactile_emb], dim=-1)
```

得到：

```text
17 + 17 + 34 = 68
```

### 9.3 gated tactile

```python
tactile = tactile * torch.sigmoid(self.gated(tactile))
```

这一步像一个门控机制：不是所有触觉信号都同等重要，模型可以学习哪些区域、哪些强度更值得相信。

### 9.4 触觉位置/区域 embedding

```python
tactile = self.pos_tac_embedd(tactile)
```

代码注释写的是：

```text
(b, 68) --> (b, 68, dim)
```

实际这里的 `timeEmb` 会把输入每个标量映射到 `dim`，所以触觉最终变成一组 tactile tokens。

### 9.5 tactile cross-attention

在 `MMAttention.forward()` 中：

```python
tactile_k = self.tactile_key(tactile)
tactile_v = self.tactile_value(tactile)
cross_attn = F.scaled_dot_product_attention(q, tactile_k, tactile_v)
attn = attn + cross_attn
```

含义：

```text
图像 token + 动作 token 作为 query，
触觉 token 作为 key/value，
模型在生成动作时主动查询触觉信息。
```

这和简单拼接触觉有明显区别。简单拼接是“触觉永远混在全局条件里”，cross-attention 则允许模型在某些动作生成阶段选择性关注触觉。

---

## 10. Plugin / LoRA adapter：如何低成本加入触觉

**输入：** 阶段一训练好的视觉策略 checkpoint、启用 `use_tactile=True` 和 `plugin=True` 的模型配置，以及可选 `adapter_model_path`。

**输出：** 一个保留原视觉策略能力、只训练新增/低秩触觉相关参数的 DECO.p / plugin tactile adapter 模型。

论文强调两阶段训练：

```text
阶段一：训练视觉 + 本体策略
阶段二：冻结已有策略，只训练触觉 adapter / LoRA
```

代码对应在 `modeling()`：

```python
if use_tactile:
    if plugin:
        if adapter_model_path:
            DeCO.load_state_dict(model_dict, strict=True)
        else:
            model_dict = torch.load(pretrain_model_path, map_location='cpu')
            pretrain_dict = {k: v for k, v in model_dict.items() if k in model_state.keys() and v.shape == model_state[k].shape}
            DeCO.load_state_dict(pretrain_dict, strict=False)
            for name, param in DeCO.named_parameters():
                if name in model_dict.keys():
                    param.requires_grad = False
                else:
                    print('trainable params:', name)
```

这段逻辑的含义：

1. 先构建一个带触觉 adapter 的 DECO。
2. 加载视觉预训练权重中能匹配上的参数。
3. 对于预训练 checkpoint 已有的参数，冻结。
4. 对于新增的触觉 encoder、cross-attention、PI adapter 等参数，保持可训练。

LoRA / adapter 模块代码：

```python
class PI_Adapter(nn.Module):
    def __init__(self, dim, out_dim, rank=32):
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, out_dim)
```

在 `MMAttention` 中，如果 `use_tactile and plugin`：

```python
self.img_qkv_pi = PI_Adapter(dim, dim*3, plugin_rank)
self.img_proj_pi = PI_Adapter(dim, dim, plugin_rank)
self.img_mlp_pi = PI_Adapter(dim, dim, plugin_rank)
self.act_qkv_pi = PI_Adapter(dim, dim*3, plugin_rank)
self.act_proj_pi = PI_Adapter(dim, dim, plugin_rank)
self.act_mlp_pi = PI_Adapter(dim, dim, plugin_rank)
```

前向时是加法注入：

```python
img_qkv += self.img_qkv_pi(img_norm)
act_qkv += self.act_qkv_pi(act_norm)
```

直观理解：

```text
原模型保持不动；
adapter 学一个低秩的“修正量”；
这个修正量让原来的视觉动作策略能够使用触觉。
```

---

## 11. Flow Matching / 去噪动作生成的代码级解释

**输入：** 训练时输入真实动作 `action`、随机噪声 `noise`、随机时间步 `t`；推理时输入随机初始化的 `sample` 和去噪 schedule。

**输出：** 训练时输出学习目标 `noise - action` 对应的速度场；推理时输出从噪声逐步更新得到的动作 chunk。

### 11.1 训练时的加噪公式

代码：

```python
noise = torch.randn_like(act)
act_noisy = (1 - t) * act + t * noise
```

可以写成：

```text
x_t = (1 - t) * x_action + t * x_noise
```

其中：

- `t = 0` 时，`x_t` 接近真实动作。
- `t = 1` 时，`x_t` 接近纯噪声。

训练目标：

```text
v = x_noise - x_action
```

代码：

```python
loss = MSE(model(x_t, observation, t), noise - action)
```

### 11.2 推理时的反向更新

推理从噪声开始：

```text
sample ≈ x_noise
```

模型预测：

```text
denoise_act ≈ x_noise - x_action
```

更新：

```python
sample = sample + (t_prev - t_curr) * denoise_act
```

因为 `t_prev < t_curr`，所以：

```text
sample = sample - 小步长 * (noise - action)
```

也就是逐步从噪声向动作靠近。

### 11.3 和传统 diffusion policy 的差异

当前仓库里也有 `models/dp/diffusion_policy.py`，它使用 `diffusers` 的 DDPM/DDIM scheduler。DECO 则自己实现了更直接的 flow-style schedule：

```python
from models.deco.denoise_schedular import get_schedule
```

DECO 的推理步数由 YAML 的 `model.inf_step` 控制，默认 `5`。这意味着它通常用较少步数生成动作 chunk，适合机器人控制中的实时性需求。

---

## 12. 一次训练 batch 在模型里的完整路线

**输入：** 一个 DataLoader batch，包括图像、状态、动作 chunk、触觉、任务编号，以及 `config/deco.yaml` 中的模型参数。

**输出：** 一个 batch 的训练损失 `MSE(out, noise - action)`，反向传播后更新模型参数。

假设使用 `config/deco.yaml` 默认配置：

```yaml
action_dim: 28
chunk_size: 32
obs_state: True
use_tactile: False
dim: 512
heads: 8
num_attn_blocks: 6
inf_step: 5
```

那么一次 batch 的路线是：

```text
dataset.py 输出：
img1:   [B, 3, 256, 256]
img2:   [B, 3, 256, 256]
obs:    [B, 28]
action: [B, 32, 28]
tac1:   [B, 1062]
tac2:   [B, 1062]

DECO.forward(training=True):
img1/img2 → ResNet34 → visual tokens: [B, ~128, 512]
action    → add_noise → action_encoder → action tokens: [B, 32, 512]
obs       → obs_encoder: [B, 512]
timestep  → time_embedd: [B, 512]
obs + timestep → AdaLN condition: [B, 512]

6 个 MMAttention blocks:
visual tokens 和 action tokens joint self-attention
AdaLN 调制视觉和动作分支

linear:
action tokens → [B, 32, 28]

loss:
MSE(out, noise - action)
```

如果 `use_tactile=True`：

```text
tac1/tac2: [B, 1062] each
→ 区域均值 17+17
→ tactile_encoder 输出 34
→ 拼成 68
→ gated
→ tactile token: [B, 68, 512]
→ 在 MMAttention 中作为 cross-attention 的 K/V
```

---

## 13. 一次实机推理在模型里的完整路线

**输入：** 实机相机图像、Unitree DDS 读取的手臂状态、Inspire Hand 状态/触觉、Dynamixel 头部状态，以及已加载权重的策略模型。

**输出：** 反归一化后的 28 维动作 chunk，并在部署程序中拆成双臂、双手和主动相机控制命令。

实机入口是 `deploy/deploy_h1.py`。

核心循环：

```text
读取头部双目图像
→ 左右切图
→ BGR 转 RGB
→ 读取手臂/手/头部状态
→ 读取触觉
→ predict_action()
→ 得到 action chunk
→ 选第一个动作或前 N 个动作
→ 发送到机器人
```

`predict_action()` 中：

```python
img1, img2, obs, tac1, tac2 = preprocess(...)
action = model(..., training=False)
action = action.cpu().squeeze(0)
action = postprocess(action, yaml_config)
```

对于 DECO/DP，部署代码使用 receding horizon：

```python
if model_name == 'deco' or model_name == 'dp':
    n_action_select = args.select_action
    receding_horizon = True
    temporal_ensembler_flag = False
```

如果 action queue 为空，就重新预测一个 chunk；否则继续执行上次预测的后续动作：

```python
action_receding = action[1:n_action_select, :]
action = action[0, :]
```

这就是机器人部署中常见的 chunk policy：

```text
不是每个控制周期都从头预测，
而是预测一段动作，执行其中若干步，再重新观测和预测。
```

好处：

- 动作更连贯。
- 推理开销更低。
- 保留一定闭环能力。

风险：

- `select_action` 太大时更接近 open-loop，环境变化时反应慢。
- `select_action` 太小时频繁重规划，可能抖动。

---

## 14. DECO 方法和代码的对应表

**输入：** 论文方法概念和当前仓库中的实现文件。

**输出：** 一张用于快速定位代码的索引表，帮助你从概念跳转到对应函数和模块。

| 论文/方法概念 | 代码位置 | 代码实现方式 | 你应该重点看什么 |
|---|---|---|---|
| Action chunk | `dataset.py` | `df_slice = ... img_idx:img_idx+chunksize` | 如何截取未来动作 |
| 动作归一化 | `dataset.py`, `inference.py` | `action_mean/std` | 训练和推理必须一致 |
| Flow Matching | `deco.py`, `train_one_epoch.py` | `add_noise()` + `MSE(out, noise-action)` | 训练目标 |
| 推理去噪 | `deco.py` | `sample = randn`, loop over schedule | `inf_step` 如何影响速度 |
| 视觉编码 | `deco.py` | ResNet34 + Conv2d + flatten | `img_encoding()` |
| 双相机区分 | `deco.py` | `pos_idx_embedd` | 图像 token 来源 |
| 视觉位置编码 | `rope.py`, `deco.py` | RoPE on image Q/K | `apply_rotary_emb()` |
| 动作 token | `deco.py` | `action_encoder` + `action_embedd` | chunk 内时序位置 |
| 本体条件 | `deco.py` | `obs_encoder`, add to time embedding | 不是 token，是 AdaLN 条件 |
| 任务条件 | `deco.py` | `task_encoder` | 多任务时启用 |
| MMDiT block | `deco.py` | `MMAttention` | joint self-attention |
| AdaLN | `deco.py` | `adaLN` 输出 scale/shift/gate | 条件如何调制 block |
| 触觉区域 | `deco.py` | `init_tac_regions()` | 17 regions / hand |
| 触觉 adapter | `deco.py` | `tactile_encoder`, `tactile_key/value` | cross-attention |
| LoRA/低秩 adapter | `deco.py` | `PI_Adapter` | plugin 训练 |
| 权重冻结 | `deco.py` | `param.requires_grad=False` | 两阶段训练 |
| 实机闭环 | `deploy_h1.py` | `predict_action()` + action slicing | chunk 如何执行 |

---

## 15. 容易误解的点

**输入：** 阅读论文和代码时最容易产生混淆的设计点、参数点和实现细节。

**输出：** 一组排雷结论，帮助你避免把 DECO 理解成普通 concat 多模态模型，或在改代码时踩维度/归一化/触觉加载问题。

### 15.1 DECO 不是把所有输入简单 concat

代码明确体现了 decoupled 注入：

- 图像：ResNet + token + self-attention。
- 动作：action token。
- 本体/任务/时间：AdaLN 条件。
- 触觉：cross-attention + plugin adapter。

如果只是 concat，代码里应该会出现把 `obs/tactile/image_feat` 拼成一个大向量再输入 MLP 的逻辑；DECO 不是这样。

### 15.2 `obs_state` 的维度被假设等于 `action_dim`

代码中：

```python
self.obs_encoder = nn.Sequential(
    nn.Linear(act_dim, dim),
    ...
)
```

也就是说当前实现假设本体状态维度和动作维度相同。默认真实机器人任务都是 28 维。如果你以后想让观测维度和动作维度不同，需要改模型构造参数。

### 15.3 `img_pretrain` 参数在当前代码里基本没有被使用

`DECO.__init__()` 里有 `img_pretrain=False`，但实际代码直接使用：

```python
resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
```

所以 YAML 里如果写 `img_pretrain`，当前代码并没有按路径加载它。

### 15.4 `freeze_backbone` 参数当前没有实际生效

`DECO.__init__()` 接收 `freeze_backbone=True`，也有 `freeze()` 方法，但初始化里没有调用 `self.freeze()`。如果你想冻结 ResNet，需要显式改代码。

### 15.5 触觉关闭时，数据集仍然会加载触觉文件

`dataset.py` 无论 `use_tactile` 是否为 False，都会执行：

```python
tact1 = np.load(tac1_path)
tact2 = np.load(tac2_path)
```

所以“无触觉训练”并不等于数据目录可以没有 tactile 文件。README 提到无触觉时可给 dummy 值，这一点需要改数据加载代码或准备 dummy `.npy`。

### 15.6 推理输出必须反归一化才能发给机器人

模型内部输出的是归一化动作空间中的动作。`inference.py` 的 `postprocess()` 会反归一化。部署时必须经过 `predict_action()`，不要直接把 `model(..., training=False)` 的输出发给机器人。

---

## 16. 推荐阅读代码顺序

**输入：** 你当前想理解 DECO 方法的学习目标，以及仓库中的核心文件。

**输出：** 一个从数据、训练目标、模型结构、推理到实机执行的阅读路线。

如果目标是理解 DECO 方法，而不是部署，建议按这个顺序读：

1. `dataset.py`

先理解一个训练样本到底是什么，尤其是：

```text
当前图像 + 当前状态 + 当前触觉 + 未来动作 chunk
```

2. `models/deco/train_one_epoch.py`

看训练目标：

```text
MSE(out, noise - action)
```

这一步能帮助你抓住 DECO 不是普通行为克隆，而是动作生成/去噪策略。

3. `models/deco/deco.py` 的 `DECO.forward()`

先读训练分支，再读推理分支：

```python
if training:
    ...
else:
    ...
```

4. `DECO.img_encoding()`

理解图像 token 从哪里来。

5. `MMAttention.forward()`

理解视觉和动作 token 怎么交互，本体条件怎么调制，触觉怎么进入。

6. `modeling()`

理解预训练权重、触觉 adapter、plugin 和冻结逻辑。

7. `inference.py`

理解训练好的模型怎么用于部署。

8. `deploy/deploy_h1.py`

理解 action chunk 在真实机器人上怎么被执行。

---

## 17. 用一句话串起整个方法

**输入：** 前面 16 个部分中关于数据、模型、训练、触觉 adapter、推理和部署的所有局部理解。

**输出：** 一段可以作为复述 DECO 方法时使用的完整总结。

DECO 先从示教数据中取出“当前多模态观测”和“未来动作 chunk”，训练模型在不同噪声时间步下预测从噪声动作到真实动作的修正方向；模型内部用 ResNet 把双图像变成视觉 token，用动作 encoder 把 noisy action chunk 变成动作 token，再通过多层 MMDiT joint self-attention 让视觉和动作交互，同时用本体状态、任务条件和时间步通过 AdaLN 调制每个 block；当使用触觉时，触觉信号被按手部区域编码成 tactile tokens，通过 cross-attention 和低秩 plugin adapter 注入已有视觉策略；推理时模型从随机动作出发迭代去噪，生成未来一段动作，再由部署程序按 receding horizon 方式发送给机器人。
