# DeCO Attention 内部交互流程

本文只展开 `deco.py` 中 `MMAttention` 内部发生的事情。这里默认各模态已经在进入 attention 前被整理好了：

```text
img:     [B, I, dim]        I = 2 * image_len, 两张图像 token 拼接
act:     [B, A, dim]        A = chunk_size, 动作 token
cond/t:  [B, dim]           time + obs + task 的条件向量
tactile: [B, T, dim]        optional, T = 68
```

本文重点回答：

- `condition` 在 attention 里怎么用
- image/action token 怎么生成 q/k/v
- image/action 怎么互相交互
- tactile 怎么参与 cross-attention
- adaLN / LayerNorm / QKNorm 分别在哪用
- plugin 模式下 `PI_Adapter` 怎么插入

## 1. 一个 MMAttention block 的总览

`MMAttention.forward(img, act, t, image_rotary_emb, tactile=None)` 一层内部可以概括为：

```text
输入:
  img  [B, I, dim]
  act  [B, A, dim]
  t    [B, dim]

image branch:
  t -> img_bais -> image adaLN 参数
  img -> LayerNorm -> adaLN 调制 -> img_qkv -> img_q/img_k/img_v
  img_q/img_k -> QKNorm -> RoPE

action branch:
  t -> act_bais -> action adaLN 参数
  act -> LayerNorm -> adaLN 调制 -> act_qkv -> act_q/act_k/act_v
  act_q/act_k -> QKNorm

joint attention:
  q = concat(img_q, act_q)    [B, heads, I + A, head_dim]
  k = concat(img_k, act_k)    [B, heads, I + A, head_dim]
  v = concat(img_v, act_v)    [B, heads, I + A, head_dim]
  attn = attention(q, k, v)   [B, heads, I + A, head_dim]

optional tactile:
  tactile -> tactile_k/tactile_v
  cross_attn = attention(q, tactile_k, tactile_v)
  attn = attn + cross_attn

拆分和回写:
  attn -> img_attn [B, I, dim], act_attn [B, A, dim]
  img = img + gated attention residual + gated MLP residual
  act = act + gated attention residual + gated MLP residual

输出:
  img [B, I, dim]
  act [B, A, dim]
```

每一层输出的 `img` 和 `act` 会作为下一层 `MMAttention` 的输入。DeCO 默认堆叠 `num_attn_blocks=6` 层。

## 2. condition 在 attention 中的角色

传入 `MMAttention` 的 `t` 已经不是原始 timestep，而是融合后的条件向量：

```text
t = time_emb
if obs_state:
    t = t + obs_emb
if use_task_condition:
    t = t + task_emb

t shape: [B, dim]
```

在 `MMAttention` 内部，`t` 不会变成 q/k/v，也不会作为 token 加到 attention 序列中。它的作用是生成每个分支的 adaptive LayerNorm 参数和残差门控。

同一个 `t` 会分别送入两个独立的 `adaLN`：

```text
image branch:  img_bais(t)
action branch: act_bais(t)
```

虽然输入是同一个 `t`，但 `img_bais` 和 `act_bais` 是两套不同参数，所以它们会生成不同的调制参数。

## 3. adaLN：condition 如何调制 token

`adaLN` 输入：

```text
t: [B, dim]
```

输出 6 个张量：

```text
scale1: [B, 1, dim]
shift1: [B, 1, dim]
gate1:  [B, 1, dim]
scale2: [B, 1, dim]
shift2: [B, 1, dim]
gate2:  [B, 1, dim]
```

其中中间的 `1` 用于广播到 token 序列维度：

```text
[B, 1, dim] 可以广播到 [B, I, dim] 或 [B, A, dim]
```

6 个参数分成两组：

| 参数 | 用在哪里 | 作用 |
|---|---|---|
| `scale1`, `shift1` | attention 前的 LayerNorm 后 | 调制进入 qkv 的 token |
| `gate1` | attention residual 回写时 | 控制 attention 分支写回强度 |
| `scale2`, `shift2` | MLP 前的 LayerNorm 后 | 调制进入 MLP 的 token |
| `gate2` | MLP residual 回写时 | 控制 MLP 分支写回强度 |

也就是说，condition 对 attention block 的影响有两处：

```text
1. 改变 q/k/v 的输入 token 表达
2. 控制 attention 输出和 MLP 输出写回原 token 的强弱
```

## 4. LayerNorm 和 adaLN 的具体组合

注意代码里的 `LayerNorm` 是：

```python
nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
```

这意味着 LayerNorm 本身不学习固定的 scale/bias。真正的 scale/shift 来自 `adaLN(t)`。

image attention 前：

```text
img_norm = img_norm1(img)                         # [B, I, dim]
img_norm = (1 + scale1_feat) * img_norm + shift1_feat
```

action attention 前：

```text
act_norm = act_norm1(act)                         # [B, A, dim]
act_norm = (1 + scale1_act) * act_norm + shift1_act
```

MLP 前也类似：

```text
img_mlp_input = (1 + scale2_feat) * img_norm2(img) + shift2_feat
act_mlp_input = (1 + scale2_act) * act_norm2(act) + shift2_act
```

所以这不是“先把 condition 拼进 token”，而是“condition 生成归一化后的缩放、平移和门控”。

## 5. image branch：图像 token 如何进入 attention

输入：

```text
img: [B, I, dim]
t:   [B, dim]
```

流程：

| 阶段 | 输入 shape | 输出 shape | 说明 |
|---|---:|---:|---|
| `img_bais(t)` | `[B,dim]` | 6 个 `[B,1,dim]` | 生成 image 分支 adaLN 参数 |
| `img_norm1(img)` | `[B,I,dim]` | `[B,I,dim]` | attention 前归一化 |
| adaLN 调制 | `[B,I,dim]` + `[B,1,dim]` | `[B,I,dim]` | 条件调制 token |
| `img_qkv` | `[B,I,dim]` | `[B,I,3*dim]` | 一次性生成 q/k/v |
| reshape 多头 | `[B,I,3*dim]` | q/k/v 各 `[B,heads,I,head_dim]` | 拆成多头 |
| `img_qknorm` | q/k/v | q/k `[B,heads,I,head_dim]` | q/k 做 RMSNorm |
| RoPE | q/k `[B,heads,I,head_dim]` | 同 shape | 只给图像 q/k 加二维位置 |

图像 token 来自两张图拼接，所以：

```text
I = 2 * image_len
feat_len = I / 2
```

RoPE 会分别作用在两段图像 token 上：

```text
img1 tokens: img_q[:, :, :feat_len, :]
img2 tokens: img_q[:, :, feat_len:, :]
```

RoPE 只作用于 `q/k`，不作用于 `v`。

## 6. action branch：动作 token 如何进入 attention

输入：

```text
act: [B, A, dim]
t:   [B, dim]
```

流程：

| 阶段 | 输入 shape | 输出 shape | 说明 |
|---|---:|---:|---|
| `act_bais(t)` | `[B,dim]` | 6 个 `[B,1,dim]` | 生成 action 分支 adaLN 参数 |
| `act_norm1(act)` | `[B,A,dim]` | `[B,A,dim]` | attention 前归一化 |
| adaLN 调制 | `[B,A,dim]` + `[B,1,dim]` | `[B,A,dim]` | 条件调制 token |
| `act_qkv` | `[B,A,dim]` | `[B,A,3*dim]` | 生成动作 q/k/v |
| reshape 多头 | `[B,A,3*dim]` | q/k/v 各 `[B,heads,A,head_dim]` | 拆成多头 |
| `act_qknorm` | q/k/v | q/k `[B,heads,A,head_dim]` | q/k 做 RMSNorm |

动作 token 不使用 RoPE。动作序列的位置已经在进入 `MMAttention` 前通过 `action_embedd` 加过。

## 7. QKNorm：为什么只处理 q/k

图像和动作分支都有自己的 `QKNorm`：

```text
img_q, img_k = img_qknorm(img_q, img_k, img_v)
act_q, act_k = act_qknorm(act_q, act_k, act_v)
```

输入/输出 shape：

```text
q: [B, heads, token_len, head_dim]
k: [B, heads, token_len, head_dim]
v: [B, heads, token_len, head_dim]

输出:
q_normed: [B, heads, token_len, head_dim]
k_normed: [B, heads, token_len, head_dim]
```

`QKNorm` 内部对 q 和 k 分别做 `RMSNorm(head_dim)`，然后把 dtype 对齐到 v。它不改变 token 数量，也不改变 head 数量，只稳定 attention score 的数值范围。

## 8. joint attention：视觉和动作如何互相交互

图像和动作各自生成 q/k/v 后，会在 token 维度拼接：

```text
q = cat([img_q, act_q], dim=2)
k = cat([img_k, act_k], dim=2)
v = cat([img_v, act_v], dim=2)
```

shape：

```text
img_q: [B, heads, I, head_dim]
act_q: [B, heads, A, head_dim]

q/k/v: [B, heads, I + A, head_dim]
```

然后执行：

```text
attn = scaled_dot_product_attention(q, k, v)
```

输出：

```text
attn: [B, heads, I + A, head_dim]
```

这里没有传入 attention mask，因此是双向、全连接的 joint attention：

| query token | 可以 attend 到 |
|---|---|
| 图像 token | 所有图像 token + 所有动作 token |
| 动作 token | 所有图像 token + 所有动作 token |

这一步是视觉和动作真正交互的位置。动作 token 可以读取视觉上下文，视觉 token 也会被动作 token 反向影响，形成联合表征。

## 9. tactile cross-attention：触觉如何进入 attention

触觉只在下面两个条件都满足时参与：

```text
self.use_tactile == True
tactile is not None
```

触觉输入：

```text
tactile: [B, T, dim]    T = 68
```

触觉只生成 key/value，不生成 query：

```text
tactile_k = tactile_key(tactile)      # [B, T, dim]
tactile_v = tactile_value(tactile)    # [B, T, dim]
```

reshape 多头：

```text
tactile_k: [B, heads, T, head_dim]
tactile_v: [B, heads, T, head_dim]
```

cross-attention 使用的 query 仍然是视觉+动作拼起来的 `q`：

```text
cross_attn = attention(q, tactile_k, tactile_v)
```

shape：

```text
q:          [B, heads, I + A, head_dim]
tactile_k: [B, heads, T,     head_dim]
tactile_v: [B, heads, T,     head_dim]

cross_attn:[B, heads, I + A, head_dim]
```

然后和 joint attention 的结果相加：

```text
attn = attn + cross_attn
```

所以 tactile 的融合方式是：

```text
视觉+动作 token 作为 query，主动去读取触觉 token；
读取到的触觉结果作为增量，加到视觉-动作 joint attention 输出上。
```

触觉 token 本身不会在这一层被更新；被更新的是 image/action token。

## 10. attention 输出如何拆回 image/action

attention 输出先从多头格式合回 token 格式：

```text
attn: [B, heads, I + A, head_dim]
  ->  [B, I + A, dim]
```

然后按原来的 token 长度拆开：

```text
img_attn = attn[:, :I, :]      # [B, I, dim]
act_attn = attn[:, I:, :]      # [B, A, dim]
```

之后 image 和 action 各自走自己的 residual update。

## 11. image token 的回写路径

image branch 有两个 residual：

```text
1. attention residual
2. MLP residual
```

attention residual：

```text
img = img + gate1_feat * img_proj(img_attn)
```

shape：

```text
img_attn:            [B, I, dim]
img_proj(img_attn):  [B, I, dim]
gate1_feat:          [B, 1, dim]
更新后 img:          [B, I, dim]
```

MLP residual：

```text
img_mlp_input = (1 + scale2_feat) * img_norm2(img) + shift2_feat
img = img + gate2_feat * img_mlp(img_mlp_input)
```

shape：

```text
img_mlp_input:       [B, I, dim]
img_mlp output:      [B, I, dim]
gate2_feat:          [B, 1, dim]
更新后 img:          [B, I, dim]
```

注意这里的 `img` 已经经过 attention residual 更新，再进入第二个 LayerNorm/MLP 分支。

## 12. action token 的回写路径

action branch 同样有两个 residual：

```text
1. attention residual
2. MLP residual
```

attention residual：

```text
act = act + gate1_act * act_proj(act_attn)
```

shape：

```text
act_attn:            [B, A, dim]
act_proj(act_attn):  [B, A, dim]
gate1_act:           [B, 1, dim]
更新后 act:          [B, A, dim]
```

MLP residual：

```text
act_mlp_input = (1 + scale2_act) * act_norm2(act) + shift2_act
act = act + gate2_act * act_mlp(act_mlp_input)
```

shape：

```text
act_mlp_input:       [B, A, dim]
act_mlp output:      [B, A, dim]
gate2_act:           [B, 1, dim]
更新后 act:          [B, A, dim]
```

最后这一层返回：

```text
return img, act
```

如果还有下一层 `MMAttention`，更新后的 `img/act` 继续进入下一层。

## 13. PI_Adapter：plugin 模式下的低秩增量路径

这里需要和 `adaLN` 区分开：

| 名称 | 作用 | 是否由 condition 生成 |
|---|---|---|
| `adaLN` | 根据 `t` 生成 scale/shift/gate，调制 LayerNorm 后的 token 和 residual gate | 是 |
| `PI_Adapter` | 给 qkv/proj/mlp 增加低秩可训练增量 | 否 |

`PI_Adapter` 只在下面条件下创建并使用：

```text
use_tactile=True and plugin=True
```

它的结构：

```text
x [..., dim]
  -> down Linear(dim, rank)
  -> up   Linear(rank, out_dim)
  -> adapter output [..., out_dim]
```

### 13.1 qkv 位置的 adapter

image qkv：

```text
img_qkv = img_qkv(img_norm)
if use_tactile and plugin:
    img_qkv += img_qkv_pi(img_norm)
```

action qkv：

```text
act_qkv = act_qkv(act_norm)
if use_tactile and plugin:
    act_qkv += act_qkv_pi(act_norm)
```

这里 adapter 会直接改变 q/k/v，因此会影响 attention 权重和 attention 读取到的 value。

### 13.2 attention 输出投影位置的 adapter

image attention residual：

```text
img = img + gate1_feat * img_proj(img_attn)
if use_tactile and plugin:
    img = img + gate1_feat * img_proj_pi(img_attn)
```

action attention residual：

```text
act = act + gate1_act * act_proj(act_attn)
if use_tactile and plugin:
    act = act + gate1_act * act_proj_pi(act_attn)
```

这里 adapter 是 attention 输出写回 token 时的额外增量，并且同样受 `gate1` 控制。

### 13.3 MLP 位置的 adapter

image MLP residual：

```text
img = img + gate2_feat * img_mlp(img_mlp_input)
if use_tactile and plugin:
    img = img + gate2_feat * img_mlp_pi(img_mlp_input)
```

action MLP residual：

```text
act = act + gate2_act * act_mlp(act_mlp_input)
if use_tactile and plugin:
    act = act + gate2_act * act_mlp_pi(act_mlp_input)
```

这里 adapter 是 MLP 分支的额外低秩增量，并且受 `gate2` 控制。

### 13.4 adapter 的总体位置

```text
attention 前:
  qkv = base_qkv(normed_token) + adapter_qkv(normed_token)

attention 后:
  token += gate1 * base_proj(attn_output)
  token += gate1 * adapter_proj(attn_output)

MLP 后:
  token += gate2 * base_mlp(normed_token)
  token += gate2 * adapter_mlp(normed_token)
```

adapter 不是 LayerNorm，也不生成 condition。它是挂在已有线性/MLP路径旁边的低秩可训练旁路。

## 14. 一个 block 的 shape 示例

以默认配置为例：

```text
dim = 512
heads = 8
head_dim = 64
img_size = 256 -> ResNet 特征约 8x8
单张图 image_len = 64
两张图 I = 128
chunk_size A = 32
tactile T = 68, optional
```

单层 `MMAttention` 中：

| 张量 | shape |
|---|---:|
| `img` | `[B,128,512]` |
| `act` | `[B,32,512]` |
| `t` | `[B,512]` |
| `scale/shift/gate` | `[B,1,512]` |
| `img_q/img_k/img_v` | `[B,8,128,64]` |
| `act_q/act_k/act_v` | `[B,8,32,64]` |
| joint `q/k/v` | `[B,8,160,64]` |
| joint `attn` | `[B,8,160,64]` |
| `tactile_k/tactile_v` | `[B,8,68,64]` |
| `cross_attn`, if tactile | `[B,8,160,64]` |
| merged `attn` | `[B,160,512]` |
| `img_attn` | `[B,128,512]` |
| `act_attn` | `[B,32,512]` |
| block 输出 `img` | `[B,128,512]` |
| block 输出 `act` | `[B,32,512]` |

## 15. 从输入到输出的简化公式

把一层 `MMAttention` 抽象成下面的计算：

```text
# image attention input
img1 = LN(img)
img1 = ada_scale_shift(img1, cond, branch="image", part=1)
img_qkv = QKV_img(img1) + optional_adapter_qkv_img(img1)

# action attention input
act1 = LN(act)
act1 = ada_scale_shift(act1, cond, branch="action", part=1)
act_qkv = QKV_act(act1) + optional_adapter_qkv_act(act1)

# joint attention
joint_qkv = concat(image_qkv, action_qkv, token_dim)
joint_attn = attention(joint_q, joint_k, joint_v)

# optional tactile
if tactile:
    joint_attn += attention(joint_q, tactile_k, tactile_v)

# split
img_attn, act_attn = split(joint_attn)

# residual update
img = img + gate_img_1(cond) * Proj_img(img_attn)
img = img + gate_img_2(cond) * MLP_img(adaLN2(LN(img), cond))

act = act + gate_act_1(cond) * Proj_act(act_attn)
act = act + gate_act_2(cond) * MLP_act(adaLN2(LN(act), cond))
```

最终，动作预测只会使用堆叠 attention 后的 `act` token：

```text
atten_forward:
  for mma in self.mmattn:
      img, act = mma(img, act, t, image_rotary_emb, tactile)

  output = linear(act)   # [B, A, act_dim]
```

