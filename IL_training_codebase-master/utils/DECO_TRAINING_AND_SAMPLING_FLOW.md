# DeCO 训练目标与推理采样数据流

本文结合 `models/deco/train_one_epoch.py`、`models/deco/deco.py` 和 `models/deco/denoise_schedular.py`，梳理 DeCO 的训练监督目标、动作加噪过程、loss 含义、验证路径，以及推理时从随机噪声采样动作的完整流程。

这里不重复展开图像/触觉/动作 token 如何进入 attention，也不展开 `MMAttention` 内部结构，只关注：

```text
训练时为什么预测 noise - action
推理时为什么可以从 random sample 迭代到 action
验证时为什么和训练 loss 不一样
```

## 1. 训练和推理的核心区别

DeCO 的 `forward(..., training=True/False)` 有两条完全不同的路径。

| 路径 | 是否有真实动作输入 | 初始动作状态 | 模型输出含义 | 最终用途 |
|---|---|---|---|---|
| 训练 `training=True` | 有，`action` | `action` 加噪后的 `noisy_action` | 预测 `noise - action` | 用 MSE 监督速度场 |
| 验证/推理 `training=False` | 不使用真实动作生成输入 | 随机高斯噪声 `sample` | 每一步预测当前 `sample` 的更新方向 | 多步迭代生成动作 |

一句话概括：

```text
训练时：给真实动作加噪，让模型学“从动作走向噪声”的方向。
推理时：从噪声开始，用负步长沿相反方向走回动作。
```

## 2. train_one_epoch.py 中训练 batch 的流向

训练循环中，每个 batch 从 dataloader 解包为：

```python
img1, img2, tac1, tac2, obs, action, mask, task_idx
```

主要 shape：

| 变量 | shape | 作用 |
|---|---:|---|
| `img1` | `[B, 3, H, W]` | 第一张图像条件 |
| `img2` | `[B, 3, H, W]` | 第二张图像条件 |
| `tac1` | `[B, 1062]` | 第一/左路触觉条件，启用触觉时使用 |
| `tac2` | `[B, 1062]` | 第二/右路触觉条件，启用触觉时使用 |
| `obs` | `[B, act_dim]` | 本体状态条件 |
| `action` | `[B, chunk, act_dim]` | 真实动作 chunk |
| `mask` | `[B, chunk]` | 有效动作 mask，训练 diffusion loss 中不用 |
| `task_idx` | `[B]` | 任务编号条件，启用任务条件时使用 |

训练时调用：

```python
out, noise = net(
    img1,
    img2,
    obs=obs,
    act=action,
    task_idx=task_idx,
    tac1=tac1,
    tac2=tac2,
    training=True,
)
loss = F.mse_loss(out, noise - action)
```

注意这里的 `mask` 没有参与训练 loss。代码里也明确注释：

```python
# NOTE: mask is not used in diffusion loss
```

因此训练阶段会对整个 `[B, chunk, act_dim]` 的预测做均方误差。

## 3. add_noise 与训练目标

训练路径中，`DECO.forward(training=True)` 先为每个样本随机采样一个噪声程度：

```python
t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))
```

shape：

```text
t: [B]
```

然后调用：

```python
act, noise = self.add_noise(act, t)
```

`add_noise` 内部做三件事：

```python
noise = torch.randn_like(act)
t = t.view(act.shape[0], 1, 1)
act = (1 - t) * act + t * noise
return act, noise
```

为了避免变量名混淆，可以把它写成：

```text
action:       真实动作                 [B, chunk, act_dim]
noise:        标准高斯噪声             [B, chunk, act_dim]
t:            噪声比例                 [B, 1, 1]
noisy_action: (1 - t) * action + t * noise
```

不同 `t` 的含义：

| `t` 值 | `noisy_action` 更接近 |
|---|---|
| 接近 0 | 真实动作 `action` |
| 接近 1 | 随机噪声 `noise` |
| 中间值 | 动作和噪声的线性插值 |

之后模型看到的动作输入不再是干净的 `action`，而是加噪后的 `noisy_action`：

```text
noisy_action + image/obs/task/tactile condition
  -> atten_forward
  -> out [B, chunk, act_dim]
```

## 4. loss = MSE(out, noise - action) 的含义

训练脚本中的监督目标是：

```python
loss = F.mse_loss(out, noise - action)
```

所以：

```text
target = noise - action
out    = model(noisy_action, t, conditions)
```

这说明 `out` 不是直接预测真实动作 `action`，也不是直接预测噪声 `noise`。它预测的是从真实动作点指向噪声点的方向：

```text
velocity = noise - action
```

可以把从真实动作到噪声的路径写成：

```text
x_t = (1 - t) * action + t * noise
```

对 `t` 求导：

```text
d x_t / d t = noise - action
```

因此模型学到的是这条线性路径上的速度场：

```text
给定当前 x_t 和条件，预测当前位置沿 t 增大方向的速度 noise - action。
```

训练目标完整写法：

```text
输入:
  x_t = noisy_action = (1 - t) * action + t * noise
  condition = image + obs + task + tactile + t

目标:
  target = noise - action

优化:
  minimize MSE(model(x_t, condition), target)
```

这也是为什么推理时可以用相反方向从噪声走回动作。

## 5. 推理采样：从随机 sample 到动作 chunk

推理路径对应：

```python
DECO.forward(..., training=False)
```

推理时没有真实动作输入，模型先随机初始化一个动作序列：

```python
sample = torch.randn(img1.shape[0], self.chunk_size, self.act_dim).to(img1.device)
```

shape：

```text
sample: [B, chunk, act_dim]
```

然后生成时间表：

```python
t = get_schedule(self.inference_step, self.chunk_size)
```

接着循环：

```python
for t_curr, t_prev in zip(t[:-1], t[1:]):
    t_vec = torch.full((B,), t_curr, dtype=img1.dtype, device=img1.device)
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
    )

    sample = sample + (t_prev - t_curr) * denoise_act
```

每一步的含义：

| 阶段 | shape | 作用 |
|---|---:|---|
| 当前 `sample` | `[B, chunk, act_dim]` | 当前待去噪动作 |
| `t_curr` | 标量 | 当前噪声时间 |
| `t_vec` | `[B, dim]` | 当前时间条件，叠加 obs/task |
| `denoise_act` | `[B, chunk, act_dim]` | 模型预测的速度方向 |
| 更新后 `sample` | `[B, chunk, act_dim]` | 更接近动作的数据点 |

推理完成后返回：

```text
sample: [B, chunk, act_dim]
```

这个 `sample` 就是模型生成的动作 chunk。

## 6. get_schedule 的作用

`get_schedule(num_steps, seq_len)` 来自 `denoise_schedular.py`。

基本逻辑：

```python
timesteps = torch.linspace(1, 0, num_steps + 1)
```

也就是先生成从 `1` 到 `0` 的时间序列。比如 `num_steps=5` 时，未 shift 前大致是：

```text
[1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
```

代码默认 `shift=True`，会根据 `seq_len` 做一次 `time_shift`，让 schedule 的分布发生变化：

```python
mu = get_lin_function(y1=base_shift, y2=max_shift)(seq_len)
timesteps = time_shift(mu, 1.0, timesteps)
```

但对理解主流程来说，最重要的是：

```text
schedule 是从高噪声时间走向低噪声时间。
```

循环中：

```python
for t_curr, t_prev in zip(t[:-1], t[1:]):
```

因此通常有：

```text
t_curr > t_prev
t_prev - t_curr < 0
```

更新公式：

```text
sample_next = sample + (t_prev - t_curr) * denoise_act
```

而模型学到的是：

```text
denoise_act ≈ noise - action
```

所以推理时乘上负步长：

```text
negative_step * (noise - action)
```

等价于把 `sample` 从噪声方向往动作方向推。

## 7. 验证阶段为什么和训练阶段不同

验证函数 `val(...)` 中调用的是：

```python
out = net(
    img1,
    img2,
    obs=obs,
    act=action,
    task_idx=task_idx,
    tac1=tac1,
    tac2=tac2,
    training=False,
)
```

虽然这里传了 `act=action`，但 `DECO.forward(training=False)` 内部并不会用这个真实动作来构造输入，而是重新初始化：

```python
sample = torch.randn(B, chunk, act_dim)
```

因此验证阶段评估的是完整采样能力：

```text
随机 sample -> 多步去噪 -> 预测动作 out
```

验证 loss：

```python
mask = mask.unsqueeze(-1).repeat(1, 1, act_dim)
loss = (mask * criterion(out, action)).sum() / mask.sum()
```

含义：

```text
只在 mask 标记为有效的时间步上，比较生成动作 out 和真实动作 action。
```

训练和验证的核心区别：

| 阶段 | 模型输入动作 | 模型输出 | loss 比较对象 | 是否使用 mask |
|---|---|---|---|---|
| 训练 | `noisy_action` | `out ≈ noise - action` | `out` vs `noise - action` | 否 |
| 验证 | 随机 `sample` 多步采样 | `out ≈ action` | `out` vs `action` | 是 |

训练阶段检查模型是否学会局部速度场；验证阶段检查用这个速度场采样出来的最终动作是否正确。

## 8. 训练/验证/推理三条路径对照表

| 项目 | 训练 `train(...)` | 验证 `val(...)` | 部署/推理 |
|---|---|---|---|
| `forward` 参数 | `training=True` | `training=False` | `training=False` |
| 是否使用真实 `action` 构造输入 | 是 | 否，传入但内部不使用 | 否 |
| 初始动作 | `noisy_action = (1-t)*action + t*noise` | `randn([B,chunk,act_dim])` | `randn([B,chunk,act_dim])` |
| 时间 `t` | 每个样本随机采样一次 | schedule 多步 | schedule 多步 |
| 模型单次输出 | `out ≈ noise - action` | 每步 `denoise_act`，最终 `out ≈ action` | 每步 `denoise_act`，最终动作 chunk |
| loss/评价 | `MSE(out, noise - action)` | masked action loss + MAE | 通常无 GT，直接执行/后处理 |
| `mask` | 不使用 | 使用 | 不使用 |

## 9. 默认配置下的 shape 示例

以 `config/deco.yaml` 常见配置为例：

```text
chunk_size = 32
act_dim = 28
dim = 512
inf_step = 5
```

训练时：

```text
action:       [B, 32, 28]
t:            [B]
noise:        [B, 32, 28]
noisy_action: [B, 32, 28]

t after time_embedd: [B, 512]
out:                 [B, 32, 28]
target:              noise - action = [B, 32, 28]
loss:                scalar
```

推理/验证时：

```text
sample init: [B, 32, 28]
schedule:    length = inf_step + 1 = 6

每一步:
  t_curr:      scalar
  t_vec:       [B, 512]
  denoise_act: [B, 32, 28]
  sample:      [B, 32, 28]

最终:
  out/sample: [B, 32, 28]
```

如果验证使用 `mask`：

```text
mask 原始:       [B, 32]
mask repeat 后:  [B, 32, 28]
criterion 输出:  [B, 32, 28]
masked loss:     scalar
```

## 10. 一句话总结 DeCO 的学习目标

DeCO 训练时不是直接学“图像到动作”的回归，而是学一个条件速度场：

```text
在给定视觉、状态、任务、触觉和当前噪声时间 t 的情况下，
预测当前 noisy_action 沿着 action -> noise 路径的速度 noise - action。
```

推理时从随机噪声动作开始，沿这个速度场的反方向积分：

```text
random noise action
  -> denoise step 1
  -> denoise step 2
  -> ...
  -> final action chunk
```

所以可以把整个训练和采样关系理解成：

```text
训练：action -> 加噪 -> 学会噪声方向
推理：noise  -> 反向走 -> 生成 action
```

