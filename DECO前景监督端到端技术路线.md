# DECO 前景监督端到端技术路线

## 1. 背景与目标

当前流程倾向于将 GroundingDINO/SAM2 作为外部工程模块，先对图像做前景分割或背景处理，再将处理后的图像输入 DECO。这个方式虽然直观，但会带来两个问题：一是部署链路变长，推理阶段依赖额外视觉模型；二是策略模型本身没有真正学到“哪些视觉信息与动作相关，哪些只是背景干扰”。

本技术路线的目标是将 DINO-SAM2 从推理时 pipeline 中移出，改为训练阶段的离线 teacher。DINO-SAM2 离线处理训练数据，生成双相机的前景 mask；DECO 在训练时利用这些 mask 学习 foreground-aware visual representation。最终推理时，模型只输入原始双相机 RGB、机器人状态和可选触觉，不再运行 DINO/SAM2，也不做背景替换。

本阶段不做以下事情：

- 不做背景替换。
- 不把分割后的图像作为 DECO 的主输入。
- 不进一步细分零件、机械手、工具等实例级或类别级 mask。
- 不在推理阶段运行 GroundingDINO 或 SAM2。

## 2. 总体架构

整体流程分为两个阶段：离线 teacher 数据构建，以及端到端策略训练。

### 2.1 离线 teacher 数据构建

对训练集中的每一帧双相机图像执行：

```text
原始 RGB 图像
  -> GroundingDINO 根据文本提示检测动作相关区域
  -> SAM2 根据检测框生成前景 mask
  -> 合并所有动作相关区域为单张二值 foreground mask
  -> 保存为 teacher supervision
```

这里的 foreground mask 是单通道二值图，只表达“动作相关前景 / 非前景背景”，不区分具体实例类别。例如机械手、目标零件、被操作物体等都合并为同一个 foreground 区域。

双相机都需要生成 teacher：

```text
episode_xxxx/
  foreground_masks/
    camera_0/
      000000_color_0.png
      000001_color_0.png
    camera_1/
      000000_color_1.png
      000001_color_1.png
```

mask 文件应与原始图像同名、同分辨率，像素值为 `0/255`。后续 dataset 会对 RGB 和 mask 使用一致的几何变换，保证监督信号与模型输入对齐。

### 2.2 DECO 端到端训练

训练时的数据流为：

```text
原始双相机 RGB + robot obs + action
  -> DECO image encoder
  -> foreground predictor 预测前景概率图
  -> foreground gate 调制视觉 token
  -> DECO action diffusion decoder
  -> action noise/action prediction
```

teacher mask 不作为模型输入给动作分支，而是只用于监督 DECO 内部的 foreground predictor。这样可以避免模型在推理阶段依赖外部分割结果。

训练目标由原来的单一 action diffusion loss 扩展为：

```text
total_loss = action_loss + lambda_fg * foreground_loss
```

其中：

- `action_loss` 保持原 DECO 的动作扩散训练目标。
- `foreground_loss` 约束模型内部预测的前景概率图接近 DINO-SAM2 teacher mask。
- `lambda_fg` 控制前景监督强度，建议初始值为 `0.05` 到 `0.1`。

本阶段暂不加入 background consistency loss，因为当前需求明确不做背景替换。

## 3. 模型设计

### 3.1 视觉编码

原 DECO 中，双相机图像经过 ResNet34 backbone 和 `img_head` 得到视觉特征图，再展平成 image tokens。新的设计在 `img_head` 后增加轻量 foreground predictor：

```text
RGB
  -> ResNet34 backbone
  -> img_head
  -> visual feature map
  -> fg_head
  -> foreground logits/probability
```

`fg_head` 输出低分辨率前景概率图，其空间尺寸与视觉 feature map 一致。teacher mask 会被下采样到同样尺寸进行监督。

### 3.2 Foreground Gate

foreground predictor 的输出用于调制视觉 token。推荐第一版使用 soft gate，而不是硬删除背景：

```text
gated_feat = feat * (1 + alpha * sigmoid(fg_logit))
```

这样做的好处是：

- 保留背景中的少量上下文信息，避免因为 teacher mask 不完美导致关键信息被完全抹掉。
- 让模型逐渐学习前景重要性，而不是被强制依赖二值裁剪。
- 对 DINO-SAM2 偶发漏检更鲁棒。

如果后续发现背景干扰仍然明显，可以升级为 background token replacement：

```text
gated_feat = feat * fg_prob + bg_token * (1 - fg_prob)
```

第一阶段建议先使用 soft gate，风险更低。

### 3.3 推理行为

推理阶段模型行为保持简单：

```text
原始 camera_0 RGB + 原始 camera_1 RGB + obs
  -> DECO 内部预测 foreground attention
  -> 输出 action chunk
```

推理阶段不需要：

- teacher mask
- GroundingDINO
- SAM2
- replaced background image
- grounding JSON

foreground predictor 是 DECO 模型内部的一部分，会随策略模型一起保存和加载。

## 4. 代码改动位置

### 4.1 DINO-SAM2 离线数据脚本

需要在现有 DINO-SAM2 离线处理脚本中增加 foreground mask 落盘能力。当前脚本已经能生成检测 JSON、debug box 图和背景替换图；本路线只需要保留 teacher mask 输出，不需要背景替换输出。

建议输出：

```text
foreground_masks/camera_0/*.png
foreground_masks/camera_1/*.png
```

每帧 mask 由所有动作相关 SAM2 masks 合并得到：

```text
merged_mask = mask_object_1 OR mask_object_2 OR ... OR mask_object_n
```

如果某帧无检测结果，建议保存全 0 mask，并在 summary 中记录 no-detection 数量，便于后续排查。

### 4.2 Dataset

`IL_training_codebase-master/dataset.py` 需要扩展读取逻辑：

```text
colors/camera image
foreground_masks/camera mask
tactiles
data.pkl action/obs
```

返回 batch 时增加：

```python
fg_mask1, fg_mask2
```

推荐返回结构：

```python
img1, img2, fg_mask1, fg_mask2, tac1, tac2, obs, action, action_mask, task_idx
```

mask transform 需要与图像 resize/letterbox 对齐，但不能应用 ColorJitter、GaussianBlur、Normalize 等颜色变换。

### 4.3 DECO 模型

`IL_training_codebase-master/models/deco/deco.py` 需要增加：

- `use_foreground_gate` 配置开关。
- `fg_head` 前景预测头。
- `foreground_gate_alpha` 控制 gate 强度。
- `img_encoding()` 返回 foreground logits 或 aux 信息。
- `forward()` 训练时返回 action prediction、noise 和 foreground auxiliary outputs。

保持推理接口兼容：如果 `training=False`，仍然只返回 action。

### 4.4 训练 Loss

`IL_training_codebase-master/models/deco/train_one_epoch.py` 需要在原 action loss 上叠加 foreground loss：

```python
action_loss = mse(pred, noise - action)
foreground_loss = bce_or_dice(fg_logits, fg_masks)
loss = action_loss + foreground_loss_weight * foreground_loss
```

建议第一版使用 `BCEWithLogitsLoss`，必要时再加入 Dice loss 处理前景面积较小的问题。

训练日志应额外记录：

- `action_loss`
- `foreground_loss`
- `total_loss`

## 5. 配置项建议

在 DECO 配置中增加：

```yaml
model:
  use_foreground_gate: true
  foreground_gate_alpha: 1.0

data:
  use_foreground_masks: true
  foreground_mask_dir_name: foreground_masks

train:
  foreground_loss_weight: 0.1
```

如果当前训练框架暂时不方便新增 `train` 段，可以先把 `foreground_loss_weight` 放到 `model` 段，由 `modeling()` 和训练 loop 读取。

## 6. 需要提供的数据与信息

为了实施这条路线，需要准备以下内容：

1. 双相机原始训练数据：

```text
episode_xxxx/colors/*_color_0.jpg
episode_xxxx/colors/*_color_1.jpg
```

2. 双相机 foreground teacher mask：

```text
episode_xxxx/foreground_masks/camera_0/*_color_0.png
episode_xxxx/foreground_masks/camera_1/*_color_1.png
```

3. DINO 文本提示词：

```text
robot hand . small black assembly component . small white plastic assembly
```

实际使用时应根据任务确认哪些物体属于动作相关前景。所有匹配到的目标会合并为一张 foreground mask。

4. mask 缺失策略：

建议第一版采用：

```text
缺失 mask 或无检测 -> 保存全 0 mask，但训练日志记录 no-detection/missing-mask 数量
```

如果无检测比例过高，需要回到 DINO prompt、box threshold、ROI 配置中调整。

## 7. 验证方案

### 7.1 数据验证

训练前应抽样检查：

- camera 0 和 camera 1 都有对应 mask。
- mask 文件名与 RGB 文件名一一对应。
- mask 与 resize/letterbox 后图像空间对齐。
- foreground 区域覆盖机械手和被操作物体。
- no-detection 比例可接受。

### 7.2 模型验证

训练后应检查：

- 原始验证集 action error 不劣于 baseline DECO。
- foreground loss 能稳定下降。
- 可视化 foreground probability，确认注意力集中在动作相关区域。
- 推理阶段不依赖 DINO/SAM2，也不读取 mask。

### 7.3 对比实验

建议保留三组实验：

```text
Baseline: 原始 DECO，只用 RGB 训练
Ours-fg: DECO + foreground mask supervision
Ours-fg-no-gate: 只加 foreground auxiliary loss，不用 gate，用于验证 gate 是否真正贡献性能
```

如果 `Ours-fg` 在新背景或复杂背景下更稳定，同时原始验证集不下降，则说明该路线有效。

## 8. 实施顺序

推荐按以下顺序推进：

1. 修改离线 DINO-SAM2 脚本，保存双相机 foreground mask。
2. 写一个数据检查脚本，确认 RGB/mask 对齐和缺失比例。
3. 扩展 dataset，使 batch 返回 `fg_mask1/fg_mask2`。
4. 在 DECO 中加入 `fg_head` 和 soft foreground gate。
5. 在训练 loop 中加入 foreground loss 和日志。
6. 训练小规模 smoke test，确认 loss、shape、保存加载都正常。
7. 训练完整模型，与 baseline DECO 做对比。
8. 可视化 foreground probability，分析模型是否学到背景无关特征。

## 9. 预期收益

该路线的核心收益是让模型内部学习背景无关的视觉表示，而不是依赖外部工程化图像预处理。最终系统部署时仍然保持 DECO 原有的简洁推理形式，但训练阶段利用 DINO-SAM2 的强视觉先验提高泛化能力。

换句话说，DINO-SAM2 不再是运行时拐杖，而是训练时老师；DECO 不再被动接收处理后的图像，而是主动学习该看哪里。
