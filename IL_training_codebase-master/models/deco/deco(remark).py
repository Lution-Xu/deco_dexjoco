"""
DeCO model with teaching comments.

这份文件保留了原始模型的计算逻辑，只额外加入较详细的中文注释。
阅读时可以把它当成一张“数据流地图”：

1. 两张图像 img1/img2 先经过 ResNet34，变成一串视觉 token。
2. 当前机器人状态 obs、任务编号 task_idx、触觉 tac1/tac2 是可选条件。
3. 动作序列 act 在训练时会被加噪声，模型学习预测“从噪声动作走回真实动作”的方向。
4. 推理时从纯随机动作开始，反复调用模型去噪，最后得到一段动作 chunk。

常见张量记号：
- B: batch size，一次送进模型的样本数量。
- C: channel，图像通道数，RGB 图通常是 3。
- H/W: 图像高和宽。
- dim: 模型内部隐藏维度，默认 512。
- heads: 多头注意力的头数，默认 8。
- head_dim: 每个注意力头的维度，等于 dim // heads。
- chunk: 一次预测的动作步数，也叫 chunk_size。
- act_dim: 每一步动作的维度，例如 28。
"""

import math
import torch
import einops
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import resnet34
from models.deco.denoise_schedular import get_schedule
from models.deco.rope import apply_rotary_emb, RotaryPosEmbed


class DECO(nn.Module):
    """
    DeCO 主模型。

    nn.Module 是 PyTorch 里所有神经网络模块的基类。
    只要继承 nn.Module，并在 __init__ 里定义层、在 forward 里写前向计算，
    PyTorch 就能自动追踪参数、计算梯度、保存/加载权重。

    这个模型的核心结构是：
    - 图像编码器：ResNet34 + Conv2d，把图片变成视觉 token。
    - 条件编码器：把 obs/task/tactile/time 编成 dim 维向量。
    - 动作编码器：把 act_dim 维动作变成 dim 维 token。
    - 多个 MMAttention：让图像 token 和动作 token 共同做注意力交互。
    - 输出头 linear：把 dim 维动作 token 还原成 act_dim 维动作预测。
    """

    def __init__(
        self,
        act_dim,
        chunk_size,
        obs_state=True,
        use_tactile=False,
        plugin=False,
        plugin_rank=32,
        use_task_condition=False,
        num_tasks=10,
        inf_step=10,
        img_pretrain=False,
        num_attn_blocks=6,
        heads=8,
        dim=512,
        rope_axes_dim=[256, 256],
        freeze_backbone=True,
    ):
        """
        构造函数：定义模型会用到的所有层和可学习参数。

        注意：这里不会真正跑数据，只是在“搭积木”。
        真正的数据流发生在 forward()。

        Args:
            act_dim: 每一步动作的维度，例如机器人关节/末端位姿等拼成 28 维。
            chunk_size: 一次预测多少个未来动作，训练输入 act 形状是 [B, chunk_size, act_dim]。
            obs_state: 是否使用机器人当前状态 obs 作为条件。
            use_tactile: 是否使用左右手触觉数据。
            plugin: 是否启用 PI_Adapter，常用于只微调小适配器参数。
            plugin_rank: PI_Adapter 的低秩瓶颈维度。
            use_task_condition: 是否用任务编号 task_idx 做条件。
            num_tasks: 任务总数，用于任务 embedding。
            inf_step: 推理时去噪迭代步数。
            img_pretrain: 当前文件中没有使用，保留是为了兼容配置。
            num_attn_blocks: MMAttention 堆叠层数。
            heads: 多头注意力头数。
            dim: 模型内部隐藏维度。
            rope_axes_dim: 旋转位置编码 RoPE 的参考网格大小。
            freeze_backbone: 当前构造函数中没有自动调用 freeze()，保留是为了兼容配置。
        """
        super().__init__()

        # 每个注意力头处理 dim 的一部分。
        # 例如 dim=512, heads=8，则每个头 head_dim=64。
        head_dim = dim // heads
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_state = obs_state
        self.use_tactile = use_tactile
        self.use_task_condition = use_task_condition
        self.inference_step = inf_step

        # RotaryPosEmbed 负责给图像 token 提供二维位置编码。
        # 它不是简单相加的位置向量，而是在注意力的 q/k 上做旋转变换。
        self.rope = RotaryPosEmbed(head_dim, rope_axes_dim)

        # resnet34(weights=...) 加载 ImageNet 预训练 ResNet34。
        # list(resnet.children())[:-2] 去掉最后的全局池化和分类层，只保留卷积特征提取部分。
        # 输入 [B, 3, H, W]，输出大致 [B, 512, H/32, W/32]。
        resnet = resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        self.img_encoder = nn.Sequential(*(list(resnet.children())[:-2]))

        # ResNet 输出通道固定是 512。这里用 3x3 卷积把通道数映射到模型隐藏维 dim。
        # Conv2d 的输入/输出仍然是图像网格形式：[B, channel, h, w]。
        self.img_head = nn.Conv2d(512, dim, kernel_size=3, padding=1)

        # 两张图像 img1/img2 会被拼成一串 token。
        # 这个 embedding 用来告诉模型“这个 token 来自第 0 张还是第 1 张图”。
        self.pos_idx_embedd = nn.Embedding(2, dim)

        if self.obs_state:
            # obs_encoder 把原始状态向量 [B, act_dim] 映射到 [B, dim]。
            # nn.Linear 是全连接层；Mish 是一种非线性激活函数。
            self.obs_encoder = nn.Sequential(
                nn.Linear(act_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )

        if self.use_tactile:
            # 触觉原始数据每只手长度 1062。
            # init_tac_regions() 建立 17 个触觉区域的切片索引，forward 里会按区域求平均。
            self.init_tac_regions()

            # gated 是一个门控层。
            # 输入 tactile 是 68 维：左手17均值 + 右手17均值 + 学习得到34维，共 68。
            # sigmoid(gated(tactile)) 生成 0~1 的权重，逐元素控制哪些触觉特征更重要。
            self.gated = nn.Linear(68, 68, bias=False)

            # 把 68 个触觉“位置/区域”编码成 68 个 dim 维 token。
            # timeEmb 在这里不是表示时间，而是借用正弦/余弦编码，把标量触觉值扩展成高维表示。
            self.pos_tac_embedd = nn.Sequential(
                timeEmb(dim),
                nn.Linear(dim, dim * 4),
                nn.Mish(),
                nn.Linear(dim * 4, dim),
            )

            # tactile_encoder 直接看完整左右手触觉原始向量，学习额外的 34 维区域特征。
            # 这和上面的“按区域平均”互补：一个是人工聚合，一个是神经网络学习聚合。
            self.tactile_encoder = nn.Sequential(
                nn.Linear(1062 * 2, 512),
                nn.Mish(),
                nn.Linear(512, 34),
            )

        if self.use_task_condition:
            # nn.Embedding 可以理解成“查表”。
            # task_idx 是整数任务编号 [B]，输出对应任务的可学习向量 [B, dim]。
            self.task_encoder = nn.Embedding(num_tasks, dim)

        # 时间 t 表示当前动作被加噪的程度。
        # t 越接近 0，越接近真实动作；t 越接近 1，越接近纯噪声。
        # time_embedd 把标量 t: [B] 编成 [B, dim]，供注意力块做条件调制。
        self.time_embedd = nn.Sequential(
            timeEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # action_embedd 是可学习的位置编码，形状 [1, chunk_size, dim]。
        # 这里的 1 会在 batch 维自动广播，让每个样本共享同一套“第几步动作”的位置向量。
        self.action_embedd = nn.Parameter(torch.zeros(1, chunk_size, dim))

        # action_encoder 把原始动作 [B, chunk, act_dim] 映射成动作 token [B, chunk, dim]。
        self.action_encoder = nn.Sequential(
            nn.Linear(act_dim, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

        # 堆叠多个联合注意力块。
        # 每个 MMAttention 都会同时更新图像 token 和动作 token。
        self.mmattn = nn.ModuleList(
            [MMAttention(heads, dim, use_tactile, plugin, plugin_rank) for _ in range(num_attn_blocks)]
        )

        # 最后的动作预测头：把每个动作 token 从 dim 维投影回 act_dim 维。
        self.linear = nn.Linear(dim, act_dim)

        # 原始代码逻辑：如果不是 plugin 模式，就初始化线性层权重。
        # plugin 模式通常会加载预训练模型并只训练 adapter，所以这里不重新初始化。
        if not plugin:
            self.initialize_weights()

    def forward(
        self,
        img1,
        img2,
        obs=None,
        act=None,
        task_idx=None,
        tac1=None,
        tac2=None,
        action_mask=None,
        training=True,
    ):
        """
        前向传播：定义输入数据如何流过模型。

        PyTorch 中调用 model(...) 时，实际执行的就是 forward(...)。
        这个函数分成训练和推理两条路径：
        - training=True: 给真实动作 act 加噪声，模型预测去噪方向，返回预测和真实噪声。
        - training=False: 从随机噪声动作开始，多步迭代去噪，返回最终动作序列。

        Args:
            img1: 第一张图像，形状 [B, C, H, W]。
            img2: 第二张图像，形状 [B, C, H, W]。
            obs: 当前机器人状态，形状通常 [B, act_dim]。
            act: 训练时的真实动作序列，形状 [B, chunk, act_dim]。
            task_idx: 任务编号，形状 [B]，每个元素是整数。
            tac1: 左手触觉，形状 [B, 1062]。
            tac2: 右手触觉，形状 [B, 1062]。
            action_mask: 当前代码没有使用，保留是为了兼容接口。
            training: 是否走训练路径。
        """
        # 1. 图像编码。
        # feat: [B, 2*img_seq_len, dim]，两张图像的 token 拼在一起。
        # image_rotary_emb: 给每张图像 token 使用的 RoPE cos/sin。
        feat, image_rotary_emb = self.img_encoding(img1, img2)

        if self.use_tactile:
            # 2. 触觉编码。
            # self.tactile_data_index 保存每个触觉区域的起止下标。
            # 对每个区域求 mean，可以把原始高维触觉压成 17 个区域强度。
            tac1_avg = torch.stack(
                [tac1[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()],
                dim=1,
            )
            tac2_avg = torch.stack(
                [tac2[:, s:e].mean(dim=1) for s, e in self.tactile_data_index.values()],
                dim=1,
            )

            # 同时让神经网络从完整触觉向量中学习 34 维补充特征。
            tactile_emb = self.tactile_encoder(torch.cat([tac1, tac2], dim=-1))

            # 拼接成 68 维触觉描述：
            # [左17区域均值, 右17区域均值, 网络学习的34维]。
            tactile = torch.cat([tac1_avg, tac2_avg, tactile_emb], dim=-1)

            # 门控：tactile * sigmoid(W tactile)。
            # 直观理解：模型自己决定哪些触觉通道要保留得多一些，哪些要压小。
            tactile = tactile * torch.sigmoid(self.gated(tactile))

            # 把 [B, 68] 变成 [B, 68, dim]。
            # 每个触觉通道现在变成一个 token，后面可被注意力读取。
            tactile = self.pos_tac_embedd(tactile)
        else:
            # 如果不使用触觉，后面的注意力块会跳过 tactile cross-attention。
            tactile = None

        if self.obs_state:
            # 把当前状态编码成 [B, dim]。
            # 之后会加到时间 embedding 上，作为去噪过程的条件。
            obs = self.obs_encoder(obs)

        if self.use_task_condition:
            # 把任务编号编码成 [B, dim]。
            task_emb = self.task_encoder(task_idx)

        if training:
            # 3A. 训练路径。
            # 为 batch 中每个样本随机采一个噪声程度 t。
            # torch.randn 先采标准正态，再 sigmoid 压到 (0, 1)。
            t = torch.sigmoid(torch.randn((act.shape[0],), device=act.device))

            # 按 t 给真实动作加噪：
            # t=0 时 act 不变；t=1 时 act 接近 noise。
            # 返回的 noise 是监督信号之一。
            act, noise = self.add_noise(act, t)

            # 把标量 t 编成 [B, dim] 条件向量。
            t = self.time_embedd(t)

            # 把 obs/task 条件加到 t 上。
            # 这里是简单相加，要求它们维度都为 [B, dim]。
            if self.obs_state:
                t = t + obs
            if self.use_task_condition:
                t = t + task_emb

            # 让图像 token 和加噪动作 token 通过联合注意力交互。
            # 输出 act 是模型预测的去噪方向，形状 [B, chunk, act_dim]。
            feat, act = self.atten_forward(
                feat,
                act,
                image_rotary_emb=image_rotary_emb,
                t=t,
                tactile=tactile,
            )

            # 训练脚本通常会用 act 和 noise/目标方向计算 loss。
            return act, noise

        else:
            # 3B. 推理路径。
            # 推理没有真实动作 act，所以先从随机噪声动作开始。
            sample = torch.randn(img1.shape[0], self.chunk_size, self.act_dim).to(img1.device)

            # get_schedule 返回从 1 递减到 0 的时间表。
            # 每一小步都让 sample 往“更像真实动作”的方向移动。
            t = get_schedule(self.inference_step, self.chunk_size)

            for t_curr, t_prev in zip(t[:-1], t[1:]):
                # 把当前时间标量 t_curr 复制成 batch 大小的向量 [B]。
                t_vec = torch.full((img1.shape[0],), t_curr, dtype=img1.dtype, device=img1.device)

                # 编码时间条件，再叠加 obs/task 条件。
                t_vec = self.time_embedd(t_vec)
                if self.obs_state:
                    t_vec = t_vec + obs
                if self.use_task_condition:
                    t_vec = t_vec + task_emb

                # 用当前 sample 作为“带噪动作”，预测去噪方向 denoise_act。
                _, denoise_act = self.atten_forward(
                    feat,
                    sample,
                    image_rotary_emb=image_rotary_emb,
                    t=t_vec,
                    tactile=tactile,
                )

                # 欧拉式更新：
                # t_prev < t_curr，所以 (t_prev - t_curr) 是负数。
                # 原始注释里 denoise_act 可理解为 noise - action，
                # 乘以负步长后，相当于把 sample 往 action 方向拉。
                sample = sample + (t_prev - t_curr) * denoise_act

            return sample

    def img_encoding(self, img1, img2):
        """
        把两张图像编码成 Transformer 可以处理的 token 序列。

        输入:
            img1/img2: [B, C, H, W]

        输出:
            feat: [B, 2*img_seq_len, dim]
                img_seq_len = ResNet 特征图的 h*w。
                两张图像 token 在序列维拼接，所以是 2 倍。
            image_rotary_emb:
                RoPE 需要的 cos/sin，用于注意力里的 q/k。
        """
        # 两张图必须形状一致，才能共享同一套图像编码流程。
        assert img1.shape == img2.shape, "img1 and img2 must have the same shape"

        # 在 batch 维拼接：[B,C,H,W] + [B,C,H,W] -> [2B,C,H,W]。
        # 这样只调用一次 ResNet，比对两张图分别调用更高效。
        img = torch.cat([img1, img2], dim=0)

        #  ResNet34 提取图像特征。
        feat = self.img_encoder(img)

        # 通道映射到 dim。
        feat = self.img_head(feat)

        # 再把 [2B, dim, h, w] 拆回两份 [B, dim, h, w]。
        feat1, feat2 = feat.chunk(2, dim=0)

        # 根据 ResNet34 特征图的高宽生成二维 RoPE。
        # 因为后续形成token需要拉平，所以位置关系会被消除，需要 RoPE 来提供位置信息。
        feat_h, feat_w = feat1.shape[-2:]
        image_rotary_emb = self.rope(feat_h, feat_w)

        # einops.rearrange 是更可读的 reshape/permute。
        # 这里把每个空间位置 h*w 展平成一个 token：
        # [B, dim, h, w] -> [B, h*w, dim]。
        feat1 = einops.rearrange(feat1, "b c h w -> b (h w) c")
        feat2 = einops.rearrange(feat2, "b c h w -> b (h w) c")

        # 为每个图像 token 加一个“图像来源 id”。
        # 前 feat1.shape[1] 个 id 是 0，后 feat2.shape[1] 个 id 是 1。
        img_id = torch.tensor([0] * feat1.shape[1] + [1] * feat2.shape[1]).to(img2.device)
        img_id = self.pos_idx_embedd(img_id).repeat(img1.shape[0], 1, 1)

        # 拼接两张图像的 token：[B, L, dim] + [B, L, dim] -> [B, 2L, dim]。
        feat = torch.cat([feat1, feat2], dim=1)

        # 加上图像 id embedding，让模型知道 token 来自哪一张图。
        feat = feat + img_id

        return feat, image_rotary_emb

    def atten_forward(self, img, act, image_rotary_emb, t, tactile=None):
        """
        运行所有 MMAttention 块。

        Args:
            img: 图像 token，[B, 2*img_len, dim]。
            act: 动作。如果刚进来通常是 [B, chunk, act_dim]。
            image_rotary_emb: 图像 RoPE 的 cos/sin。
            t: 条件向量，[B, dim]，包含时间/状态/任务信息。
            tactile: 可选触觉 token，[B, 68, dim]。

        Returns:
            img: 更新后的图像 token。
            act: 动作预测，[B, chunk, act_dim]。
        """
        # 把原始动作维度 act_dim 映射到模型隐藏维 dim。
        act = self.action_encoder(act)

        # 加动作位置 embedding，让模型知道这是第几个未来动作。
        act = act + self.action_embedd

        # 逐层更新 img/act token。
        for mma in self.mmattn:
            img, act = mma(img, act, t, image_rotary_emb, tactile)

        # 把动作 token 从 dim 映射回 act_dim。
        act = self.linear(act)
        return img, act

    def add_noise(self, act: torch.Tensor, t: torch.Tensor):
        """
        给动作序列加噪声。

        公式:
            noisy_act = (1 - t) * act + t * noise

        直观理解:
            t 越小，noisy_act 越像真实动作；
            t 越大，noisy_act 越像随机噪声。

        Args:
            act: 真实动作，[B, chunk, act_dim]。
            t: 每个样本的噪声程度，[B]。

        Returns:
            act: 加噪后的动作。
            noise: 本次采样的随机噪声。
        """
        # 生成和 act 形状完全一样的标准正态噪声。
        noise = torch.randn_like(act).to(act.device)

        # 把 [B] 变成 [B,1,1]，以便自动广播到 [B,chunk,act_dim]。
        t = t.view(act.shape[0], 1, 1)

        # 线性插值：真实动作和噪声按 t 混合。
        act = (1 - t) * act + t * noise
        return act, noise

    def freeze(self):
        """
        冻结图像骨干网络参数。

        requires_grad=False 表示反向传播时不为这些参数计算梯度，
        优化器也就不会更新它们。常用于保留预训练视觉特征。
        """
        for param in self.img_encoder.parameters():
            param.requires_grad = False

    def initialize_weights(self):
        """
        初始化模型权重。

        self.apply(_basic_init) 会递归遍历当前模块下的所有子模块。
        这里对所有 nn.Linear 用 Xavier uniform 初始化，并把 bias 置 0。
        最后一层 linear 被额外置 0，让模型刚开始训练时输出接近 0，
        这在扩散/流匹配类模型里是常见的稳定训练技巧。
        """

        def _basic_init(module):
            # 只处理全连接层，Conv2d/LayerNorm/Embedding 等保持 PyTorch 默认初始化。
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.linear.weight, 0)
        nn.init.constant_(self.linear.bias, 0)

    def init_tac_regions(self):
        """
        建立触觉数据的区域索引表。

        每只手原始触觉向量长度是 1062。
        这里列出 17 个区域，每个区域有自己的长度。
        forward() 中会用这些起止下标切片，然后对每个区域求平均。
        """
        _touch_regions = [
            ("fingerone_tip_touch", 3 * 3, (3, 3)),
            ("fingerone_top_touch", 12 * 8, (8, 12)),
            ("fingerone_palm_touch", 10 * 8, (8, 10)),
            ("fingertwo_tip_touch", 3 * 3, (3, 3)),
            ("fingertwo_top_touch", 12 * 8, (8, 12)),
            ("fingertwo_palm_touch", 10 * 8, (8, 10)),
            ("fingerthree_tip_touch", 3 * 3, (3, 3)),
            ("fingerthree_top_touch", 12 * 8, (8, 12)),
            ("fingerthree_palm_touch", 10 * 8, (8, 10)),
            ("fingerfour_tip_touch", 3 * 3, (3, 3)),
            ("fingerfour_top_touch", 12 * 8, (8, 12)),
            ("fingerfour_palm_touch", 10 * 8, (8, 10)),
            ("fingerfive_tip_touch", 3 * 3, (3, 3)),
            ("fingerfive_top_touch", 12 * 8, (8, 12)),
            ("fingerfive_middle_touch", 3 * 3, (3, 3)),
            ("fingerfive_palm_touch", 12 * 8, (8, 12)),
            ("palm_touch", 8 * 14, (8, 14)),
        ]

        # 当前区域的起点下标。
        _touch_start = 0

        # 保存成字典：region_name -> [start, end]。
        self.tactile_data_index = {}

        for region_name, region_size, _ in _touch_regions:
            self.tactile_data_index[region_name] = [_touch_start, _touch_start + region_size]
            _touch_start += region_size

        # 防止区域长度写错。所有区域长度加起来必须正好是 1062。
        assert _touch_start == 1062, "Total tactile data length should be 1062"


class PI_Adapter(nn.Module):
    """
    低秩适配器模块。

    它的结构是:
        dim -> rank -> out_dim

    如果 rank 远小于 dim，那么新增参数量会很小。
    常见用途是：冻结大模型主体，只训练这些 adapter 参数，以较低成本适配新模态或新任务。
    """

    def __init__(self, dim, out_dim, rank=32):
        super().__init__()

        # down 降维到 rank，up 再升维到 out_dim。
        self.down = nn.Linear(dim, rank)
        self.up = nn.Linear(rank, out_dim)

        # 初始化策略：
        # down 用小随机数；up 初始化为 0。
        # 因此 adapter 刚开始输出接近 0，不会突然破坏预训练模型行为。
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        """
        前向计算：先降维，再升维。

        x 的最后一维必须是 dim，其它前缀维度如 batch/sequence 会自动保留。
        """
        x = self.down(x)
        x = self.up(x)
        return x


class MMAttention(nn.Module):
    """
    Multi-Modal Attention block，多模态联合注意力块。

    这个模块同时处理两类 token：
    - img token: 来自两张图像。
    - act token: 来自动作序列。

    它做的事情大致是：
    1. 对 img/act 分别做 LayerNorm。
    2. 用时间条件 t 通过 adaLN 生成 scale/shift/gate，调制 img/act。
    3. 分别生成 q/k/v。
    4. 把 img 和 act 的 q/k/v 拼起来做一次 joint attention。
    5. 如果有 tactile，再额外做一次从 img+act 到 tactile 的 cross-attention。
    6. 拆回 img/act，各自走残差连接和 MLP。
    """

    def __init__(self, heads=8, dim=512, use_tactile=False, plugin=False, plugin_rank=32):
        super().__init__()
        head_dim = dim // heads
        self.head_dim = dim // heads
        self.head = heads
        self.use_tactile = use_tactile
        self.plugin = plugin

        # -------------------------
        # image branch
        # -------------------------
        # adaLN 根据条件向量 t 生成 LayerNorm 后的缩放/平移和残差门控。
        # 原代码变量名写作 bais，应是 bias 的拼写变体，这里不改名以保持兼容。
        self.img_bais = adaLN(dim)

        # LayerNorm 对最后一维 dim 做归一化。
        # elementwise_affine=False 表示 LayerNorm 自己不学习 scale/bias，
        # 因为 scale/shift 会由 adaLN 根据 t 动态生成。
        self.img_norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

        # 一次性生成 q/k/v，输出最后一维是 3*dim。
        self.img_qkv = nn.Linear(dim, dim * 3)

        # 对 q/k 做 RMSNorm，稳定注意力计算。
        self.img_qknorm = QKNorm(head_dim)

        # 注意力输出投影层。
        self.img_proj = nn.Linear(dim, dim)

        # 第二个 LayerNorm + MLP，类似 Transformer block 的 feed-forward network。
        self.img_norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim * 4, dim, bias=True),
        )

        # -------------------------
        # action branch
        # -------------------------
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

        # -------------------------
        # tactile branch
        # -------------------------
        if self.use_tactile:
            # cross-attention 里，img+act 作为 query，tactile 作为 key/value。
            self.tactile_key = nn.Linear(dim, dim)
            self.tactile_value = nn.Linear(dim, dim)

            if plugin:
                # plugin=True 时给若干线性层旁边加低秩 adapter。
                # 原主干输出 + adapter 输出，达到小参数微调效果。
                self.img_qkv_pi = PI_Adapter(dim, dim * 3, plugin_rank)
                self.img_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.img_mlp_pi = PI_Adapter(dim, dim, plugin_rank)

                self.act_qkv_pi = PI_Adapter(dim, dim * 3, plugin_rank)
                self.act_proj_pi = PI_Adapter(dim, dim, plugin_rank)
                self.act_mlp_pi = PI_Adapter(dim, dim, plugin_rank)

    def forward(self, img, act, t, image_rotary_emb, tactile=None):
        """
        运行一个多模态注意力块。

        Args:
            img: 图像 token，[B, 2*img_len, dim]。
            act: 动作 token，[B, chunk, dim]。
            t: 条件向量，[B, dim]。
            image_rotary_emb: 图像 RoPE 的 cos/sin。
            tactile: 可选触觉 token，[B, 68, dim]。
        """
        total_img_len = img.shape[1]

        # ===== image attention path =====
        # adaLN 输出 6 个 [B,1,dim] 张量。
        # scale/shift 用来调制归一化结果；gate 用来控制残差分支强度。
        scale1_feat, shift1_feat, gate1_feat, scale2_feat, shift2_feat, gate2_feat = self.img_bais(t)

        # 先对 img token 做 LayerNorm，再做条件调制。
        img_norm = self.img_norm1(img)
        img_norm = (1 + scale1_feat) * img_norm + shift1_feat

        # 生成图像分支的 q/k/v。
        img_qkv = self.img_qkv(img_norm)
        if self.use_tactile and self.plugin:
            # adapter 输出作为增量加到主线性层输出上。
            img_qkv += self.img_qkv_pi(img_norm)

        # 把 [B, L, 3*dim] 拆成 q/k/v，并拆多头：
        # 输出形状 [3, B, H, L, D]，第一维 K=3 分别对应 q/k/v。
        img_q, img_k, img_v = einops.rearrange(
            img_qkv,
            "B L (K H D) -> K B H L D",
            K=3,
            H=self.head,
            D=self.head_dim,
        )

        # 对 q/k 归一化，v 用来提供 dtype 参考。
        img_q, img_k = self.img_qknorm(img_q, img_k, img_v)

        # 两张图的 token 是拼起来的，所以总长度一半是 img1，一半是 img2。
        feat_len = int(total_img_len / 2)

        # 对两张图分别应用同一套二维 RoPE。
        # RoPE 只作用于 q/k，不作用于 v。
        img_q[:, :, :feat_len, :] = apply_rotary_emb(img_q[:, :, :feat_len, :], image_rotary_emb)
        img_k[:, :, :feat_len, :] = apply_rotary_emb(img_k[:, :, :feat_len, :], image_rotary_emb)
        img_q[:, :, feat_len:, :] = apply_rotary_emb(img_q[:, :, feat_len:, :], image_rotary_emb)
        img_k[:, :, feat_len:, :] = apply_rotary_emb(img_k[:, :, feat_len:, :], image_rotary_emb)

        # ===== action attention path =====
        scale1_act, shift1_act, gate1_act, scale2_act, shift2_act, gate2_act = self.act_bais(t)

        # 动作 token 已经在 DECO.atten_forward() 里加过可学习位置编码。
        act_norm = self.act_norm1(act)
        act_norm = (1 + scale1_act) * act_norm + shift1_act

        # 生成动作分支的 q/k/v。
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

        # ===== joint attention =====
        # 在序列维 L 上拼接 img 和 act。
        # 这样注意力可以同时看到图像 token 和动作 token：
        # - 动作可以关注图像，理解场景。
        # - 图像 token 也可以被动作 token 反向影响，形成联合表征。
        q = torch.cat([img_q, act_q], dim=2)
        k = torch.cat([img_k, act_k], dim=2)
        v = torch.cat([img_v, act_v], dim=2)

        # PyTorch 内置的 scaled dot-product attention。
        # 输入 [B,H,L,D]，输出同样是 [B,H,L,D]。
        attn = F.scaled_dot_product_attention(q, k, v)

        # ===== optional tactile cross-attention =====
        if self.use_tactile and tactile is not None:
            # 触觉 token 生成 key/value。
            tactile_k = self.tactile_key(tactile)
            tactile_v = self.tactile_value(tactile)

            # 拆成多头格式 [B,H,L,D]。
            tactile_k = einops.rearrange(tactile_k, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)
            tactile_v = einops.rearrange(tactile_v, "B L (H D) -> B H L D", H=self.head, D=self.head_dim)

            # query 仍然是 img+act，key/value 来自 tactile。
            # 直观理解：每个图像/动作 token 都可以从触觉 token 中读取信息。
            cross_attn = F.scaled_dot_product_attention(q, tactile_k, tactile_v)

            # 把视觉-动作联合注意力和触觉注意力相加融合。
            attn = attn + cross_attn

        # 把多头重新合并：
        # [B,H,L,D] -> [B,L,H*D]，而 H*D = dim。
        attn = einops.rearrange(attn, "B H L D -> B L (H D)")

        # 拆回图像 token 和动作 token。
        img_attn, act_attn = attn[:, :total_img_len, :], attn[:, total_img_len:, :]

        # ===== image residual update =====
        # Transformer 常见结构：x = x + attention_branch(x)。
        # 这里多了 gate1_feat，让条件 t 控制该残差分支强弱。
        img = img + gate1_feat * self.img_proj(img_attn)
        if self.use_tactile and self.plugin:
            img = img + gate1_feat * self.img_proj_pi(img_attn)

        # 第二个残差分支是 MLP。
        # 进入 MLP 前再次 LayerNorm，并用 adaLN 的 scale2/shift2 调制。
        img = img + gate2_feat * self.img_mlp((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)
        if self.use_tactile and self.plugin:
            img = img + gate2_feat * self.img_mlp_pi((1 + scale2_feat) * self.img_norm2(img) + shift2_feat)

        # ===== action residual update =====
        act = act + gate1_act * self.act_proj(act_attn)
        if self.use_tactile and self.plugin:
            act = act + gate1_act * self.act_proj_pi(act_attn)

        act = act + gate2_act * self.act_mlp((1 + scale2_act) * self.act_norm2(act) + shift2_act)
        if self.use_tactile and self.plugin:
            act = act + gate2_act * self.act_mlp_pi((1 + scale2_act) * self.act_norm2(act) + shift2_act)

        return img, act


class adaLN(nn.Module):
    """
    Adaptive LayerNorm 条件生成器。

    普通 LayerNorm 的 scale/bias 是固定可学习参数。
    adaLN 的思想是：根据条件向量 vec 动态生成 scale/shift/gate。

    本文件中 vec 通常是 time embedding + obs embedding + task embedding。
    所以注意力块会根据当前去噪时间、机器人状态、任务编号调整行为。
    """

    def __init__(self, dim: int):
        super().__init__()

        # 一个 dim 维条件向量要生成 6 组 dim 维参数。
        self.linear = nn.Linear(dim, dim * 6)

        # SiLU 是一种平滑激活函数，也叫 swish。
        self.silu = nn.SiLU()

    def forward(self, vec: Tensor):
        """
        Args:
            vec: 条件向量，[B, dim]。

        Returns:
            scale1, shift1, gate1, scale2, shift2, gate2:
            每个形状都是 [B, 1, dim]，中间的 1 方便和 [B, L, dim] token 广播相加/相乘。
        """
        out = self.linear(self.silu(vec))

        # out[:, None, :] 把 [B, 6*dim] 变成 [B, 1, 6*dim]。
        # chunk(6, dim=-1) 沿最后一维平均切成 6 份。
        scale1, shift1, gate1, scale2, shift2, gate2 = out[:, None, :].chunk(6, dim=-1)
        return scale1, shift1, gate1, scale2, shift2, gate2


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization。

    和 LayerNorm 类似，都是归一化最后一维。
    不同点是 RMSNorm 不减均值，只按均方根缩放：
        x / sqrt(mean(x^2) + eps)

    这里用于 q/k normalization，让注意力分数更稳定。
    """

    def __init__(self, dim: int):
        super().__init__()

        # 每个通道一个可学习缩放参数，初始为 1。
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        # 先记录 dtype，例如 float16/bfloat16。
        x_dtype = x.dtype

        # 用 float32 做归一化计算更稳定。
        x = x.float()

        # torch.rsqrt(y) 等于 1 / sqrt(y)。
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)

        # 归一化后转回原 dtype，再乘可学习 scale。
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """
    对 attention 的 query/key 分别做 RMSNorm。

    注意力分数来自 q @ k^T。
    如果 q/k 数值尺度不稳定，attention 也会不稳定。
    QKNorm 用于缓解这个问题。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        # 分别归一化 q 和 k。
        q = self.query_norm(q)
        k = self.key_norm(k)

        # q.to(v), k.to(v) 把 dtype 对齐到 v。
        # 混合精度训练时，这能减少 dtype 不一致的问题。
        return q.to(v), k.to(v)


class timeEmb(nn.Module):
    """
    一维正弦/余弦位置编码。

    经典 Transformer 论文 Attention Is All You Need 使用过这种编码。
    输入是一个标量 x，例如时间 t 或触觉标量；
    输出是 dim 维向量，前半部分是 sin，后半部分是 cos。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: 可以是 [B]，也可以是 [B, L]。

        Returns:
            如果 x 是 [B]，返回 [B, dim]。
            如果 x 是 [B, L]，返回 [B, L, dim]。
        """
        device = x.device

        # sin/cos 各占一半维度。
        half_dim = self.dim // 2

        # 生成不同频率，频率按指数间隔分布。
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # x.unsqueeze(-1) 在最后加一维，让 x 可以和频率向量相乘。
        # 例如 [B] -> [B,1]，乘 [half_dim] 后广播为 [B,half_dim]。
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)

        # 拼接 sin 和 cos，得到 dim 维编码。
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def modeling(
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
):
    """
    模型工厂函数。

    训练/评估脚本通常不直接写 DECO(...)，而是根据 yaml 配置调用 modeling(**config['model'])。
    这样不同模型可以暴露统一入口。

    这个函数还负责按配置加载预训练权重、adapter 权重，以及控制哪些参数需要训练。
    """
    # 下面这段被注释的 helper 原本用于只保存 requires_grad=True 的参数。
    # def get_trainable_state_dict(model):
    #     return {
    #         name: param.detach().cpu()
    #         for name, param in model.named_parameters()
    #         if param.requires_grad
    #     }

    # 先按配置创建一个 DeCO 模型实例。
    DeCO = DECO(
        act_dim=action_dim,
        chunk_size=chunk_size,
        obs_state=obs_state,
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
    )

    if pretrain_model_path:
        # pretrain_model_path 不为 False/空字符串时，加载已有权重。
        if use_tactile:
            # 使用触觉时，可能有两种情况：
            # 1. 加载完整视觉-触觉模型。
            # 2. 加载视觉预训练模型，再训练触觉 adapter。
            if plugin:
                if adapter_model_path:
                    # adapter 推理或 adapter 继续微调：
                    # adapter_model_path 中保存的是完整模型权重，所以 strict=True。
                    print("loading adapter weights from {} for adapter inference".format(adapter_model_path))
                    model_dict = torch.load(adapter_model_path, map_location="cpu")
                    DeCO.load_state_dict(model_dict, strict=True)
                else:
                    # adapter 微调：
                    # 加载视觉预训练权重，能匹配上的参数先加载。
                    # 新增的 tactile/plugin 参数没有预训练权重，会保持初始化状态并参与训练。
                    print("loading vision pretrained weights from {} for adapter finetuning".format(pretrain_model_path))
                    model_dict = torch.load(pretrain_model_path, map_location="cpu")
                    model_state = DeCO.state_dict()

                    # 只加载“名字存在且形状一致”的参数，避免新旧结构不同导致报错。
                    pretrain_dict = {
                        k: v for k, v in model_dict.items() if k in model_state.keys() and v.shape == model_state[k].shape
                    }
                    DeCO.load_state_dict(pretrain_dict, strict=False)

                    # 冻结预训练里已有的参数，只训练新增参数。
                    for name, param in DeCO.named_parameters():
                        if name in model_dict.keys():
                            param.requires_grad = False
                        else:
                            print("trainable params:", name)

            else:
                # 使用触觉但不使用 plugin：期望加载完整视觉-触觉模型权重。
                print("loading vision-tactile pretrained weights from {} for inference".format(pretrain_model_path))
                model_dict = torch.load(pretrain_model_path)
                DeCO.load_state_dict(model_dict, strict=True)

        else:
            # 不使用触觉：加载纯视觉模型权重。
            print("loading vision pretrained weights from {} for inference".format(pretrain_model_path))
            model_dict = torch.load(pretrain_model_path)
            DeCO.load_state_dict(model_dict, strict=True)

        # save_dict = get_trainable_state_dict(DeCO)
        # torch.save(save_dict, './test.pth')

    return DeCO


if __name__ == "__main__":
    """
    这个区域只在直接运行 python deco.py 时执行。
    被 import 时不会执行。

    它主要用于本地调试：读配置、创建模型、查看权重或 summary。
    """
    # model = torch.load('/root/yusun/ICML_codes/IL_training_codebase/models/mmrdt/1.pth')
    # total_params = 0
    # for k, v in model.items():
    #     print(k, v.shape)
    #     num = v.numel()
    #     total_params += num
    # print(f"\nTotal params: {total_params:,}")

    import yaml

    # 注意：这里使用相对路径 './deco.yaml'。
    # 如果你从别的目录运行，可能需要改成 config/deco.yaml 的实际路径。
    with open("./deco.yaml", "r") as f:
        config = yaml.safe_load(f)
    model = modeling(**config["model"])

    # 下面是一些手动测试示例。取消注释后可以检查模型输入/输出形状。
    # from torchinfo import summary
    # # mmdit = DECO(act_dim=28, chunk_size=16, num_attn_blocks=3, inf_step=1, use_task_condition=True)
    # img1 = torch.randn(1, 3, 224, 224)
    # img2 = torch.randn(1, 3, 224, 224)
    # obs = torch.randn(1, 28)
    # act = torch.randn(1, 32, 28)
    # task_idx = torch.randint(0, 8, (1,))
    # tac1 = torch.randn(1, 1062)
    # tac2 = torch.randn(1, 1062)

    # act_train, _ = mmdit(img1, img2, obs=obs, act=act, task_idx=task_idx, training=True)
    # act_pred = mmdit(img1, img2, obs=obs, task_idx=task_idx, training=False)
    # print(act_train.shape, act_pred.shape)

    # summary(model, input_data=(img1, img2, obs, act, task_idx, tac1, tac2), device='cpu')
