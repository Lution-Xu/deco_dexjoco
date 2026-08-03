import argparse
import importlib
import json
import os
import random
import time
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import yaml
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from dataset import letterbox
from dexjoco_dataset_unify import DexJoCoUnifiedDataset
from dexjoco_stats_unify import load_or_compute_stats

try:
    from torch.optim.lr_scheduler import _LRScheduler
except Exception:
    from torch.optim.lr_scheduler import LRScheduler as _LRScheduler

warnings.filterwarnings("ignore")

# 转化为true和false
def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}")


class WarmUpLR(_LRScheduler):
    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


class LossHistory():
    def __init__(self, log_dir, val_loss_flag=True):
        self.log_dir = log_dir
        self.val_loss_flag = val_loss_flag
        self.losses = []
        if self.val_loss_flag:
            self.val_loss = []
        os.makedirs(self.log_dir, exist_ok=True)

    def append_loss(self, loss, val_loss=None):
        self.losses.append(loss)
        has_val_loss = self.val_loss_flag and val_loss is not None
        if has_val_loss:
            self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "train_loss.txt"), 'a') as f:
            f.write(f"{loss}\n")
        if has_val_loss:
            with open(os.path.join(self.log_dir, "val_loss.txt"), 'a') as f:
                f.write(f"{val_loss}\n")
        self.plot_train_loss()
        if has_val_loss:
            self.plot_val_loss()

    def plot_train_loss(self):
        iters = range(len(self.losses))
        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth=2, label='train loss')
        try:
            if len(self.losses) >= 5:
                window = 5 if len(self.losses) < 25 else 15
                smooth = scipy.signal.savgol_filter(self.losses, window, 3)
                plt.plot(iters, smooth, 'green', linestyle='--', linewidth=2, label='smooth train loss')
        except Exception:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Train Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "train_loss.png"))
        plt.cla()
        plt.close("all")

    def plot_val_loss(self):
        iters = range(len(self.val_loss))
        plt.figure()
        plt.plot(iters, self.val_loss, 'blue', linewidth=2, label='val loss')
        try:
            if len(self.val_loss) >= 5:
                window = 5 if len(self.val_loss) < 25 else 15
                smooth = scipy.signal.savgol_filter(self.val_loss, window, 3)
                plt.plot(iters, smooth, 'purple', linestyle='--', linewidth=2, label='smooth val loss')
        except Exception:
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Val Loss')
        plt.legend()
        plt.savefig(os.path.join(self.log_dir, "val_loss.png"))
        plt.cla()
        plt.close("all")


def ensure_stats(config):
    data_cfg = config['data']
    stats_path = data_cfg.get('stats_path')
    regimes = data_cfg.get('regimes') or data_cfg['regime']
    stats_already_exists = bool(stats_path and os.path.exists(stats_path))
    stats = load_or_compute_stats(
        data_cfg['data_root'],
        data_cfg.get('task_group', 'unify'),
        regimes,
        stats_path,
        data_cfg.get('chunk_size', config['model'].get('chunk_size', 30)),
        task=data_cfg.get('task'),
        tasks=data_cfg.get('tasks'),
    )
    if stats_path and not stats_already_exists:
        os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Wrote unified DexJoCo stats to {stats_path}")

    data_cfg['observation_mean'] = stats['observation_mean']
    data_cfg['observation_std'] = stats['observation_std']
    data_cfg['observation_min'] = stats['observation_min']
    data_cfg['observation_max'] = stats['observation_max']
    data_cfg['action_mean'] = stats['action_mean']
    data_cfg['action_std'] = stats['action_std']
    data_cfg['action_min'] = stats['action_min']
    data_cfg['action_max'] = stats['action_max']


def dataset_regimes(data_cfg):
    return data_cfg.get('regimes') or [data_cfg['regime']]


def dataset_kwargs_for_regime(data_cfg, regime, mixed_regimes):
    excluded = {'data_root', 'task_group', 'regime', 'regimes', 'image_roots'}
    kwargs = {key: value for key, value in data_cfg.items() if key not in excluded}
    image_roots = data_cfg.get('image_roots')
    if image_roots:
        kwargs.pop('image_root', None)
        image_root = image_roots.get(regime)
        if image_root:
            kwargs['image_root'] = image_root
    elif mixed_regimes:
        kwargs.pop('image_root', None)
    return kwargs


def build_dataset(data_cfg, train, transform):
    regimes = dataset_regimes(data_cfg)
    mixed_regimes = len(regimes) > 1
    datasets = [
        DexJoCoUnifiedDataset(
            data_root=data_cfg['data_root'],
            task_group=data_cfg.get('task_group', 'unify'),
            regime=regime,
            train=train,
            transform=transform,
            **dataset_kwargs_for_regime(data_cfg, regime, mixed_regimes),
        )
        for regime in regimes
    ]
    if len(datasets) == 1:
        return datasets[0]
    return torch.utils.data.ConcatDataset(datasets)


def to_device(tensor, local_rank):
    if torch.cuda.is_available():
        return tensor.cuda(local_rank, non_blocking=True)
    return tensor


def unpack_unify_batch(batch, local_rank):
    (
        images,
        camera_mask,
        tac1,
        tac2,
        obs,
        obs_mask,
        action,
        action_mask,
        time_mask,
        task_idx,
    ) = batch
    return (
        to_device(images, local_rank),
        to_device(camera_mask, local_rank),
        to_device(tac1, local_rank),
        to_device(tac2, local_rank),
        to_device(obs, local_rank),
        to_device(obs_mask, local_rank),
        to_device(action, local_rank),
        to_device(action_mask, local_rank),
        to_device(time_mask, local_rank),
        to_device(task_idx, local_rank),
    )


def masked_mse_loss(out, target, action_mask):
    # Match the original DECO training behavior: repeated tail actions are
    # training targets rather than temporal padding. Only mask action
    # dimensions that do not belong to the current robot embodiment.
    loss_mask = action_mask[:, None, :].to(out.dtype).expand_as(out)
    loss = torch.square(out - target) * loss_mask
    return loss.sum() / loss_mask.sum().clamp_min(1.0)


def masked_l1_loss(out, target, time_mask, action_mask):
    loss_mask = time_mask[:, :, None].to(out.dtype) * action_mask[:, None, :].to(out.dtype)
    loss = torch.abs(out - target) * loss_mask
    return loss.sum() / loss_mask.sum().clamp_min(1.0)


def train(net, net_without_ddp, train_loader, optimizer, warmup_scheduler, epoch, opt, scaler, local_rank):
    if local_rank == 0:
        print("Start unified training")
        pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.train()
    total_loss = 0

    for batch_idx, batch in enumerate(train_loader):
        images, camera_mask, tac1, tac2, obs, obs_mask, action, action_mask, time_mask, task_idx = unpack_unify_batch(batch, local_rank)
        optimizer.zero_grad()
        if epoch <= opt.warm_up_epoch:
            warmup_scheduler.step()

        if not opt.amp or scaler is None:
            out, noise = net(
                images=images,
                camera_mask=camera_mask,
                obs=obs,
                obs_mask=obs_mask,
                act=action,
                task_idx=task_idx,
                tac1=tac1,
                tac2=tac2,
                action_mask=action_mask,
                training=True,
            )
            target = (noise - action) * action_mask[:, None, :].to(action.dtype)
            loss = masked_mse_loss(out, target, action_mask)
            loss.backward()
            optimizer.step()
        else:
            from torch.amp import autocast
            with autocast(device_type='cuda', enabled=True, dtype=torch.float16):
                out, noise = net(
                    images=images,
                    camera_mask=camera_mask,
                    obs=obs,
                    obs_mask=obs_mask,
                    act=action,
                    task_idx=task_idx,
                    tac1=tac1,
                    tac2=tac2,
                    action_mask=action_mask,
                    training=True,
                )
                target = (noise - action) * action_mask[:, None, :].to(action.dtype)
                loss = masked_mse_loss(out, target, action_mask)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        if local_rank == 0:
            pbar.set_postfix(**{
                'total_loss': total_loss / (batch_idx + 1),
                'lr': optimizer.state_dict()['param_groups'][0]['lr'],
            })
            pbar.update(1)
            with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
                f.writelines("Epoch:%d [%d|%d] loss:%f \n" % (epoch, batch_idx + 1, len(train_loader), loss.mean()))

    if dist.is_initialized():
        dist.barrier()
    epoch_loss = total_loss / len(train_loader)
    if epoch % opt.save_period == 0 and local_rank == 0:
        print('save model to logs')
        torch.save(net_without_ddp.state_dict(), os.path.join(opt.logs, 'epoch_%d_loss_%f.pth') % (epoch, total_loss))
    if local_rank == 0:
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nEpoch: %d, total loss: %f, epoch loss: %f' % (epoch, total_loss, epoch_loss))
    return epoch_loss


def val(net, test_loader, epoch, opt, act_dim, chunksize, local_rank):
    if local_rank == 0:
        print("Start unified validation")
        pbar = tqdm(total=len(test_loader), desc=f'Epoch {epoch}/{opt.epochs}', postfix=dict, mininterval=0.3)
    net.eval()
    total_loss = 0
    mae_sum = to_device(torch.zeros(chunksize, act_dim), local_rank)
    mae_count = to_device(torch.zeros(chunksize, act_dim), local_rank)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images, camera_mask, tac1, tac2, obs, obs_mask, action, action_mask, time_mask, task_idx = unpack_unify_batch(batch, local_rank)
            out = net(
                images=images,
                camera_mask=camera_mask,
                obs=obs,
                obs_mask=obs_mask,
                act=action,
                task_idx=task_idx,
                tac1=tac1,
                tac2=tac2,
                action_mask=action_mask,
                training=False,
            )
            loss = masked_l1_loss(out, action, time_mask, action_mask)
            total_loss += loss.item()

            loss_mask = time_mask[:, :, None].to(out.dtype) * action_mask[:, None, :].to(out.dtype)
            ae = (torch.abs(out - action) * loss_mask).sum(0)
            counts = loss_mask.sum(0)
            if dist.is_initialized():
                dist.all_reduce(ae)
                dist.all_reduce(counts)
            mae_sum += ae
            mae_count += counts
            if local_rank == 0:
                dim_mae = (mae_sum.sum(0) / mae_count.sum(0).clamp_min(1.0)).tolist()
                dim_mae = [round(x, 2) for x in dim_mae]
                pbar.set_postfix(**{'val_loss': total_loss / (batch_idx + 1), 'AE': dim_mae})
                pbar.update(1)

    if dist.is_initialized():
        dist.barrier()
    mae = mae_sum / mae_count.clamp_min(1.0)
    mae = torch.round(mae * 100) / 100
    epoch_loss = total_loss / len(test_loader)
    if local_rank == 0:
        print("\nVal epoch loss: %f" % epoch_loss)
        print('\nmasked mae: ', mae)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('\nVal epoch: %d, total loss: %f, epoch loss: %f \n MAE: %s \n\n' % (epoch, total_loss, epoch_loss, str(mae)))
    return epoch_loss


def main(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.device_id
    Cuda = True if torch.cuda.is_available() else False
    ngpus_per_node = torch.cuda.device_count()
    old_dir = opt.logs
    # 下面这个if是什么意思
    if opt.distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = int(os.environ["RANK"])
        dist.init_process_group(backend="nccl")
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) unified training...")
            print("GPU Device Count : ", ngpus_per_node)
    else:
        local_rank = 0

    if local_rank == 0:
        if os.path.exists(opt.logs):
            print('log dir exists, change to a new dir')
            opt.logs = opt.logs + str(random.randint(0, 1000))
        print('create log dir:', opt.logs)
        loss_dir = os.path.join(opt.logs, 'loss')
        os.makedirs(loss_dir, exist_ok=True)
        loss_history = LossHistory(loss_dir)
        with open(os.path.join(opt.logs, 'result.txt'), 'a+') as f:
            f.writelines('----------------unified training logs----------------\n %s \n \n' % str(opt))
    else:
        loss_history = None
    # 下面加载config的内容，这个config是一个字典吗？,是的，是字典
    config = yaml.safe_load(open(opt.config, 'r'))
    if opt.data is not None:
        config['data']['data_root'] = opt.data
    ensure_stats(config)

    act_dim = config['model']['action_dim']
    chunksize = config['model']['chunk_size']
    img_size = config['img']['img_size']
    img_mean, img_std = config['img']['img_mean'], config['img']['img_std']
    model_name = config['model_name']
    importmodule = importlib.import_module(f"models.{model_name}")
    # 判断图像长宽，如果不相等就使用transforms.Resize，如果相等就使用letterbox，这两个分别是什么作用，
    if img_size[0] != img_size[1]:
        print('use transform.Resize to ', img_size)
        resize_transform = transforms.Resize(img_size)
    else:
        print('use letterbox to ', img_size)
        resize_transform = letterbox(img_size[0], fill=128)
    # transforms 是 torchvision 的图像预处理工具，对图像进行一系列的变换操作，这里定义了训练集和测试集的图像预处理流程。
    # train_transform 包含了随机颜色抖动、随机高斯模糊、转换为图像、转换为张量并归一化等操作，而 test_transform 只包含了调整大小、转换为图像、转换为张量并归一化等操作。
    train_transform = transforms.Compose([
        resize_transform,
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.7, 1.3), contrast=(0.8, 1.2), saturation=(0.8, 1.2))], p=0.5),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=(random.choice([3, 5, 7])), sigma=random.uniform(0.1, 2))], p=0.5),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    test_transform = transforms.Compose([
        resize_transform,
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])

    data_cfg = config['data']
    #train和test是怎么进行区分的，在dataset.py中有train参数，train=True表示训练集，train=False表示测试集
    train_dataset = build_dataset(data_cfg, train=True, transform=train_transform)
    test_dataset = build_dataset(data_cfg, train=False, transform=test_transform)
    # 看不懂这块，if的语句这块
    if opt.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        batch_size = opt.batch_size // ngpus_per_node
        shuffle = False
    else:
        batch_size = opt.batch_size
        train_sampler, val_sampler = None, None
        shuffle = True
    # 这块是分别创建训练集和验证集的dataloader，batch_size是每个batch的大小，shuffle是是否打乱数据，num_workers是加载数据的线程数，pin_memory是是否将数据放到GPU上，drop_last是是否丢弃最后一个不完整的batch，sampler是采样器
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=val_sampler,
    )
    # 下面这些看不懂，为我解释
    net = importmodule.modeling(**config['model'])
    if Cuda:
        if opt.distributed:
            net = net.cuda(local_rank)
            for param in net.parameters():
                dist.broadcast(param.data, src=0)
            net = torch.nn.parallel.DistributedDataParallel(
                net,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
            net_without_ddp = net.module
            cudnn.benchmark = True
        else:
            if torch.cuda.device_count() > 1:
                net = torch.nn.DataParallel(net)
                cudnn.benchmark = True
                net = net.cuda()
                net_without_ddp = net.module
            else:
                cudnn.benchmark = True
                net = net.cuda()
                net_without_ddp = net
    else:
        print('Please try to use GPU, otherwise it will take long time for training')
        net_without_ddp = net
# 看不懂
    if opt.amp and Cuda:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    else:
        scaler = None
# 看不懂
    if not opt.adamw:
        if local_rank == 0:
            print('use sgd')
        optimizer = torch.optim.SGD(
            params=(p for p in net_without_ddp.parameters() if p.requires_grad),
            lr=opt.lr,
            momentum=0.843,
            weight_decay=0.00036,
        )
    else:
        if local_rank == 0:
            print('use adamw')
        optimizer = torch.optim.AdamW(
            params=(p for p in net_without_ddp.parameters() if p.requires_grad),
            lr=opt.lr,
            betas=(0.95, 0.999),
            weight_decay=1e-6,
        )
# 这个是分别调整学习率和预热的，为我解释这两个参数分别是什么意思
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr_f)
    warmup_scheduler = WarmUpLR(optimizer, len(train_loader) * opt.warm_up_epoch)
# 这个是判断是否从上次的权重继续训练，如果是就加载上次的权重和优化器状态，如果不是就从头开始训练，local_rank是什么
    if opt.resume:
        if local_rank == 0:
            print('resume from last weights')
        checkpoint = torch.load(os.path.join(old_dir, "last_weights.pth"), map_location='cpu')
        net_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        if opt.amp and scaler is not None:
            scaler.load_state_dict(checkpoint["scaler"])
    else:
        start_epoch = 1

    loss = 1000
    val_loss = 1000
    save_epoch = 0
    for epoch in range(start_epoch, opt.epochs + 1):
        if opt.distributed:
            train_sampler.set_epoch(epoch)
        train_loss = train(net, net_without_ddp, train_loader, optimizer, warmup_scheduler, epoch, opt, scaler, local_rank)
        should_validate = epoch % opt.val_per_epoch == 0
        if should_validate:
            val_loss = val(net, test_loader, epoch, opt, act_dim, chunksize, local_rank)
        if should_validate and val_loss < loss:
            loss = val_loss
            save_epoch = epoch
            if local_rank == 0:
                print('save best model to logs!')
                torch.save(net_without_ddp.state_dict(), os.path.join(opt.logs, 'best.pth'))
        if epoch > opt.warm_up_epoch:
            lr_scheduler.step()
        if local_rank == 0:
            loss_history.append_loss(train_loss, val_loss if should_validate else None)
            print('current best epoch:', save_epoch, '\n')
            print('lr:', optimizer.state_dict()['param_groups'][0]['lr'], '\n')
            save_file = {
                'model': net_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            if opt.amp and scaler is not None:
                save_file["scaler"] = scaler.state_dict()
            torch.save(save_file, os.path.join(opt.logs, "last_weights.pth"))

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/deco_dexjoco_unify_rand_obj.yaml', help='Path to unified DexJoCo YAML')
    parser.add_argument('--data', type=str, default=None, help='Path to DexJoCo dataset root')
    parser.add_argument('--adamw', default=True, type=str2bool, help='Use AdamW optimizer instead of SGD')
    parser.add_argument('--lr', type=float, default=2e-4, help='Initial learning rate')
    parser.add_argument('--lr_f', type=float, default=5e-6, help='Final learning rate')
    parser.add_argument('--batch-size', type=int, default=128*8, help='Total batch size across all GPUs')
    parser.add_argument('--warm_up_epoch', type=int, default=1, help='Number of warmup epochs')
    parser.add_argument('--epochs', type=int, default=200, help='Total epochs')
    parser.add_argument('--val_per_epoch', type=int, default=5, help='Validation interval')
    parser.add_argument('--logs', type=str, default='./logs/log_dexjoco_unify', help='Directory to save logs and models')
    parser.add_argument('--save_period', type=int, default=20, help='Checkpoint interval')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint')
    parser.add_argument('--local_rank', default=-1, type=int, help='Local rank for DDP')
    parser.add_argument('--distributed', default=True, type=str2bool, help='Enable DDP')
    parser.add_argument('--amp', default=True, type=str2bool, help='Enable AMP')
    parser.add_argument('--num-workers', type=int, default=16, help='DataLoader workers')
    parser.add_argument('--device_id', default='0, 1, 2, 3, 4, 5, 6, 7', type=str, help='CUDA device IDs')
# 加载超参数
    opt = parser.parse_args()
    main(opt)
    run_time_in_seconds = time.time() - start_time
    hours, remainder = divmod(run_time_in_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)} h {int(minutes)} m {int(seconds)} s")
