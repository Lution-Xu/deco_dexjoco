import argparse
import importlib
import json
import os
import random
import shutil
import time

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import scipy
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import yaml
from torch.amp import GradScaler, autocast
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

from dataset import letterbox
from dexjoco_dataset import DexJoCoLeRobotDataset
from dexjoco_stats import compute_dexjoco_stats


class LossHistory:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        os.makedirs(self.log_dir, exist_ok=True)

    def append_loss(self, loss, val_loss):
        self.losses.append(loss)
        self.val_loss.append(val_loss)
        with open(os.path.join(self.log_dir, "train_loss.txt"), "a") as f:
            f.write(f"{loss}\n")
        with open(os.path.join(self.log_dir, "val_loss.txt"), "a") as f:
            f.write(f"{val_loss}\n")
        self._plot(self.losses, "train_loss.png", "Train Loss")
        self._plot(self.val_loss, "val_loss.png", "Val Loss")

    def _plot(self, values, name, ylabel):
        plt.figure()
        plt.plot(range(len(values)), values, linewidth=2)
        try:
            if len(values) >= 5:
                window = 5 if len(values) < 25 else 15
                smooth = scipy.signal.savgol_filter(values, window, 3)
                plt.plot(range(len(values)), smooth, linestyle="--", linewidth=2)
        except Exception:
            pass
        plt.grid(True)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.savefig(os.path.join(self.log_dir, name))
        plt.close("all")


def ensure_stats(config):
    data_cfg = config["data"]
    stats_path = data_cfg.get("stats_path")
    if stats_path and os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = compute_dexjoco_stats(
            data_cfg["data_root"],
            data_cfg["task_group"],
            data_cfg["regime"],
        )
        if stats_path:
            os.makedirs(os.path.dirname(os.path.abspath(stats_path)), exist_ok=True)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Wrote DexJoCo stats to {stats_path}")

    data_cfg["observation_mean"] = stats["observation_mean"]
    data_cfg["observation_std"] = stats["observation_std"]
    data_cfg["observation_min"] = stats["observation_min"]
    data_cfg["observation_max"] = stats["observation_max"]
    data_cfg["action_mean"] = stats["action_mean"]
    data_cfg["action_std"] = stats["action_std"]
    data_cfg["action_min"] = stats["action_min"]
    data_cfg["action_max"] = stats["action_max"]
    return stats


def build_transform(config, train):
    img_size = config["img"]["img_size"]
    img_mean = config["img"]["img_mean"]
    img_std = config["img"]["img_std"]
    resize = transforms.Resize(img_size) if img_size[0] != img_size[1] else letterbox(img_size[0], fill=128)
    ops = [resize]
    if train:
        ops.extend([
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=(0.7, 1.3),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                )
            ], p=0.5),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=random.choice([3, 5, 7]), sigma=random.uniform(0.1, 2))
            ], p=0.5),
        ])
    ops.extend([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean=img_mean, std=img_std),
    ])
    return transforms.Compose(ops)


def train_one_epoch(model, loader, optimizer, epoch, opt, device, scaler):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch}/{opt.epochs}", mininterval=0.3)
    for img1, img2, tac1, tac2, obs, action, _mask, task_idx in pbar:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        obs = obs.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        tac1 = tac1.to(device, non_blocking=True)
        tac2 = tac2.to(device, non_blocking=True)
        task_idx = task_idx.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if scaler is None:
            out, noise = model(img1, img2, obs=obs, act=action, task_idx=task_idx, tac1=tac1, tac2=tac2, training=True)
            loss = F.mse_loss(out, noise - action)
            loss.backward()
            optimizer.step()
        else:
            with autocast(device_type="cuda", enabled=True, dtype=torch.float16):
                out, noise = model(img1, img2, obs=obs, act=action, task_idx=task_idx, tac1=tac1, tac2=tac2, training=True)
                loss = F.mse_loss(out, noise - action)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        pbar.set_postfix(loss=total_loss / (pbar.n + 1), lr=optimizer.param_groups[0]["lr"])
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def validate(model, loader, opt, device, action_dim):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Validation", mininterval=0.3)
    for img1, img2, tac1, tac2, obs, action, mask, task_idx in pbar:
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        obs = obs.to(device, non_blocking=True)
        action = action.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        tac1 = tac1.to(device, non_blocking=True)
        tac2 = tac2.to(device, non_blocking=True)
        task_idx = task_idx.to(device, non_blocking=True)

        pred = model(img1, img2, obs=obs, act=action, task_idx=task_idx, tac1=tac1, tac2=tac2, training=False)
        mask = mask.unsqueeze(-1).repeat(1, 1, action_dim)
        loss = (mask * torch.abs(pred - action)).sum() / mask.sum().clamp_min(1)
        total_loss += loss.item()
        pbar.set_postfix(val_loss=total_loss / (pbar.n + 1))
    return total_loss / max(len(loader), 1)


def main(opt):
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.device_id
    with open(opt.config, "r") as f:
        config = yaml.safe_load(f)
    if opt.data_root is not None:
        config["data"]["data_root"] = opt.data_root
    if opt.task_group is not None:
        config["data"]["task_group"] = opt.task_group
    if opt.regime is not None:
        config["data"]["regime"] = opt.regime

    stats = ensure_stats(config)
    os.makedirs(opt.logs, exist_ok=True)
    os.makedirs(os.path.join(opt.logs, "loss"), exist_ok=True)
    with open(os.path.join(opt.logs, "resolved_config.yaml"), "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    with open(os.path.join(opt.logs, "dexjoco_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(opt.logs, "task_mapping.json"), "w") as f:
        json.dump({task: idx for idx, task in enumerate(stats["tasks"])}, f, indent=2)
    shutil.copy2(opt.config, os.path.join(opt.logs, os.path.basename(opt.config)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = device.type == "cuda"
    data_cfg = config["data"]
    dataset_kwargs = {
        key: value
        for key, value in data_cfg.items()
        if key not in {"data_root", "task_group", "regime"}
    }

    train_dataset = DexJoCoLeRobotDataset(
        data_root=data_cfg["data_root"],
        task_group=data_cfg["task_group"],
        regime=data_cfg["regime"],
        train=True,
        transform=build_transform(config, train=True),
        **dataset_kwargs,
    )
    val_dataset = DexJoCoLeRobotDataset(
        data_root=data_cfg["data_root"],
        task_group=data_cfg["task_group"],
        regime=data_cfg["regime"],
        train=False,
        transform=build_transform(config, train=False),
        **dataset_kwargs,
    )
    print(f"Train samples: {len(train_dataset)}; val samples: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    module = importlib.import_module(f"models.{config['model_name']}")
    model = module.modeling(**config["model"]).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model_without_dp = model.module if isinstance(model, torch.nn.DataParallel) else model

    optimizer = torch.optim.AdamW(
        (p for p in model_without_dp.parameters() if p.requires_grad),
        lr=opt.lr,
        betas=(0.95, 0.999),
        weight_decay=1e-6,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=opt.lr_f)
    scaler = GradScaler("cuda") if opt.amp and device.type == "cuda" else None
    loss_history = LossHistory(os.path.join(opt.logs, "loss"))

    start_epoch = 1
    best_loss = float("inf")
    if opt.resume:
        checkpoint = torch.load(os.path.join(opt.logs, "last_weights.pth"), map_location="cpu")
        model_without_dp.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["lr_scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])

    action_dim = config["model"]["action_dim"]
    for epoch in range(start_epoch, opt.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, epoch, opt, device, scaler)
        val_loss = validate(model, val_loader, opt, device, action_dim) if epoch % opt.val_per_epoch == 0 else best_loss
        scheduler.step()
        loss_history.append_loss(train_loss, val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model_without_dp.state_dict(), os.path.join(opt.logs, "best.pth"))
            print(f"Saved best.pth with val_loss={val_loss:.6f}")

        save_file = {
            "model": model_without_dp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_loss": best_loss,
        }
        if scaler is not None:
            save_file["scaler"] = scaler.state_dict()
        torch.save(save_file, os.path.join(opt.logs, "last_weights.pth"))
        if epoch % opt.save_period == 0:
            torch.save(model_without_dp.state_dict(), os.path.join(opt.logs, f"epoch_{epoch}.pth"))
        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, best={best_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config/deco_dexjoco_single_rand_obj.yaml")
    parser.add_argument("--data-root", dest="data_root", default=None)
    parser.add_argument("--task-group", dest="task_group", choices=["single", "dual"], default=None)
    parser.add_argument("--regime", choices=["rand_obj", "rand_full"], default=None)
    parser.add_argument("--device_id", default="0")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=64)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--val_per_epoch", type=int, default=1)
    parser.add_argument("--save_period", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_f", type=float, default=5e-6)
    parser.add_argument("--logs", default=f"./logs/dexjoco_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    main(parser.parse_args())
