"""Executable checks for the minimal DECO reproduction.

Examples:
    python deco_from_scratch/checks.py shape-v0
    python deco_from_scratch/checks.py shape-vision
    python deco_from_scratch/checks.py gradient
    python deco_from_scratch/checks.py toy-overfit --steps 80
    python deco_from_scratch/checks.py dataset-smoke --data ./Deco-50/task1_merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from minimal_deco import MMAttention, MinimalDECO


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def shape_v0() -> None:
    device = choose_device()
    model = MinimalDECO(
        action_dim=28,
        chunk_size=32,
        dim=128,
        heads=4,
        num_attn_blocks=2,
        use_image_encoder=False,
        image_token_len=128,
    ).to(device)
    bsz = 2
    image_tokens = torch.randn(bsz, 128, 128, device=device)
    obs = torch.randn(bsz, 28, device=device)
    action = torch.randn(bsz, 32, 28, device=device)

    pred, noise = model(image_tokens=image_tokens, obs=obs, act=action, training=True)
    sample = model(image_tokens=image_tokens, obs=obs, training=False)

    assert pred.shape == (bsz, 32, 28), pred.shape
    assert noise.shape == (bsz, 32, 28), noise.shape
    assert sample.shape == (bsz, 32, 28), sample.shape
    print("shape-v0 ok")
    print(f"  pred:   {tuple(pred.shape)}")
    print(f"  noise:  {tuple(noise.shape)}")
    print(f"  sample: {tuple(sample.shape)}")


def shape_vision() -> None:
    device = choose_device()
    model = MinimalDECO(
        action_dim=28,
        chunk_size=32,
        dim=64,
        heads=4,
        num_attn_blocks=1,
        use_image_encoder=True,
        pretrained_backbone=False,
    ).to(device)
    model.eval()

    bsz = 2
    img1 = torch.randn(bsz, 3, 64, 64, device=device)
    img2 = torch.randn(bsz, 3, 64, 64, device=device)
    obs = torch.randn(bsz, 28, device=device)
    action = torch.randn(bsz, 32, 28, device=device)

    with torch.no_grad():
        tokens, rotary = model.encode_image(img1=img1, img2=img2, image_tokens=None)
        pred, noise = model(img1=img1, img2=img2, obs=obs, act=action, training=True)

    assert tokens.ndim == 3, tokens.shape
    assert tokens.shape[0] == bsz, tokens.shape
    assert tokens.shape[-1] == 64, tokens.shape
    assert rotary is not None
    assert pred.shape == (bsz, 32, 28), pred.shape
    assert noise.shape == (bsz, 32, 28), noise.shape
    print("shape-vision ok")
    print(f"  image tokens: {tuple(tokens.shape)}")
    print(f"  rope cos/sin: {tuple(rotary[0].shape)}, {tuple(rotary[1].shape)}")
    print(f"  pred/noise:   {tuple(pred.shape)}, {tuple(noise.shape)}")


def gradient() -> None:
    device = choose_device()
    torch.manual_seed(7)
    model = MinimalDECO(
        action_dim=28,
        chunk_size=32,
        dim=128,
        heads=4,
        num_attn_blocks=1,
        use_image_encoder=False,
        image_token_len=64,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    image_tokens = torch.randn(4, 64, 128, device=device)
    obs = torch.randn(4, 28, device=device)
    action = torch.randn(4, 32, 28, device=device)

    first_loss = None
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        pred, noise = model(image_tokens=image_tokens, obs=obs, act=action, training=True)
        loss = F.mse_loss(pred, noise - action)
        loss.backward()
        optimizer.step()
        if first_loss is None:
            first_loss = loss.item()

    grad_names = [
        name
        for name, param in model.named_parameters()
        if param.grad is not None and torch.isfinite(param.grad).all() and param.grad.abs().sum().item() > 0
    ]
    assert grad_names, "No non-zero finite gradients found."
    print("gradient ok")
    print(f"  first loss: {first_loss:.6f}")
    print(f"  non-zero gradient tensors: {len(grad_names)}")
    print(f"  examples: {', '.join(grad_names[:5])}")


def toy_overfit(steps: int, batch_size: int, lr: float) -> None:
    device = choose_device()
    torch.manual_seed(42)
    model = MinimalDECO(
        action_dim=28,
        chunk_size=32,
        dim=96,
        heads=4,
        num_attn_blocks=1,
        inference_steps=4,
        use_image_encoder=False,
        image_token_len=64,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_samples = 32
    image_tokens = torch.randn(num_samples, 64, 96, device=device)
    obs = torch.randn(num_samples, 28, device=device)
    base = torch.tanh(obs[:, None, :].repeat(1, 32, 1))
    ramp = torch.linspace(-0.2, 0.2, 32, device=device)[None, :, None]
    actions = base + ramp

    losses: list[float] = []
    for step in range(steps):
        idx = torch.randint(0, num_samples, (batch_size,), device=device)
        pred, noise = model(
            image_tokens=image_tokens[idx],
            obs=obs[idx],
            act=actions[idx],
            training=True,
        )
        loss = F.mse_loss(pred, noise - actions[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  step {step:04d}: loss={loss.item():.6f}")

    start = sum(losses[: max(1, steps // 10)]) / max(1, steps // 10)
    end = sum(losses[-max(1, steps // 10) :]) / max(1, steps // 10)
    assert torch.isfinite(torch.tensor(losses)).all(), "Loss contains NaN or Inf."
    print("toy-overfit ok")
    print(f"  avg start loss: {start:.6f}")
    print(f"  avg end loss:   {end:.6f}")
    print("  note: stochastic noise makes this a smoke test, not a strict convergence proof.")


def dataset_smoke(data: str, config: str, batch_size: int, num_workers: int) -> None:
    import yaml
    from torchvision.transforms import v2 as transforms

    repo_root = Path(__file__).resolve().parents[1]
    il_root = repo_root / "IL_training_codebase-master"
    sys.path.insert(0, str(il_root))

    from dataset import letterbox, my_Dataset  # pylint: disable=import-error,import-outside-toplevel

    cfg = yaml.safe_load(open(config, "r", encoding="utf-8"))
    img_size = cfg["img"]["img_size"]
    resize = transforms.Resize(img_size) if img_size[0] != img_size[1] else letterbox(img_size[0], fill=128)
    transform = transforms.Compose(
        [
            resize,
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=cfg["img"]["img_mean"], std=cfg["img"]["img_std"]),
        ]
    )
    dataset = my_Dataset(data_dir=data, train=True, transform=transform, **cfg["data"])
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    img1, img2, tac1, tac2, obs, action, mask, task_idx = batch

    assert img1.ndim == 4 and img2.ndim == 4
    assert obs.shape[-1] == cfg["model"]["action_dim"]
    assert action.shape[1:] == (cfg["model"]["chunk_size"], cfg["model"]["action_dim"])
    assert mask.shape[1] == cfg["model"]["chunk_size"]

    print("dataset-smoke ok")
    print(f"  dataset size: {len(dataset)}")
    print(f"  img1/img2:    {tuple(img1.shape)}, {tuple(img2.shape)}")
    print(f"  tac1/tac2:    {tuple(tac1.shape)}, {tuple(tac2.shape)}")
    print(f"  obs/action:   {tuple(obs.shape)}, {tuple(action.shape)}")
    print(f"  mask/task:    {tuple(mask.shape)}, {tuple(task_idx.shape)}")


def inspect_block() -> None:
    device = choose_device()
    block = MMAttention(heads=4, dim=64).to(device)
    img = torch.randn(2, 16, 64, device=device)
    act = torch.randn(2, 8, 64, device=device)
    cond = torch.randn(2, 64, device=device)
    img_out, act_out = block(img, act, cond)
    assert img_out.shape == img.shape
    assert act_out.shape == act.shape
    print("inspect-block ok")
    print(f"  image block output:  {tuple(img_out.shape)}")
    print(f"  action block output: {tuple(act_out.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("shape-v0")
    sub.add_parser("shape-vision")
    sub.add_parser("gradient")
    sub.add_parser("inspect-block")

    toy = sub.add_parser("toy-overfit")
    toy.add_argument("--steps", type=int, default=80)
    toy.add_argument("--batch-size", type=int, default=8)
    toy.add_argument("--lr", type=float, default=2e-3)

    data = sub.add_parser("dataset-smoke")
    data.add_argument("--data", required=True)
    data.add_argument("--config", default="IL_training_codebase-master/config/deco.yaml")
    data.add_argument("--batch-size", type=int, default=2)
    data.add_argument("--num-workers", type=int, default=0)

    args = parser.parse_args()
    if args.command == "shape-v0":
        shape_v0()
    elif args.command == "shape-vision":
        shape_vision()
    elif args.command == "gradient":
        gradient()
    elif args.command == "inspect-block":
        inspect_block()
    elif args.command == "toy-overfit":
        toy_overfit(steps=args.steps, batch_size=args.batch_size, lr=args.lr)
    elif args.command == "dataset-smoke":
        dataset_smoke(
            data=args.data,
            config=args.config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
