"""
train.py - train Graph WaveNet on the traffic forecasting task.

Designed to run on the training box (GPU). Falls back to CPU if no CUDA.

Usage:
    uv run python train.py                          # default config
    uv run python train.py --epochs 50 --batch-size 32
    uv run python train.py --max-grad-norm 5        # gradient clipping
    uv run python train.py --amp                     # mixed precision

Outputs:
    checkpoints/model_best.pt         (lowest val MSE)
    checkpoints/model_last.pt         (latest epoch)
    logs/train_log.json               (per-epoch metrics)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import (
    HIST,
    HORIZONS,
    N_ROADS,
    build_splits,
    build_node_features,
    build_test,
    load_adj,
    TrafficDataset,
)
from model import GraphWaveNet, masked_mse_loss


ROOT = Path(__file__).resolve().parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"
CKPT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--val-stride", type=int, default=5)
    p.add_argument("--val-frac", type=float, default=0.2)
    # Model
    p.add_argument("--hidden-dim", type=int, default=32)
    p.add_argument("--out-dim", type=int, default=32)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--emb-dim", type=int, default=10)
    p.add_argument("--dropout", type=float, default=0.3)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    # Save/resume
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--save-every", type=int, default=5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def per_horizon_mse(preds, targets, mask):
    """Return [3] array of per-horizon masked MSE."""
    # preds: [B, N, H] -> [B, H, N]
    preds_a = preds.permute(0, 2, 1)
    out = []
    for h in range(3):
        m = mask[:, h]
        out.append(float(((preds_a[:, h] - targets[:, h]) ** 2 * m).sum() / m.sum()))
    return np.array(out)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, scaler, node_feat, adj, dev, amp, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_mask = 0.0
    n_batches = 0

    for hist, evt, tgt, mask in loader:
        hist = hist.to(dev, non_blocking=True)        # [B, T, N]
        evt = evt.to(dev, non_blocking=True)          # [B, N, E]
        tgt = tgt.to(dev, non_blocking=True)          # [B, H, N]
        mask = mask.to(dev, non_blocking=True)        # [B, H, N]

        optimizer.zero_grad(set_to_none=True)

        if amp:
            with torch.autocast(device_type=dev.type, dtype=torch.float16):
                preds = model(hist, node_feat, evt, adj)
                loss = masked_mse_loss(preds, tgt, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(hist, node_feat, evt, adj)
            loss = masked_mse_loss(preds, tgt, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

        total_loss += loss.item() * mask.sum().item()
        total_mask += mask.sum().item()
        n_batches += 1

    return total_loss / max(n_batches, 1)   # avg loss per batch


@torch.no_grad()
def evaluate(model, loader, node_feat, adj, dev):
    model.eval()
    # Accumulate per-horizon sums
    sq_sum = torch.zeros(3, device=dev)
    mask_sum = torch.zeros(3, device=dev)

    for hist, evt, tgt, mask in loader:
        hist = hist.to(dev, non_blocking=True)
        evt = evt.to(dev, non_blocking=True)
        tgt = tgt.to(dev, non_blocking=True)
        mask = mask.to(dev, non_blocking=True)

        preds = model(hist, node_feat, evt, adj)        # [B, N, H]
        preds_a = preds.permute(0, 2, 1)                # [B, H, N]
        sq = (preds_a - tgt) ** 2 * mask
        sq_sum += sq.sum(dim=(0, 2))
        mask_sum += mask.sum(dim=(0, 2))

    per_h = (sq_sum / mask_sum.clamp(min=1.0)).cpu().numpy()
    total = float(sq_sum.sum() / mask_sum.sum().clamp(min=1.0))
    return total, per_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    set_seed(args.seed)
    dev = device()
    print(f"device: {dev}")

    # --- Data ---
    print("=== Loading splits ===")
    train_split, val_split = build_splits(
        val_frac=args.val_frac,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
    )
    print(f"train: {train_split.hist.shape}, val: {val_split.hist.shape}")

    train_ds = TrafficDataset(train_split)
    val_ds = TrafficDataset(val_split)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(dev.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev.type == "cuda"),
    )

    # --- Static tensors ---
    print("=== Building static features ===")
    node_feat_np = build_node_features()
    adj_np = load_adj()
    node_feat = torch.from_numpy(node_feat_np).to(dev)
    adj = torch.from_numpy(adj_np).to(dev)
    n_node_feat = node_feat_np.shape[1]
    n_event_feat = train_split.event_feat.shape[-1]
    print(f"node_feat: {node_feat.shape}, adj: {adj.shape}")

    # --- Model ---
    model = GraphWaveNet(
        n_nodes=N_ROADS,
        in_channels=1,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        n_blocks=args.n_blocks,
        n_node_feat=n_node_feat,
        n_event_feat=n_event_feat,
        n_horizons=3,
        emb_dim=args.emb_dim,
        kernel_size=2,
        dropout=args.dropout,
    ).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {type(model).__name__}, params: {n_params:,}")

    # --- Optimizer / loss / amp scaler ---
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = args.amp and dev.type == "cuda"
    scaler = torch.amp.GradScaler() if amp_enabled else None

    # --- Resume ---
    start_epoch = 0
    best_val_mse = math.inf
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=dev)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_mse = ckpt["best_val_mse"]
        print(f"resumed from {args.resume} at epoch {start_epoch}, best_val_mse={best_val_mse:.4f}")

    # --- Train loop ---
    log = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler,
            node_feat, adj, dev, amp_enabled, args.max_grad_norm,
        )
        val_mse, val_per_h = evaluate(model, val_loader, node_feat, adj, dev)
        scheduler.step()
        dt = time.time() - t0

        improved = val_mse < best_val_mse
        best_val_mse = min(best_val_mse, val_mse)

        print(
            f"epoch {epoch+1:3d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  "
            f"val_mse={val_mse:.4f}  "
            f"(h5={val_per_h[0]:.2f} h10={val_per_h[1]:.2f} h15={val_per_h[2]:.2f})  "
            f"best={best_val_mse:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"time={dt:.1f}s"
            f"  *IMPROVED" if improved else ""
        )

        log.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_mse": val_mse,
            "val_mse_per_horizon": val_per_h.tolist(),
            "best_val_mse": best_val_mse,
            "lr": scheduler.get_last_lr()[0],
            "time_s": dt,
        })

        # Save best
        if improved:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mse": best_val_mse,
                "args": vars(args),
            }, CKPT_DIR / "model_best.pt")

        # Periodic snapshot
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_val_mse": best_val_mse,
                "args": vars(args),
            }, CKPT_DIR / "model_last.pt")

    # Final log dump
    with open(LOG_DIR / "train_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n=== Training complete. best_val_mse = {best_val_mse:.4f} ===")
    print(f"Best checkpoint: {CKPT_DIR / 'model_best.pt'}")
    print(f"Log: {LOG_DIR / 'train_log.json'}")


if __name__ == "__main__":
    main()
