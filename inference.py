"""
inference.py - generate submission.csv from a trained Graph WaveNet checkpoint.

Feeds BOTH inputs to the model:
  - test_X_hist.npy      (15-step speed history, [540, 15, 1260])
  - test_texts.json      (event text, aggregated per sample)
    -> parsed via inverse_map.json into per-road event features [540, 1260, E]

Usage:
    uv run python inference.py --ckpt checkpoints/model_best.pt
    uv run python inference.py --ckpt checkpoints/model_best.pt --out submissions/gwn_v1.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import (
    HORIZONS,
    N_ROADS,
    build_node_features,
    build_test,
    load_adj,
    TestDataset,
)
from model import GraphWaveNet


ROOT = Path(__file__).resolve().parent
SUBMISSION_TEMPLATE = ROOT / "dataset-task1" / "sample_submission.csv"
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="path to model_best.pt")
    p.add_argument("--out", type=str, default=None, help="output csv (default: submissions/<ckpt_stem>.csv)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def predict(model, loader, node_feat, adj, dev, n_samples):
    """Run the model on all test samples. Returns preds [N, N_ROADS, 3]."""
    model.eval()
    preds = torch.empty(n_samples, N_ROADS, 3, dtype=torch.float32, device=dev)
    cursor = 0
    for hist, evt in loader:
        hist = hist.to(dev, non_blocking=True)       # [B, T, N]
        evt = evt.to(dev, non_blocking=True)         # [B, N, E]
        out = model(hist, node_feat, evt, adj)       # [B, N, 3]
        B = out.shape[0]
        preds[cursor : cursor + B] = out
        cursor += B
    return preds.cpu().numpy()


def fill_zeros_with_road_mean(preds: np.ndarray, hist: np.ndarray) -> np.ndarray:
    """
    For roads that are always zero across the test history (dead sensors),
    replace predictions with per-road mean computed over all samples' last
    history step. Avoids emitting 0 for roads the model has no signal on.

    preds: [N, n_roads, 3]
    hist:  [N, 15, n_roads]
    """
    # Identify always-zero roads (no non-zero reading across all samples' history)
    nonzero_per_road = (hist > 0).any(axis=(0, 1))           # [n_roads] bool
    always_zero = ~nonzero_per_road                          # dead sensors

    if always_zero.sum() == 0:
        return preds

    # Compute per-road mean speed from non-dead roads (used as fallback)
    last_step = hist[:, -1, :]                                # [N, n_roads]
    road_means = np.zeros(N_ROADS, dtype=np.float32)
    for r in range(N_ROADS):
        if nonzero_per_road[r]:
            vals = last_step[:, r]
            vals = vals[vals > 0]
            road_means[r] = vals.mean() if len(vals) else 0.0
    # For dead roads, use global mean
    global_mean = road_means[nonzero_per_road].mean()
    road_means[always_zero] = global_mean

    # Fill predictions for always-zero roads
    for r in np.where(always_zero)[0]:
        preds[:, r, :] = road_means[r]

    print(f"[fill] replaced predictions for {always_zero.sum()} always-zero roads with mean speeds")
    return preds


def write_submission(preds: np.ndarray, out_path: Path) -> None:
    """
    preds: [540, 3, 1260] -> writes submission CSV in the exact id order
    of sample_submission.csv.
    """
    # The model outputs preds as [N_samples, N_roads, N_horizons]; we want
    # to write rows in the order: sample -> horizon -> road.
    # Re-arrange to match submission's id iteration order.
    template = pd.read_csv(SUBMISSION_TEMPLATE)
    n_samples, n_roads, n_horizons = preds.shape
    assert n_samples == 540 and n_roads == N_ROADS and n_horizons == 3, (
        f"unexpected preds shape {preds.shape}"
    )

    horizon_order = [5, 10, 15]
    hmap = {h: i for i, h in enumerate(horizon_order)}

    ids = template["id"].tolist()
    values = np.empty(len(ids), dtype=np.float32)
    for i, rid in enumerate(ids):
        parts = rid.split("_")
        s = int(parts[1])
        h = int(parts[2][1:])
        r = int(parts[3][1:])
        values[i] = preds[s, r, hmap[h]]

    template["speed"] = values
    template.to_csv(out_path, index=False)
    print(f"[submit] wrote {len(ids)} rows -> {out_path}")


def main():
    args = parse_args()
    dev = device()
    print(f"device: {dev}")

    # --- Load checkpoint ---
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    ckpt_args = ckpt.get("args", {})
    print(f"loaded ckpt: {ckpt_path} (epoch={ckpt.get('epoch')} val_mse={ckpt.get('best_val_mse')})")

    # --- Build static tensors ---
    node_feat_np = build_node_features()
    adj_np = load_adj()
    node_feat = torch.from_numpy(node_feat_np).to(dev)
    adj = torch.from_numpy(adj_np).to(dev)

    # --- Build model from checkpoint args ---
    model = GraphWaveNet(
        n_nodes=N_ROADS,
        in_channels=1,
        hidden_dim=ckpt_args.get("hidden_dim", 32),
        out_dim=ckpt_args.get("out_dim", 32),
        n_blocks=ckpt_args.get("n_blocks", 4),
        n_node_feat=node_feat_np.shape[1],
        n_event_feat=8,                        # EVENT_TYPES has 8 entries
        n_horizons=3,
        emb_dim=ckpt_args.get("emb_dim", 10),
        kernel_size=2,
        dropout=ckpt_args.get("dropout", 0.3),
    ).to(dev)
    model.load_state_dict(ckpt["model"])
    print(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    # --- Load test data (BOTH hist AND event_feat from text) ---
    print("=== Loading test set (history + text-derived event features) ===")
    test_split = build_test()
    print(f"test: hist={test_split.hist.shape}, event_feat={test_split.event_feat.shape}")
    assert test_split.event_feat.shape == (540, N_ROADS, 8), (
        f"expected event_feat shape (540, {N_ROADS}, 8), got {test_split.event_feat.shape}"
    )

    test_ds = TestDataset(test_split)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(dev.type == "cuda"),
    )

    # --- Run inference ---
    print("=== Running inference ===")
    preds = predict(
        model, test_loader, node_feat, adj, dev,
        n_samples=len(test_split.sample_ids),
    )                                                       # [540, N_ROADS, 3]
    print(f"raw preds: shape={preds.shape}, mean={preds.mean():.2f}, std={preds.std():.2f}")

    # --- Fill always-zero roads with per-road mean ---
    preds = fill_zeros_with_road_mean(preds, test_split.hist)

    # --- Clip to physically plausible range [0, 200] ---
    preds = np.clip(preds, 0.0, 200.0)

    # --- Write submission ---
    out_path = Path(args.out) if args.out else OUT_DIR / f"{ckpt_path.stem}.csv"
    write_submission(preds, out_path)
    print(f"\n=== Done. Submission: {out_path} ===")


if __name__ == "__main__":
    main()
