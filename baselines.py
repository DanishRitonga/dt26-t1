"""
baselines.py - Iteration 1: minimal baselines that establish the MSE floor.

Two baselines, both CPU-only, both run in seconds:

  1. Persistence: pred = hist[:, -1, :] for all 3 horizons.
  2. Per-road linear: ridge regression on flattened history per (road, horizon).

Both report masked-MSE on the val split and produce submission CSVs.

Usage:
    uv run python baselines.py                # train all baselines, report val MSE
    uv run python baselines.py --submit       # also write submission CSVs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from dataset import (
    HIST,
    HORIZONS,
    N_ROADS,
    build_splits,
    build_test,
    Split,
    TestSplit,
)

ROOT = Path(__file__).resolve().parent
SUBMISSION_TEMPLATE = ROOT / "dataset-task1" / "sample_submission.csv"
OUT_DIR = ROOT / "submissions"
OUT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """
    pred, target, mask: [N, 3, 1260] (horizons, roads per sample).
    Returns scalar MSE.
    """
    return float(((pred - target) ** 2 * mask).sum() / mask.sum())


def per_horizon_masked_mse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Returns [3] array of masked MSE per horizon."""
    out = np.zeros(3, dtype=np.float64)
    for h in range(3):
        m = mask[:, h]
        out[h] = ((pred[:, h] - target[:, h]) ** 2 * m).sum() / m.sum()
    return out


# ---------------------------------------------------------------------------
# Baseline 1: Persistence
# ---------------------------------------------------------------------------
def persistence_predict(hist: np.ndarray) -> np.ndarray:
    """hist: [N, 15, 1260] -> [N, 3, 1260] (last value repeated for each horizon)."""
    last = hist[:, -1, :]                          # [N, 1260]
    pred = np.broadcast_to(last[:, None, :], (last.shape[0], 3, last.shape[1]))
    return np.ascontiguousarray(pred)


def eval_persistence(train: Split, val: Split) -> dict:
    pred_val = persistence_predict(val.hist)
    mse_total = masked_mse(pred_val, val.targets, val.target_mask)
    mse_per = per_horizon_masked_mse(pred_val, val.targets, val.target_mask)
    return {
        "name": "persistence",
        "val_mse": mse_total,
        "val_mse_per_horizon": mse_per.tolist(),
    }


# ---------------------------------------------------------------------------
# Baseline 2: Per-road ridge regression
# ---------------------------------------------------------------------------
def ridge_per_road(
    train: Split,
    val: Split,
    alpha: float = 1.0,
) -> np.ndarray:
    """
    Fit per-road ridge. For each road r, regress y[h, r] on hist[:, :, r] (15 feats).
    Return pred_val [N_val, 3, 1260].

    Batched over roads via einsum, so it stays fast.
    """
    X_tr = train.hist                      # [N_tr, 15, R]
    Y_tr = train.targets                   # [N_tr, 3, R]
    X_va = val.hist                        # [N_va, 15, R]

    # Add bias by augmenting to [N, 16, R]
    ones_tr = np.ones((X_tr.shape[0], 1, X_tr.shape[2]), dtype=np.float32)
    ones_va = np.ones((X_va.shape[0], 1, X_va.shape[2]), dtype=np.float32)
    Xb_tr = np.concatenate([X_tr, ones_tr], axis=1)
    Xb_va = np.concatenate([X_va, ones_va], axis=1)

    # Closed-form ridge, batched across roads:
    #   W[r,b,h] = (sum_n X[n,b,r] X[n,c,r] + alpha*I[b,c])^{-1} sum_n X[n,b,r] Y[n,h,r]
    XtX = np.einsum("nbr,ncr->rbc", Xb_tr, Xb_tr)        # [R, 16, 16]
    XtY = np.einsum("nbr,nhr->rbh", Xb_tr, Y_tr)         # [R, 16, 3]
    reg = alpha * np.eye(XtX.shape[1], dtype=np.float32)  # [16, 16]
    W = np.linalg.solve(XtX + reg[None], XtY)             # [R, 16, 3]

    # Predict: Y_hat[n,h,r] = sum_b X[n,b,r] W[r,b,h]
    pred_val = np.einsum("nbr,rbh->nrh", Xb_va, W)        # [N_va, R, 3]
    pred_val = np.transpose(pred_val, (0, 2, 1))          # [N_val, 3, R]
    return pred_val


def eval_ridge(train: Split, val: Split, alpha: float = 1.0) -> dict:
    pred_val = ridge_per_road(train, val, alpha=alpha)
    mse_total = masked_mse(pred_val, val.targets, val.target_mask)
    mse_per = per_horizon_masked_mse(pred_val, val.targets, val.target_mask)
    return {
        "name": f"ridge_per_road(alpha={alpha})",
        "val_mse": mse_total,
        "val_mse_per_horizon": mse_per.tolist(),
        "pred_val": pred_val,
    }


# ---------------------------------------------------------------------------
# Submission writer
# ---------------------------------------------------------------------------
def write_submission(pred: np.ndarray, name: str) -> Path:
    """
    pred: [540, 3, 1260] -> writes submissions/{name}.csv in the exact id order
    of sample_submission.csv.
    """
    out = OUT_DIR / f"{name}.csv"
    template = pd.read_csv(SUBMISSION_TEMPLATE)
    # template['id'] is already in the correct order. Parse and index.
    n_samples, n_horizons, n_roads = pred.shape
    assert n_samples == 540 and n_horizons == 3 and n_roads == N_ROADS

    # Build a flat array indexed by [sample*3*1260 + horizon_idx*1260 + road]
    # matching id order test_{s:05d}_h{5|10|15}_r{r}
    # Easiest: iterate template rows in order, parse id, index pred.
    ids = template["id"].tolist()
    horizon_order = [5, 10, 15]
    hmap = {h: i for i, h in enumerate(horizon_order)}

    values = np.empty(len(ids), dtype=np.float32)
    for i, rid in enumerate(ids):
        parts = rid.split("_")
        s = int(parts[1])
        h = int(parts[2][1:])    # strip leading 'h'
        r = int(parts[3][1:])    # strip leading 'r'
        values[i] = pred[s, hmap[h], r]

    template["speed"] = values
    template.to_csv(out, index=False)
    print(f"[submit] wrote {len(ids)} rows -> {out}")
    return out


def predict_test_with_ridge(train_full: Split, test: TestSplit, alpha: float = 1.0) -> np.ndarray:
    """
    Train per-road ridge on the FULL train data (no val holdout),
    predict on test. Returns [540, 3, 1260].
    """
    X_tr = train_full.hist                      # [N_tr, 15, R]
    Y_tr = train_full.targets                   # [N_tr, 3, R]
    X_va = test.hist                            # [N_va, 15, R]

    ones_tr = np.ones((X_tr.shape[0], 1, X_tr.shape[2]), dtype=np.float32)
    ones_va = np.ones((X_va.shape[0], 1, X_va.shape[2]), dtype=np.float32)
    Xb_tr = np.concatenate([X_tr, ones_tr], axis=1)
    Xb_va = np.concatenate([X_va, ones_va], axis=1)

    XtX = np.einsum("nbr,ncr->rbc", Xb_tr, Xb_tr)        # [R, 16, 16]
    XtY = np.einsum("nbr,nhr->rbh", Xb_tr, Y_tr)         # [R, 16, 3]
    reg = alpha * np.eye(XtX.shape[1], dtype=np.float32)
    W = np.linalg.solve(XtX + reg[None], XtY)
    pred_test = np.einsum("nbr,rbh->nrh", Xb_va, W)       # [N_test, R, 3]
    pred_test = np.transpose(pred_test, (0, 2, 1))
    return pred_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="write submission CSVs")
    parser.add_argument("--alpha", type=float, default=1.0, help="ridge regularization")
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--val-stride", type=int, default=5)
    args = parser.parse_args()

    print("=== Loading splits ===")
    train, val = build_splits(val_frac=0.2, train_stride=args.train_stride, val_stride=args.val_stride)
    print(f"train: {train.hist.shape}, val: {val.hist.shape}")

    results = []

    # Baseline 1: Persistence
    print("\n=== Baseline 1: Persistence ===")
    r1 = eval_persistence(train, val)
    print(f"val MSE = {r1['val_mse']:.4f}")
    print(f"per-horizon MSE: {[round(x, 4) for x in r1['val_mse_per_horizon']]}")
    results.append(r1)

    # Baseline 2: Per-road ridge
    print("\n=== Baseline 2: Per-road Ridge ===")
    r2 = eval_ridge(train, val, alpha=args.alpha)
    print(f"val MSE = {r2['val_mse']:.4f}")
    print(f"per-horizon MSE: {[round(x, 4) for x in r2['val_mse_per_horizon']]}")
    results.append(r2)

    # Summary
    print("\n=== Summary ===")
    for r in results:
        print(f"  {r['name']:40s}  val_mse = {r['val_mse']:.4f}")

    if args.submit:
        print("\n=== Generating submissions on test set ===")
        test = build_test()
        print(f"test: {test.hist.shape}")

        # Persistence on test
        pred_persist = persistence_predict(test.hist)
        write_submission(pred_persist, "persistence")

        # Ridge on test (train on full train+val pooled, no holdout)
        print("\n[ridge] rebuilding with full train+val pooled for test prediction")
        train_full, _ = build_splits(val_frac=0.0, train_stride=args.train_stride)
        pred_ridge = predict_test_with_ridge(train_full, test, alpha=args.alpha)
        write_submission(pred_ridge, f"ridge_alpha{args.alpha}")


if __name__ == "__main__":
    main()
