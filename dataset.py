"""
dataset.py - data pipeline for Graph WaveNet traffic forecasting.

Produces tensors matching the spec in windowing.md section 12.2:
  speed_hist : [B, 1260, 15, 1]
  adj_fixed  : [1260, 1260]
  node_feat  : [1260, F_static]
  event_feat : [B, 1260, E_event]   (per-sample)
  targets    : [B, 1260, 3]
  target_mask: [B, 1260, 3]

Designed to run identically on dev box (CPU) and training box (GPU).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIST = 15
HORIZONS = (5, 10, 15)         # +5, +10, +15 steps past t_end
H_MAX = max(HORIZONS)          # 15
N_ROADS = 1260

EVENT_TYPES = [
    "road closure",
    "construction",
    "a general traffic accident",
    "road traffic control",
    "prohibit left turn",
    "an announcement",
    "road obstruction",
    "a broken down vehicle",
]
EVENT_PATTERN = re.compile(r"^(.+?)\s+on\s+(.+)$")

# Default roadclass / formway vocabularies (computed from data, kept stable)
ROADCLASS_VOCAB = [0, 1, 2, 3, 6]
FORMWAY_TOP = [1, 6, 7, 15, 3]   # 5 most frequent; rest -> "other"


# ---------------------------------------------------------------------------
# Paths (resolved relative to this file so it works on any machine)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset-task1"
ART = ROOT / "data"              # artifacts from translate.py

TRAIN_BLOCKS = [
    {
        "prefix": "m1",
        "speed": DATA / "train" / "train_speed_m1_1_11160.npy",
        "text":  DATA / "train" / "train_text_m1_1_11160.json",
    },
    {
        "prefix": "m2",
        "speed": DATA / "train" / "train_speed_m2_1_5039.npy",
        "text":  DATA / "train" / "train_text_m2_1_5039.json",
    },
]
TEST_HIST  = DATA / "test" / "test_X_hist.npy"
TEST_TEXT  = DATA / "test" / "test_texts.json"
ADJ_PATH   = DATA / "static" / "matrix.npy"
ROADS_JSON = DATA / "static" / "Roads1260.json"

INVERSE_MAP = ART / "inverse_map.json"


# ---------------------------------------------------------------------------
# Static: adjacency
# ---------------------------------------------------------------------------
def load_adj(strip_self_loops: bool = True, dtype=np.float32) -> np.ndarray:
    """Load [1260, 1260] adjacency. Optionally strip self-loops."""
    adj = np.load(ADJ_PATH).astype(dtype)
    if strip_self_loops:
        adj = adj - np.diag(np.diag(adj))
    return adj


# ---------------------------------------------------------------------------
# Static: per-road features from Roads1260.json
# ---------------------------------------------------------------------------
def _centroid(seg: dict) -> tuple[float, float]:
    """Return (lng, lat) centroid of a sub-segment."""
    coords = seg.get("coordList", [])
    if not coords:
        return (0.0, 0.0)
    lngs = coords[0::2]
    lats = coords[1::2]
    return (float(np.mean(lngs)), float(np.mean(lats)))


def build_node_features() -> np.ndarray:
    """
    Returns [1260, F_static] numpy float32.
    Features:
      0   length_total (log-scaled meters)
      1   n_segments (log-scaled)
      2-6 roadclass one-hot (5 dims)
      7-11 formway one-hot (top-5 + other = 6 dims)
      12  centroid_lng (normalized)
      13  centroid_lat (normalized)
    Total F_static = 14.
    """
    with open(ROADS_JSON, encoding="utf-8") as f:
        roads = json.load(f)

    F = 2 + len(ROADCLASS_VOCAB) + (len(FORMWAY_TOP) + 1) + 2
    feats = np.zeros((N_ROADS, F), dtype=np.float32)

    # collect raw values for normalization
    lengths_all = []
    centroids_lng, centroids_lat = [], []

    for j, group in enumerate(roads):
        if not group:
            continue
        total_len = float(sum(s.get("length", 0) for s in group))
        feats[j, 0] = total_len
        feats[j, 1] = len(group)
        lengths_all.append(total_len)

        # mode of roadclass across sub-segments
        cls_count = Counter(s.get("roadclass", 0) for s in group)
        cls_mode = cls_count.most_common(1)[0][0]
        if cls_mode in ROADCLASS_VOCAB:
            feats[j, 2 + ROADCLASS_VOCAB.index(cls_mode)] = 1.0

        # mode of formway
        fw_count = Counter(s.get("formway", 0) for s in group)
        fw_mode = fw_count.most_common(1)[0][0]
        if fw_mode in FORMWAY_TOP:
            feats[j, 2 + len(ROADCLASS_VOCAB) + FORMWAY_TOP.index(fw_mode)] = 1.0
        else:
            feats[j, 2 + len(ROADCLASS_VOCAB) + len(FORMWAY_TOP)] = 1.0  # "other"

        # centroid across sub-segments
        lngs, lats = [], []
        for s in group:
            lng, lat = _centroid(s)
            lngs.append(lng); lats.append(lat)
        clng, clat = float(np.mean(lngs)), float(np.mean(lats))
        centroids_lng.append(clng); centroids_lat.append(clat)

    # Normalize: log + z-score for length and n_segments; z-score for coords
    log_lens = np.log1p(feats[:, 0])
    feats[:, 0] = (log_lens - log_lens.mean()) / (log_lens.std() + 1e-6)
    log_ns = np.log1p(feats[:, 1])
    feats[:, 1] = (log_ns - log_ns.mean()) / (log_ns.std() + 1e-6)

    clng = np.array(centroids_lng)
    clat = np.array(centroids_lat)
    feats[:, -2] = (clng - clng.mean()) / (clng.std() + 1e-6)
    feats[:, -1] = (clat - clat.mean()) / (clat.std() + 1e-6)

    return feats


# ---------------------------------------------------------------------------
# Per-sample: event features from text
# ---------------------------------------------------------------------------
def build_event_feat(
    text: str,
    inv_map: dict[str, list[int]],
    n_roads: int = N_ROADS,
) -> np.ndarray:
    """Returns [n_roads, len(EVENT_TYPES)] binary event mask."""
    feat = np.zeros((n_roads, len(EVENT_TYPES)), dtype=np.float32)
    for sent in text.split("."):
        s = sent.strip()
        if not s:
            continue
        m = EVENT_PATTERN.match(s)
        if not m:
            continue
        etype = m.group(1).strip().lower()
        phrase = m.group(2).strip().lower()
        roads = inv_map.get(phrase)
        if not roads:
            continue
        for i, canon in enumerate(EVENT_TYPES):
            if canon in etype:
                feat[roads, i] = 1.0
    return feat


def load_inverse_map() -> dict[str, list[int]]:
    with open(INVERSE_MAP, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Windowing (per-block)
# ---------------------------------------------------------------------------
def window_block(
    speed: np.ndarray,
    text_dict: dict[str, str],
    prefix: str,
    inv_map: dict[str, list[int]],
    t_end_lo: int,
    t_end_hi: int,
    stride: int = 1,
) -> dict[str, np.ndarray]:
    """
    Slice a block into windows for t_end in [t_end_lo, t_end_hi] inclusive,
    every `stride` steps. Returns dict of stacked arrays.

    Outputs:
      hist        : [N, 15, 1260]
      event_feat  : [N, 1260, E_event]
      targets     : [N, 3, 1260]
      target_mask : [N, 3, 1260]   (1 where target != 0, else 0)
    """
    t_ends = range(t_end_lo, t_end_hi + 1, stride)
    n = len(t_ends)

    hist_buf = np.empty((n, HIST, N_ROADS), dtype=np.float32)
    tgt_buf = np.empty((n, len(HORIZONS), N_ROADS), dtype=np.float32)
    mask_buf = np.empty((n, len(HORIZONS), N_ROADS), dtype=np.float32)
    evt_buf = np.empty((n, N_ROADS, len(EVENT_TYPES)), dtype=np.float32)

    for i, t_end in enumerate(t_ends):
        hist_buf[i] = speed[t_end - HIST + 1 : t_end + 1]
        for h_idx, hp in enumerate(HORIZONS):
            y = speed[t_end + hp]
            tgt_buf[i, h_idx] = y
            mask_buf[i, h_idx] = (y != 0).astype(np.float32)
        # aggregated text for this window (train: join per-step texts)
        parts, prev = [], None
        for k in range(t_end - HIST + 1, t_end + 1):
            s = text_dict[f"{prefix}_{k + 1}"]
            if s != prev:
                parts.append(s)
                prev = s
        agg = ". ".join(parts)
        evt_buf[i] = build_event_feat(agg, inv_map)

    return {
        "hist": hist_buf,
        "event_feat": evt_buf,
        "targets": tgt_buf,
        "target_mask": mask_buf,
    }


def split_block(T: int, val_frac: float = 0.2, gap: int = HIST + H_MAX):
    """Returns (train_t_end_range, val_t_end_range) for a block of length T."""
    val_start_T = int(T * (1 - val_frac))
    train_lo, train_hi = HIST - 1, val_start_T - gap - 1
    val_lo, val_hi = val_start_T, T - H_MAX - 1
    return (train_lo, train_hi), (val_lo, val_hi)


# ---------------------------------------------------------------------------
# Build full train/val tensors
# ---------------------------------------------------------------------------
@dataclass
class Split:
    hist: np.ndarray            # [N, 15, 1260]
    event_feat: np.ndarray      # [N, 1260, E]
    targets: np.ndarray         # [N, 3, 1260]
    target_mask: np.ndarray     # [N, 3, 1260]


def build_splits(val_frac: float = 0.2, train_stride: int = 1, val_stride: int = 5) -> tuple[Split, Split]:
    """Build train + val splits pooled across both blocks."""
    inv_map = load_inverse_map()
    train_chunks: list[dict] = []
    val_chunks: list[dict] = []

    for blk in TRAIN_BLOCKS:
        speed = np.load(blk["speed"])
        with open(blk["text"], encoding="utf-8") as f:
            text_dict = json.load(f)
        T = speed.shape[0]
        (tr_lo, tr_hi), (va_lo, va_hi) = split_block(T, val_frac=val_frac)
        train_chunks.append(window_block(
            speed, text_dict, blk["prefix"], inv_map,
            tr_lo, tr_hi, stride=train_stride,
        ))
        val_chunks.append(window_block(
            speed, text_dict, blk["prefix"], inv_map,
            va_lo, va_hi, stride=val_stride,
        ))

    def stack(chunks, key):
        return np.concatenate([c[key] for c in chunks], axis=0)

    train = Split(
        hist=stack(train_chunks, "hist"),
        event_feat=stack(train_chunks, "event_feat"),
        targets=stack(train_chunks, "targets"),
        target_mask=stack(train_chunks, "target_mask"),
    )
    val = Split(
        hist=stack(val_chunks, "hist"),
        event_feat=stack(val_chunks, "event_feat"),
        targets=stack(val_chunks, "targets"),
        target_mask=stack(val_chunks, "target_mask"),
    )
    return train, val


# ---------------------------------------------------------------------------
# Test set loader
# ---------------------------------------------------------------------------
@dataclass
class TestSplit:
    hist: np.ndarray            # [540, 15, 1260]
    event_feat: np.ndarray      # [540, 1260, E]
    sample_ids: list[int]       # [0..539]


def build_test() -> TestSplit:
    hist = np.load(TEST_HIST).astype(np.float32)        # [N, 15, 1260]
    with open(TEST_TEXT, encoding="utf-8") as f:
        text_dict = json.load(f)
    inv_map = load_inverse_map()

    n = hist.shape[0]
    evt = np.empty((n, N_ROADS, len(EVENT_TYPES)), dtype=np.float32)
    ids: list[int] = []
    for i in range(n):
        key = f"test_{i:05d}"
        evt[i] = build_event_feat(text_dict[key], inv_map)
        ids.append(i)
    return TestSplit(hist=hist, event_feat=evt, sample_ids=ids)


# ---------------------------------------------------------------------------
# Torch Dataset wrappers
# ---------------------------------------------------------------------------
class TrafficDataset(Dataset):
    """Wraps a Split into a torch Dataset.

    Each item returns:
      hist        : [15, 1260]
      event_feat  : [1260, E]
      targets     : [3, 1260]
      target_mask : [3, 1260]
    """

    def __init__(self, split: Split):
        self.hist = torch.from_numpy(split.hist)
        self.event_feat = torch.from_numpy(split.event_feat)
        self.targets = torch.from_numpy(split.targets)
        self.target_mask = torch.from_numpy(split.target_mask)

    def __len__(self) -> int:
        return self.hist.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.hist[idx],
            self.event_feat[idx],
            self.targets[idx],
            self.target_mask[idx],
        )


class TestDataset(Dataset):
    def __init__(self, split: TestSplit):
        self.hist = torch.from_numpy(split.hist)
        self.event_feat = torch.from_numpy(split.event_feat)

    def __len__(self) -> int:
        return self.hist.shape[0]

    def __getitem__(self, idx: int):
        return self.hist[idx], self.event_feat[idx]


# ---------------------------------------------------------------------------
# CLI sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== Static features ===")
    adj = load_adj()
    print(f"adj: shape={adj.shape}, nnz={(adj != 0).sum()}, dtype={adj.dtype}")
    nf = build_node_features()
    print(f"node_feat: shape={nf.shape}, mean={nf.mean():.3f}, std={nf.std():.3f}")

    print("\n=== Train/Val splits (val_frac=0.2, train_stride=5, val_stride=5) ===")
    train, val = build_splits(val_frac=0.2, train_stride=5, val_stride=5)
    print(f"train: hist={train.hist.shape}, evt={train.event_feat.shape}, targets={train.targets.shape}")
    print(f"val:   hist={val.hist.shape}, evt={val.event_feat.shape}, targets={val.targets.shape}")
    print(f"train mask coverage: {train.target_mask.mean():.4f} (m1+m2 pooled)")
    print(f"val mask coverage:   {val.target_mask.mean():.4f}")

    print("\n=== Test set ===")
    test = build_test()
    print(f"test: hist={test.hist.shape}, evt={test.event_feat.shape}, n_ids={len(test.sample_ids)}")

    print("\n=== Torch Dataset smoke test ===")
    ds = TrafficDataset(train)
    h, e, y, m = ds[0]
    print(f"item 0: hist={tuple(h.shape)}, event_feat={tuple(e.shape)}, targets={tuple(y.shape)}, mask={tuple(m.shape)}")
