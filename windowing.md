# Windowing & Train/Val Split Strategy

This document captures the time-windowing rules for the traffic forecasting task,
the leakage trap to avoid, and the concrete slicing recipe.

---

## 1. Time conventions (from `overview.md`)

| Quantity | Value |
|---|---|
| History window length | 15 steps |
| History wall-clock span | 1 hour |
| **1 step** | **4 minutes** (60 min / 15) |
| Prediction horizons | +20, +40, +60 min |
| Horizon in steps | +5, +10, +15 (= `h5`, `h10`, `h15`) |

All time is **relative to the window**. There is no wall-clock timestamp anywhere
in the dataset (filenames, text, or speed series). m1 and m2 are independent
recording periods and are NOT aligned in absolute time — and do not need to be.

---

## 2. What a "history window" is

A history window is the **input** given to the model for one prediction:

- Shape: `[15, 1260]` (15 timesteps × 1,260 roads)
- = 1 hour of past speeds ending at "now" (step index 14, 0-indexed)

For each history window, the model must predict **3 future points**:

```
step index:    0 ... 14 | 15 16 17 18  19  20 21 22 23  24  25 26 27 28  29
              \_ input _/   .  .  .  .   ↑   .  .  .  .   ↑   .  .  .  .   ↑
                           (unused)      h5  (unused)      h10 (unused)      h15
                                         +5                +10               +15
```

- `h5`  → step 14 + 5  = **step 19** (+20 min)
- `h10` → step 14 + 10 = **step 24** (+40 min)
- `h15` → step 14 + 15 = **step 29** (+60 min)

Steps 15–18, 20–23, 25–28 are **unused** for the scoring targets. (They may be
used as auxiliary training signal, but are not part of `submission.csv`.)

### Output volume per sample
3 horizons × 1,260 roads = **3,780 speed predictions per sample**.

---

## 3. Windowing a continuous block (m1 / m2)

For a block of length `T` (rows = timesteps, columns = roads), choose `t_end` =
the **last index of the history** (i.e. step 14 within that window).

### Valid range of `t_end`

- History must fit:        `t_end - 14 >= 0`           →  `t_end >= 14`
- Farthest target (`h15`): `t_end + 15 <= T - 1`        →  `t_end <= T - 16`

Combined:  **`t_end ∈ [14, T - 16]`**, giving **`T - 29` valid windows** per block.

| Block | T | Windows |
|---|---|---|
| m1 (`train_speed_m1_1_11160.npy`) | 11,160 | 11,131 |
| m2 (`train_speed_m2_1_5039.npy`)  | 5,039  | 5,010  |
| **Total** | | **16,141** |

### Slicing (one window)

For a chosen `t_end`:

```python
hist  = speed[t_end - 14 : t_end + 1]   # [15, 1260]   (indices t_end-14 .. t_end)
y_h5  = speed[t_end + 5]                # [1260]       step 19 (within this window's frame)
y_h10 = speed[t_end + 10]               # [1260]       step 24
y_h15 = speed[t_end + 15]               # [1260]       step 29
```

### Critical rule: never cross the m1 ↔ m2 boundary
m1 and m2 are different recording periods. A window that starts in m1 and ends
in m2 would mix unrelated time periods and produce garbage. Process each block
independently.

---

## 4. The leakage trap (why naive random splits fail)

If train and validation windows overlap in time, **the validation ground truth
can appear inside training inputs**, and vice versa.

### Example of leakage (stride 5, same block)

```
Window A:  hist=[0..14]   targets at 19, 24, 29
Window B:  hist=[5..19]   targets at 24, 29, 34
                              ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
                              B's history contains A's targets h5 and h10!
```

If A ∈ train and B ∈ val, the model has **literally seen val ground truth** as
part of its training input. Local CV becomes meaningless and the model collapses
on the real test set (which has no such overlap with train).

This is **data leakage**, which is worse than overfitting:
- Overfitting → model memorizes patterns → still generalizes a bit.
- Leakage → model sees the answer → generalizes to nothing.

---

## 5. Correct train/val split: by time, not by window index

### Overlap *within* train is fine
Windows sharing history/targets **inside the same split** is just augmentation.
The model sees slightly shifted versions of the same traffic pattern — usually
helpful.

### What must NOT happen
A train window's targets must not appear inside any val window's history, and
vice versa.

### Recipe
For each block independently:

1. Pick a time-based split point, e.g. last 20% of steps → val.
2. Leave a safety gap of `HIST + H_MAX = 15 + 15 = 30` steps at the boundary so
   no train window's `h15` lands inside val's history.
3. Window each side independently with any stride you like (stride 1 is fine).
4. Pool train windows from (m1-train, m2-train).
5. Pool val windows from (m1-val, m2-val).

### Visual

```
m1 block (11160 steps):
[0 ................. 8000]  gap  [8031 ........ 11160]
        TRAIN (windowable)        VAL (windowable)

m2 block (5039 steps):
[0 ............ 4000]  gap  [4031 .... 5039]
       TRAIN                   VAL
```

Then combine:
- train = m1_train_windows + m2_train_windows
- val   = m1_val_windows   + m2_val_windows

---

## 6. Reference slicing code (vectorized)

```python
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

HIST = 15
HORIZONS = (5, 10, 15)         # +5, +10, +15 steps past t_end
H_MAX = max(HORIZONS)          # 15

def make_windows(speed: np.ndarray, t_start: int, t_end_range: tuple[int, int]):
    """
    speed: [T, 1260] continuous series from one block.
    Windows are produced for t_end in [t_end_range[0], t_end_range[1]] inclusive,
    using full history [t_end-14 .. t_end] from `speed`.
    Returns dict of arrays.
    """
    t_lo, t_hi = t_end_range
    # Pre-slice the minimal region we need: [t_lo - 14 .. t_hi + 15]
    region = speed[t_lo - 14 : t_hi + 15 + 1]   # [W, 1260]
    W = region.shape[0]
    assert W >= HIST + H_MAX + 1, f"region too short: {W}"

    # Sliding window of length (HIST + H_MAX) over the region
    view = sliding_window_view(region, HIST + H_MAX + 1, axis=0)   # [N, 1260, W]
    view = np.moveaxis(view, -1, 1)                                # [N, W, 1260]

    hists = view[:, :HIST, :]                                       # [N, 15, 1260]
    ys = {f"h{hp}": view[:, HIST - 1 + hp, :] for hp in HORIZONS}   # h5/h10/h15
    # HIST - 1 + hp because t_end sits at index (HIST-1) inside the window,
    # and the target is hp steps further.
    return {"hist": hists, **ys}


def split_block(T: int, val_frac: float = 0.2, gap: int = HIST + H_MAX):
    """
    Returns (train_t_end_range, val_t_end_range) for a block of length T.
    Each range is [lo, hi] inclusive for valid t_end values.
    """
    val_start_T = int(T * (1 - val_frac))   # split point in step index
    train_lo, train_hi = HIST - 1, val_start_T - gap - 1
    val_lo,   val_hi   = val_start_T, T - H_MAX - 1
    return (train_lo, train_hi), (val_lo, val_hi)
```

Usage:

```python
m1 = np.load("dataset-task1/train/train_speed_m1_1_11160.npy")
m2 = np.load("dataset-task1/train/train_speed_m2_1_5039.npy")

train_chunks, val_chunks = [], []
for block in (m1, m2):
    T = block.shape[0]
    (tr_lo, tr_hi), (va_lo, va_hi) = split_block(T, val_frac=0.2)
    train_chunks.append(make_windows(block, tr_lo, (tr_lo, tr_hi)))
    val_chunks.append(make_windows(block, va_lo, (va_lo, va_hi)))

def stack(chunks, key):
    return np.concatenate([c[key] for c in chunks], axis=0)

train_hist = stack(train_chunks, "hist")
train_h5   = stack(train_chunks, "h5")
train_h10  = stack(train_chunks, "h10")
train_h15  = stack(train_chunks, "h15")
# ... same for val
```

---

## 7. Connecting windows to `matrix.npy` and `train_text`

The two connections are different in nature.

### 7.1 `matrix.npy` — static, global (NOT per window)

The adjacency matrix is a fixed property of the road network. It is **the same
for every window, every sample, every split**.

- Shape: `[1260, 1260]`, binary, **directed** (asymmetric), 5,122 edges, with
  some self-loops.
- Column `j` of any speed array = row/column `j` of `matrix.npy` = **same road**.

Connection is by **index identity**: load once, attach to model, never slice.

```python
import numpy as np

ADJ = np.load("dataset-task1/static/matrix.npy").astype(np.float32)  # [1260,1260]
ADJ = ADJ - np.diag(np.diag(ADJ))   # optional: strip self-loops
# optional: symmetrize for undirected GCN: ADJ = np.maximum(ADJ, ADJ.T)
```

The model takes `ADJ` as a fixed constructor argument. You do NOT slice it per
window — it doesn't depend on `t_end`.

### 7.2 `train_text_*.json` — per window, aligned by `t_end`

Train text keys are **1-indexed step numbers**:

| Speed array row (0-indexed) | Text key (1-indexed) |
|---|---|
| 0              | `m1_1` |
| 14             | `m1_15` |
| `i`            | `m1_{i+1}` |
| 11159          | `m1_11160` |

So for a window ending at `t_end` (0-indexed in the speed array), the **15
aligned text keys** are `m1_{t_end-13}` … `m1_{t_end+1}`.

### 7.3 The text aggregation problem

Train gives **15 per-step texts** per window. Test gives **1 aggregated text**
per sample. To use the same text encoder at train and test time, collapse the
15 train texts into a single string per window:

```python
def window_text(text_dict: dict, prefix: str, t_end: int, hist: int = 15) -> str:
    """
    Aggregate the `hist` per-step texts aligned to a window ending at t_end.
    Dedup adjacent identical texts (event narratives often repeat for several
    steps). Returns one string — same shape as test_texts[sample].
    """
    parts, prev = [], None
    for i in range(t_end - hist + 1, t_end + 1):     # 0-indexed step range
        s = text_dict[f"{prefix}_{i + 1}"]           # 1-indexed key
        if s != prev:                                # dedup runs of identical text
            parts.append(s)
            prev = s
    return ". ".join(parts)
```

### 7.4 Per-window sample shape

Each training sample is a 5-tuple:

```python
@dataclass
class WindowSample:
    hist:  np.ndarray   # [15, 1260]   float32
    text:  str          # aggregated event text for this window
    y_h5:  np.ndarray   # [1260]
    y_h10: np.ndarray   # [1260]
    y_h15: np.ndarray   # [1260]
```

The adjacency `ADJ` is **not** per-sample — it lives on the model.

### 7.5 Reference: extended windowing with text

```python
def make_windows_with_text(speed, text_dict, prefix, t_end_range):
    """
    Yields WindowSample objects for t_end in [t_end_range[0], t_end_range[1]].
    """
    t_lo, t_hi = t_end_range
    for t_end in range(t_lo, t_hi + 1):
        hist  = speed[t_end - 14 : t_end + 1].copy()      # [15, 1260]
        y_h5  = speed[t_end + 5].copy()                   # [1260]
        y_h10 = speed[t_end + 10].copy()                  # [1260]
        y_h15 = speed[t_end + 15].copy()                  # [1260]
        text  = window_text(text_dict, prefix, t_end)     # str
        yield WindowSample(hist, text, y_h5, y_h10, y_h15)
```

For test, the equivalent is just:
```python
test_hist = np.load("dataset-task1/test/test_X_hist.npy")     # [540, 15, 1260]
with open("dataset-task1/test/test_texts.json") as f:
    test_texts = json.load(f)                                 # {test_00000: "...", ...}
# sample i:  hist = test_hist[i],  text = test_texts[f"test_{i:05d}"]
```

---

## 8. Test-set inference

Test windows are **already pre-cut** in `test_X_hist.npy` with shape
`[540, 15, 1260]`. No windowing is needed at inference time:

```python
test_hist = np.load("dataset-task1/test/test_X_hist.npy")   # [540, 15, 1260]
preds = model(test_hist)   # {h5: [540,1260], h10: ..., h15: ...}
```

For submission, flatten to 3 horizons × 1,260 roads per sample in the exact id
order of `sample_submission.csv`:
`test_{sample:05d}_h{5|10|15}_r{0..1259}`.

---

## 9. Data-quality caveats

1. **m2 has ~17% zeros** (vs ~1% in m1). These are missing-data masks, not real
   traffic. Decide upfront:
   - drop windows whose targets contain zeros,
   - mask zeros in the loss, or
   - impute (forward-fill / per-road mean from m1) before windowing.

2. **Test has ~5.9% zeros**, and 4 roads are always zero in test. You still must
   produce predictions for those roads — fill with per-road mean (or speed limit
   if available). Don't drop them.

3. **Never concatenate m1 and m2** before windowing. Window each separately and
   pool the resulting samples.

4. **Event text differs structurally between train and test**:
   - Train: 15 per-step text strings per window (one per timestep).
   - Test:  1 aggregated text string per sample.

   To keep the text encoder identical, aggregate train to one string per window
   (concatenate per-step texts, optionally dedup adjacent duplicates).

---

## 10. Road metadata: `Roads1260.json`

### 10.1 Structure

Top-level: a list of **1,260 entries** (one per road column). Each entry is a
list of **sub-segments** that compose that road. So `roads[j]` = full geometry
of the road at speed column `j`.

| Field | Meaning |
|---|---|
| `coordList` | Flat `[lng, lat, lng, lat, ...]` polyline of the segment's geometry |
| `length`    | Segment length in meters (5–3198 m, mean ~173 m) |
| `roadName`  | Chinese name (e.g. `大瓦窑桥`, `S50东五环`, `南二环`) |
| `roadclass` | Road category {0,1,2,3,6} — likely expressway/arterial/local |
| `formway`   | Road form {1,3,4,6,7,8,9,10,11,15} — physical type |
| `linkId`    | Original map provider ID |

Per road column (j = 0..1259): on average **5.65 sub-segments**, max 36.

### 10.2 What this unlocks

**(a) Static per-road features** (global, indexed by `j`). From each `roads[j]`:
- total `length` of road `j` (sum across sub-segments)
- `roadclass` / `formway` (categorical, mode across sub-segments)
- centroid lat/lng (from `coordList`)
- bearing / direction (from polyline endpoints)
- number of sub-segments (complexity proxy)

Builds a `[1260, F]` static feature matrix, attached to the model like `ADJ`.
Does not depend on `t_end`.

**(b) Linking event text → specific road indices.** The event text uses English
translations/transliterations of the Chinese road names in `Roads1260.json`. So
`"wufang bridge"` ↔ `望和桥` ↔ roads [16, 17, 18, 19, 20]. With a name lookup,
each event phrase can be attributed to specific road columns.

---

## 11. Translating Chinese road names to Latin

### 11.1 Data inspection findings

**191 unique Chinese names** in `Roads1260.json`. The names are **almost
entirely official / structural nomenclature** — not colloquial nicknames. Their
suffixes follow a small set of patterns:

| Chinese suffix | Meaning   | Count | Example |
|---|---|---|---|
| 路             | road      | 78    | 京良路, 庑殿路, 复兴路 |
| 桥             | bridge (interchange) | 55 | 望和桥, 四惠桥, 大瓦窑桥 |
| 入口 / 出口    | entry / exit | 22  | 京开高速入口, S50北五环出口 |
| 大街           | avenue    | 13    | 朝阳门外大街, 建国门外大街 |
| 高速           | expressway | 13   | G4京港澳高速, S12机场高速 |
| 环             | ring (alone) | 8   | S50东五环, 东二环 |
| other          | —         | 2     | 加油站 (gas station) |

Because the names are structural, **rule-based translation is feasible** — no
ML or paid API needed.

### 11.2 The English text is hybrid (translation + transliteration)

The 174 unique English road phrases in the event text are **NOT pure pinyin**
and **NOT pure translation**:

- **~63 are translated (semantic)**:
  `east fourth ring middle road`, `g4 beijing hong kong macao expressway`,
  `107 national highway`, `entrance of jingkai expressway`.
- **~111 are pinyin (transliterated)**:
  `wufang bridge`, `dujiakan bridge`, `fuxing road`, `chaoyangmenwai street`.

So `大瓦窑桥 → dawayao bridge` (pinyin + translated suffix), while
`西四环北路 → west fourth ring north road` (fully translated). Strategy depends
on which token is being processed.

### 11.3 Translation strategy: hybrid, three rules

#### Rule A — Tokenize the Chinese name structurally

Every name decomposes into a prefix + suffix:

| Chinese | Decomp | English |
|---|---|---|
| 京良**路** | 京良 + 路 | jingliang + road |
| 望和**桥** | 望和 + 桥 | wanghe + bridge |
| 朝阳门**外大街** | 朝阳门外 + 大街 | chaoyangmenwai + street |
| 东**二环** | 东 + 二环 | east + second ring |
| S50东五环 | S50 + 东 + 五环 | S50 + east + fifth ring |
| 京开高速**入口** | 京开高速 + 入口 | jingkai expressway + entrance |

#### Rule B — Translate structural tokens by dictionary

Small lookup tables cover all 191 names:

```python
SUFFIX_DICT = {
    '路':       'road',
    '桥':       'bridge',
    '大街':     'street',
    '高速':     'expressway',
    '入口':     'entrance',
    '出口':     'exit',
    '辅路':     'auxiliary road',
    '环':       'ring',
    '二环':     'second ring',
    '三环':     'third ring',
    '四环':     'fourth ring',
    '五环':     'fifth ring',
    '北':       'north',
    '南':       'south',
    '东':       'east',
    '西':       'west',
    '中':       'middle',
    '东路':     'east road',
    '西路':     'west road',
    '南路':     'south road',
    '北路':     'north road',
    '中路':     'middle road',
    '内':       'inner',
    '外':       'outer',
}

# Special full-name translations for expressways (translated, not transliterated)
EXPRESSWAY_DICT = {
    '京哈高速':   'beijing harbin expressway',
    '京沪高速':   'beijing shanghai expressway',
    '京台高速':   'beijing taiwan expressway',
    '京港澳高速': 'beijing hong kong macao expressway',
    '京藏高速':   'beijing tibet expressway',
    '京新高速':   'beijing urumqi expressway',
    '京承高速':   'beijing chengde expressway',
    '机场高速':   'airport expressway',
    '京津高速':   'beijing tianjin expressway',
    '京通快速路': 'beijing tongzhou expressway',
    '京开高速':   'jingkai expressway',
    '京雄高速':   'jingxiong expressway',
    '大兴机场高速': 'daxing airport expressway',
    '机场第二高速': 'airport second expressway',
}
```

#### Rule C — Pinyin-transliterate the proper-name part

For proper-name parts (望和, 大瓦窑, 庑殿, 朝阳门外), use `pypinyin`:

```bash
uv add pypinyin
```

```python
from pypinyin import lazy_pinyin, Style

def to_pinyin(han: str) -> str:
    # 朝阳门外 -> chaoyangmenwai ; 望和 -> wanghe ; 大瓦窑 -> dawayao
    return ''.join(lazy_pinyin(han, style=Style.NORMAL))
```

### 11.4 Full algorithm

```python
def cn_to_en(name: str) -> str:
    # 1. Expressway full-name lookup (highest priority)
    for cn, en in EXPRESSWAY_DICT.items():
        if name.startswith(cn):
            rest = name[len(cn):].strip()
            if not rest:
                return en
            return f'{en} {SUFFIX_DICT.get(rest, to_pinyin(rest))}'.strip()

    # 2. Find longest structural suffix match
    for suffix_len in (4, 3, 2, 1):
        suffix = name[-suffix_len:]
        if suffix in SUFFIX_DICT:
            head = name[:-suffix_len]
            en_head = to_pinyin(head) if head else ''
            en_suffix = SUFFIX_DICT[suffix]
            return f'{en_head} {en_suffix}'.strip()

    # 3. Fallback: pure pinyin
    return to_pinyin(name)
```

### 11.5 Why this works here

- **No nicknames** to worry about (data is structural / administrative).
- **Only 191 unique names** → manually verify/fix the table offline and ship it
  as a static JSON. No runtime translation cost.
- **174 English phrases in text** → only 174 successful matches needed for
  per-road event attribution, not 1260.
- Once you have `english_phrase → road_indices`, parse each event text in
  O(events × phrase_length) and attribute closures/accidents to specific columns.

### 11.6 Recommended workflow

1. `uv add pypinyin`.
2. Build mapping script once: for each of 191 Chinese names, run `cn_to_en`,
   store as `name_map.json` (`{chinese_name: english_phrase}`).
3. Manually inspect/patch any mismatches against the 174 English phrases in the
   text — a few dozen at most.
4. Invert mapping: `english_phrase → [road_indices]`.
5. At training/inference: parse text → split by `.` → match each event's road
   phrase → set per-road event flags/embeddings.

### 11.7 Coverage check (do this before committing to the approach)

Before writing the full pipeline, run a quick coverage test:
- For each of the 191 Chinese names, generate the candidate English phrase.
- Check whether ANY English phrase in the event text starts with / equals it.
- Report the match count — that is the ceiling on per-road event attribution.

This tells you exactly how many of the 1,260 road columns can receive direct
event signal from the text (vs. only indirect signal via the adjacency graph).

**Result (verified end-to-end):**
- 185 / 191 road names match the event text exactly.
- The remaining 6 are roads that simply never appear in any event text
  (e.g. 京福路, 旧桥路) — they're harmless dead-weight rules.
- **100% event coverage**: all 320,819 train events + 16,497 test events map
  to specific road indices via `inverse_map.json`.
- Linguistic note: the event text uses `zhaoyang` for 朝阳路 / 朝阳门桥
  (wrong reading — should be *cháo*yáng), but `chaoyang` for 朝阳门外大街.
  Both forms are preserved as-is to match the text.

---

## 12. Model specification

### 12.1 Chosen architecture: **Graph WaveNet** (Wu et al., IJCAI 2019)

Rationale:
- Within ~5-10% of true SOTA on METR-LA / PEMS-BAY benchmarks.
- Simple, well-tested, trains in ~1 hour on RTX 5070 Ti for our dataset size.
- Supports both fixed adjacency (`matrix.npy`) and a learned adaptive
  adjacency (source/target node embeddings).
- Stacked gated temporal convolutions + graph convolution layers.
- Output is multi-horizon in a single forward pass (matches our h5/h10/h15
  requirement).

Future upgrade path (not for first iteration):
- **AGCRN** (no graph needed, learns per-node patterns).
- **STAEformer** (transformer + spatio-temporal adaptive embeddings, SOTA).

### 12.2 Tensor shapes (contract for `dataset.py`)

| Tensor | Shape | Source |
|---|---|---|
| `speed_hist`  | `[B, 1260, 15, 1]` | `test_X_hist.npy` / sliced train windows |
| `adj_fixed`   | `[1260, 1260]`     | `matrix.npy` (binary, directed, self-loops stripped) |
| `node_feat`   | `[1260, F_static]` | `Roads1260.json` (length, roadclass, formway, lat, lng, n_segs) |
| `event_feat`  | `[B, 1260, E_event]` | parsed from text via `inverse_map.json` |
| `targets`     | `[B, 1260, 3]`     | h5, h10, h15 speeds |
| `target_mask` | `[B, 1260, 3]`     | 1 where target is valid (non-zero in m2's missing-data case), 0 elsewhere |

Where:
- `B` = batch size (typical 16-64).
- `F_static ≈ 8-12` (length, roadclass one-hot ~5, formway one-hot ~5, lat, lng).
- `E_event ≈ 6-10` (one-hot event-type counts × intensity per road).

### 12.3 Static per-road features (`node_feat`)

Extracted once from `Roads1260.json`, indexed by road column `j`:

| Feature | Type | Source field |
|---|---|---|
| `length_total` | float | sum of `length` across sub-segments |
| `n_segments` | int | `len(roads[j])` |
| `roadclass` | one-hot(5) | mode of `roadclass` across sub-segments |
| `formway` | one-hot(top-5) | mode of `formway` across sub-segments |
| `centroid_lng` | float | mean of lng coords across sub-segments |
| `centroid_lat` | float | mean of lat coords across sub-segments |

Final shape: `[1260, ~13]`, then optionally MLP-projected to model dim.

### 12.4 Per-road event features (`event_feat`)

Built per-sample by parsing the aggregated text via `inverse_map.json`:

```python
EVENT_TYPES = [
    "road closure", "construction", "a general traffic accident",
    "road traffic control", "prohibit left turn", "an announcement",
    "road obstruction", "a broken down vehicle",
]

def build_event_feat(text: str, inv_map: dict, n_roads=1260) -> np.ndarray:
    """Returns [n_roads, len(EVENT_TYPES)] binary event mask."""
    feat = np.zeros((n_roads, len(EVENT_TYPES)), dtype=np.float32)
    for sent in text.split("."):
        s = sent.strip()
        if not s: continue
        m = EVENT_PATTERN.match(s)
        if not m: continue
        etype = m.group(1).strip().lower()
        phrase = m.group(2).strip().lower()
        roads = inv_map.get(phrase, [])
        # match etype against canonical event types (substring ok for compound events)
        for i, canon in enumerate(EVENT_TYPES):
            if canon in etype:
                feat[roads, i] = 1.0
    return feat
```

This produces `[B, 1260, 8]` per batch.

### 12.5 Loss: masked MSE

The test set has ~6% zero readings (missing data), and m2's training windows have
up to ~17% zeros. Predictions on those positions should not contribute to the
loss.

```python
def masked_mse(pred, target, mask):
    # pred, target, mask: [B, n_roads, n_horizons]
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp(min=1)
```

At inference, always-zero roads (4 in test) are filled with per-road mean.

### 12.6 Iteration plan

**Device split:**
- **This machine (dev box)**: data prep, baselines (CPU-only), validation
  plots, submission generation, debugging.
- **Training box (separate)**: Graph WaveNet training on GPU. Code must be
  self-contained and runnable there without manual setup beyond `uv sync`.

Implications:
- `dataset.py`, `model.py`, `train.py` must be portable (no hardcoded paths
  outside the repo).
- Pre-computed artifacts (`name_map.json`, `inverse_map.json`, static feature
  matrix) are committed so the training box doesn't need to rebuild them.
- Baselines (`baselines.py`) run on this machine to give the MSE floor before
  any GPU time is spent.

**Iteration 1 (minimal baseline, ~15 min, CPU on dev box):**
1. **Last-value persistence**: `pred = hist[:, -1, :]` for all 3 horizons.
   - Establishes the MSE floor.
2. **Per-road linear**: `y_h = W_h · flatten(hist)` per road, three separate
   heads. Trains in seconds on CPU.

**Iteration 2 (Graph WaveNet, ~1 hour on training box GPU):**
- Train on pooled (m1-train, m2-train) windows.
- Validate on (m1-val, m2-val) windows.
- Fixed + adaptive adjacency.
- Static features + per-road event features.
- Target: beat Iteration 1's MSE by ≥20%.

**Iteration 3 (push the leaderboard, optional):**
- AGCRN ensemble, or
- STAEformer single model.

### 12.7 Submission format

Final inference produces `[540, 1260, 3]` (samples × roads × horizons),
flattened to match `sample_submission.csv` id order
`test_{sample:05d}_h{5|10|15}_r{0..1259}`.

```python
# samples, horizons, roads -> long DataFrame
preds_hw = preds  # [540, 1260, 3]
for s in range(540):
    for h_idx, h in enumerate([5, 10, 15]):
        for r in range(1260):
            row_id = f"test_{s:05d}_h{h}_r{r}"
            submission[row_id] = preds_hw[s, r, h_idx]
```
