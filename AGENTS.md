# AGENTS.md

Compact guide for OpenCode sessions in this repo. Read `windowing.md` for the
full spec — this file only captures things an agent would otherwise get wrong.

## What this repo is

Traffic forecasting task: predict speed (km/h) on 1,260 Beijing road segments
at +20/+40/+60 min horizons (h5/h10/h15) from a 15-step history window + event
text + road graph. Scored by MSE. See `overview.md` and `data.md`.

## Environment

- Python 3.13, managed by `uv` (not pip / venv directly).
- All commands: `uv run python <script>.py`, NOT `python <script>.py`.
- `uv sync` installs everything; `uv add <pkg>` adds deps.
- Dev box has no CUDA — for CPU baselines and pipeline checks only.
- Training happens on a separate GPU box. Code must be portable (no hardcoded
  paths outside the repo, no machine-specific setup beyond `uv sync`).

## Dataset (NOT in git — re-download separately)

`dataset-task1/` is gitignored (~100 MB). Layout:

- `train/train_speed_m{1,2}_*.npy` — continuous speed series `[T, 1260]`
- `train/train_text_m{1,2}_*.json` — per-step event text, 1-indexed keys
- `test/test_X_hist.npy` — pre-cut windows `[540, 15, 1260]`
- `test/test_texts.json` — one aggregated text string per sample
- `static/matrix.npy` — directed adjacency `[1260, 1260]`, has self-loops
- `static/Roads1260.json` — per-road geometry + Chinese names
- `sample_submission.csv` — exact id order to preserve

## Critical conventions (these are easy to break)

- **1 step = 4 minutes** (15 steps = 1 hour). Horizons `h5/h10/h15` = `+5/+10/+15`
  steps past the last history step (= +20/+40/+60 min). Not +20/+40/+60 steps.
- **Targets are at steps 19, 24, 29** within a 30-step slice (history ends at 14).
- **m1 and m2 are NOT temporally aligned.** Different recording periods, different
  incidents. Never cross-window across the m1/m2 boundary.
- **Train/val split must be time-based**, not by window index, with a 30-step gap
  at the boundary. Random splits leak targets into train inputs. See `windowing.md` §5.
- **m2 has ~17% zeros** = missing-data masks, NOT real 0 km/h traffic. Use masked
  MSE; don't let the model learn "0 is a valid speed".
- **Train text is per-step (15 strings/window); test text is aggregated (1 string).**
  `dataset.window_block` aggregates train to match test.
- **Event text uses English** (transliterated + translated Chinese road names).
  Use `data/inverse_map.json` to map `"wufang bridge"` → road indices. Do not
  retranslate at runtime — `data/translate.py` produces the static map.

## Commands

```bash
uv sync                                      # install deps
uv run python dataset.py                     # sanity-check data pipeline
uv run python baselines.py                   # CPU baselines + val MSE
uv run python baselines.py --submit          # also write submission CSVs
uv run python model.py                       # smoke-test Graph WaveNet
uv run python train.py --epochs 50 --batch-size 32   # train on GPU box
uv run python inference.py --ckpt checkpoints/model_best.pt
```

No test framework, no linter, no formatter configured. Verification = run the
script and check output. Smoke tests are at the bottom of each module under
`if __name__ == "__main__":`.

## Tensor contract (Graph WaveNet)

Per window (train or test), the model expects:

| Tensor | Shape | Notes |
|---|---|---|
| `hist` | `[B, 15, 1260]` | speeds |
| `node_feat` | `[1260, F≈15]` | static per-road features (global) |
| `event_feat` | `[B, 1260, 8]` | per-road event mask from text |
| `adj` | `[1260, 1260]` | from `matrix.npy`, self-loops stripped |
| output | `[B, 1260, 3]` | h5/h10/h15 predictions |

Loss: `masked_mse_loss(preds, targets, mask)` from `model.py`. Targets/masks are
`[B, 3, 1260]` (horizon-major); the function handles the axis swap.

## Files

| File | Role |
|---|---|
| `windowing.md` | The spec. Read this first for any data question. |
| `dataset.py` | Windowing, splits, static + event features. Source of truth for tensor shapes. |
| `model.py` | Graph WaveNet (Wu et al. IJCAI 2019). |
| `train.py` | Training loop, checkpoints, logs. |
| `inference.py` | Checkpoint → submission CSV. Feeds BOTH hist and event_feat. |
| `baselines.py` | Persistence + per-road ridge. Floor to beat: val_mse = 37.70. |
| `data/translate.py` | Rebuilds Chinese↔English road name maps. |
| `data/inverse_map.json` | `english_phrase → [road_indices]`. Verified 100% text coverage. |
| `data/name_patches.json` | 88 manual translation overrides. Edit if a name mismatches. |

## Gitignore rules

`dataset-task1/`, `.env`, `submissions/`, `checkpoints/`, `logs/` are all
gitignored. Do NOT commit large binaries or per-machine artifacts. Commit only
source code, `windowing.md`, `data/*.json` + `data/translate.py`, and
`pyproject.toml` / `uv.lock`.

## Things to NOT do

- Don't assume m1/m2 share a start time. They don't.
- Don't add wall-clock features — no timestamps exist anywhere in the data.
- Don't trust `0` as a real speed reading (it's a missing-data mask).
- Don't concatenate m1 and m2 before windowing.
- Don't use random or k-fold splits — leakage. Time-based only.
- Don't translate Chinese names at runtime. Use the precomputed maps in `data/`.
- Don't emit predictions outside `[0, 200]` km/h. Clip.
- Don't skip the always-zero roads in submission — they still need a value
  (fill with per-road mean; `inference.fill_zeros_with_road_mean` does this).
