# Data Description

## Data
- 1,260 road segments, indexed 0 to 1259. Speeds are in km/h.
- Training data is given as continuous speed series; the test set is given as fixed history windows.
- Each history window spans 1 hour of consecutive readings.

## train/
| File | Description |
| -- | -- |
|train_speed_m1_*.npy | Continuous speed series, block 1. Rows = timesteps, columns = the 1,260 roads.|
|train_speed_m2_*.npy | Continuous speed series, block 2 (same roads).|
|train_text_m1_*.json | Event text aligned to the timesteps of block 1.|
|train_text_m2_*.json | Event text aligned to the timesteps of block 2.|

## test/
| File | Description |
| -- | -- |
| test_X_hist.npy | The test history windows: one recent-history window per sample, for all roads. |
| test_texts.json | Event text, one entry per test sample. | 
| sample_submission.csv | The exact id list to predict, pre-filled with 0.0. |

## static| /
| File | Description |
| -- | -- |
| matrix.npy | Road-network adjacency between the 1,260 segments.|
|Roads1260.json | Geometry and metadata per segment (coordinates, length, id fields, etc.).|
|active_mask.npy | Boolean mask relating the underlying spatial grid to the 1,260 active segments. |

## Submission format.

Predict a speed for every id listed in sample_submission.csv. Ids follow the pattern test_XXXXX_h{H}_r{R} (sample, horizon, road). Keep the id order intact.
