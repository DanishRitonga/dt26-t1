# Overview

Traffic does not move at random. It responds to the road network, to the time of day, and to events unfolding across the city, accidents, closures, weather, gatherings, all described in a running stream of text.

Given one hour of recent speed history across 1,260 connected road segments, and a live feed of event text describing what is happening on the ground, you are challenged to forecast how fast traffic will be moving 20, 40, and 60 minutes into the future.

## Description

**What you get**
For each sample, you receive three sources of signal:

1. Speed history
A 15-step window of recent speed readings (one hour of history) for all 1,260 road segments.

2. Event text
A stream of natural-language event descriptions reporting what is happening across the network, the kind of context a human dispatcher would read before predicting how traffic will shift.

3. Road network
A 1,260 x 1,260 adjacency matrix describing how segments connect, plus road geometry and metadata for every segment.

**What you need to do**
Predict the traffic speed (in km/h) for every road segment at three horizons:

- +20 minutes (h5)
- +40 minutes (h10)
- +60 minutes (h15)

## Evaluation
Submissions are scored by Mean Squared Error (MSE) between your predicted speeds and the held-out ground truth, averaged across all road segments and all three horizons:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2$$

where 
 \hat{y}_i is your predicted speed, 
 y_i is the true speed, and 
 N is the total number of scored predictions (samples x horizons x roads).

## Submission Format
For every sample, predict the speed of all 1,260 roads at all 3 horizons. When creating submission.csv, use this specific id format with one prediction per row:
```
id,speed
test_00000_h5_r0,44.0
test_00000_h5_r1,73.0
...

```

The id format encodes sample, horizon, and road:
```
test_{sample}_h{horizon}_r{road}
  horizon in {5, 10, 15}   (= +20, +40, +60 minutes)
  road    in {0 ... 1259}
```

You can also see sample_submission.csv for the expected format.
