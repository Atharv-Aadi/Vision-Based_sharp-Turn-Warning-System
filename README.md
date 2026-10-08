# Road Curvature & Sharp Turn Warning — 50% Milestone

This folder contains the working 50% version of the road detection project.

## Implemented

### Level 1 — Stable road detection
- ROI-based road region
- Gaussian blur + Canny edges
- Morphological closing
- Hough line detection
- Left/right road-boundary filtering
- Weighted straight-line fitting

### Level 2 — Corrected curvature calculation
- Fits quadratic curves to both road boundaries
- Builds the road-center curve
- Calculates curvature over a look-ahead region
- Converts curvature into an approximate image-space radius
- Determines left/right curve direction

### Level 3 — Temporal smoothing
- Median filtering of recent radius values
- Exponential moving average
- Protection against sudden one-frame radius jumps
- Three-state classification with persistence/hysteresis

## On-screen output

The final window displays:

- `Road Status: STRAIGHT`
- `Road Status: MODERATE TURN`
- `Road Status: SHARP TURN`
- `Curve radius: XXX px`
- Turn direction when available
- `!! SHARP TURN AHEAD !!` during a sharp turn

## Important note about radius

The displayed radius is currently an **approximate image-space radius in pixels**, not a real-world radius in metres.

To obtain a physically meaningful value such as `394 m`, the project still needs camera calibration / perspective transformation and road-scale calibration. Those belong to later levels.

## Not included yet

These are intentionally left for the next stages:

- RANSAC outlier rejection
- Bird's-eye perspective transformation
- Real-world metre calibration
- Vehicle masking
- Advanced lane/road segmentation

## Run

Put your input video at the path configured in `config.py`, or pass a video directly:

```bash
python main.py "dataset/1st turn.mp4"
```

Final output only:

```bash
python main.py "dataset/1st turn.mp4" --no-debug
```

Save processed video:

```bash
python main.py "dataset/1st turn.mp4" --no-display --save results/output.mp4
```

Press `Esc` during display to stop processing.

## Main files

- `config.py` — all thresholds and parameters
- `preprocessing.py` — ROI and edge processing
- `lane_detection.py` — Hough detection and road-boundary filtering
- `curvature.py` — road-center curve and radius calculation
- `smoothing.py` — temporal radius smoothing
- `warning.py` — stable STRAIGHT/MODERATE/SHARP classification
- `vehicle_detection.py` — placeholder for a later vehicle-masking stage
- `main.py` — complete pipeline and display
