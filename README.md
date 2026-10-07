# Sharp Turn Warning — Refactored Baseline

This project is a direct modular refactor of the original prototype.

## Important

The algorithm has NOT been upgraded yet.

The following original behavior is intentionally preserved:

- ROI_TOP_RATIO = 0.40
- ROI polygon ratios
- Canny thresholds
- HoughLinesP settings
- slope-based left/right separation
- degree-1 `np.polyfit`
- image-space center-shift curvature metric
- 0.10 straight threshold
- 0.30 sharp-turn threshold
- debug windows
- command-line arguments
- one-third-FPS saved output
- 100 ms display delay

## Modules

- `main.py` — application flow, visualization, CLI
- `config.py` — all original constants
- `preprocessing.py` — ROI, polygon mask, grayscale, blur, Canny
- `lane_detection.py` — Hough detection, separation, line fitting
- `curvature.py` — original center-shift curvature metric
- `vehicle_detection.py` — placeholder only; YOLO was not in the supplied prototype
- `warning.py` — original classification logic

## Run

From this folder:

```bash
python main.py
```

Or:

```bash
python main.py "dataset/1st turn.mp4"
```

Options:

```bash
python main.py --no-debug
python main.py --no-display --save results/output.mp4
```

## Next development stage

Only after verifying that this refactored version behaves like the original should we add:

1. True curved polynomial fitting
2. YOLO vehicle masking
3. Temporal smoothing
4. AI turn prediction
5. Early-warning logic
6. Quantitative evaluation
