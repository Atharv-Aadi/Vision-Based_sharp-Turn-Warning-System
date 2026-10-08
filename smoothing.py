"""Temporal smoothing for curvature/radius values."""
from collections import deque
import numpy as np

from config import (
    SMOOTHING_WINDOW,
    MEDIAN_WINDOW,
    EMA_ALPHA,
    MAX_RADIUS_JUMP_RATIO,
)


class TemporalSmoother:
    """Median + bounded EMA smoothing for frame-to-frame radius."""

    def __init__(self, window_size=SMOOTHING_WINDOW, median_window=MEDIAN_WINDOW):
        self.window_size = max(1, int(window_size))
        self.median_window = max(1, min(int(median_window), self.window_size))
        self.history = deque(maxlen=self.window_size)
        self.ema = None

    def update(self, radius):
        if radius is None:
            return self.get_smoothed_value()

        radius = float(radius)
        if not np.isfinite(radius):
            return self.get_smoothed_value()

        # Prevent a single bad frame from moving the output wildly.
        if self.ema is not None:
            limit = max(self.ema * MAX_RADIUS_JUMP_RATIO, 150.0)
            lower = self.ema - limit
            upper = self.ema + limit
            radius = float(np.clip(radius, lower, upper))

        self.history.append(radius)
        recent = list(self.history)[-self.median_window:]
        median = float(np.median(recent))

        if self.ema is None:
            self.ema = median
        else:
            self.ema = EMA_ALPHA * median + (1.0 - EMA_ALPHA) * self.ema

        return self.ema

    def get_smoothed_value(self):
        return self.ema

    def reset(self):
        self.history.clear()
        self.ema = None

    def get_history(self):
        return list(self.history)
