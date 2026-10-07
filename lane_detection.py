"""Hough detection, left/right separation, and line fitting."""
import cv2
import numpy as np

from config import (
    HOUGH_RHO,
    HOUGH_THETA,
    HOUGH_THRESHOLD,
    MIN_LINE_LENGTH_RATIO,
    MAX_LINE_GAP_RATIO,
    SLOPE_THRESHOLD,
    CENTER_MARGIN_RATIO,
    MIN_SEGMENTS_PER_SIDE,
    LINE_TOP_RATIO,
)


def detect_lines(edges):
    """Find line segments using HoughLinesP."""
    h = edges.shape[0]

    lines = cv2.HoughLinesP(
        edges,
        HOUGH_RHO,
        HOUGH_THETA,
        HOUGH_THRESHOLD,
        minLineLength=max(
            5,
            int(h * MIN_LINE_LENGTH_RATIO)
        ),
        maxLineGap=max(
            2,
            int(h * MAX_LINE_GAP_RATIO)
        )
    )

    if lines is None:
        return []

    lines = np.asarray(lines).reshape(-1, 4)

    return [
        (int(x1), int(y1), int(x2), int(y2))
        for x1, y1, x2, y2 in lines
    ]


def separate_lines(lines, frame_width):
    """Put detected segments into left and right groups."""
    left = []
    right = []

    center_x = frame_width / 2
    margin = frame_width * CENTER_MARGIN_RATIO

    for x1, y1, x2, y2 in lines:
        if x2 == x1:
            continue

        slope = (y2 - y1) / (x2 - x1)

        if abs(slope) <= SLOPE_THRESHOLD:
            continue

        mid_x = (x1 + x2) / 2

        if slope < 0 and mid_x < center_x + margin:
            left.append((x1, y1, x2, y2))

        elif slope > 0 and mid_x > center_x - margin:
            right.append((x1, y1, x2, y2))

    return left, right


def fit_lane_line(lines, frame_height):
    """Fit a single line to one side of the road."""
    if len(lines) < MIN_SEGMENTS_PER_SIDE:
        return None

    xs = []
    ys = []

    for x1, y1, x2, y2 in lines:
        xs += [x1, x2]
        ys += [y1, y2]

    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)

    if np.ptp(ys) < 1:
        return None

    # Preserve original algorithm: fit x against y with degree 1.
    a, b = np.polyfit(ys, xs, 1)

    y_bottom = frame_height - 1
    y_top = int(frame_height * LINE_TOP_RATIO)

    x_bottom = a * y_bottom + b
    x_top = a * y_top + b

    return (
        x_bottom,
        float(y_bottom),
        x_top,
        float(y_top)
    )
