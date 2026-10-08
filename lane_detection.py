"""Level 1 road-boundary detection using filtered Hough segments."""
import cv2
import numpy as np

from config import (
    HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
    MIN_LINE_LENGTH_RATIO, MAX_LINE_GAP_RATIO,
    MIN_ABS_SLOPE, MAX_ABS_SLOPE,
    CENTER_MARGIN_RATIO, MIN_VERTICAL_SPAN_RATIO,
    MIN_SEGMENTS_PER_SIDE, MIN_SIDE_POINTS, LINE_TOP_RATIO,
)


def detect_lines(edges):
    h = edges.shape[0]
    lines = cv2.HoughLinesP(
        edges, HOUGH_RHO, HOUGH_THETA, HOUGH_THRESHOLD,
        minLineLength=max(5, int(h * MIN_LINE_LENGTH_RATIO)),
        maxLineGap=max(2, int(h * MAX_LINE_GAP_RATIO)),
    )
    if lines is None:
        return []
    lines = np.asarray(lines).reshape(-1, 4)
    return [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in lines]


def _features(line):
    x1, y1, x2, y2 = line
    dx, dy = x2 - x1, y2 - y1
    if dx == 0:
        return None
    return float(np.hypot(dx, dy)), dy / dx, abs(dy)


def _useful(line, frame_height):
    f = _features(line)
    if f is None:
        return False
    _, slope, vspan = f
    return (
        MIN_ABS_SLOPE <= abs(slope) <= MAX_ABS_SLOPE
        and vspan >= frame_height * MIN_VERTICAL_SPAN_RATIO
    )


def separate_lines(lines, frame_width, frame_height):
    left, right = [], []
    center = frame_width / 2.0
    margin = frame_width * CENTER_MARGIN_RATIO

    for line in lines:
        if not _useful(line, frame_height):
            continue
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1)
        mid_x = (x1 + x2) / 2.0

        if slope < 0 and mid_x < center + margin:
            left.append(line)
        elif slope > 0 and mid_x > center - margin:
            right.append(line)

    return left, right


def get_lane_points(lines, points_per_segment=10):
    if not lines:
        return None
    points = []
    for x1, y1, x2, y2 in lines:
        xs = np.linspace(x1, x2, points_per_segment)
        ys = np.linspace(y1, y2, points_per_segment)
        points.extend(zip(xs, ys))
    if len(points) < MIN_SIDE_POINTS:
        return None
    return np.asarray(points, dtype=np.float64)


def fit_lane_line(lines, frame_height):
    if len(lines) < MIN_SEGMENTS_PER_SIDE:
        return None

    xs, ys, weights = [], [], []
    for line in lines:
        x1, y1, x2, y2 = line
        length = float(np.hypot(x2 - x1, y2 - y1))
        xs.extend([x1, x2])
        ys.extend([y1, y2])
        weights.extend([length, length])

    xs, ys, weights = map(lambda a: np.asarray(a, dtype=float), (xs, ys, weights))
    if np.ptp(ys) < 1:
        return None

    try:
        a, b = np.polyfit(ys, xs, 1, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None

    y_bottom = frame_height - 1
    y_top = int(frame_height * LINE_TOP_RATIO)
    return (float(a * y_bottom + b), float(y_bottom),
            float(a * y_top + b), float(y_top))
