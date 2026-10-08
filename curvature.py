"""Road-center curvature and approximate radius estimation.

The previous implementation reported a normalized polynomial curvature
value whose magnitude was not directly interpretable as a road radius.
This version fits the road center in coordinates normalized by frame
height and converts curvature to an approximate image-space radius.
"""
import numpy as np

from config import (
    CURVATURE_Y_MIN,
    CURVATURE_Y_MAX,
    CURVATURE_SAMPLES,
    MIN_VALID_RADIUS_PX,
    MAX_VALID_RADIUS_PX,
)


def fit_polynomial(points):
    if points is None or len(points) < 3:
        return None
    points = np.asarray(points, dtype=np.float64)
    x = points[:, 0]
    y = points[:, 1]
    try:
        return np.polyfit(y, x, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None


def evaluate_polynomial(coefficients, y):
    if coefficients is None:
        return None
    a, b, c = coefficients
    return a * y ** 2 + b * y + c


def _center_points(left_points, right_points, frame_height):
    if left_points is None or right_points is None:
        return None

    left = np.asarray(left_points, dtype=np.float64)
    right = np.asarray(right_points, dtype=np.float64)

    if len(left) < 3 or len(right) < 3:
        return None

    # Fit each side first so the center is not biased by different
    # numbers of Hough segments on the two sides.
    left_coeff = fit_polynomial(left)
    right_coeff = fit_polynomial(right)
    if left_coeff is None or right_coeff is None:
        return None

    y_px = np.linspace(
        frame_height * CURVATURE_Y_MIN,
        frame_height * CURVATURE_Y_MAX,
        CURVATURE_SAMPLES,
    )

    left_x = evaluate_polynomial(left_coeff, y_px)
    right_x = evaluate_polynomial(right_coeff, y_px)
    center_x = (left_x + right_x) / 2.0

    # Normalize both coordinates by frame height. This keeps the
    # curvature numerically stable across resolutions.
    y = y_px / float(frame_height)
    x = center_x / float(frame_height)

    try:
        center_coeff = np.polyfit(y, x, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None

    return center_coeff, left_coeff, right_coeff


def calculate_polynomial_curvature(left_points, right_points, frame_width, frame_height):
    """Return (radius_px, direction, left_coeff, right_coeff)."""
    result = _center_points(left_points, right_points, frame_height)
    if result is None:
        return None, None, None, None

    center_coeff, left_coeff, right_coeff = result
    a, b, _ = center_coeff

    y_values = np.linspace(
        CURVATURE_Y_MIN,
        CURVATURE_Y_MAX,
        CURVATURE_SAMPLES,
    )

    radii = []
    signed_curvatures = []

    for y in y_values:
        first = 2.0 * a * y + b
        second = 2.0 * a
        denominator = (1.0 + first ** 2) ** 1.5
        if denominator <= 1e-12:
            continue
        signed_kappa = second / denominator
        kappa = abs(signed_kappa)
        if kappa > 1e-9:
            radii.append(1.0 / kappa)
            signed_curvatures.append(signed_kappa)

    if not radii:
        # A nearly perfectly straight fitted centerline is effectively
        # infinite radius rather than a failed detection.
        return MAX_VALID_RADIUS_PX, "STRAIGHT", left_coeff, right_coeff

    radius_normalized = float(np.median(radii))
    radius_px = radius_normalized * frame_height
    radius_px = float(np.clip(radius_px, MIN_VALID_RADIUS_PX, MAX_VALID_RADIUS_PX))

    # With image coordinates increasing downward, the sign is opposite
    # the usual Cartesian y-axis convention.
    median_signed = float(np.median(signed_curvatures))
    direction = "LEFT" if median_signed < 0 else "RIGHT"

    return radius_px, direction, left_coeff, right_coeff


def get_road_center(left_line, right_line):
    if left_line is None or right_line is None:
        return None
    cx_bottom = (left_line[0] + right_line[0]) / 2.0
    cx_top = (left_line[2] + right_line[2]) / 2.0
    return (cx_bottom, left_line[1]), (cx_top, left_line[3])


def calculate_curvature(left_line, right_line, frame_width, frame_height):
    """Legacy straight-line metric retained for compatibility."""
    center = get_road_center(left_line, right_line)
    if center is None:
        return None, None
    (cx_b, y_b), (cx_t, y_t) = center
    vertical_span = abs(y_b - y_t)
    if vertical_span < 1:
        return None, None
    shift = cx_t - cx_b
    metric = (abs(shift) / frame_width) / (vertical_span / frame_height)
    direction = "RIGHT" if shift > 0 else "LEFT"
    return float(metric), direction
