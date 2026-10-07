"""Road curvature estimation.

Uses quadratic polynomial fitting:

    x = a*y^2 + b*y + c

for the left and right road boundaries, then estimates
curvature and turn direction from the fitted road center.
"""

import numpy as np


def fit_polynomial(points):
    """Fit x = a*y^2 + b*y + c to detected lane points.

    Args:
        points: numpy array of shape (N, 2), containing [x, y].

    Returns:
        numpy array [a, b, c], or None if there are insufficient points.
    """

    if points is None or len(points) < 3:
        return None

    points = np.asarray(points, dtype=np.float32)

    x = points[:, 0]
    y = points[:, 1]

    # Quadratic polynomial:
    # x = a*y^2 + b*y + c
    try:
        coefficients = np.polyfit(y, x, 2)
    except (np.linalg.LinAlgError, ValueError):
        return None

    return coefficients


def evaluate_polynomial(coefficients, y):
    """Evaluate x = a*y^2 + b*y + c at y."""

    if coefficients is None:
        return None

    a, b, c = coefficients

    return (
        a * y**2
        + b * y
        + c
    )


def calculate_polynomial_curvature(
    left_points,
    right_points,
    frame_width,
    frame_height
):
    """
    Fit quadratic road boundaries and calculate
    image-space road curvature.

    Polynomial:

        x = a*y^2 + b*y + c
    """

    if (
        left_points is None
        or right_points is None
    ):
        return None, None, None, None

    left_coeff = fit_polynomial(
        left_points
    )

    right_coeff = fit_polynomial(
        right_points
    )

    if (
        left_coeff is None
        or right_coeff is None
    ):
        return None, None, None, None

    # --------------------------------------------------
    # Road center polynomial
    # --------------------------------------------------

    left_a, left_b, left_c = left_coeff
    right_a, right_b, right_c = right_coeff

    center_coeff = np.array([
        (left_a + right_a) / 2.0,
        (left_b + right_b) / 2.0,
        (left_c + right_c) / 2.0
    ])

    center_a, center_b, center_c = center_coeff

    # --------------------------------------------------
    # Calculate curvature at multiple look-ahead points
    # --------------------------------------------------

    y_positions = np.linspace(
        frame_height * 0.45,
        frame_height * 0.75,
        10
    )

    curvature_values = []

    for y in y_positions:

        slope = (
            2.0 * center_a * y
            + center_b
        )

        second_derivative = (
            2.0 * center_a
        )

        denominator = (
            1.0 + slope ** 2
        ) ** 1.5

        if denominator < 1e-9:
            continue

        kappa = (
            abs(second_derivative)
            / denominator
        )

        curvature_values.append(kappa)

    if not curvature_values:
        return None, None, None, None

    raw_curvature = np.median(curvature_values)

    # Normalize by image height
    normalized_curvature = (
        raw_curvature * frame_height
    )

    # --------------------------------------------------
    # Direction
    # --------------------------------------------------

    if center_a > 0:
        direction = "LEFT"

    elif center_a < 0:
        direction = "RIGHT"

    else:
        direction = None

    return (
        float(normalized_curvature),
        direction,
        left_coeff,
        right_coeff
    )

# ---------------------------------------------------------
# Existing baseline method
# ---------------------------------------------------------

def get_road_center(left_line, right_line):
    """Calculate road center from the existing straight lines."""

    cx_bottom = (
        left_line[0]
        + right_line[0]
    ) / 2

    cx_top = (
        left_line[2]
        + right_line[2]
    ) / 2

    return (
        (cx_bottom, left_line[1]),
        (cx_top, left_line[3])
    )


def calculate_curvature(
    left_line,
    right_line,
    frame_width,
    frame_height
):
    """Original center-shift curvature metric.

    Kept as the baseline for comparison.
    """

    if (
        left_line is None
        or right_line is None
    ):
        return None, None

    (cx_b, y_b), (cx_t, y_t) = get_road_center(
        left_line,
        right_line
    )

    vertical_span = abs(
        y_b - y_t
    )

    if vertical_span < 1:
        return None, None

    shift = cx_t - cx_b

    metric = (
        (abs(shift) / frame_width)
        /
        (vertical_span / frame_height)
    )

    direction = (
        "RIGHT"
        if shift > 0
        else "LEFT"
    )

    return float(metric), direction



def get_road_center(left_line, right_line):
    cx_bottom = (
        left_line[0] +
        right_line[0]
    ) / 2

    cx_top = (
        left_line[2] +
        right_line[2]
    ) / 2

    return (
        (cx_bottom, left_line[1]),
        (cx_top, left_line[3])
    )


def calculate_curvature(
    left_line,
    right_line,
    frame_width,
    frame_height
):
    """
    Preserve the original approximate image-space curvature metric.

    It measures normalized horizontal center shift from the bottom
    of the frame toward the top.
    """
    if left_line is None or right_line is None:
        return None, None

    (cx_b, y_b), (cx_t, y_t) = get_road_center(
        left_line,
        right_line
    )

    vertical_span = abs(y_b - y_t)

    if vertical_span < 1:
        return None, None

    shift = cx_t - cx_b

    metric = (
        (abs(shift) / frame_width)
        /
        (vertical_span / frame_height)
    )

    direction = (
        "RIGHT"
        if shift > 0
        else "LEFT"
    )

    return float(metric), direction
