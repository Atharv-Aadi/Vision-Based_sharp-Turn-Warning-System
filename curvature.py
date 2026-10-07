import numpy as np

from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def fit_polynomial_ransac(points):
    """
    Robustly fit:

        x = a*y^2 + b*y + c

    using RANSAC to reject outlier points.
    """

    if points is None or len(points) < 8:
        return None

    points = np.asarray(points, dtype=np.float64)

    x = points[:, 0]
    y = points[:, 1]

    y_min = np.min(y)
    y_max = np.max(y)

    if y_max - y_min < 30:
        return None

    # Normalize Y for numerical stability
    y_center = (y_min + y_max) / 2.0
    y_scale = (y_max - y_min) / 2.0

    if y_scale < 1e-9:
        return None

    t = (y - y_center) / y_scale

    # Polynomial features:
    # t -> [t, t²]
    X = np.column_stack([
        t,
        t ** 2
    ])

    try:
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=12.0,
            min_samples=0.5,
            max_trials=100,
            random_state=42
        )

        ransac.fit(X, x)

    except (ValueError, np.linalg.LinAlgError):
        return None

    # Which points were accepted?
    inlier_mask = ransac.inlier_mask_

    if inlier_mask is None:
        return None

    if np.sum(inlier_mask) < 5:
        return None

    # Refit a clean polynomial using ONLY RANSAC inliers
    t_inliers = t[inlier_mask]
    x_inliers = x[inlier_mask]

    try:
        quadratic_norm = np.polyfit(
            t_inliers,
            x_inliers,
            2
        )
    except (np.linalg.LinAlgError, ValueError):
        return None

    A, B, C = quadratic_norm

    # Convert back to original image Y coordinates
    a = A / (y_scale ** 2)

    b = (
        B / y_scale
        - (2.0 * A * y_center) / (y_scale ** 2)
    )

    c = (
        A * (y_center ** 2) / (y_scale ** 2)
        - B * y_center / y_scale
        + C
    )

    return {
        "quadratic": np.array([a, b, c]),
        "y_min": y_min,
        "y_max": y_max,
        "inlier_mask": inlier_mask,
        "inlier_count": int(np.sum(inlier_mask)),
        "outlier_count": int(np.sum(~inlier_mask))
    }


def fit_polynomial(points):
    """
    Fit:

        x = a*y^2 + b*y + c

    using the detected lane points.

    Normalization is used internally for numerical stability,
    but the returned coefficients are converted back to the
    original image Y coordinate so main.py can draw them.
    """

    if points is None or len(points) < 5:
        return None

    points = np.asarray(
        points,
        dtype=np.float64
    )

    x = points[:, 0]
    y = points[:, 1]

    y_min = np.min(y)
    y_max = np.max(y)

    if y_max - y_min < 30:
        return None

    # Normalize Y internally
    y_center = (y_min + y_max) / 2.0
    y_scale = (y_max - y_min) / 2.0

    if y_scale < 1e-9:
        return None

    t = (
        y - y_center
    ) / y_scale

    try:
        quadratic_norm = np.polyfit(
            t,
            x,
            2
        )

        linear_norm = np.polyfit(
            t,
            x,
            1
        )

    except (np.linalg.LinAlgError, ValueError):
        return None

    # --------------------------------------------------
    # Convert normalized quadratic back to:
    #
    # x = a*y^2 + b*y + c
    # --------------------------------------------------

    A, B, C = quadratic_norm

    a = A / (y_scale ** 2)

    b = (
        B / y_scale
        - (2.0 * A * y_center)
        / (y_scale ** 2)
    )

    c = (
        A * (y_center ** 2)
        / (y_scale ** 2)
        - B * y_center
        / y_scale
        + C
    )

    quadratic = np.array([
        a,
        b,
        c
    ])

    # Linear model in original Y coordinates
    linear_pred = np.polyval(
        linear_norm,
        t
    )

    quadratic_pred = np.polyval(
        quadratic_norm,
        t
    )

    linear_rmse = np.sqrt(
        np.mean(
            (x - linear_pred) ** 2
        )
    )

    quadratic_rmse = np.sqrt(
        np.mean(
            (x - quadratic_pred) ** 2
        )
    )

    if linear_rmse > 1e-9:
        improvement = (
            linear_rmse
            - quadratic_rmse
        ) / linear_rmse
    else:
        improvement = 0.0

    return {
        "quadratic": quadratic,
        "y_min": y_min,
        "y_max": y_max,
        "linear_rmse": linear_rmse,
        "quadratic_rmse": quadratic_rmse,
        "improvement": improvement
    }


def evaluate_polynomial(
    model,
    y
):
    """Evaluate x = ay² + by + c."""

    if model is None:
        return None

    a, b, c = model["quadratic"]

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
    Calculate curvature from the ROAD CENTERLINE.

    Left and right lane boundaries are first fitted separately.
    Their midpoint is then used to construct the road centerline.

    The centerline is compared against a linear model.

    If the centerline is effectively straight:
        curvature = 0
        direction = None
    """

    # --------------------------------------------------
    # Basic validation
    # --------------------------------------------------

    if (
        left_points is None
        or right_points is None
    ):
        return None, None, None, None

    # --------------------------------------------------
    # Fit left and right boundaries
    # --------------------------------------------------

    left_model = fit_polynomial_ransac(
        left_points
    )

    right_model = fit_polynomial_ransac(
        right_points
    )

    if (
        left_model is None
        or right_model is None
    ):
        return None, None, None, None

    # --------------------------------------------------
    # Find common Y region
    # --------------------------------------------------

    y_start = max(
        left_model["y_min"],
        right_model["y_min"]
    )

    y_end = min(
        left_model["y_max"],
        right_model["y_max"]
    )

    if y_end - y_start < 50:
        return None, None, None, None

    # Ignore unstable extreme ends
    valid_start = (
        y_start
        + 0.10 * (y_end - y_start)
    )

    valid_end = (
        y_end
        - 0.10 * (y_end - y_start)
    )

    if valid_end <= valid_start:
        return None, None, None, None

    # --------------------------------------------------
    # Evaluate both boundaries at common Y positions
    # --------------------------------------------------

    y_positions = np.linspace(
        valid_start,
        valid_end,
        30
    )

    left_x = np.array([
        evaluate_polynomial(
            left_model,
            y
        )
        for y in y_positions
    ])

    right_x = np.array([
        evaluate_polynomial(
            right_model,
            y
        )
        for y in y_positions
    ])

    # --------------------------------------------------
    # ROAD CENTERLINE
    # --------------------------------------------------

    center_x = (
        left_x + right_x
    ) / 2.0

    # --------------------------------------------------
    # Fit centerline
    #
    # x = ay² + by + c
    # --------------------------------------------------

    try:
        center_quadratic = np.polyfit(
            y_positions,
            center_x,
            2
        )

        center_linear = np.polyfit(
            y_positions,
            center_x,
            1
        )

    except (np.linalg.LinAlgError, ValueError):
        return None, None, None, None

    # --------------------------------------------------
    # Compare curved model vs straight model
    # --------------------------------------------------

    quadratic_prediction = np.polyval(
        center_quadratic,
        y_positions
    )

    linear_prediction = np.polyval(
        center_linear,
        y_positions
    )

    quadratic_rmse = np.sqrt(
        np.mean(
            (center_x - quadratic_prediction) ** 2
        )
    )

    linear_rmse = np.sqrt(
        np.mean(
            (center_x - linear_prediction) ** 2
        )
    )

    if linear_rmse > 1e-9:
        improvement = (
            linear_rmse
            - quadratic_rmse
        ) / linear_rmse
    else:
        improvement = 0.0

    # --------------------------------------------------
    # How far does the centerline actually bend?
    # --------------------------------------------------

    center_deviation = np.max(
        np.abs(
            quadratic_prediction
            - linear_prediction
        )
    )

    normalized_deviation = (
        center_deviation
        / frame_width
    )

    # --------------------------------------------------
    # STRAIGHT ROAD TEST
    # --------------------------------------------------

    MIN_IMPROVEMENT = 0.05

    is_curved = (
        improvement >= MIN_IMPROVEMENT
    )

    if not is_curved:
        return (
            0.0,
            None,
            left_model["quadratic"],
            right_model["quadratic"]
        )

    # --------------------------------------------------
    # Curvature of centerline
    # --------------------------------------------------

    a, b, c = center_quadratic

    curvature_values = []

    for y in y_positions:

        dx_dy = (
            2.0 * a * y
            + b
        )

        d2x_dy2 = (
            2.0 * a
        )

        denominator = (
            1.0 + dx_dy**2
        ) ** 1.5

        if denominator < 1e-9:
            continue

        kappa = (
            abs(d2x_dy2)
            / denominator
        )

        if np.isfinite(kappa):
            curvature_values.append(
                kappa
            )

    if not curvature_values:
        return (
            0.0,
            None,
            left_model["quadratic"],
            right_model["quadratic"]
        )

    # Median is more robust than maximum.
    raw_curvature = np.median(
        curvature_values
    )

    normalized_curvature = (
        raw_curvature
        * frame_height
    )

    # --------------------------------------------------
    # TURN DIRECTION
    # --------------------------------------------------

    y_far = valid_start
    y_near = valid_end

    center_far = np.polyval(
        center_quadratic,
        y_far
    )

    center_near = np.polyval(
        center_quadratic,
        y_near
    )

    center_shift = (
        center_far
        - center_near
    )

    DIRECTION_THRESHOLD = (
        frame_width * 0.02
    )

    if center_shift > DIRECTION_THRESHOLD:
        direction = "RIGHT"

    elif center_shift < -DIRECTION_THRESHOLD:
        direction = "LEFT"

    else:
        direction = None

    return (
        float(normalized_curvature),
        direction,
        left_model["quadratic"],
        right_model["quadratic"]
    )


# =========================================================
# Original baseline functions
# =========================================================

def get_road_center(
    left_line,
    right_line
):
    """Calculate road center from straight lines."""

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
    """Original center-shift metric."""

    if (
        left_line is None
        or right_line is None
    ):
        return None, None

    (
        (cx_b, y_b),
        (cx_t, y_t)
    ) = get_road_center(
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

    return (
        float(metric),
        direction
    )
