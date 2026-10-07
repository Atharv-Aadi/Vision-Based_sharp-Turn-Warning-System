"""Road-center and original image-space curvature metric."""
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
