"""Vision-Based Road Curvature & Sharp Turn Warning System.

Refactored from the original prototype into modules.
The algorithm and parameter values are intentionally unchanged.
"""

import argparse
import sys

import cv2
import numpy as np

from config import (
    VIDEO_PATH,
    DEBUG,
    COLOR_LEFT,
    COLOR_RIGHT,
    COLOR_CENTER,
    COLOR_HOUGH,
    STATUS_COLORS,
)

from preprocessing import (
    get_roi,
    apply_polygon_mask,
    preprocess,
)

from lane_detection import (
    detect_lines,
    separate_lines,
    fit_lane_line,
    get_lane_points,
)

from curvature import (
    get_road_center,
    calculate_curvature,
    calculate_polynomial_curvature,
)

from warning import classify_turn
from vehicle_detection import VehicleDetector


def _draw_line(img, line, color, thickness=4):
    if line is None:
        return

    x_b, y_b, x_t, y_t = line

    cv2.line(
        img,
        (int(x_b), int(y_b)),
        (int(x_t), int(y_t)),
        color,
        thickness
    )

def _draw_polynomial(
    img,
    coefficients,
    y_start,
    y_end,
    color,
    thickness=4
):
    """
    Draw x = a*y^2 + b*y + c on the image.
    """

    if coefficients is None:
        return

    a, b, c = coefficients

    points = []

    for y in np.linspace(
        y_start,
        y_end,
        100
    ):

        x = (
            a * y**2
            + b * y
            + c
        )

        x = int(round(x))
        y = int(round(y))

        if (
            0 <= x < img.shape[1]
            and 0 <= y < img.shape[0]
        ):
            points.append(
                (x, y)
            )

    if len(points) >= 2:

        cv2.polylines(
            img,
            [np.array(points)],
            False,
            color,
            thickness
        )


def draw_results(
    frame,
    left_line,
    right_line,
    curvature,
    status,
    direction=None
):
    out = frame.copy()

    # _draw_line(
    #     out,
    #     left_line,
    #     COLOR_LEFT
    # )

    # _draw_line(
    #     out,
    #     right_line,
    #     COLOR_RIGHT
    # )

    if (
        left_line is not None
        and right_line is not None
    ):
        (cx_b, y_b), (cx_t, y_t) = get_road_center(
            left_line,
            right_line
        )

        cv2.line(
            out,
            (int(cx_b), int(y_b)),
            (int(cx_t), int(y_t)),
            COLOR_CENTER,
            2
        )

        cv2.circle(
            out,
            (int(cx_t), int(y_t)),
            6,
            COLOR_CENTER,
            -1
        )

    scale = max(
        0.5,
        frame.shape[0] / 720
    )

    thick = max(
        1,
        int(round(2 * scale))
    )

    color = STATUS_COLORS.get(
        status,
        (255, 255, 255)
    )

    x0 = int(20 * scale)
    y0 = int(40 * scale)
    step = int(38 * scale)

    cv2.putText(
        out,
        f"Road Status: {status}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0 * scale,
        color,
        thick + 1
    )

    metric_text = (
        "n/a"
        if curvature is None
        else f"{curvature:.2f}"
    )

    cv2.putText(
        out,
        f"Curvature Metric: {metric_text}",
        (x0, y0 + step),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8 * scale,
        (255, 255, 255),
        thick
    )

    cv2.putText(
        out,
        "(approx. image-space metric)",
        (
            x0,
            y0 + 2 * step - int(8 * scale)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5 * scale,
        (200, 200, 200),
        max(1, thick - 1)
    )

    if (
        direction
        and status in (
            "MODERATE TURN",
            "SHARP TURN"
        )
    ):
        cv2.putText(
            out,
            f"Bends: {direction}",
            (
                x0,
                y0 + 3 * step - int(8 * scale)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8 * scale,
            color,
            thick
        )

    return out


def process_frame(frame):
    height, width = frame.shape[:2]

    # Original pipeline:
    # ROI -> preprocessing -> polygon mask
    roi, y_offset, polygon = get_roi(frame)

    edges = apply_polygon_mask(
        preprocess(roi),
        polygon
    )

    # Hough detection
    roi_lines = detect_lines(edges)

    # Shift ROI coordinates back into full-frame coordinates
    frame_lines = [
        (
            x1,
            y1 + y_offset,
            x2,
            y2 + y_offset
        )
        for x1, y1, x2, y2 in roi_lines
    ]

    # Separate left and right Hough segments
    left_lines, right_lines = separate_lines(
        frame_lines,
        frame.shape[1]
    )

    # NEW:
    # Extract actual Hough points for future polynomial fitting
    left_points = get_lane_points(left_lines)
    right_points = get_lane_points(right_lines)

    # Existing straight-line fitting
    left_line = fit_lane_line(
        left_lines,
        frame.shape[0]
    )

    right_line = fit_lane_line(
        right_lines,
        frame.shape[0]
    )

    # Existing center-shift curvature metric
    # Polynomial curvature calculation
    (
        curvature,
        direction,
        left_coeff,
        right_coeff
    ) = calculate_polynomial_curvature(
        left_points,
        right_points,
        width,
        height
    )

    status = classify_turn(curvature)

    output = frame.copy()

    _draw_polynomial(
        output,
        left_coeff,
        int(height * 0.40),
        height,
        COLOR_LEFT,
        4
    )

    _draw_polynomial(
        output,
        right_coeff,
        int(height * 0.40),
        height,
        COLOR_RIGHT,
        4
    )

    output = draw_results(
        output,
        left_line,
        right_line,
        curvature,
        status,
        direction
    )

    debug = {}

    if DEBUG:

        roi_vis = roi.copy()

        cv2.polylines(
            roi_vis,
            [polygon],
            True,
            (0, 255, 255),
            2
        )

        hough_vis = frame.copy()

        shifted_poly = (
            polygon +
            np.array(
                [0, y_offset],
                dtype=np.int32
            )
        )

        cv2.polylines(
            hough_vis,
            [shifted_poly],
            True,
            (0, 255, 255),
            2
        )

        # Draw accepted LEFT Hough segments
        for x1, y1, x2, y2 in left_lines:
            cv2.line(
                hough_vis,
                (x1, y1),
                (x2, y2),
                COLOR_LEFT,
                2
            )

        # Draw accepted RIGHT Hough segments
        for x1, y1, x2, y2 in right_lines:
            cv2.line(
                hough_vis,
                (x1, y1),
                (x2, y2),
                COLOR_RIGHT,
                2
            )

        # Draw all raw Hough lines
        for x1, y1, x2, y2 in frame_lines:
            cv2.line(
                hough_vis,
                (x1, y1),
                (x2, y2),
                COLOR_HOUGH,
                1
            )

        fitted_vis = np.zeros_like(frame)

        _draw_line(
            fitted_vis,
            left_line,
            COLOR_LEFT
        )

        _draw_line(
            fitted_vis,
            right_line,
            COLOR_RIGHT
        )

        if (
            left_line is not None
            and right_line is not None
        ):
            (cx_b, y_b), (cx_t, y_t) = get_road_center(
                left_line,
                right_line
            )

            cv2.line(
                fitted_vis,
                (int(cx_b), int(y_b)),
                (int(cx_t), int(y_t)),
                COLOR_CENTER,
                2
            )

        debug = {
            "1 ROI": roi_vis,
            "2 Canny edges (masked)": edges,
            "3 Hough lines (green=raw, blue/red=accepted)": hough_vis,
            "4 Fitted boundaries": fitted_vis,
        }

    print(
        f"Curvature: {curvature:.6f} | "
        f"Direction: {direction}"
        if curvature is not None
        else "Curvature: None"
    )

    return output, debug

def parse_args():
    p = argparse.ArgumentParser(
        description="Road curvature & sharp turn detector (prototype)"
    )

    p.add_argument(
        "video",
        nargs="?",
        default=VIDEO_PATH,
        help="input video path"
    )

    p.add_argument(
        "--no-debug",
        action="store_true",
        help="show only the final output window"
    )

    p.add_argument(
        "--no-display",
        action="store_true",
        help="do not open windows (use with --save)"
    )

    p.add_argument(
        "--save",
        metavar="FILE",
        help="write the processed video to FILE"
    )

    return p.parse_args()


def main():
    global DEBUG

    args = parse_args()

    if args.no_debug:
        DEBUG = False

    # Keep the original no-op vehicle module available without changing
    # the actual baseline processing pipeline.
    _ = VehicleDetector()

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        print(
            f"ERROR: could not open video '{args.video}'"
        )
        sys.exit(1)

    writer = None
    frame_count = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_count += 1

            output, debug = process_frame(
                frame
            )

            if args.save:
                if writer is None:
                    h, w = output.shape[:2]

                    fps = (
                        cap.get(
                            cv2.CAP_PROP_FPS
                        )
                        or 30.0
                    )

                    # Preserve original behavior:
                    # save processed video at one-third FPS.
                    slow_fps = fps / 3

                    writer = cv2.VideoWriter(
                        args.save,
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),
                        slow_fps,
                        (w, h)
                    )

                writer.write(output)

            if not args.no_display:
                cv2.imshow(
                    "Road Curvature - Final Output",
                    output
                )

                for name, img in debug.items():
                    cv2.imshow(
                        name,
                        img
                    )

                # Preserve original frame viewing delay.
                if cv2.waitKey(100) & 0xFF == 27:
                    break

    finally:
        cap.release()

        if writer is not None:
            writer.release()

        if not args.no_display:
            cv2.destroyAllWindows()

    print(
        f"Processed {frame_count} frames."
    )


if __name__ == "__main__":
    main()
