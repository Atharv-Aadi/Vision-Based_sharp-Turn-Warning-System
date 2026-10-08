"""Road curvature / turn classification - 50% milestone.

Implemented:
    Level 1: cleaner road-boundary detection
    Level 2: corrected curvature/radius calculation
    Level 3: temporal smoothing and stable 3-state classification

Not yet implemented:
    RANSAC, bird's-eye perspective transform, vehicle masking,
    real-world camera calibration.
"""
import argparse
import os
import sys

import cv2
import numpy as np

from config import (
    VIDEO_PATH, DEBUG, COLOR_LEFT, COLOR_RIGHT, COLOR_CENTER,
    COLOR_HOUGH, COLOR_RADIUS, STATUS_COLORS,
    CURVATURE_Y_MIN, MAX_VALID_RADIUS_PX,
    SMOOTHING_WINDOW,
)
from preprocessing import get_roi, apply_polygon_mask, preprocess
from lane_detection import detect_lines, separate_lines, fit_lane_line, get_lane_points
from curvature import get_road_center, calculate_polynomial_curvature
from smoothing import TemporalSmoother
from warning import TurnClassifier


def draw_line(img, line, color, thickness=4):
    if line is None:
        return
    xb, yb, xt, yt = line
    cv2.line(img, (int(xb), int(yb)), (int(xt), int(yt)), color, thickness)


def draw_polynomial(img, coefficients, y_start, y_end, color, thickness=4):
    if coefficients is None:
        return
    a, b, c = coefficients
    points = []
    for y in np.linspace(y_start, y_end, 120):
        x = a * y * y + b * y + c
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < img.shape[1] and 0 <= yi < img.shape[0]:
            points.append((xi, yi))
    if len(points) >= 2:
        cv2.polylines(img, [np.asarray(points, dtype=np.int32)], False, color, thickness)


def draw_results(frame, left_line, right_line, radius, status, direction):
    out = frame.copy()
    scale = max(0.5, frame.shape[0] / 720.0)
    thick = max(1, int(round(2 * scale)))
    color = STATUS_COLORS.get(status, (255, 255, 255))
    x = int(20 * scale)
    y = int(42 * scale)
    step = int(40 * scale)

    center = get_road_center(left_line, right_line)
    if center is not None:
        (xb, yb), (xt, yt) = center
        cv2.line(out, (int(xb), int(yb)), (int(xt), int(yt)), COLOR_CENTER, 2)
        cv2.circle(out, (int(xb), int(yb)), 5, COLOR_CENTER, -1)

    radius_text = "n/a" if radius is None else f"{radius:.0f} px"
    direction_text = "" if direction is None else f"Direction: {direction}"

    cv2.putText(out, f"Road Status: {status}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.95 * scale, color, thick + 1, cv2.LINE_AA)
    cv2.putText(out, f"Curve radius: {radius_text}", (x, y + step),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78 * scale, COLOR_RADIUS, thick, cv2.LINE_AA)

    if direction_text:
        cv2.putText(out, direction_text, (x, y + 2 * step),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68 * scale, color, thick, cv2.LINE_AA)

    if status == "SHARP TURN":
        text = "!! SHARP TURN AHEAD !!"
        font_scale = 0.9 * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick + 1)
        tx = max(10, (frame.shape[1] - tw) // 2)
        ty = frame.shape[0] - int(45 * scale)
        cv2.rectangle(out, (tx - 12, ty - th - 12), (tx + tw + 12, ty + 10), (0, 0, 0), -1)
        cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thick + 1, cv2.LINE_AA)

    return out


def process_frame(frame, smoother, classifier):
    height, width = frame.shape[:2]
    roi, y_offset, polygon = get_roi(frame)
    edges = apply_polygon_mask(preprocess(roi), polygon)

    roi_lines = detect_lines(edges)
    frame_lines = [(x1, y1 + y_offset, x2, y2 + y_offset)
                   for x1, y1, x2, y2 in roi_lines]

    left_lines, right_lines = separate_lines(frame_lines, width, height)
    left_points = get_lane_points(left_lines)
    right_points = get_lane_points(right_lines)

    left_line = fit_lane_line(left_lines, height)
    right_line = fit_lane_line(right_lines, height)

    radius, direction, left_coeff, right_coeff = calculate_polynomial_curvature(
        left_points, right_points, width, height
    )

    smoothed_radius = smoother.update(radius)
    status = classifier.update(smoothed_radius)

    output = frame.copy()
    draw_polynomial(output, left_coeff, int(height * CURVATURE_Y_MIN), height,
                    COLOR_LEFT, 4)
    draw_polynomial(output, right_coeff, int(height * CURVATURE_Y_MIN), height,
                    COLOR_RIGHT, 4)
    output = draw_results(output, left_line, right_line,
                          smoothed_radius, status, direction)

    debug = {}
    if DEBUG:
        roi_vis = roi.copy()
        cv2.polylines(roi_vis, [polygon], True, (0, 255, 255), 2)

        hough_vis = frame.copy()
        shifted_poly = polygon + np.array([0, y_offset], dtype=np.int32)
        cv2.polylines(hough_vis, [shifted_poly], True, (0, 255, 255), 2)

        for x1, y1, x2, y2 in frame_lines:
            cv2.line(hough_vis, (x1, y1), (x2, y2), COLOR_HOUGH, 1)
        for x1, y1, x2, y2 in left_lines:
            cv2.line(hough_vis, (x1, y1), (x2, y2), COLOR_LEFT, 3)
        for x1, y1, x2, y2 in right_lines:
            cv2.line(hough_vis, (x1, y1), (x2, y2), COLOR_RIGHT, 3)

        fitted_vis = np.zeros_like(frame)
        draw_line(fitted_vis, left_line, COLOR_LEFT)
        draw_line(fitted_vis, right_line, COLOR_RIGHT)
        draw_polynomial(fitted_vis, left_coeff, int(height * CURVATURE_Y_MIN), height, COLOR_LEFT, 4)
        draw_polynomial(fitted_vis, right_coeff, int(height * CURVATURE_Y_MIN), height, COLOR_RIGHT, 4)

        debug = {
            "1 ROI": roi_vis,
            "2 Canny + mask": edges,
            "3 Hough (green=raw, blue/red=accepted)": hough_vis,
            "4 Fitted road curves": fitted_vis,
        }

    return output, debug, radius, smoothed_radius, status, direction, len(left_lines), len(right_lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Road curvature and turn classifier - 50% milestone")
    parser.add_argument("video", nargs="?", default=VIDEO_PATH)
    parser.add_argument("--no-debug", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save", metavar="FILE")
    return parser.parse_args()


def main():
    global DEBUG
    args = parse_args()
    if args.no_debug:
        DEBUG = False

    smoother = TemporalSmoother(window_size=SMOOTHING_WINDOW)
    classifier = TurnClassifier()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video '{args.video}'")
        sys.exit(1)

    writer = None
    frame_count = 0
    last_status = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1

            output, debug, raw_radius, smoothed_radius, status, direction, nl, nr = process_frame(
                frame, smoother, classifier
            )

            if status != last_status:
                print(f"Frame {frame_count}: {status} | radius={smoothed_radius} | direction={direction}")
                last_status = status

            if args.save:
                if writer is None:
                    h, w = output.shape[:2]
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                    writer = cv2.VideoWriter(
                        args.save,
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (w, h),
                    )
                writer.write(output)

            if not args.no_display:
                cv2.imshow("Road Curvature - Final Output", output)
                for name, img in debug.items():
                    cv2.imshow(name, img)
                if cv2.waitKey(200) & 0xFF == 27:
                    break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames.")


if __name__ == "__main__":
    main()
