"""ROI and edge preprocessing."""
import cv2
import numpy as np

from config import (
    ROI_TOP_RATIO,
    ROI_POLYGON_RATIOS,
    BLUR_KERNEL,
    CANNY_LOW,
    CANNY_HIGH,
    MORPH_KERNEL_SIZE,
    MORPH_CLOSE_ITERATIONS,
)


def get_roi(frame):
    height, width = frame.shape[:2]
    y_offset = int(height * ROI_TOP_RATIO)
    roi = frame[y_offset:height, :]
    roi_h, roi_w = roi.shape[:2]

    polygon = np.array(
        [[int(xr * (roi_w - 1)), int(yr * (roi_h - 1))]
         for xr, yr in ROI_POLYGON_RATIOS],
        dtype=np.int32,
    ).reshape(-1, 1, 2)

    return roi, y_offset, polygon


def apply_polygon_mask(image, polygon):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return cv2.bitwise_and(image, image, mask=mask)


def preprocess(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
    edges = cv2.Canny(blur, CANNY_LOW, CANNY_HIGH)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE),
    )
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_CLOSE_ITERATIONS,
    )
    return edges
