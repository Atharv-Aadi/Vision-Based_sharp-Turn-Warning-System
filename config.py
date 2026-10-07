"""Configuration copied from the original prototype without changing algorithmic values."""
import numpy as np

VIDEO_PATH = "dataset/1st turn.mp4"
DEBUG = True

ROI_TOP_RATIO = 0.40

LINE_TOP_RATIO = 0.20

ROI_POLYGON_RATIOS = [
    (0.00, 1.0),
    (0.30, 0.0),
    (0.70, 0.0),
    (1.00, 1.0)
]

BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30

MIN_LINE_LENGTH_RATIO = 0.06
MAX_LINE_GAP_RATIO = 0.04

SLOPE_THRESHOLD = 0.5
CENTER_MARGIN_RATIO = 0.10
MIN_SEGMENTS_PER_SIDE = 1

STRAIGHT_THRESHOLD = 0.15
SHARP_TURN_THRESHOLD = 1.50

COLOR_LEFT = (255, 128, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_CENTER = (0, 255, 255)
COLOR_HOUGH = (0, 255, 0)

STATUS_COLORS = {
    "STRAIGHT": (0, 200, 0),
    "MODERATE TURN": (0, 165, 255),
    "SHARP TURN": (0, 0, 255),
    "NO DETECTION": (180, 180, 180),
}
