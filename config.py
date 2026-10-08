"""Configuration for the 50% road-curvature milestone."""
import numpy as np

VIDEO_PATH = "dataset/2nd_turn.mp4"
DEBUG = True

# ------------------------- ROI -------------------------
ROI_TOP_RATIO = 0.40
LINE_TOP_RATIO = 0.20
ROI_POLYGON_RATIOS = [
    (0.00, 1.00),
    (0.30, 0.00),
    (0.70, 0.00),
    (1.00, 1.00),
]

# ---------------------- preprocessing -------------------
BLUR_KERNEL = (5, 5)
CANNY_LOW = 50
CANNY_HIGH = 150
MORPH_KERNEL_SIZE = 3
MORPH_CLOSE_ITERATIONS = 1

# ------------------------- Hough ------------------------
HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 30
MIN_LINE_LENGTH_RATIO = 0.05
MAX_LINE_GAP_RATIO = 0.035

# ------------------- segment filtering ------------------
MIN_ABS_SLOPE = 0.45
MAX_ABS_SLOPE = 5.0
CENTER_MARGIN_RATIO = 0.18
MIN_VERTICAL_SPAN_RATIO = 0.06
MIN_SEGMENTS_PER_SIDE = 2
MIN_SIDE_POINTS = 6

# --------------------- curvature -------------------------
# The curve is fitted in coordinates normalized by frame height.
# radius is then reported in approximate image-space pixels.
CURVATURE_Y_MIN = 0.45
CURVATURE_Y_MAX = 0.82
CURVATURE_SAMPLES = 15

# Radius thresholds are intentionally conservative for the current
# camera/video geometry. They are the values to tune later after
# perspective transformation / real-world calibration.
SHARP_RADIUS_PX = 650.0
MODERATE_RADIUS_PX = 1600.0

# Reject mathematically unstable curvature values.
MIN_VALID_RADIUS_PX = 80.0
MAX_VALID_RADIUS_PX = 100000.0

# ------------------ temporal smoothing -------------------
SMOOTHING_WINDOW = 15
MEDIAN_WINDOW = 9
EMA_ALPHA = 0.20
MAX_RADIUS_JUMP_RATIO = 0.60

# Status persistence / hysteresis.
STATUS_ENTER_FRAMES = 8
STATUS_EXIT_FRAMES = 8

# ---------------------- visualization --------------------
COLOR_LEFT = (255, 128, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_CENTER = (0, 255, 255)
COLOR_HOUGH = (0, 255, 0)
COLOR_RADIUS = (255, 255, 255)

STATUS_COLORS = {
    "STRAIGHT": (0, 200, 0),
    "MODERATE TURN": (0, 165, 255),
    "SHARP TURN": (0, 0, 255),
    "NO DETECTION": (180, 180, 180),
}
