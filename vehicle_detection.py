"""Vehicle-detection placeholder.

The original supplied prototype does not contain YOLO/vehicle masking.
This module is intentionally a no-op so the refactored project does not
change the original algorithm.

YOLO can be integrated here later without modifying the lane/curvature
baseline.
"""


class VehicleDetector:
    def get_vehicle_mask(self, frame):
        return None
