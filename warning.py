"""Original turn classification logic."""
from config import STRAIGHT_THRESHOLD, SHARP_TURN_THRESHOLD


def classify_turn(curvature):
    if curvature is None:
        return "NO DETECTION"

    if curvature < STRAIGHT_THRESHOLD:
        return "STRAIGHT"

    if curvature < SHARP_TURN_THRESHOLD:
        return "MODERATE TURN"

    return "SHARP TURN"
