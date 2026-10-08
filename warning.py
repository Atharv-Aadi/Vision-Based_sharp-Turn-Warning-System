"""Stable three-state road classification with hysteresis."""
from config import (
    SHARP_RADIUS_PX,
    MODERATE_RADIUS_PX,
    STATUS_ENTER_FRAMES,
    STATUS_EXIT_FRAMES,
)


class TurnClassifier:
    def __init__(self):
        self.status = None
        self.candidate = None
        self.candidate_count = 0
        self.exit_count = 0

    @staticmethod
    def raw_classification(radius):
        if radius is None:
            return None
        radius = abs(float(radius))
        if radius <= SHARP_RADIUS_PX:
            return "SHARP TURN"
        if radius <= MODERATE_RADIUS_PX:
            return "MODERATE TURN"
        return "STRAIGHT"

    def update(self, radius):
        raw = self.raw_classification(radius)
        if raw is None:
            return self.status or "NO DETECTION"

        if self.status is None:
            self.status = raw
            self.candidate = None
            self.candidate_count = 0
            return self.status

        if raw == self.status:
            self.candidate = None
            self.candidate_count = 0
            self.exit_count = 0
            return self.status

        # Require several consecutive frames before changing state.
        if raw != self.candidate:
            self.candidate = raw
            self.candidate_count = 1
        else:
            self.candidate_count += 1

        if self.candidate_count >= STATUS_ENTER_FRAMES:
            self.status = raw
            self.candidate = None
            self.candidate_count = 0
            self.exit_count = 0

        return self.status

    def reset(self):
        self.status = None
        self.candidate = None
        self.candidate_count = 0
        self.exit_count = 0


_classifier = TurnClassifier()


def classify_turn(radius):
    return _classifier.update(radius)


def reset_warning_state():
    _classifier.reset()
