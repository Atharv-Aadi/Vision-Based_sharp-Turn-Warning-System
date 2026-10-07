"""
Temporal smoothing for road curvature estimation.

This module reduces frame-to-frame fluctuations in the curvature
value produced by the curvature estimation module.
"""

from collections import deque


class TemporalSmoother:
    """
    Maintains a rolling history of curvature values and returns
    a smoothed curvature using a moving average.
    """

    def __init__(self, window_size=5):
        """
        Create a temporal smoother.

        Parameters
        ----------
        window_size : int
            Number of recent curvature values used for smoothing.
        """

        if window_size < 1:
            raise ValueError("window_size must be at least 1")

        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def update(self, curvature):
        """
        Add a new curvature value and return the smoothed value.

        Parameters
        ----------
        curvature : float or None
            Curvature value obtained from the curvature estimator.

        Returns
        -------
        float or None
            Smoothed curvature value.
        """

        # If curvature could not be calculated for this frame,
        # do not add anything to the history.
        if curvature is None:
            return self.get_smoothed_value()

        # Store the new curvature value.
        self.history.append(float(curvature))

        # Calculate moving average.
        return sum(self.history) / len(self.history)

    def get_smoothed_value(self):
        """
        Return the current smoothed curvature.

        Returns
        -------
        float or None
            Current moving average, or None if no values exist.
        """

        if not self.history:
            return None

        return sum(self.history) / len(self.history)

    def reset(self):
        """
        Clear all stored curvature values.
        """

        self.history.clear()

    def get_history(self):
        """
        Return the current curvature history.

        Returns
        -------
        list
            List of recent curvature values.
        """

        return list(self.history)


# Test the temporal smoother when this file is run directly.
if __name__ == "__main__":
    smoother = TemporalSmoother(window_size=5)

    test_values = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60
    ]

    for value in test_values:
        smoothed = smoother.update(value)

        print(
            f"Raw: {value:.2f}  "
            f"Smoothed: {smoothed:.2f}"
        )