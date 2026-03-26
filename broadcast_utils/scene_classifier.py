"""
Scene classifier for broadcast streams.
Detects hard cuts and classifies frames as gameplay vs non-gameplay
using frame differencing + table color heuristics (no segmentation model needed).
"""

import cv2
import numpy as np

# States
GAMEPLAY = "gameplay"
CUT_DETECTED = "cut_detected"
NON_GAMEPLAY = "non_gameplay"

# HSV ranges for common ITTF table colors
TABLE_COLOR_PRESETS = {
    "blue": {
        "lower": np.array([100, 50, 50]),
        "upper": np.array([130, 255, 255]),
    },
    "green": {
        "lower": np.array([35, 50, 50]),
        "upper": np.array([85, 255, 255]),
    },
}

# Resolution for frame diff computation
DIFF_SIZE = (320, 180)

# Threshold for mean absolute frame difference to detect a hard cut
CUT_THRESHOLD = 30.0

# Minimum fraction of center region with table color to count as gameplay
MIN_TABLE_FRACTION = 0.04

# How often to poll for gameplay return during NON_GAMEPLAY state (in frames)
POLL_INTERVAL = 15


class SceneClassifier:
    """
    Classifies broadcast frames as gameplay or non-gameplay.

    Usage:
        classifier = SceneClassifier(table_color="blue")
        for frame in frames:
            classifier.update(frame)
            if classifier.state == GAMEPLAY:
                # process frame
            elif classifier.is_cut:
                # reset tracker state
    """

    def __init__(self, table_color="blue"):
        self.table_color = table_color
        self.color_preset = TABLE_COLOR_PRESETS.get(table_color, TABLE_COLOR_PRESETS["blue"])
        self.state = GAMEPLAY
        self.prev_gray = None
        self.frames_in_state = 0
        self.is_cut = False  # True on the first frame after a cut is detected

    def _frame_diff(self, gray):
        """Compute mean absolute difference between current and previous frame."""
        if self.prev_gray is None:
            return 0.0
        return float(np.mean(cv2.absdiff(self.prev_gray, gray)))

    def _has_table_color(self, frame):
        """Check if center region of frame contains enough table color."""
        h, w = frame.shape[:2]
        y1, y2 = h // 4, 3 * h // 4
        x1, x2 = w // 4, 3 * w // 4
        center = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.color_preset["lower"], self.color_preset["upper"])
        fraction = np.count_nonzero(mask) / mask.size
        return fraction >= MIN_TABLE_FRACTION

    def update(self, frame):
        """
        Update state based on the current frame.

        Returns the current state (GAMEPLAY, CUT_DETECTED, or NON_GAMEPLAY).
        Also sets self.is_cut = True on the frame where a cut transitions to GAMEPLAY.
        """
        small = cv2.resize(frame, DIFF_SIZE, interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        diff = self._frame_diff(gray)
        self.prev_gray = gray
        self.is_cut = False
        self.frames_in_state += 1

        if self.state == GAMEPLAY:
            if diff > CUT_THRESHOLD:
                self.state = CUT_DETECTED
                self.frames_in_state = 0

        elif self.state == CUT_DETECTED:
            # Check if we're back to gameplay
            if self._has_table_color(frame):
                self.state = GAMEPLAY
                self.frames_in_state = 0
                self.is_cut = True  # signal to reset tracker
            else:
                self.state = NON_GAMEPLAY
                self.frames_in_state = 0

        elif self.state == NON_GAMEPLAY:
            # Poll periodically to check if gameplay resumed
            if self.frames_in_state % POLL_INTERVAL == 0:
                if self._has_table_color(frame):
                    self.state = GAMEPLAY
                    self.frames_in_state = 0
                    self.is_cut = True

        return self.state

    @property
    def should_process(self):
        """True if the current frame should be processed by the tracking pipeline."""
        return self.state == GAMEPLAY
