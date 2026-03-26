"""
Optical flow-based table corner tracker for broadcast streams.
Tracks 4 manually-marked table corners frame-to-frame using Lucas-Kanade
optical flow, with EMA smoothing and quality checks.
"""

import cv2
import numpy as np


# EMA smoothing factor (0 = no smoothing, 1 = no memory)
EMA_ALPHA = 0.3

# If a corner jumps more than this many pixels in one frame, treat as re-detection
JUMP_THRESHOLD = 50.0

# Quality check thresholds
MIN_QUAD_AREA = 500          # minimum pixel area for the quadrilateral
ASPECT_RATIO_MIN = 1.2       # table is roughly 2:1, allow some perspective distortion
ASPECT_RATIO_MAX = 5.0

# Lucas-Kanade parameters
LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


def _order_corners(pts):
    """
    Order 4 points as TL, TR, BR, BL based on their position.
    Assumes the table is roughly horizontal in the frame.
    """
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    # Sort by y to get top-2 and bottom-2
    sorted_by_y = pts[np.argsort(pts[:, 1])]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    # Sort each pair by x
    tl, tr = top[np.argsort(top[:, 0])]
    bl, br = bottom[np.argsort(bottom[:, 0])]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _is_convex(pts):
    """Check if 4 points form a convex quadrilateral."""
    pts = pts.reshape(4, 2).astype(np.float32)
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])
        if i == 0:
            sign = cross > 0
        elif (cross > 0) != sign:
            return False
    return True


def _quad_area(pts):
    """Compute area of a quadrilateral using the shoelace formula."""
    pts = pts.reshape(4, 2)
    n = len(pts)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1]
        area -= pts[j][0] * pts[i][1]
    return abs(area) / 2.0


class TableTracker:
    """
    Tracks 4 table corners using Lucas-Kanade optical flow.

    Usage:
        tracker = TableTracker(initial_corners)  # 4 points: TL, TR, BR, BL
        for frame in frames:
            corners = tracker.update(frame)
            if tracker.is_valid:
                calibration = TableCalibration(corners)
    """

    def __init__(self, initial_corners):
        """
        initial_corners: list/array of 4 (x, y) points in order TL, TR, BR, BL.
        """
        self.corners = np.array(initial_corners, dtype=np.float32).reshape(4, 2)
        self.smoothed_corners = self.corners.copy()
        self.prev_gray = None
        self.is_valid = True
        self.frames_since_valid = 0
        self._initial_area = _quad_area(self.corners)

    def update(self, frame):
        """
        Track corners in the new frame. Returns smoothed corner positions (4x2 array).
        Updates self.is_valid based on quality checks.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return self.smoothed_corners.copy()

        # Run Lucas-Kanade optical flow
        pts_prev = self.smoothed_corners.reshape(-1, 1, 2)
        pts_next, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, pts_prev, None, **LK_PARAMS
        )

        self.prev_gray = gray

        if pts_next is None or status is None:
            self._mark_invalid()
            return self.smoothed_corners.copy()

        pts_next = pts_next.reshape(4, 2)
        status = status.reshape(4)

        # Check if all 4 points were tracked successfully
        if not np.all(status == 1):
            self._mark_invalid()
            return self.smoothed_corners.copy()

        # Check for large jumps (indicates tracking failure)
        deltas = np.linalg.norm(pts_next - self.smoothed_corners, axis=1)
        if np.any(deltas > JUMP_THRESHOLD):
            self._mark_invalid()
            return self.smoothed_corners.copy()

        # Apply EMA smoothing
        new_smoothed = EMA_ALPHA * pts_next + (1 - EMA_ALPHA) * self.smoothed_corners

        # Quality checks on the smoothed result
        if self._quality_check(new_smoothed):
            self.smoothed_corners = new_smoothed
            self.is_valid = True
            self.frames_since_valid = 0
        else:
            self._mark_invalid()

        return self.smoothed_corners.copy()

    def _quality_check(self, pts):
        """Run quality checks on proposed corner positions."""
        pts = pts.reshape(4, 2)

        # Convexity
        if not _is_convex(pts):
            return False

        # Minimum area
        area = _quad_area(pts)
        if area < MIN_QUAD_AREA:
            return False

        # Area shouldn't change too drastically from initial (camera zoom is limited)
        if self._initial_area > 0:
            ratio = area / self._initial_area
            if ratio < 0.3 or ratio > 3.0:
                return False

        # Approximate aspect ratio (avg width / avg height)
        w_top = np.linalg.norm(pts[1] - pts[0])
        w_bot = np.linalg.norm(pts[2] - pts[3])
        h_left = np.linalg.norm(pts[3] - pts[0])
        h_right = np.linalg.norm(pts[2] - pts[1])
        avg_w = (w_top + w_bot) / 2
        avg_h = (h_left + h_right) / 2
        if avg_h < 1:
            return False
        aspect = avg_w / avg_h
        if aspect < ASPECT_RATIO_MIN or aspect > ASPECT_RATIO_MAX:
            return False

        return True

    def _mark_invalid(self):
        """Mark tracking as degraded (keep last good corners)."""
        self.is_valid = False
        self.frames_since_valid += 1

    def reinitialize(self, new_corners, frame=None):
        """
        Re-initialize tracking with new corner positions.
        Call this after a scene cut or when the user re-marks corners.
        """
        self.corners = np.array(new_corners, dtype=np.float32).reshape(4, 2)
        self.smoothed_corners = self.corners.copy()
        self._initial_area = _quad_area(self.corners)
        self.is_valid = True
        self.frames_since_valid = 0
        if frame is not None:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            self.prev_gray = None

    def get_corners_int(self):
        """Return corners as integer pixel coordinates (for drawing)."""
        return self.smoothed_corners.astype(np.int32)
