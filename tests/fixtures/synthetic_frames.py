"""
Synthetic frame and mock data generators for testing.
No real video or GPU required — all data is fabricated with numpy/OpenCV.
"""

import cv2
import numpy as np
from collections import namedtuple


def make_blank_frame(w=1280, h=720, color=(0, 0, 0)):
    """Return a solid-color BGR frame."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def make_frame_with_ball(cx, cy, radius=8, w=1280, h=720, bg_color=(0, 0, 0), ball_color=(255, 255, 255)):
    """Return a frame with a white circle (simulated ball)."""
    frame = make_blank_frame(w, h, bg_color)
    cv2.circle(frame, (int(cx), int(cy)), radius, ball_color, -1)
    return frame


def make_frame_sequence_with_moving_ball(n_frames=30, w=1280, h=720, start=(200, 360), end=(1080, 360)):
    """Generate a list of frames with a ball moving in a straight line."""
    frames = []
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        cx = start[0] + t * (end[0] - start[0])
        cy = start[1] + t * (end[1] - start[1])
        frames.append(make_frame_with_ball(cx, cy, w=w, h=h))
    return frames


def make_scoreboard_crop(score, w=100, h=60):
    """Render a digit image for a given score (0-30). Returns a BGR image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    text = str(score)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.5
    thickness = 3
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (w - tw) // 2
    y = (h + th) // 2
    cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness)
    return img


def make_test_video(path, n_frames=150, fps=30.0, w=640, h=360):
    """Write a short .mp4 test video with a ball moving left to right."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    frames = make_frame_sequence_with_moving_ball(n_frames, w=w, h=h, start=(50, 180), end=(590, 180))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


# =============================================================================
# MOCK YOLO RESULTS (for unit tests that bypass real model)
# =============================================================================

class MockBox:
    """Mimics a single ultralytics box object."""
    def __init__(self, x1, y1, x2, y2, cls_id, conf):
        self.xyxy = [np.array([x1, y1, x2, y2])]
        self.cls = _ScalarTensor(cls_id)
        self.conf = _ScalarTensor(conf)


class _ScalarTensor:
    """Mimics a PyTorch scalar tensor with .item()."""
    def __init__(self, val):
        self._val = val

    def item(self):
        return self._val


class MockBoxes:
    """List-like container of MockBox objects, mimicking ultralytics Results.boxes."""
    def __init__(self, boxes=None):
        self._boxes = boxes or []

    def __len__(self):
        return len(self._boxes)

    def __iter__(self):
        return iter(self._boxes)

    def __bool__(self):
        return len(self._boxes) > 0


class MockResult:
    """Mimics a single ultralytics Result object."""
    def __init__(self, boxes=None):
        self.boxes = MockBoxes(boxes)


class MockYOLOModel:
    """
    A mock YOLO model that returns preset detections.

    Usage:
        model = MockYOLOModel(detections=[
            (100, 200, 120, 220, 0, 0.95),  # x1,y1,x2,y2,cls,conf
        ])
        results = model(frame, conf=0.4, verbose=False)
    """
    def __init__(self, detections=None):
        """detections: list of (x1, y1, x2, y2, cls, conf) tuples."""
        self._detections = detections or []

    def __call__(self, img, **kwargs):
        conf_thresh = kwargs.get('conf', 0.0)
        boxes = []
        for x1, y1, x2, y2, cls_id, conf in self._detections:
            if conf >= conf_thresh:
                boxes.append(MockBox(x1, y1, x2, y2, cls_id, conf))
        return [MockResult(boxes)]


# =============================================================================
# MOCK POSE LANDMARKS
# =============================================================================

MockLandmark = namedtuple('MockLandmark', ['x', 'y', 'z', 'visibility'])


def make_pose_landmarks(pose='neutral', visibility=0.9):
    """
    Create a list of 33 fake MediaPipe pose landmarks.
    pose: 'neutral' (standing upright) or 'serving' (arm raised).
    """
    landmarks = []
    for i in range(33):
        landmarks.append(MockLandmark(x=0.5, y=0.5, z=0.0, visibility=visibility))

    if pose == 'neutral':
        # Key joints in normalized coords (within a half-frame crop)
        landmarks[11] = MockLandmark(0.4, 0.3, 0, visibility)    # left_shoulder
        landmarks[12] = MockLandmark(0.6, 0.3, 0, visibility)    # right_shoulder
        landmarks[13] = MockLandmark(0.35, 0.45, 0, visibility)  # left_elbow
        landmarks[14] = MockLandmark(0.65, 0.45, 0, visibility)  # right_elbow
        landmarks[15] = MockLandmark(0.3, 0.55, 0, visibility)   # left_wrist
        landmarks[16] = MockLandmark(0.7, 0.55, 0, visibility)   # right_wrist
        landmarks[23] = MockLandmark(0.45, 0.6, 0, visibility)   # left_hip
        landmarks[24] = MockLandmark(0.55, 0.6, 0, visibility)   # right_hip
        landmarks[25] = MockLandmark(0.43, 0.78, 0, visibility)  # left_knee
        landmarks[26] = MockLandmark(0.57, 0.78, 0, visibility)  # right_knee
        landmarks[27] = MockLandmark(0.4, 0.95, 0, visibility)   # left_ankle
        landmarks[28] = MockLandmark(0.6, 0.95, 0, visibility)   # right_ankle

    return landmarks
