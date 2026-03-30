"""
Synthetic frame generators for testing.
Produces NumPy BGR frames without requiring real video files or camera access.
"""

import cv2
import numpy as np


def make_blank_frame(w: int = 640, h: int = 480, color: tuple = (50, 50, 50)) -> np.ndarray:
    """Return a solid-color BGR frame."""
    frame = np.full((h, w, 3), color, dtype=np.uint8)
    return frame


def make_gameplay_frame(w: int = 640, h: int = 480, table_color: str = "blue") -> np.ndarray:
    """
    Return a frame that looks like gameplay: a colored rectangle in the center
    that matches the table HSV range used by SceneClassifier.
    """
    frame = make_blank_frame(w, h, color=(30, 20, 20))

    # Table region covers the center 50% of the frame
    x1, y1 = w // 4, h // 4
    x2, y2 = 3 * w // 4, 3 * h // 4

    if table_color == "blue":
        # HSV ~(115, 180, 180) -> BGR
        bgr = (180, 50, 50)
    elif table_color == "green":
        # HSV ~(60, 180, 180) -> BGR
        bgr = (50, 180, 50)
    else:
        bgr = (180, 50, 50)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, -1)
    return frame


def make_non_gameplay_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """
    Return a frame that does NOT contain table color (e.g. crowd/replay).
    Uses a brownish/reddish palette far from blue or green HSV ranges.
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Fill with a crowd-like orange-brown
    frame[:, :] = (30, 80, 160)
    # Add some noise to avoid being completely uniform
    noise = np.random.randint(0, 20, (h, w, 3), dtype=np.uint8)
    frame = cv2.add(frame, noise)
    return frame


def make_frame_with_ball(
    w: int = 640,
    h: int = 480,
    ball_x: int = 320,
    ball_y: int = 240,
    ball_r: int = 10,
    bg_color: tuple = (30, 30, 30),
) -> np.ndarray:
    """Return a frame with a white circle representing the ball."""
    frame = make_blank_frame(w, h, color=bg_color)
    cv2.circle(frame, (ball_x, ball_y), ball_r, (255, 255, 255), -1)
    return frame


def make_shifted_frame(base: np.ndarray, dx: int = 0, dy: int = 0) -> np.ndarray:
    """
    Return a copy of `base` translated by (dx, dy) pixels.
    Used to simulate camera motion / hard cuts with large frame differences.
    """
    h, w = base.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(base, M, (w, h))


def make_large_diff_frame(base: np.ndarray) -> np.ndarray:
    """Return a frame that is maximally different from `base` (inverted)."""
    return cv2.bitwise_not(base)


def make_synthetic_video_frames(
    n_frames: int = 30,
    w: int = 640,
    h: int = 480,
    table_color: str = "blue",
) -> list:
    """Return a list of n_frames gameplay frames (useful for pipeline smoke tests)."""
    return [make_gameplay_frame(w, h, table_color) for _ in range(n_frames)]


def write_synthetic_video(path: str, n_frames: int = 30, fps: float = 30.0,
                           w: int = 640, h: int = 480) -> str:
    """
    Write a short synthetic MP4 to `path` and return the path.
    Requires OpenCV with video-write support.
    """
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in make_synthetic_video_frames(n_frames, w, h):
        writer.write(frame)
    writer.release()
    return path
