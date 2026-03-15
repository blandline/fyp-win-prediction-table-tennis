"""
Ball Tracking and Score Detection System for Table Tennis
==========================================================
This script provides:
1. Interactive first-frame setup with zoom capability
2. Manual marking of:
   - Player 1 score region
   - Player 2 score region
   - Rounds won indicators for each player
3. Ball trajectory tracking with speed calculation
4. Real-time score detection using YOLO digit model
"""

import cv2
import numpy as np
import json
import sys
import os
from collections import deque
from datetime import datetime
from pathlib import Path

# Add sort directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sort'))
from sort import Sort

from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================
BALL_MODEL_PATH = "runs/detect/runs/ball_detector/weights/best.pt"
DIGIT_MODEL_PATH = "runs/detect/runs/detect/digits_v2/weights/best.pt"


def resolve_model_path(pt_path, require_engine=False):
    """
    Return the path to use for YOLO model loading. If a TensorRT .engine file
    exists next to the .pt file, use it for faster inference (same API).
    require_engine: if True, require the .engine to exist and return its path;
                    use for --use-tensorrt (caller should check existence).
    """
    p = Path(pt_path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent / p
    engine_path = p.with_suffix(".engine")
    if require_engine:
        return str(engine_path)
    if engine_path.exists():
        return str(engine_path)
    return str(p)


# -----------------------------------------------------------------------------
# TensorRT export (one-time, on the same GPU you run inference on)
# -----------------------------------------------------------------------------
# Ball detector (match your --ball-inference-size, e.g. 640x360):
#   from ultralytics import YOLO
#   m = YOLO("runs/detect/runs/ball_detector/weights/best.pt")
#   m.export(format="engine", imgsz=(360, 640), half=True, device=0, workspace=2)
#
# Digit/score model (runs on 320-size crops):
#   m = YOLO("runs/detect/runs/detect/digits_v2/weights/best.pt")
#   m.export(format="engine", imgsz=320, half=True, device=0, workspace=2)
#
# Engines are GPU-specific. Use the resulting .engine paths automatically
# when present, or pass --use-tensorrt to require them.
# -----------------------------------------------------------------------------

# Detection settings
BALL_CONF_THRESHOLD = 0.35
DIGIT_CONF_THRESHOLD = 0.35

# Tracker settings
TRACKER_MAX_AGE = 5  # Frames to keep tracking without detection
TRACKER_MIN_HITS = 2  # Minimum detections before track is valid
TRACKER_IOU_THRESHOLD = 0.1  # IOU threshold for matching (lower for fast ball)

# Trajectory visualization
TRAJECTORY_LENGTH = 50  # Number of points to show in trajectory
SPEED_SMOOTHING_WINDOW = 5  # Frames to average speed over

# ITTF table dimensions (meters)
TABLE_LENGTH_M = 2.74
TABLE_WIDTH_M = 1.525

# Physical limits for filtering
TABLE_MARGIN_M = 1.5  # accept ball positions up to this far outside the table
MAX_BALL_SPEED_MPS = 40.0  # world-record smash is ~32 m/s; generous cap

# =============================================================================
# INFERENCE RESCALING CONFIGURATION
# =============================================================================
# Ball detector runs every frame and is the primary throughput bottleneck.
# Rescaling the frame before inference reduces per-frame YOLO cost significantly
# while (hopefully) preserving detection quality.
#
# Recommended values to evaluate (width, height):
#   None          -> pass the original frame unchanged (baseline)
#   (1280, 720)   -> moderate downscale from typical 1920x1080 source (~44 % fewer pixels)
#   (960, 540)    -> aggressive but still reasonable (75 % fewer pixels vs 1080p)
#   (640, 360)    -> maximum downscale worth trying for a small ball
#
# NOTE: The table-tennis ball is tiny in frame.  Values below 960x540 risk
# destroying detection accuracy.  Start conservative (1280x720) and work down.
#
# Override via CLI: --ball-inference-size 1280x720
BALL_INFERENCE_SIZE = None   # (width, height) or None for full resolution

# =============================================================================
# TABLE CALIBRATION (HOMOGRAPHY FOR PIXEL -> METERS)
# =============================================================================
class TableCalibration:
    """
    Maps pixel coordinates to table-plane coordinates in meters using homography.
    Expects 4 corners in order: top-left, top-right, bottom-right, bottom-left.
    """
    # Real-world table corners (meters): TL, TR, BR, BL
    DST_CORNERS = np.array([
        [0, 0],
        [TABLE_LENGTH_M, 0],
        [TABLE_LENGTH_M, TABLE_WIDTH_M],
        [0, TABLE_WIDTH_M]
    ], dtype=np.float32)

    def __init__(self, pixel_corners):
        """
        pixel_corners: list of 4 (x, y) in pixel coords, order TL, TR, BR, BL.
        """
        self.pixel_corners = np.array(pixel_corners, dtype=np.float32).reshape(4, 2)
        self.H, self._valid = self._compute_homography()
        self._reprojection_error = None
        if self._valid:
            self._reprojection_error = self._check_reprojection_error()

    def _compute_homography(self):
        """Compute homography from pixel plane to table plane (meters)."""
        if len(self.pixel_corners) != 4:
            return None, False
        try:
            H, status = cv2.findHomography(
                self.pixel_corners, self.DST_CORNERS,
                method=cv2.RANSAC, ransacReprojThreshold=5.0
            )
            if H is None or status is None:
                return None, False
            return H, True
        except Exception:
            return None, False

    def _check_reprojection_error(self):
        """Reproject pixel corners to meter plane and return mean error in meters."""
        if self.H is None:
            return float('inf')
        src = self.pixel_corners.reshape(-1, 1, 2).astype(np.float32)
        dst_reproj = cv2.perspectiveTransform(src, self.H)
        dst_reproj = dst_reproj.reshape(4, 2)
        real = self.DST_CORNERS
        return float(np.mean(np.linalg.norm(dst_reproj - real, axis=1)))

    def is_valid(self):
        """True if homography was computed and sanity checks pass."""
        if not self._valid or self.H is None:
            return False
        # Reprojection error should be small (in meters)
        if self._reprojection_error is not None and self._reprojection_error > 0.15:
            return False
        # Check aspect ratio of mapped table (from transformed corners)
        pts = np.array([[0, 0], [TABLE_LENGTH_M, 0], [TABLE_LENGTH_M, TABLE_WIDTH_M], [0, TABLE_WIDTH_M]], dtype=np.float32)
        # Map back to pixel and compute side lengths in pixels to check for degenerate
        p = self.pixel_corners
        w1 = np.linalg.norm(p[1] - p[0])
        w2 = np.linalg.norm(p[2] - p[3])
        h1 = np.linalg.norm(p[3] - p[0])
        h2 = np.linalg.norm(p[2] - p[1])
        if w1 < 10 or w2 < 10 or h1 < 10 or h2 < 10:
            return False
        return True

    def pixel_to_meters(self, u, v):
        """
        Transform a single point (u, v) in pixel coords to (x_m, y_m) on table plane.
        Returns (None, None) if the projected point is unreasonably far from the table,
        which happens when the ball is airborne (above the table plane) and the
        homography extrapolates to a divergent position.
        """
        if not self._valid or self.H is None:
            return None, None
        pt = np.array([[[float(u), float(v)]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self.H)
        x_m, y_m = float(out[0, 0, 0]), float(out[0, 0, 1])
        # Reject positions that are too far from the table surface.
        # The ball is often airborne, and the 2D homography projects such
        # points to wildly wrong locations on the table plane.
        if (x_m < -TABLE_MARGIN_M or x_m > TABLE_LENGTH_M + TABLE_MARGIN_M or
                y_m < -TABLE_MARGIN_M or y_m > TABLE_WIDTH_M + TABLE_MARGIN_M):
            return None, None
        return x_m, y_m

    def get_reprojection_error(self):
        """Mean reprojection error in meters (for logging)."""
        return self._reprojection_error

# =============================================================================
# GLOBAL STATE FOR INTERACTIVE MARKING
# =============================================================================
class MarkingState:
    def __init__(self):
        self.zoom_factor = 1.0
        self.zoom_center = None
        self.pan_offset = [0, 0]
        self.dragging = False
        self.drag_start = None
        
        # ROI marking
        self.current_roi = []
        self.marking_mode = None  # 'player1_score', 'player2_score', 'player1_rounds', 'player2_rounds'
        
        # Stored ROIs
        self.rois = {
            'player1_score': None,
            'player2_score': None,
            'player1_rounds': None,
            'player2_rounds': None
        }
        # Table corners for homography: 4 points (TL, TR, BR, BL) in original frame coords
        self.table_corners = []

        self.done = False
        self.original_frame = None
        self.display_frame = None

marking_state = MarkingState()

# =============================================================================
# INTERACTIVE FRAME SETUP
# =============================================================================
def get_zoomed_frame(frame, zoom_factor, pan_offset):
    """Apply zoom and pan to frame."""
    h, w = frame.shape[:2]
    
    if zoom_factor == 1.0 and pan_offset == [0, 0]:
        return frame.copy(), (0, 0, w, h)
    
    # Calculate visible region
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    
    # Center of zoom
    cx = w // 2 + pan_offset[0]
    cy = h // 2 + pan_offset[1]
    
    # Bounds
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    
    # Adjust if we hit boundaries
    if x2 - x1 < new_w:
        x1 = max(0, x2 - new_w)
    if y2 - y1 < new_h:
        y1 = max(0, y2 - new_h)
    
    # Crop and resize
    cropped = frame[y1:y2, x1:x2]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed, (x1, y1, x2, y2)


def screen_to_original_coords(x, y, zoom_factor, pan_offset, frame_shape):
    """Convert screen coordinates to original frame coordinates."""
    h, w = frame_shape[:2]
    
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    
    cx = w // 2 + pan_offset[0]
    cy = h // 2 + pan_offset[1]
    
    x1 = max(0, cx - new_w // 2)
    y1 = max(0, cy - new_h // 2)
    
    # Scale coordinates
    orig_x = x1 + int(x / zoom_factor)
    orig_y = y1 + int(y / zoom_factor)
    
    return orig_x, orig_y


def draw_rois_on_frame(frame, rois, zoom_factor, pan_offset, frame_shape):
    """Draw all marked ROIs on the frame."""
    colors = {
        'player1_score': (0, 255, 0),      # Green
        'player2_score': (0, 255, 255),    # Yellow
        'player1_rounds': (255, 0, 0),     # Blue
        'player2_rounds': (255, 0, 255)    # Magenta
    }
    
    labels = {
        'player1_score': 'P1 Score',
        'player2_score': 'P2 Score',
        'player1_rounds': 'P1 Rounds',
        'player2_rounds': 'P2 Rounds'
    }
    
    h, w = frame_shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    cx = w // 2 + pan_offset[0]
    cy = h // 2 + pan_offset[1]
    x1_off = max(0, cx - new_w // 2)
    y1_off = max(0, cy - new_h // 2)
    
    for roi_name, roi in rois.items():
        if roi is not None:
            x1, y1, x2, y2 = roi
            
            # Convert to screen coordinates
            sx1 = int((x1 - x1_off) * zoom_factor)
            sy1 = int((y1 - y1_off) * zoom_factor)
            sx2 = int((x2 - x1_off) * zoom_factor)
            sy2 = int((y2 - y1_off) * zoom_factor)
            
            color = colors.get(roi_name, (255, 255, 255))
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)
            cv2.putText(frame, labels.get(roi_name, roi_name), 
                       (sx1, sy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def draw_table_corners_on_frame(frame, table_corners, zoom_factor, pan_offset, frame_shape):
    """Draw table corner points and polygon on the display frame (zoomed coords)."""
    if not table_corners:
        return
    h, w = frame_shape[:2]
    new_w = int(w / zoom_factor)
    new_h = int(h / zoom_factor)
    cx = w // 2 + pan_offset[0]
    cy = h // 2 + pan_offset[1]
    x1_off = max(0, cx - new_w // 2)
    y1_off = max(0, cy - new_h // 2)
    pts = []
    for (ox, oy) in table_corners:
        sx = int((ox - x1_off) * zoom_factor)
        sy = int((oy - y1_off) * zoom_factor)
        pts.append((sx, sy))
        cv2.circle(frame, (sx, sy), 8, (0, 255, 255), 2)
    if len(pts) >= 2:
        for i in range(len(pts)):
            p1, p2 = pts[i], pts[(i + 1) % len(pts)]
            cv2.line(frame, p1, p2, (0, 255, 255), 2)
    if len(pts) == 4:
        cv2.putText(frame, "Table", (pts[0][0], pts[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def draw_instructions(frame, marking_mode, current_roi):
    """Draw instruction overlay."""
    h, w = frame.shape[:2]
    
    # Instructions (wider box for table line)
    instructions = [
        "CONTROLS:",
        "  Mouse Wheel: Zoom in/out",
        "  Right-click drag: Pan",
        "  Left-click: Draw ROI corners (or table corners for 5)",
        "",
        "KEYS:",
        "  1: Mark Player 1 Score Area",
        "  2: Mark Player 2 Score Area",
        "  3: Mark Player 1 Rounds Area",
        "  4: Mark Player 2 Rounds Area",
        "  5: Mark Table (4 corners: TL, TR, BR, BL)",
        "  R: Reset all markings",
        "  ENTER: Confirm and start tracking",
        "  ESC: Cancel"
    ]
    overlay_w = 480
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (overlay_w, 230), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    y = 30
    for line in instructions:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 15

    # Current mode indicator
    if marking_mode:
        mode_names = {
            'player1_score': 'MARKING: Player 1 Score',
            'player2_score': 'MARKING: Player 2 Score',
            'player1_rounds': 'MARKING: Player 1 Rounds',
            'player2_rounds': 'MARKING: Player 2 Rounds',
            'table_corners': 'MARKING: Table (TL, TR, BR, BL)'
        }
        cv2.putText(frame, mode_names.get(marking_mode, ''), 
                   (w - 320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw current ROI in progress
    if marking_mode != 'table_corners' and len(current_roi) == 1:
        cv2.circle(frame, current_roi[0], 5, (0, 0, 255), -1)
        cv2.putText(frame, "Click second corner", (current_roi[0][0] + 10, current_roi[0][1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def mouse_callback_setup(event, x, y, flags, param):
    """Handle mouse events for interactive setup."""
    global marking_state
    
    if event == cv2.EVENT_MOUSEWHEEL:
        # Zoom with mouse wheel
        if flags > 0:
            marking_state.zoom_factor = min(5.0, marking_state.zoom_factor * 1.1)
        else:
            marking_state.zoom_factor = max(1.0, marking_state.zoom_factor / 1.1)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Start panning
        marking_state.dragging = True
        marking_state.drag_start = (x, y)
    
    elif event == cv2.EVENT_RBUTTONUP:
        # Stop panning
        marking_state.dragging = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if marking_state.dragging and marking_state.drag_start:
            dx = marking_state.drag_start[0] - x
            dy = marking_state.drag_start[1] - y
            marking_state.pan_offset[0] += int(dx / marking_state.zoom_factor)
            marking_state.pan_offset[1] += int(dy / marking_state.zoom_factor)
            marking_state.drag_start = (x, y)
    
    elif event == cv2.EVENT_LBUTTONDOWN:
        if marking_state.marking_mode:
            # Convert screen coords to original frame coords
            orig_x, orig_y = screen_to_original_coords(
                x, y, marking_state.zoom_factor, marking_state.pan_offset,
                marking_state.original_frame.shape
            )

            if marking_state.marking_mode == 'table_corners':
                marking_state.table_corners.append((orig_x, orig_y))
                corner_names = ['TL', 'TR', 'BR', 'BL']
                idx = len(marking_state.table_corners) - 1
                print(f"Table corner {corner_names[idx]}: ({orig_x}, {orig_y})")
                if len(marking_state.table_corners) >= 4:
                    marking_state.marking_mode = None
                    print("Table corners complete.")
            else:
                marking_state.current_roi.append((x, y))  # Store screen coords for display

                if len(marking_state.current_roi) == 2:
                    # Complete ROI
                    p1 = screen_to_original_coords(
                        marking_state.current_roi[0][0], marking_state.current_roi[0][1],
                        marking_state.zoom_factor, marking_state.pan_offset,
                        marking_state.original_frame.shape
                    )
                    p2 = screen_to_original_coords(
                        marking_state.current_roi[1][0], marking_state.current_roi[1][1],
                        marking_state.zoom_factor, marking_state.pan_offset,
                        marking_state.original_frame.shape
                    )

                    x1, y1 = min(p1[0], p2[0]), min(p1[1], p2[1])
                    x2, y2 = max(p1[0], p2[0]), max(p1[1], p2[1])

                    marking_state.rois[marking_state.marking_mode] = (x1, y1, x2, y2)
                    marking_state.current_roi = []
                    marking_state.marking_mode = None
                    print(f"ROI marked: ({x1}, {y1}) to ({x2}, {y2})")


def interactive_frame_setup(frame):
    """Interactive first frame setup with zoom and ROI marking."""
    global marking_state
    
    marking_state.original_frame = frame.copy()
    marking_state.zoom_factor = 1.0
    marking_state.pan_offset = [0, 0]
    
    window_name = "Setup - Mark Score Regions"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback_setup)
    
    print("\n" + "="*60)
    print("INTERACTIVE SETUP MODE")
    print("="*60)
    print("Mark the score regions for each player.")
    print("Use mouse wheel to zoom, right-click drag to pan.")
    print("Press number keys (1-4) to select what to mark, then click two corners.")
    print("Press ENTER when done, ESC to cancel.")
    print("="*60 + "\n")
    
    while not marking_state.done:
        # Get zoomed frame
        display, visible_region = get_zoomed_frame(
            marking_state.original_frame, 
            marking_state.zoom_factor, 
            marking_state.pan_offset
        )
        
        # Draw ROIs
        draw_rois_on_frame(
            display, marking_state.rois, 
            marking_state.zoom_factor, marking_state.pan_offset,
            marking_state.original_frame.shape
        )
        # Draw table corners
        draw_table_corners_on_frame(
            display, marking_state.table_corners,
            marking_state.zoom_factor, marking_state.pan_offset,
            marking_state.original_frame.shape
        )
        
        # Draw current ROI in progress
        if len(marking_state.current_roi) == 1:
            cv2.circle(display, marking_state.current_roi[0], 5, (0, 0, 255), -1)
        
        # Draw instructions
        draw_instructions(display, marking_state.marking_mode, marking_state.current_roi)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            return None
        elif key == 13:  # ENTER
            marking_state.done = True
        elif key == ord('1'):
            marking_state.marking_mode = 'player1_score'
            marking_state.current_roi = []
            print("Mode: Marking Player 1 Score Area")
        elif key == ord('2'):
            marking_state.marking_mode = 'player2_score'
            marking_state.current_roi = []
            print("Mode: Marking Player 2 Score Area")
        elif key == ord('3'):
            marking_state.marking_mode = 'player1_rounds'
            marking_state.current_roi = []
            print("Mode: Marking Player 1 Rounds Area")
        elif key == ord('4'):
            marking_state.marking_mode = 'player2_rounds'
            marking_state.current_roi = []
            print("Mode: Marking Player 2 Rounds Area")
        elif key == ord('5'):
            marking_state.marking_mode = 'table_corners'
            marking_state.table_corners = []
            marking_state.current_roi = []
            print("Mode: Mark Table. Click 4 corners in order: TL, TR, BR, BL")
        elif key == ord('r'):
            marking_state.rois = {k: None for k in marking_state.rois}
            marking_state.table_corners = []
            marking_state.current_roi = []
            marking_state.marking_mode = None
            print("All markings reset")
    
    cv2.destroyWindow(window_name)
    return marking_state.rois, marking_state.table_corners


# =============================================================================
# BALL TRACKING AND SPEED CALCULATION
# =============================================================================
class BallTracker:
    def __init__(self, model_path, fps, table_calibration=None, inference_size=None):
        """
        inference_size: (width, height) to rescale the frame before YOLO inference,
                        or None to run inference at the original resolution.
                        Detections are always mapped back to the original-frame
                        coordinate space before returning.
        """
        self.model = YOLO(model_path)
        self.tracker = Sort(
            max_age=TRACKER_MAX_AGE,
            min_hits=TRACKER_MIN_HITS,
            iou_threshold=TRACKER_IOU_THRESHOLD
        )
        self.fps = fps
        self.table_calibration = table_calibration  # TableCalibration or None
        self.inference_size = inference_size  # (w, h) or None

        # Trajectory history per track ID (pixel coords)
        self.trajectories = {}
        self.speed_history = {}
        # Meter-space trajectory and speed when calibration is set
        self.trajectories_meters = {}
        self.speed_history_mps = {}

    def _prepare_inference_frame(self, frame):
        """
        Return (inference_frame, scale_x, scale_y).
        scale_x / scale_y are the factors needed to map inference-frame pixel
        coords back to original-frame pixel coords.
        If inference_size is None the original frame is returned unchanged with
        scale factors of 1.0.
        """
        if self.inference_size is None:
            return frame, 1.0, 1.0
        iw, ih = self.inference_size
        oh, ow = frame.shape[:2]
        # Avoid an unnecessary copy if the frame is already the target size
        if ow == iw and oh == ih:
            return frame, 1.0, 1.0
        small = cv2.resize(frame, (iw, ih), interpolation=cv2.INTER_LINEAR)
        return small, ow / iw, oh / ih

    def detect_and_track(self, frame):
        """Detect ball and update tracking."""
        inf_frame, sx, sy = self._prepare_inference_frame(frame)
        results = self.model(inf_frame, conf=BALL_CONF_THRESHOLD, verbose=False)

        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                # Scale coordinates back to original frame space
                detections.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy, conf])
        
        # Convert to numpy array
        if detections:
            dets = np.array(detections)
        else:
            dets = np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(dets)
        
        # Update trajectories
        dt_sec = 1.0 / self.fps if self.fps > 0 else 1.0
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = deque(maxlen=TRAJECTORY_LENGTH)
                self.speed_history[track_id] = deque(maxlen=SPEED_SMOOTHING_WINDOW)
                self.trajectories_meters[track_id] = deque(maxlen=TRAJECTORY_LENGTH)
                self.speed_history_mps[track_id] = deque(maxlen=SPEED_SMOOTHING_WINDOW)
            
            self.trajectories[track_id].append((cx, cy))
            
            # Calculate speed (pixels per second)
            if len(self.trajectories[track_id]) >= 2:
                prev_x, prev_y = self.trajectories[track_id][-2]
                dist_pixels = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                speed_pps = dist_pixels * self.fps
                self.speed_history[track_id].append(speed_pps)
            
            # Meter-space position and speed when calibration is available
            if self.table_calibration and self.table_calibration.is_valid():
                x_m, y_m = self.table_calibration.pixel_to_meters(cx, cy)
                if x_m is not None:
                    self.trajectories_meters[track_id].append((x_m, y_m))
                    if len(self.trajectories_meters[track_id]) >= 2:
                        px, py = self.trajectories_meters[track_id][-2]
                        dist_m = np.sqrt((x_m - px)**2 + (y_m - py)**2)
                        speed_mps = dist_m / dt_sec
                        if speed_mps <= MAX_BALL_SPEED_MPS:
                            self.speed_history_mps[track_id].append(speed_mps)
        
        return tracks
    
    def get_smoothed_speed(self, track_id):
        """Get smoothed speed for a track (pixels per second)."""
        if track_id in self.speed_history and self.speed_history[track_id]:
            return np.mean(self.speed_history[track_id])
        return 0
    
    def get_smoothed_speed_mps(self, track_id):
        """Get smoothed speed in m/s (median for outlier robustness)."""
        if track_id in self.speed_history_mps and self.speed_history_mps[track_id]:
            return float(np.median(self.speed_history_mps[track_id]))
        return None
    
    def get_position_meters(self, track_id):
        """Get latest ball position in table-plane meters (x_m, y_m) or (None, None)."""
        if not self.table_calibration or track_id not in self.trajectories_meters or not self.trajectories_meters[track_id]:
            return None, None
        x_m, y_m = self.trajectories_meters[track_id][-1]
        return x_m, y_m
    
    def draw_trajectories(self, frame, tracks):
        """Draw ball trajectories and speed on frame."""
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Draw trajectory
            if track_id in self.trajectories:
                points = list(self.trajectories[track_id])
                for i in range(1, len(points)):
                    # Color gradient from red (old) to green (new)
                    alpha = i / len(points)
                    color = (0, int(255 * alpha), int(255 * (1 - alpha)))
                    thickness = max(1, int(3 * alpha))
                    
                    pt1 = (int(points[i-1][0]), int(points[i-1][1]))
                    pt2 = (int(points[i][0]), int(points[i][1]))
                    cv2.line(frame, pt1, pt2, color, thickness)
            
            # Draw speed (m/s when table calibration is valid, else px/s)
            speed_mps = self.get_smoothed_speed_mps(track_id)
            if speed_mps is not None:
                speed_text = f"{speed_mps:.2f} m/s"
            else:
                speed = self.get_smoothed_speed(track_id)
                speed_text = f"{speed:.0f} px/s"
            cv2.putText(frame, speed_text, (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Draw ball center
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)


# =============================================================================
# SCORE DETECTION WITH YOLO + PREPROCESSING
# =============================================================================
class ScoreDetector:
    def __init__(self, model_path):
        """Initialize YOLO digit model with image preprocessing."""
        self.model = YOLO(model_path)
        
        self.score_history = {
            'player1': deque(maxlen=5),
            'player2': deque(maxlen=5)
        }
        self.current_scores = {
            'player1': None,
            'player2': None
        }
        self.rounds = {
            'player1': 0,
            'player2': 0
        }
    
    def stop(self):
        """Cleanup (no-op for YOLO, kept for API compatibility)."""
        pass
    
    def preprocess_for_detection(self, crop):
        """
        Apply preprocessing to make digits clearer for YOLO detection.
        Returns both the processed image and the original for flexibility.
        """
        if crop is None or crop.size == 0:
            return None
        
        h, w = crop.shape[:2]
        
        # Resize small crops for better detection
        min_size = 64
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        
        # Convert to grayscale for processing
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Otsu's thresholding for automatic threshold selection
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Determine if we need to invert (digits should be visible)
        # Check if the image is mostly dark or light
        if np.mean(binary) < 127:
            binary = cv2.bitwise_not(binary)
        
        # Convert back to BGR for YOLO (it expects 3 channels)
        processed = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        
        return processed
    
    def detect_digits_in_roi(self, frame, roi, use_preprocessing=True, debug=False):
        """Detect digits in a region of interest using YOLO."""
        if roi is None:
            return None
        
        x1, y1, x2, y2 = roi
        
        # Ensure valid crop bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        crop = frame[y1:y2, x1:x2]
        
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return None
        
        # Try with preprocessing first, then without if no result
        images_to_try = []
        
        if use_preprocessing:
            processed = self.preprocess_for_detection(crop)
            if processed is not None:
                images_to_try.append(processed)
        
        # Also try original image
        images_to_try.append(crop)
        
        for img in images_to_try:
            results = self.model(img, conf=DIGIT_CONF_THRESHOLD, imgsz=320, verbose=False)
            
            if not results or len(results[0].boxes) == 0:
                continue
            
            digits = []
            for box in results[0].boxes:
                bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                cls = int(box.cls.item())
                conf = box.conf.item()
                
                # Only accept digits 0-9
                if cls < 0 or cls > 9:
                    continue
                
                cx = (bx1 + bx2) / 2
                digits.append({
                    'digit': str(cls),
                    'x': cx,
                    'conf': conf,
                    'bbox': (bx1, by1, bx2, by2)
                })
            
            if not digits:
                continue
            
            # Sort by x position (left to right)
            digits.sort(key=lambda d: d['x'])
            
            # Combine digits (max 2 for score)
            score_str = ''.join(d['digit'] for d in digits[:2])
            
            try:
                score = int(score_str)
                if score > 30:  # Invalid table tennis score
                    continue
                return score
            except ValueError:
                continue
        
        return None
    
    def update_scores(self, frame, rois, frame_idx):
        """Update scores from detected digits."""
        for player, roi_key in [('player1', 'player1_score'), ('player2', 'player2_score')]:
            roi = rois.get(roi_key)
            if roi:
                score = self.detect_digits_in_roi(frame, roi)
                if score is not None:
                    self.score_history[player].append(score)
                    
                    # Use majority voting for stability
                    if len(self.score_history[player]) >= 3:
                        from collections import Counter
                        counts = Counter(self.score_history[player])
                        most_common = counts.most_common(1)[0]
                        if most_common[1] >= 2:  # At least 2 occurrences
                            self.current_scores[player] = most_common[0]
        
        self.last_processed_frame = frame_idx
        return self.current_scores
    
    def detect_rounds(self, frame, rois):
        """Detect rounds won from round indicator ROIs."""
        for player, roi_key in [('player1', 'player1_rounds'), ('player2', 'player2_rounds')]:
            roi = rois.get(roi_key)
            if roi:
                rounds = self.detect_digits_in_roi(frame, roi)
                if rounds is not None and rounds <= 5:  # Max 5 games in a match
                    self.rounds[player] = rounds
        
        return self.rounds
    
    def draw_scores(self, frame, rois):
        """Draw score overlays on frame."""
        h, w = frame.shape[:2]
        
        # Draw score boxes at top of frame
        overlay = frame.copy()
        
        # Player 1 score box (left)
        cv2.rectangle(overlay, (20, 20), (200, 100), (0, 100, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        p1_score = self.current_scores['player1']
        p1_score_text = str(p1_score) if p1_score is not None else "--"
        cv2.putText(frame, "Player 1", (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, p1_score_text, (30, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Sets: {self.rounds['player1']}", (120, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Player 2 score box (right)
        overlay = frame.copy()
        cv2.rectangle(overlay, (w - 200, 20), (w - 20, 100), (100, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        p2_score = self.current_scores['player2']
        p2_score_text = str(p2_score) if p2_score is not None else "--"
        cv2.putText(frame, "Player 2", (w - 190, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, p2_score_text, (w - 190, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(frame, f"Sets: {self.rounds['player2']}", (w - 90, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Draw ROI boxes on frame
        for roi_name, roi in rois.items():
            if roi:
                x1, y1, x2, y2 = roi
                if 'player1' in roi_name:
                    color = (0, 255, 0)
                else:
                    color = (255, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)


# =============================================================================
# RALLY AGGREGATION (SCORE-CHANGE TO SCORE-CHANGE)
# =============================================================================
class RallyAggregator:
    """
    Aggregates ball trajectory in meter space per rally (score change to score change).
    Outputs one row per rally for the prediction module.
    """
    def __init__(self, fps, data_logger, table_calibration=None):
        self.fps = fps
        self.logger = data_logger
        self.table_calibration = table_calibration
        self.rally_id = 0
        self.rally_start_frame = 0
        self.rally_start_time = 0.0
        self.rally_start_scores = (None, None)  # score at start of current rally
        self.rally_start_sets = (None, None)
        self.samples = []  # list of (frame_idx, x_m, y_m, speed_mps)
        self.last_scores = (None, None)  # (p1, p2) previous frame
        self.last_sets = (None, None)
    
    def _table_contains(self, x_m, y_m):
        """True if (x_m, y_m) is inside table plane (with small margin)."""
        if x_m is None or y_m is None:
            return False
        margin = 0.1
        return (-margin <= x_m <= TABLE_LENGTH_M + margin and
                -margin <= y_m <= TABLE_WIDTH_M + margin)
    
    def _landing_zone_index(self, x_m, y_m):
        """3x3 grid over table. Returns 0-8 or None if outside."""
        if not self._table_contains(x_m, y_m):
            return None
        x = max(0, min(TABLE_LENGTH_M, x_m))
        y = max(0, min(TABLE_WIDTH_M, y_m))
        col = min(2, int(x / (TABLE_LENGTH_M / 3)))
        row = min(2, int(y / (TABLE_WIDTH_M / 3)))
        return row * 3 + col
    
    def add_frame(self, frame_idx, timestamp_sec, tracks_with_meters, scores, rounds):
        """
        Call each frame. tracks_with_meters: list of (track_id, x_m, y_m, speed_mps).
        When score changes, flush current rally and start next.
        """
        p1, p2 = scores.get('player1'), scores.get('player2')
        r1, r2 = rounds.get('player1', 0), rounds.get('player2', 0)
        current_scores = (p1, p2)
        current_sets = (r1, r2)
        
        # First time we have scores: set rally start
        if self.rally_start_scores[0] is None and p1 is not None and p2 is not None:
            self.rally_start_scores = current_scores
            self.rally_start_sets = current_sets
        
        # Detect score change (point ended)
        if self.last_scores[0] is not None and self.last_scores[1] is not None:
            if (p1, p2) != self.last_scores:
                # Flush previous rally: winner = whoever's score increased
                if p1 != self.last_scores[0] and p2 == self.last_scores[1]:
                    point_winner = 'p1'
                elif p2 != self.last_scores[1] and p1 == self.last_scores[0]:
                    point_winner = 'p2'
                else:
                    point_winner = 'unknown'
                self._flush_rally(frame_idx, timestamp_sec, point_winner)
                self.rally_id += 1
                self.rally_start_frame = frame_idx
                self.rally_start_time = timestamp_sec
                self.rally_start_scores = current_scores
                self.rally_start_sets = current_sets
                self.samples = []
        
        self.last_scores = current_scores
        self.last_sets = current_sets
        
        # Accumulate samples for current rally (only if we have meter-space data)
        if self.table_calibration and self.table_calibration.is_valid() and tracks_with_meters:
            for (track_id, x_m, y_m, speed_mps) in tracks_with_meters:
                if x_m is not None and y_m is not None:
                    self.samples.append((frame_idx, x_m, y_m, speed_mps or 0.0))
    
    def _flush_rally(self, end_frame, end_time, point_winner):
        """Write one rally row and reset buffer."""
        record = {
            'rally_id': self.rally_id,
            'rally_start_frame': self.rally_start_frame,
            'rally_end_frame': end_frame,
            'rally_start_time': self.rally_start_time,
            'rally_end_time': end_time,
            'p1_score_start': self.rally_start_scores[0] if self.rally_start_scores[0] is not None else '',
            'p2_score_start': self.rally_start_scores[1] if self.rally_start_scores[1] is not None else '',
            'p1_sets_start': self.rally_start_sets[0] if self.rally_start_sets[0] is not None else '',
            'p2_sets_start': self.rally_start_sets[1] if self.rally_start_sets[1] is not None else '',
            'point_winner': point_winner
        }
        if not self.samples:
            record['mean_speed_mps'] = ''
            record['max_speed_mps'] = ''
            record['speed_std_mps'] = ''
            for i in range(9):
                record[f'landing_zone_{i}'] = 0
        else:
            speeds = [s[3] for s in self.samples if s[3] is not None and s[3] > 0]
            record['mean_speed_mps'] = f"{np.mean(speeds):.4f}" if speeds else ''
            record['max_speed_mps'] = f"{max(speeds):.4f}" if speeds else ''
            record['speed_std_mps'] = f"{np.std(speeds):.4f}" if len(speeds) > 1 else ''
            # Landing zone histogram (3x3)
            zones = [0] * 9
            for _, x_m, y_m, _ in self.samples:
                idx = self._landing_zone_index(x_m, y_m)
                if idx is not None:
                    zones[idx] += 1
            for i in range(9):
                record[f'landing_zone_{i}'] = zones[i]
        self.logger.log_rally(record)
    
    def flush_final(self, end_frame, end_time):
        """Call at end of video to write the last rally if any."""
        if self.samples or self.last_scores[0] is not None:
            self._flush_rally(end_frame, end_time, 'unknown')


# =============================================================================
# DATA LOGGING
# =============================================================================
class DataLogger:
    def __init__(self, output_dir, with_meters=False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.with_meters = with_meters  # If True, trajectory CSV includes x_m, y_m, speed_mps
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Trajectory log
        self.trajectory_file = self.output_dir / f"trajectories_{timestamp}.csv"
        header = "frame,track_id,x,y,speed_pps"
        if with_meters:
            header += ",x_m,y_m,speed_mps"
        with open(self.trajectory_file, 'w') as f:
            f.write(header + "\n")
        
        # Score log
        self.score_file = self.output_dir / f"scores_{timestamp}.csv"
        with open(self.score_file, 'w') as f:
            f.write("frame,timestamp_sec,player1_score,player2_score,player1_sets,player2_sets\n")
        
        # Rally log (Stage 1: ball + score + placement)
        self.rally_file = self.output_dir / f"rallies_{timestamp}.csv"
        self._write_rally_header()
        
        # ROI config
        self.config_file = self.output_dir / f"config_{timestamp}.json"
    
    def _write_rally_header(self):
        with open(self.rally_file, 'w') as f:
            f.write(
                "rally_id,rally_start_frame,rally_end_frame,rally_start_time,rally_end_time,"
                "p1_score_start,p2_score_start,p1_sets_start,p2_sets_start,"
                "mean_speed_mps,max_speed_mps,speed_std_mps,"
                "landing_zone_0,landing_zone_1,landing_zone_2,landing_zone_3,landing_zone_4,"
                "landing_zone_5,landing_zone_6,landing_zone_7,landing_zone_8,"
                "point_winner\n"
            )
    
    def log_trajectory(self, frame_idx, track_id, x, y, speed, x_m=None, y_m=None, speed_mps=None):
        line = f"{frame_idx},{track_id},{x:.2f},{y:.2f},{speed:.2f}"
        if self.with_meters and x_m is not None and y_m is not None:
            sm = f"{speed_mps:.4f}" if speed_mps is not None else ""
            line += f",{x_m:.4f},{y_m:.4f},{sm}"
        with open(self.trajectory_file, 'a') as f:
            f.write(line + "\n")
    
    def log_score(self, frame_idx, timestamp, scores, rounds):
        with open(self.score_file, 'a') as f:
            p1 = scores['player1'] if scores['player1'] is not None else ''
            p2 = scores['player2'] if scores['player2'] is not None else ''
            f.write(f"{frame_idx},{timestamp:.2f},{p1},{p2},{rounds['player1']},{rounds['player2']}\n")
    
    def log_rally(self, record):
        """Append one rally feature row (dict with keys matching CSV header)."""
        with open(self.rally_file, 'a') as f:
            parts = [
                record.get('rally_id', ''),
                record.get('rally_start_frame', ''),
                record.get('rally_end_frame', ''),
                record.get('rally_start_time', ''),
                record.get('rally_end_time', ''),
                record.get('p1_score_start', ''),
                record.get('p2_score_start', ''),
                record.get('p1_sets_start', ''),
                record.get('p2_sets_start', ''),
                record.get('mean_speed_mps', ''),
                record.get('max_speed_mps', ''),
                record.get('speed_std_mps', ''),
            ]
            for i in range(9):
                parts.append(record.get(f'landing_zone_{i}', 0))
            parts.append(record.get('point_winner', ''))
            f.write(','.join(str(p) for p in parts) + "\n")
    
    def save_config(self, rois, video_path, fps, table_corners=None):
        config = {
            'video_path': str(video_path),
            'fps': fps,
            'rois': {k: list(v) if v else None for k, v in rois.items()},
            'ball_model': BALL_MODEL_PATH,
            'digit_model': DIGIT_MODEL_PATH
        }
        if table_corners is not None and len(table_corners) == 4:
            config['table_corners'] = [[float(p[0]), float(p[1])] for p in table_corners]
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)


# =============================================================================
# REAL-TIME STATS DISPLAY
# =============================================================================
class RealtimeStats:
    """Track and display real-time performance stats."""
    def __init__(self, video_fps):
        self.video_fps = video_fps
        self.frame_times = deque(maxlen=30)
        self.last_time = None
        self.current_fps = 0
        self.ball_detected = False
        self.current_speed = 0
        self.max_speed = 0
        self.current_speed_mps = None  # m/s when table calibration is used
        self.max_speed_mps = None

        # Per-phase timing (milliseconds, rolling 30-frame averages)
        self.t_ball_infer = deque(maxlen=30)   # ball YOLO inference + tracker
        self.t_score_infer = deque(maxlen=30)  # score YOLO (when it runs)
        self.t_draw = deque(maxlen=30)         # drawing overlays
        self.t_write = deque(maxlen=30)        # video writer

    def update(self):
        import time
        current = time.time()
        if self.last_time is not None:
            self.frame_times.append(current - self.last_time)
            if self.frame_times:
                self.current_fps = 1.0 / np.mean(self.frame_times)
        self.last_time = current

    def record_phase(self, phase, elapsed_sec):
        """Record elapsed time (seconds) for a named phase."""
        ms = elapsed_sec * 1000.0
        if phase == 'ball_infer':
            self.t_ball_infer.append(ms)
        elif phase == 'score_infer':
            self.t_score_infer.append(ms)
        elif phase == 'draw':
            self.t_draw.append(ms)
        elif phase == 'write':
            self.t_write.append(ms)

    def update_ball_stats(self, detected, speed, speed_mps=None):
        self.ball_detected = detected
        if detected and speed > 0:
            self.current_speed = speed
            self.max_speed = max(self.max_speed, speed)
        if speed_mps is not None and speed_mps > 0:
            self.current_speed_mps = speed_mps
            self.max_speed_mps = max(self.max_speed_mps or 0, speed_mps)

    def draw(self, frame, frame_idx, total_frames, paused, playback_speed):
        h, w = frame.shape[:2]

        # Stats panel background (bottom-left) — extended for timing rows
        overlay = frame.copy()
        panel_h = 155
        cv2.rectangle(overlay, (10, h - panel_h - 10), (310, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        y = h - panel_h + 5
        line_height = 18

        # Processing FPS
        fps_color = (0, 255, 0) if self.current_fps >= self.video_fps * 0.9 else (0, 255, 255) if self.current_fps >= self.video_fps * 0.5 else (0, 0, 255)
        cv2.putText(frame, f"Processing: {self.current_fps:.1f} FPS", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1)
        y += line_height

        # Video FPS
        cv2.putText(frame, f"Video: {self.video_fps:.1f} FPS | Speed: {playback_speed:.1f}x",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y += line_height

        # Ball status
        ball_color = (0, 255, 0) if self.ball_detected else (100, 100, 100)
        ball_status = "DETECTED" if self.ball_detected else "NOT FOUND"
        cv2.putText(frame, f"Ball: {ball_status}", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ball_color, 1)
        y += line_height

        # Speed (m/s when table calibrated, else px/s)
        if self.max_speed_mps is not None and self.max_speed_mps > 0:
            cur = f"{self.current_speed_mps:.2f}" if self.current_speed_mps is not None else "0.00"
            cv2.putText(frame, f"Speed: {cur} m/s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += line_height
            cv2.putText(frame, f"Max Speed: {self.max_speed_mps:.2f} m/s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += line_height
        else:
            cv2.putText(frame, f"Speed: {self.current_speed:.0f} px/s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            y += line_height
            cv2.putText(frame, f"Max Speed: {self.max_speed:.0f} px/s", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
            y += line_height

        # Per-phase timings
        def _avg_ms(q):
            return f"{np.mean(q):.1f}" if q else "---"

        cv2.putText(frame, f"BallInfer: {_avg_ms(self.t_ball_infer)}ms  Score: {_avg_ms(self.t_score_infer)}ms",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (160, 200, 255), 1)
        y += line_height
        cv2.putText(frame, f"Draw: {_avg_ms(self.t_draw)}ms  Write: {_avg_ms(self.t_write)}ms",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (160, 200, 255), 1)
        y += line_height

        # Progress
        progress = frame_idx / total_frames if total_frames > 0 else 0
        time_sec = frame_idx / self.video_fps
        total_time = total_frames / self.video_fps
        cv2.putText(frame, f"Time: {time_sec:.1f}s / {total_time:.1f}s ({progress*100:.1f}%)",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Progress bar (top)
        bar_y = 5
        bar_h = 8
        cv2.rectangle(frame, (10, bar_y), (w - 10, bar_y + bar_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, bar_y), (10 + int((w - 20) * progress), bar_y + bar_h), (0, 200, 0), -1)

        # Pause indicator
        if paused:
            cv2.putText(frame, "PAUSED", (w // 2 - 50, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        # Controls hint (top-right)
        cv2.putText(frame, "Q:Quit P:Pause +/-:Speed S:Screenshot",
                    (w - 350, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)


# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main(video_path, output_dir="tracking_output", save_video=True,
         inference_size=None, benchmark_frames=0,
         score_interval_sec=2.0, output_size=None, use_tensorrt=False):
    """
    Main entry point for ball tracking and score detection.

    inference_size:     (width, height) for ball-detector rescaling, or None.
    benchmark_frames:   if > 0, run headless for this many frames and exit.
    score_interval_sec: seconds between score-detector runs (default 2.0).
    output_size:        (width, height) to downscale the saved output video,
                        or None to write at the original resolution.
    use_tensorrt:       if True, require .engine models (fail if missing).
    """
    import time

    # Resolve model paths (TensorRT .engine preferred when present)
    ball_model_path = resolve_model_path(BALL_MODEL_PATH, require_engine=use_tensorrt)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH, require_engine=use_tensorrt)
    if use_tensorrt:
        for name, path in [("Ball", ball_model_path), ("Digit", digit_model_path)]:
            if not Path(path).exists():
                print(f"Error: --use-tensorrt set but {name} engine not found: {path}")
                print("Export with: model.export(format='engine', imgsz=..., half=True, device=0)")
                return

    print("\n" + "="*60)
    print("BALL TRACKING AND SCORE DETECTION SYSTEM")
    print("="*60)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Ball model:   {ball_model_path}")
    print(f"Digit model:  {digit_model_path}")
    if inference_size:
        print(f"Ball inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print(f"Ball inference size: {frame_width}x{frame_height} (full resolution / baseline)")
    if output_size:
        print(f"Output video size:   {output_size[0]}x{output_size[1]}")
    else:
        print(f"Output video size:   {frame_width}x{frame_height} (full resolution)")

    # -------------------------------------------------------------------------
    # BENCHMARK MODE: headless, no interactive setup, no display
    # -------------------------------------------------------------------------
    if benchmark_frames > 0:
        _run_benchmark(cap, fps, frame_width, frame_height,
                       inference_size, benchmark_frames, ball_model_path)
        cap.release()
        return

    # Read first frame for setup
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return

    # Interactive setup
    print("\nStarting interactive setup...")
    result = interactive_frame_setup(first_frame)
    if result is None:
        print("Setup cancelled.")
        return
    rois, table_corners = result

    print("\nROIs configured:")
    for name, roi in rois.items():
        print(f"  {name}: {roi}")
    if len(table_corners) == 4:
        print("  Table corners: 4 points set (homography enabled)")
    else:
        print("  Table corners: not set (ball speed in px/s only)")

    # Table calibration (optional)
    table_calibration = None
    if len(table_corners) == 4:
        table_calibration = TableCalibration(table_corners)
        if table_calibration.is_valid():
            print(f"  Table calibration OK (reprojection error ~{table_calibration.get_reprojection_error():.4f} m)")
        else:
            print("  Table calibration failed sanity checks; continuing without meter-space features.")
            table_calibration = None

    # Initialize components
    print("\nLoading models...")
    print("  - Loading ball detection model...")
    ball_tracker = BallTracker(ball_model_path, fps,
                               table_calibration=table_calibration,
                               inference_size=inference_size)
    print("  - Loading digit detection model...")
    score_detector = ScoreDetector(digit_model_path)
    logger = DataLogger(output_dir, with_meters=(table_calibration is not None and table_calibration.is_valid()))
    rally_aggregator = RallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    print("  - Models loaded!")

    # Save configuration
    logger.save_config(rois, video_path, fps, table_corners=table_corners)

    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Create output window
    cv2.namedWindow("Ball Tracking - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracking - Real-Time", 1280, 720)

    # Video writer for output
    video_writer = None
    output_video_path = None
    write_w = output_size[0] if output_size else frame_width
    write_h = output_size[1] if output_size else frame_height
    if save_video:
        output_video_path = Path(output_dir) / f"tracked_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps,
            (write_w, write_h)
        )
        print(f"Output video will be saved to: {output_video_path} ({write_w}x{write_h})")

    # Score detection interval (configurable to reduce expensive YOLO calls)
    score_detect_interval = max(1, int(fps * score_interval_sec))
    last_score_log = None

    print("\n" + "="*60)
    print("REAL-TIME TRACKING STARTED")
    print("="*60)
    print("Controls:")
    print("  Q      : Quit")
    print("  P      : Pause/Resume")
    print("  +/=    : Speed up playback")
    print("  -      : Slow down playback")
    print("  0      : Reset to 1x speed")
    print("  S      : Save screenshot")
    print("  </>    : Skip backward/forward 5 seconds")
    print("  SPACE  : Step frame (when paused)")
    print("="*60 + "\n")

    frame_idx = 0
    paused = False
    playback_speed = 1.0  # 1.0 = real-time
    frame_delay_base = 1.0 / fps  # Base delay between frames

    while cap.isOpened():
        loop_start = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                # End of video - flush last rally and stop
                print("\nEnd of video reached.")
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break

            # Ball detection and tracking (timed)
            t0 = time.perf_counter()
            tracks = ball_tracker.detect_and_track(frame)
            stats.record_phase('ball_infer', time.perf_counter() - t0)

            # Update stats
            ball_detected = len(tracks) > 0
            max_speed = 0

            # Log trajectories (with optional meter-space columns)
            tracks_with_meters = []
            max_speed_mps = None
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                speed = ball_tracker.get_smoothed_speed(track_id)
                max_speed = max(max_speed, speed)
                x_m, y_m = ball_tracker.get_position_meters(track_id)
                speed_mps = ball_tracker.get_smoothed_speed_mps(track_id)
                if speed_mps is not None:
                    max_speed_mps = max(max_speed_mps or 0, speed_mps)
                logger.log_trajectory(frame_idx, track_id, cx, cy, speed, x_m=x_m, y_m=y_m, speed_mps=speed_mps)
                tracks_with_meters.append((track_id, x_m, y_m, speed_mps))

            stats.update_ball_stats(ball_detected, max_speed, max_speed_mps)

            # Score detection (every N frames for performance, timed)
            if frame_idx % score_detect_interval == 0:
                t0 = time.perf_counter()
                scores = score_detector.update_scores(frame, rois, frame_idx)
                rounds = score_detector.detect_rounds(frame, rois)
                stats.record_phase('score_infer', time.perf_counter() - t0)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds

            # Rally aggregation (per-frame; flushes on score change)
            rally_aggregator.add_frame(frame_idx, frame_idx / fps, tracks_with_meters, scores, rounds)

            # Log if scores changed
            current_score_log = (scores['player1'], scores['player2'],
                                 rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                timestamp = frame_idx / fps
                logger.log_score(frame_idx, timestamp, scores, rounds)
                last_score_log = current_score_log

            # Draw overlays (timed)
            t0 = time.perf_counter()
            ball_tracker.draw_trajectories(frame, tracks)
            score_detector.draw_scores(frame, rois)
            stats.record_phase('draw', time.perf_counter() - t0)

            # Update FPS stats
            stats.update()

            # Write to output video (downscale if requested, timed)
            if video_writer:
                t0 = time.perf_counter()
                write_frame = (cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR)
                               if output_size else frame)
                video_writer.write(write_frame)
                stats.record_phase('write', time.perf_counter() - t0)

            frame_idx += 1

        # Draw real-time stats overlay (after saving clean frame)
        stats.draw(frame, frame_idx, total_frames, paused, playback_speed)

        # Display
        cv2.imshow("Ball Tracking - Real-Time", frame)
        
        # Calculate wait time for real-time playback
        processing_time = time.time() - loop_start
        wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))
        
        if paused:
            wait_time = 0  # Block until key press when paused
        
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord(' ') and paused:
            # Step one frame when paused
            ret, frame = cap.read()
            if ret:
                tracks = ball_tracker.detect_and_track(frame)
                ball_tracker.draw_trajectories(frame, tracks)
                score_detector.draw_scores(frame, rois)
                frame_idx += 1
        elif key in [ord('+'), ord('=')]:
            playback_speed = min(4.0, playback_speed + 0.25)
            print(f"Playback speed: {playback_speed}x")
        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed - 0.25)
            print(f"Playback speed: {playback_speed}x")
        elif key == ord('0'):
            playback_speed = 1.0
            print("Playback speed: 1.0x (real-time)")
        elif key == ord('s'):
            screenshot_path = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"Screenshot saved: {screenshot_path}")
        elif key == ord('.') or key == ord('>'):
            # Skip forward 5 seconds
            skip_frames = int(fps * 5)
            new_pos = min(frame_idx + skip_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_idx = new_pos
            print(f"Skipped to frame {frame_idx}")
        elif key == ord(',') or key == ord('<'):
            # Skip backward 5 seconds
            skip_frames = int(fps * 5)
            new_pos = max(0, frame_idx - skip_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_idx = new_pos
            # Reset tracker for new position
            ball_tracker.tracker = Sort(
                max_age=TRACKER_MAX_AGE,
                min_hits=TRACKER_MIN_HITS,
                iou_threshold=TRACKER_IOU_THRESHOLD
            )
            ball_tracker.trajectories.clear()
            ball_tracker.speed_history.clear()
            ball_tracker.trajectories_meters.clear()
            ball_tracker.speed_history_mps.clear()
            print(f"Skipped to frame {frame_idx}")
    
    # Cleanup
    score_detector.stop()  # Stop OCR thread
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log: {logger.score_file}")
    print(f"Rally log: {logger.rally_file}")
    print(f"Configuration: {logger.config_file}")
    if save_video and output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"\nMax ball speed recorded: {stats.max_speed:.0f} px/s")


def load_config_and_run(video_path, config_path, output_dir="tracking_output",
                        save_video=True, inference_size=None,
                        score_interval_sec=2.0, output_size=None, use_tensorrt=False):
    """
    Run tracking with pre-saved ROI configuration (skip interactive setup).
    Useful for batch processing or re-running with same settings.

    inference_size:     (width, height) for ball-detector rescaling, or None.
    score_interval_sec: seconds between score-detector runs (default 2.0).
    output_size:        (width, height) to downscale the saved output video,
                        or None to write at the original resolution.
    use_tensorrt:       if True, require .engine models (fail if missing).
    """
    import time

    # Resolve model paths (TensorRT .engine preferred when present)
    ball_model_path = resolve_model_path(BALL_MODEL_PATH, require_engine=use_tensorrt)
    digit_model_path = resolve_model_path(DIGIT_MODEL_PATH, require_engine=use_tensorrt)
    if use_tensorrt:
        for name, path in [("Ball", ball_model_path), ("Digit", digit_model_path)]:
            if not Path(path).exists():
                print(f"Error: --use-tensorrt set but {name} engine not found: {path}")
                print("Export with: model.export(format='engine', imgsz=..., half=True, device=0)")
                return

    print("\n" + "="*60)
    print("BALL TRACKING - USING SAVED CONFIGURATION")
    print("="*60)

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    rois = {}
    for k, v in config['rois'].items():
        rois[k] = tuple(v) if v else None

    table_corners = config.get('table_corners')
    table_calibration = None
    if table_corners and len(table_corners) == 4:
        table_calibration = TableCalibration(table_corners)
        if not table_calibration.is_valid():
            table_calibration = None

    print(f"Loaded configuration from: {config_path}")
    print("\nROIs:")
    for name, roi in rois.items():
        print(f"  {name}: {roi}")
    print("  Table calibration:", "enabled" if table_calibration else "disabled")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\nVideo: {video_path}")
    print(f"Resolution: {frame_width}x{frame_height}")
    print(f"FPS: {fps}")
    if inference_size:
        print(f"Ball inference size: {inference_size[0]}x{inference_size[1]}")
    else:
        print(f"Ball inference size: {frame_width}x{frame_height} (full resolution / baseline)")
    if output_size:
        print(f"Output video size:   {output_size[0]}x{output_size[1]}")
    else:
        print(f"Output video size:   {frame_width}x{frame_height} (full resolution)")
    print(f"Ball model:   {ball_model_path}")
    print(f"Digit model:  {digit_model_path}")

    # Initialize components
    print("\nLoading models...")
    print("  - Loading ball detection model...")
    ball_tracker = BallTracker(ball_model_path, fps,
                               table_calibration=table_calibration,
                               inference_size=inference_size)
    print("  - Loading digit detection model...")
    score_detector = ScoreDetector(digit_model_path)
    logger = DataLogger(output_dir, with_meters=(table_calibration is not None and table_calibration.is_valid()))
    rally_aggregator = RallyAggregator(fps, logger, table_calibration)
    stats = RealtimeStats(fps)
    print("  - Models loaded!")
    logger.save_config(rois, video_path, fps, table_corners=table_corners)

    # Video writer (optionally downscaled)
    video_writer = None
    output_video_path = None
    write_w = output_size[0] if output_size else frame_width
    write_h = output_size[1] if output_size else frame_height
    if save_video:
        output_video_path = Path(output_dir) / f"tracked_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (write_w, write_h))
        print(f"Output video will be saved to: {output_video_path} ({write_w}x{write_h})")

    cv2.namedWindow("Ball Tracking - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracking - Real-Time", 1280, 720)

    score_detect_interval = max(1, int(fps * score_interval_sec))
    last_score_log = None
    frame_idx = 0
    paused = False
    playback_speed = 1.0
    frame_delay_base = 1.0 / fps

    print("\nReal-time tracking started...")
    print("Controls: Q=Quit, P=Pause, +/-=Speed, S=Screenshot, </.>=Skip")

    while cap.isOpened():
        loop_start = time.time()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                rally_aggregator.flush_final(frame_idx, frame_idx / fps if fps > 0 else 0)
                break

            # Ball detection (timed)
            t0 = time.perf_counter()
            tracks = ball_tracker.detect_and_track(frame)
            stats.record_phase('ball_infer', time.perf_counter() - t0)

            ball_detected = len(tracks) > 0
            max_speed = 0
            max_speed_mps = None
            tracks_with_meters = []
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                speed = ball_tracker.get_smoothed_speed(track_id)
                max_speed = max(max_speed, speed)
                x_m, y_m = ball_tracker.get_position_meters(track_id)
                speed_mps = ball_tracker.get_smoothed_speed_mps(track_id)
                if speed_mps is not None:
                    max_speed_mps = max(max_speed_mps or 0, speed_mps)
                logger.log_trajectory(frame_idx, track_id, cx, cy, speed, x_m=x_m, y_m=y_m, speed_mps=speed_mps)
                tracks_with_meters.append((track_id, x_m, y_m, speed_mps))

            stats.update_ball_stats(ball_detected, max_speed, max_speed_mps)

            # Score detection (every N frames, timed)
            if frame_idx % score_detect_interval == 0:
                t0 = time.perf_counter()
                scores = score_detector.update_scores(frame, rois, frame_idx)
                rounds = score_detector.detect_rounds(frame, rois)
                stats.record_phase('score_infer', time.perf_counter() - t0)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds

            rally_aggregator.add_frame(frame_idx, frame_idx / fps, tracks_with_meters, scores, rounds)

            # Draw overlays (timed)
            t0 = time.perf_counter()
            ball_tracker.draw_trajectories(frame, tracks)
            current_score_log = (scores['player1'], scores['player2'], rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                timestamp = frame_idx / fps
                logger.log_score(frame_idx, timestamp, scores, rounds)
                last_score_log = current_score_log
            score_detector.draw_scores(frame, rois)
            stats.record_phase('draw', time.perf_counter() - t0)

            stats.update()

            if video_writer:
                t0 = time.perf_counter()
                write_frame = (cv2.resize(frame, (write_w, write_h), interpolation=cv2.INTER_LINEAR)
                               if output_size else frame)
                video_writer.write(write_frame)
                stats.record_phase('write', time.perf_counter() - t0)

            frame_idx += 1

        stats.draw(frame, frame_idx, total_frames, paused, playback_speed)
        cv2.imshow("Ball Tracking - Real-Time", frame)

        processing_time = time.time() - loop_start
        wait_time = max(1, int((frame_delay_base / playback_speed - processing_time) * 1000))
        if paused:
            wait_time = 0

        key = cv2.waitKey(wait_time) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord(' ') and paused:
            ret, frame = cap.read()
            if ret:
                tracks = ball_tracker.detect_and_track(frame)
                ball_tracker.draw_trajectories(frame, tracks)
                score_detector.draw_scores(frame, rois)
                frame_idx += 1
        elif key in [ord('+'), ord('=')]:
            playback_speed = min(4.0, playback_speed + 0.25)
        elif key == ord('-'):
            playback_speed = max(0.25, playback_speed - 0.25)
        elif key == ord('0'):
            playback_speed = 1.0
        elif key == ord('s'):
            screenshot_path = Path(output_dir) / f"screenshot_{frame_idx}.jpg"
            cv2.imwrite(str(screenshot_path), frame)
            print(f"Screenshot saved: {screenshot_path}")
        elif key == ord('.'):
            skip_frames = int(fps * 5)
            new_pos = min(frame_idx + skip_frames, total_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_idx = new_pos
        elif key == ord(','):
            skip_frames = int(fps * 5)
            new_pos = max(0, frame_idx - skip_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            frame_idx = new_pos
            ball_tracker.tracker = Sort(max_age=TRACKER_MAX_AGE, min_hits=TRACKER_MIN_HITS, iou_threshold=TRACKER_IOU_THRESHOLD)
            ball_tracker.trajectories.clear()
            ball_tracker.speed_history.clear()
            ball_tracker.trajectories_meters.clear()
            ball_tracker.speed_history_mps.clear()

    # Cleanup
    score_detector.stop()
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log: {logger.score_file}")
    print(f"Rally log: {logger.rally_file}")
    if output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"Max ball speed: {stats.max_speed:.0f} px/s")


# =============================================================================
# BENCHMARK HELPER
# =============================================================================
def _run_benchmark(cap, fps, frame_width, frame_height, inference_size, n_frames,
                  ball_model_path=None):
    """
    Headless benchmark: measure ball-inference FPS over n_frames frames.
    No display, no video writing, no score detection.
    Prints a concise summary on completion.
    ball_model_path: path to ball model (.pt or .engine); default from resolve_model_path.
    """
    import time

    if ball_model_path is None:
        ball_model_path = resolve_model_path(BALL_MODEL_PATH, require_engine=False)

    inf_label = (f"{inference_size[0]}x{inference_size[1]}"
                 if inference_size else f"{frame_width}x{frame_height} (full)")
    print(f"\n{'='*60}")
    print(f"BENCHMARK MODE  —  inference size: {inf_label}")
    print(f"Ball model: {ball_model_path}")
    print(f"Frames to process: {n_frames}")
    print(f"{'='*60}")

    ball_tracker = BallTracker(ball_model_path, fps, inference_size=inference_size)

    # Warm-up (1 frame to load CUDA kernels / model cache)
    ret, frame = cap.read()
    if not ret:
        print("Error: could not read frame for warm-up.")
        return
    ball_tracker.detect_and_track(frame)
    print("Warm-up done. Starting benchmark...")

    ball_times = []
    total_start = time.perf_counter()

    for i in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            print(f"  Video ended after {i} frames.")
            break
        t0 = time.perf_counter()
        ball_tracker.detect_and_track(frame)
        ball_times.append(time.perf_counter() - t0)

    total_elapsed = time.perf_counter() - total_start
    n = len(ball_times)
    if n == 0:
        print("No frames processed.")
        return

    avg_ms = np.mean(ball_times) * 1000
    p50_ms = np.median(ball_times) * 1000
    p95_ms = np.percentile(ball_times, 95) * 1000
    throughput_fps = n / total_elapsed

    print(f"\n{'='*60}")
    print(f"BENCHMARK RESULTS  ({n} frames, inference size: {inf_label})")
    print(f"{'='*60}")
    print(f"  Ball-infer avg latency : {avg_ms:.1f} ms")
    print(f"  Ball-infer p50 latency : {p50_ms:.1f} ms")
    print(f"  Ball-infer p95 latency : {p95_ms:.1f} ms")
    print(f"  Overall throughput     : {throughput_fps:.1f} FPS  "
          f"(wall-clock over {total_elapsed:.1f}s)")
    print(f"  Source video FPS       : {fps:.0f}")
    ratio = throughput_fps / fps if fps > 0 else 0
    status = "REAL-TIME OK" if ratio >= 1.0 else f"DEFICIT  ({ratio:.2f}x source)"
    print(f"  Real-time ratio        : {ratio:.2f}x  ->  {status}")
    print(f"{'='*60}\n")


# =============================================================================
# ENTRY POINT
# =============================================================================
def _parse_inference_size(s):
    """Parse 'WxH' or 'WxH' string to (int, int) tuple."""
    try:
        w, h = s.lower().split('x')
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError(
            f"Invalid inference size '{s}'. Expected format: WxH  e.g. 1280x720"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ball Tracking and Score Detection for Table Tennis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (mark ROIs on first frame)
  python ball_tracking_analysis.py game_1.mp4

  # Recommended fast config: 640x360 inference, 960x540 output, score every 2s
  python ball_tracking_analysis.py game_1.mp4 --ball-inference-size 640x360 --output-size 960x540

  # Use previously saved configuration
  python ball_tracking_analysis.py game_2.mp4 --config tracking_output/config_20260126.json

  # Disable video output (fastest, no write overhead)
  python ball_tracking_analysis.py game_1.mp4 --no-video

  # Check score only every 3 seconds instead of default 2
  python ball_tracking_analysis.py game_1.mp4 --score-interval 3.0

  # Headless benchmark: measure ball-infer FPS at different resolutions
  python ball_tracking_analysis.py game_1.mp4 --benchmark 300
  python ball_tracking_analysis.py game_1.mp4 --benchmark 300 --ball-inference-size 640x360

  # Use TensorRT engines (export .engine once; then use --use-tensorrt or rely on auto-detect)
  python ball_tracking_analysis.py game_1.mp4 --use-tensorrt --ball-inference-size 640x360
        """
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="tracking_output",
                        help="Output directory for logs and video (default: tracking_output)")
    parser.add_argument("--config", "-c", default=None,
                        help="Path to saved config JSON (skip interactive setup)")
    parser.add_argument("--no-video", action="store_true",
                        help="Don't save output video (faster processing)")
    parser.add_argument(
        "--ball-inference-size", metavar="WxH", default=None,
        type=_parse_inference_size,
        help=(
            "Resize frame to WxH before ball-detector inference, then map "
            "boxes back to original resolution.  Suggested: 960x540 or 640x360."
        )
    )
    parser.add_argument(
        "--output-size", metavar="WxH", default=None,
        type=_parse_inference_size,
        help=(
            "Downscale the saved output video to WxH.  The live display window "
            "always shows the original resolution.  Suggested: 960x540 or 1280x720.  "
            "Reduces video-write cost significantly on high-res sources."
        )
    )
    parser.add_argument(
        "--score-interval", metavar="SEC", type=float, default=2.0,
        help=(
            "How often (in seconds) to run the score detector (default: 2.0).  "
            "Higher values reduce YOLO score calls; scores change slowly so "
            "2–3 seconds is safe.  Use 0.5 for the original behaviour."
        )
    )
    parser.add_argument(
        "--benchmark", metavar="N", type=int, default=0,
        help=(
            "Run a headless ball-inference benchmark over N frames and exit.  "
            "No display, no logging, no interactive setup required.  "
            "Combine with --ball-inference-size to compare resolutions."
        )
    )
    parser.add_argument(
        "--use-tensorrt", action="store_true",
        help=(
            "Use TensorRT .engine models for inference (faster on NVIDIA GPU).  "
            "Requires that you have exported .engine files next to the .pt weights.  "
            "If omitted, .engine is used automatically when present."
        )
    )

    args = parser.parse_args()

    # Resolve inference size: CLI flag overrides module-level constant
    inf_size = args.ball_inference_size if args.ball_inference_size is not None else BALL_INFERENCE_SIZE
    out_size = args.output_size  # None means full resolution

    if args.config:
        load_config_and_run(args.video, args.config, args.output,
                            not args.no_video, inference_size=inf_size,
                            score_interval_sec=args.score_interval,
                            output_size=out_size, use_tensorrt=args.use_tensorrt)
    else:
        main(args.video, args.output, not args.no_video,
             inference_size=inf_size, benchmark_frames=args.benchmark,
             score_interval_sec=args.score_interval, output_size=out_size,
             use_tensorrt=args.use_tensorrt)
