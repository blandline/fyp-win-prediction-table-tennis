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


def draw_instructions(frame, marking_mode, current_roi):
    """Draw instruction overlay."""
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay for instructions
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (450, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Instructions
    instructions = [
        "CONTROLS:",
        "  Mouse Wheel: Zoom in/out",
        "  Right-click drag: Pan",
        "  Left-click: Draw ROI corners",
        "",
        "KEYS:",
        "  1: Mark Player 1 Score Area",
        "  2: Mark Player 2 Score Area",
        "  3: Mark Player 1 Rounds Area",
        "  4: Mark Player 2 Rounds Area",
        "  R: Reset all markings",
        "  ENTER: Confirm and start tracking",
        "  ESC: Cancel"
    ]
    
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
            'player2_rounds': 'MARKING: Player 2 Rounds'
        }
        cv2.putText(frame, mode_names.get(marking_mode, ''), 
                   (w - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw current ROI in progress
    if len(current_roi) == 1:
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
        elif key == ord('r'):
            marking_state.rois = {k: None for k in marking_state.rois}
            marking_state.current_roi = []
            marking_state.marking_mode = None
            print("All markings reset")
    
    cv2.destroyWindow(window_name)
    return marking_state.rois


# =============================================================================
# BALL TRACKING AND SPEED CALCULATION
# =============================================================================
class BallTracker:
    def __init__(self, model_path, fps):
        self.model = YOLO(model_path)
        self.tracker = Sort(
            max_age=TRACKER_MAX_AGE,
            min_hits=TRACKER_MIN_HITS,
            iou_threshold=TRACKER_IOU_THRESHOLD
        )
        self.fps = fps
        
        # Trajectory history per track ID
        self.trajectories = {}
        self.speed_history = {}
        
        # Pixel to meter conversion (approximate for table tennis)
        # Table tennis table is 2.74m x 1.525m
        # This should be calibrated for your camera setup
        self.pixels_per_meter = None  # Will be estimated or set manually
    
    def detect_and_track(self, frame):
        """Detect ball and update tracking."""
        results = self.model(frame, conf=BALL_CONF_THRESHOLD, verbose=False)
        
        detections = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                detections.append([x1, y1, x2, y2, conf])
        
        # Convert to numpy array
        if detections:
            dets = np.array(detections)
        else:
            dets = np.empty((0, 5))
        
        # Update tracker
        tracks = self.tracker.update(dets)
        
        # Update trajectories
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            track_id = int(track_id)
            
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            if track_id not in self.trajectories:
                self.trajectories[track_id] = deque(maxlen=TRAJECTORY_LENGTH)
                self.speed_history[track_id] = deque(maxlen=SPEED_SMOOTHING_WINDOW)
            
            self.trajectories[track_id].append((cx, cy))
            
            # Calculate speed
            if len(self.trajectories[track_id]) >= 2:
                prev_x, prev_y = self.trajectories[track_id][-2]
                dist_pixels = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                
                # Speed in pixels per second
                speed_pps = dist_pixels * self.fps
                self.speed_history[track_id].append(speed_pps)
        
        return tracks
    
    def get_smoothed_speed(self, track_id):
        """Get smoothed speed for a track."""
        if track_id in self.speed_history and self.speed_history[track_id]:
            return np.mean(self.speed_history[track_id])
        return 0
    
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
            
            # Draw speed
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
# DATA LOGGING
# =============================================================================
class DataLogger:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Trajectory log
        self.trajectory_file = self.output_dir / f"trajectories_{timestamp}.csv"
        with open(self.trajectory_file, 'w') as f:
            f.write("frame,track_id,x,y,speed_pps\n")
        
        # Score log
        self.score_file = self.output_dir / f"scores_{timestamp}.csv"
        with open(self.score_file, 'w') as f:
            f.write("frame,timestamp_sec,player1_score,player2_score,player1_sets,player2_sets\n")
        
        # ROI config
        self.config_file = self.output_dir / f"config_{timestamp}.json"
    
    def log_trajectory(self, frame_idx, track_id, x, y, speed):
        with open(self.trajectory_file, 'a') as f:
            f.write(f"{frame_idx},{track_id},{x:.2f},{y:.2f},{speed:.2f}\n")
    
    def log_score(self, frame_idx, timestamp, scores, rounds):
        with open(self.score_file, 'a') as f:
            p1 = scores['player1'] if scores['player1'] is not None else ''
            p2 = scores['player2'] if scores['player2'] is not None else ''
            f.write(f"{frame_idx},{timestamp:.2f},{p1},{p2},{rounds['player1']},{rounds['player2']}\n")
    
    def save_config(self, rois, video_path, fps):
        config = {
            'video_path': str(video_path),
            'fps': fps,
            'rois': {k: list(v) if v else None for k, v in rois.items()},
            'ball_model': BALL_MODEL_PATH,
            'digit_model': DIGIT_MODEL_PATH
        }
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
    
    def update(self):
        import time
        current = time.time()
        if self.last_time is not None:
            self.frame_times.append(current - self.last_time)
            if self.frame_times:
                self.current_fps = 1.0 / np.mean(self.frame_times)
        self.last_time = current
    
    def update_ball_stats(self, detected, speed):
        self.ball_detected = detected
        if detected and speed > 0:
            self.current_speed = speed
            self.max_speed = max(self.max_speed, speed)
    
    def draw(self, frame, frame_idx, total_frames, paused, playback_speed):
        h, w = frame.shape[:2]
        
        # Stats panel background (bottom-left)
        overlay = frame.copy()
        panel_h = 120
        cv2.rectangle(overlay, (10, h - panel_h - 10), (280, h - 10), (0, 0, 0), -1)
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
        
        # Speed
        cv2.putText(frame, f"Speed: {self.current_speed:.0f} px/s", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y += line_height
        
        # Max speed
        cv2.putText(frame, f"Max Speed: {self.max_speed:.0f} px/s", (20, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
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
def main(video_path, output_dir="tracking_output", save_video=True):
    """Main entry point for ball tracking and score detection."""
    import time
    
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
    
    # Read first frame for setup
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    # Interactive setup
    print("\nStarting interactive setup...")
    rois = interactive_frame_setup(first_frame)
    
    if rois is None:
        print("Setup cancelled.")
        return
    
    print("\nROIs configured:")
    for name, roi in rois.items():
        print(f"  {name}: {roi}")
    
    # Initialize components
    print("\nLoading models...")
    print("  - Loading ball detection model...")
    ball_tracker = BallTracker(BALL_MODEL_PATH, fps)
    print("  - Loading digit detection model...")
    score_detector = ScoreDetector(DIGIT_MODEL_PATH)
    logger = DataLogger(output_dir)
    stats = RealtimeStats(fps)
    print("  - Models loaded!")
    
    # Save configuration
    logger.save_config(rois, video_path, fps)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Create output window
    cv2.namedWindow("Ball Tracking - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracking - Real-Time", 1280, 720)
    
    # Video writer for output
    video_writer = None
    output_video_path = None
    if save_video:
        output_video_path = Path(output_dir) / f"tracked_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path), fourcc, fps, 
            (frame_width, frame_height)
        )
        print(f"Output video will be saved to: {output_video_path}")
    
    # Processing settings - detect score every ~0.5 second (YOLO is fast)
    score_detect_interval = max(1, int(fps * 0.5))
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
                # End of video - loop or stop
                print("\nEnd of video reached.")
                break
            
            # Ball detection and tracking
            tracks = ball_tracker.detect_and_track(frame)
            
            # Update stats
            ball_detected = len(tracks) > 0
            max_speed = 0
            
            # Log trajectories
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                speed = ball_tracker.get_smoothed_speed(track_id)
                max_speed = max(max_speed, speed)
                logger.log_trajectory(frame_idx, track_id, cx, cy, speed)
            
            stats.update_ball_stats(ball_detected, max_speed)
            
            # Draw ball trajectories
            ball_tracker.draw_trajectories(frame, tracks)
            
            # Score detection (every N frames for performance)
            if frame_idx % score_detect_interval == 0:
                scores = score_detector.update_scores(frame, rois, frame_idx)
                rounds = score_detector.detect_rounds(frame, rois)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds
            
            # Log if scores changed
            current_score_log = (scores['player1'], scores['player2'], 
                                rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                timestamp = frame_idx / fps
                logger.log_score(frame_idx, timestamp, scores, rounds)
                last_score_log = current_score_log
            
            # Draw score overlay
            score_detector.draw_scores(frame, rois)
            
            # Update FPS stats
            stats.update()
            
            # Write to output video (without stats overlay for clean output)
            if video_writer:
                video_writer.write(frame.copy())
            
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
    print(f"Configuration: {logger.config_file}")
    if save_video and output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"\nMax ball speed recorded: {stats.max_speed:.0f} px/s")


def load_config_and_run(video_path, config_path, output_dir="tracking_output", save_video=True):
    """
    Run tracking with pre-saved ROI configuration (skip interactive setup).
    Useful for batch processing or re-running with same settings.
    """
    import time
    
    print("\n" + "="*60)
    print("BALL TRACKING - USING SAVED CONFIGURATION")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    rois = {}
    for k, v in config['rois'].items():
        rois[k] = tuple(v) if v else None
    
    print(f"Loaded configuration from: {config_path}")
    print("\nROIs:")
    for name, roi in rois.items():
        print(f"  {name}: {roi}")
    
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
    
    # Initialize components
    print("\nLoading models...")
    print("  - Loading ball detection model...")
    ball_tracker = BallTracker(BALL_MODEL_PATH, fps)
    print("  - Loading digit detection model...")
    score_detector = ScoreDetector(DIGIT_MODEL_PATH)
    logger = DataLogger(output_dir)
    stats = RealtimeStats(fps)
    print("  - Models loaded!")
    logger.save_config(rois, video_path, fps)
    
    # Video writer
    video_writer = None
    output_video_path = None
    if save_video:
        output_video_path = Path(output_dir) / f"tracked_{Path(video_path).stem}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (frame_width, frame_height))
    
    cv2.namedWindow("Ball Tracking - Real-Time", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Ball Tracking - Real-Time", 1280, 720)
    
    score_detect_interval = int(fps * 1.0)  # Detect score every ~1 second
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
                break
            
            tracks = ball_tracker.detect_and_track(frame)
            
            ball_detected = len(tracks) > 0
            max_speed = 0
            
            for track in tracks:
                x1, y1, x2, y2, track_id = track
                track_id = int(track_id)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                speed = ball_tracker.get_smoothed_speed(track_id)
                max_speed = max(max_speed, speed)
                logger.log_trajectory(frame_idx, track_id, cx, cy, speed)
            
            stats.update_ball_stats(ball_detected, max_speed)
            ball_tracker.draw_trajectories(frame, tracks)
            
            # Score detection (every N frames for performance)
            if frame_idx % score_detect_interval == 0:
                scores = score_detector.update_scores(frame, rois, frame_idx)
                rounds = score_detector.detect_rounds(frame, rois)
            else:
                scores = score_detector.current_scores
                rounds = score_detector.rounds
            current_score_log = (scores['player1'], scores['player2'], rounds['player1'], rounds['player2'])
            if current_score_log != last_score_log:
                timestamp = frame_idx / fps
                logger.log_score(frame_idx, timestamp, scores, rounds)
                last_score_log = current_score_log
            
            score_detector.draw_scores(frame, rois)
            stats.update()
            
            if video_writer:
                video_writer.write(frame.copy())
            
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
    
    # Cleanup
    score_detector.stop()  # Stop OCR thread
    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"Trajectory log: {logger.trajectory_file}")
    print(f"Score log: {logger.score_file}")
    if output_video_path:
        print(f"Output video: {output_video_path}")
    print(f"Max ball speed: {stats.max_speed:.0f} px/s")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ball Tracking and Score Detection for Table Tennis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (mark ROIs on first frame)
  python ball_tracking_analysis.py game_1.mp4
  
  # With custom output directory
  python ball_tracking_analysis.py game_1.mp4 -o my_output
  
  # Use previously saved configuration
  python ball_tracking_analysis.py game_2.mp4 --config tracking_output/config_20260126.json
  
  # Disable video output (faster processing)
  python ball_tracking_analysis.py game_1.mp4 --no-video
        """
    )
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output", "-o", default="tracking_output", 
                       help="Output directory for logs and video (default: tracking_output)")
    parser.add_argument("--config", "-c", default=None,
                       help="Path to saved config JSON (skip interactive setup)")
    parser.add_argument("--no-video", action="store_true",
                       help="Don't save output video (faster processing)")
    
    args = parser.parse_args()
    
    if args.config:
        load_config_and_run(args.video, args.config, args.output, not args.no_video)
    else:
        main(args.video, args.output, not args.no_video)
