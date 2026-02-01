# """
# Review and correct bounding boxes for table tennis ball detection.

# This script allows you to visualize bounding boxes on images and correct them
# if needed. Use arrow keys to navigate, and mouse to adjust bounding boxes.
# """

# import json
# import os
# import cv2
# import numpy as np
# from pathlib import Path
# from typing import Optional, Tuple

# # Configuration
# GAME_NAME = "game_1"
# IMAGES_DIR = Path("dataset/training/images") / GAME_NAME
# ANNOTATIONS_DIR = Path("output") / GAME_NAME  # Directory with generated annotations
# OUTPUT_DIR = Path("output") / GAME_NAME  # Where to save corrected annotations

# # Colors (BGR format for OpenCV)
# BALL_COLOR = (0, 255, 0)  # Green
# TEXT_COLOR = (255, 255, 255)  # White
# TEXT_BG_COLOR = (0, 0, 0)  # Black

# # Bounding box size (should match the one used in conversion script)
# BALL_BBOX_SIZE = 40

# # Mouse interaction state
# dragging = False
# resizing = False
# start_point = None
# current_bbox = None
# current_image_idx = 0
# images = []
# annotations = {}


# def yolo_to_pixel_bbox(yolo_bbox, img_width, img_height):
#     """
#     Convert YOLOv8 normalized bbox to pixel coordinates.
    
#     Args:
#         yolo_bbox: (center_x, center_y, width, height) normalized [0, 1]
#         img_width, img_height: Image dimensions
    
#     Returns:
#         (x_min, y_min, x_max, y_max) in pixel coordinates
#     """
#     center_x, center_y, width, height = yolo_bbox
    
#     # Convert to pixel coordinates
#     center_x_px = center_x * img_width
#     center_y_px = center_y * img_height
#     width_px = width * img_width
#     height_px = height * img_height
    
#     # Calculate corners
#     x_min = int(center_x_px - width_px / 2)
#     y_min = int(center_y_px - height_px / 2)
#     x_max = int(center_x_px + width_px / 2)
#     y_max = int(center_y_px + height_px / 2)
    
#     return x_min, y_min, x_max, y_max


# def pixel_to_yolo_bbox(x_min, y_min, x_max, y_max, img_width, img_height):
#     """
#     Convert pixel bbox to YOLOv8 normalized format.
    
#     Args:
#         x_min, y_min, x_max, y_max: Pixel coordinates
#         img_width, img_height: Image dimensions
    
#     Returns:
#         (center_x, center_y, width, height) normalized [0, 1]
#     """
#     # Clamp to image boundaries
#     x_min = max(0, min(x_min, img_width))
#     y_min = max(0, min(y_min, img_height))
#     x_max = max(0, min(x_max, img_width))
#     y_max = max(0, min(y_max, img_height))
    
#     # Calculate center and dimensions
#     center_x = (x_min + x_max) / 2.0
#     center_y = (y_min + y_max) / 2.0
#     width = x_max - x_min
#     height = y_max - y_min
    
#     # Normalize
#     center_x_norm = center_x / img_width
#     center_y_norm = center_y / img_height
#     width_norm = width / img_width
#     height_norm = height / img_height
    
#     return center_x_norm, center_y_norm, width_norm, height_norm


# def load_annotation(image_name):
#     """Load annotation for an image (TXT or JSON format)."""
#     txt_path = ANNOTATIONS_DIR / image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#     json_path = ANNOTATIONS_DIR / image_name.replace('.jpg', '.json').replace('.png', '.json')
    
#     if txt_path.exists():
#         with open(txt_path, 'r') as f:
#             line = f.readline().strip()
#             if line:
#                 parts = line.split()
#                 if len(parts) >= 5:
#                     return tuple(map(float, parts[1:5]))  # Skip class_id
#     elif json_path.exists():
#         with open(json_path, 'r') as f:
#             data = json.load(f)
#             bbox = data.get('bbox', {})
#             return (bbox.get('center_x', 0.5), bbox.get('center_y', 0.5),
#                    bbox.get('width', 0.1), bbox.get('height', 0.1))
    
#     return None


# def save_annotation(image_name, bbox, output_format='txt'):
#     """Save annotation in the specified format."""
#     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
#     if output_format == 'txt':
#         txt_path = OUTPUT_DIR / image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#         with open(txt_path, 'w') as f:
#             f.write(f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
#     else:
#         json_path = OUTPUT_DIR / image_name.replace('.jpg', '.json').replace('.png', '.json')
#         annotation = {
#             "image": image_name,
#             "bbox": {
#                 "center_x": bbox[0],
#                 "center_y": bbox[1],
#                 "width": bbox[2],
#                 "height": bbox[3]
#             },
#             "class_id": 0
#         }
#         with open(json_path, 'w') as f:
#             json.dump(annotation, f, indent=2)


# def is_near_corner(x, y, corner_x, corner_y, threshold=10):
#     """Check if point is near a corner."""
#     return abs(x - corner_x) < threshold and abs(y - corner_y) < threshold


# def mouse_callback(event, x, y, flags, param):
#     """Handle mouse events for adjusting bounding boxes."""
#     global dragging, resizing, start_point, current_bbox
    
#     img = param['image']
#     img_height, img_width = img.shape[:2]
    
#     # If no bbox exists, create one on click
#     if current_bbox is None:
#         if event == cv2.EVENT_LBUTTONDOWN:
#             # Create new bbox centered on click
#             half_size = BALL_BBOX_SIZE // 2
#             x_min = max(0, x - half_size)
#             y_min = max(0, y - half_size)
#             x_max = min(img_width, x + half_size)
#             y_max = min(img_height, y + half_size)
#             current_bbox = pixel_to_yolo_bbox(x_min, y_min, x_max, y_max, img_width, img_height)
#             param['bbox_changed'] = True
#         return
    
#     x_min, y_min, x_max, y_max = yolo_to_pixel_bbox(
#         current_bbox, img_width, img_height
#     )
    
#     if event == cv2.EVENT_LBUTTONDOWN:
#         dragging = True
#         start_point = (x, y)
        
#         # Check if clicking near a corner (for resizing)
#         corners = [
#             (x_min, y_min, 'tl'),  # top-left
#             (x_max, y_min, 'tr'),  # top-right
#             (x_min, y_max, 'bl'),  # bottom-left
#             (x_max, y_max, 'br')   # bottom-right
#         ]
        
#         resize_corner = None
#         for corner_x, corner_y, corner_name in corners:
#             if is_near_corner(x, y, corner_x, corner_y, threshold=15):
#                 resize_corner = corner_name
#                 break
        
#         if resize_corner:
#             param['resizing'] = True
#             param['resize_corner'] = resize_corner
#             param['start_bbox'] = (x_min, y_min, x_max, y_max)
#         elif x_min <= x <= x_max and y_min <= y <= y_max:
#             # Start dragging the bbox
#             param['dragging_bbox'] = True
#             param['drag_offset'] = (x - (x_min + x_max) // 2, y - (y_min + y_max) // 2)
    
#     elif event == cv2.EVENT_MOUSEMOVE:
#         if dragging:
#             if 'resizing' in param and param['resizing']:
#                 # Resize bbox
#                 sx_min, sy_min, sx_max, sy_max = param['start_bbox']
#                 corner = param['resize_corner']
                
#                 if corner == 'tl':
#                     new_x_min = max(0, min(x, sx_max - 20))
#                     new_y_min = max(0, min(y, sy_max - 20))
#                     x_min, y_min = new_x_min, new_y_min
#                 elif corner == 'tr':
#                     new_x_max = min(img_width, max(x, sx_min + 20))
#                     new_y_min = max(0, min(y, sy_max - 20))
#                     x_max, y_min = new_x_max, new_y_min
#                 elif corner == 'bl':
#                     new_x_min = max(0, min(x, sx_max - 20))
#                     new_y_max = min(img_height, max(y, sy_min + 20))
#                     x_min, y_max = new_x_min, new_y_max
#                 elif corner == 'br':
#                     new_x_max = min(img_width, max(x, sx_min + 20))
#                     new_y_max = min(img_height, max(y, sy_min + 20))
#                     x_max, y_max = new_x_max, new_y_max
                
#                 current_bbox = pixel_to_yolo_bbox(x_min, y_min, x_max, y_max, img_width, img_height)
#                 param['bbox_changed'] = True
                
#             elif 'dragging_bbox' in param and param['dragging_bbox']:
#                 # Move bbox
#                 center_x = x - param['drag_offset'][0]
#                 center_y = y - param['drag_offset'][1]
                
#                 bbox_width = x_max - x_min
#                 bbox_height = y_max - y_min
#                 center_x = max(bbox_width // 2, min(center_x, img_width - bbox_width // 2))
#                 center_y = max(bbox_height // 2, min(center_y, img_height - bbox_height // 2))
                
#                 current_bbox = pixel_to_yolo_bbox(
#                     center_x - bbox_width // 2,
#                     center_y - bbox_height // 2,
#                     center_x + bbox_width // 2,
#                     center_y + bbox_height // 2,
#                     img_width, img_height
#                 )
#                 param['bbox_changed'] = True
    
#     elif event == cv2.EVENT_LBUTTONUP:
#         dragging = False
#         if 'dragging_bbox' in param:
#             param['dragging_bbox'] = False
#         if 'resizing' in param:
#             param['resizing'] = False


# def draw_bbox(image, bbox, img_width, img_height):
#     """Draw bounding box on image with resize handles."""
#     if bbox is None:
#         return image
    
#     x_min, y_min, x_max, y_max = yolo_to_pixel_bbox(bbox, img_width, img_height)
    
#     # Draw rectangle
#     cv2.rectangle(image, (x_min, y_min), (x_max, y_max), BALL_COLOR, 2)
    
#     # Draw center point
#     center_x = (x_min + x_max) // 2
#     center_y = (y_min + y_max) // 2
#     cv2.circle(image, (center_x, center_y), 3, BALL_COLOR, -1)
    
#     # Draw resize handles at corners
#     handle_size = 8
#     corners = [
#         (x_min, y_min),  # top-left
#         (x_max, y_min),  # top-right
#         (x_min, y_max),  # bottom-left
#         (x_max, y_max)   # bottom-right
#     ]
#     for corner_x, corner_y in corners:
#         cv2.rectangle(image, 
#                      (corner_x - handle_size, corner_y - handle_size),
#                      (corner_x + handle_size, corner_y + handle_size),
#                      BALL_COLOR, -1)
    
#     return image


# def display_image(image_path, bbox, image_idx=0, total_images=0):
#     """Display image with bounding box and controls."""
#     image = cv2.imread(str(image_path))
#     if image is None:
#         print(f"Error: Could not load image {image_path}")
#         return None
    
#     img_height, img_width = image.shape[:2]
    
#     # Draw bounding box
#     if bbox is not None:
#         image = draw_bbox(image, bbox, img_width, img_height)
    
#     # Add text overlay with instructions
#     bbox_status = "Has bbox" if bbox is not None else "No bbox - Click to create"
#     instructions = [
#         "Arrow Keys: Navigate | 'D': Delete | 'S': Save | 'Q': Quit",
#         f"Mouse: {bbox_status} | Drag center to move | Drag corners to resize",
#         f"Image: {os.path.basename(image_path)} ({image_idx + 1}/{total_images})"
#     ]
    
#     y_offset = 30
#     for i, text in enumerate(instructions):
#         # Draw background for text
#         (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
#         cv2.rectangle(image, (10, y_offset - text_height - 5), 
#                      (10 + text_width + 10, y_offset + 5), TEXT_BG_COLOR, -1)
#         cv2.putText(image, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
#         y_offset += 30
    
#     return image


# def main():
#     """Main review function."""
#     global current_bbox, current_image_idx, images
    
#     # Get all image files
#     if not IMAGES_DIR.exists():
#         print(f"Error: Images directory not found: {IMAGES_DIR}")
#         return
    
#     images = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(('.jpg', '.png'))])
    
#     if not images:
#         print(f"No images found in {IMAGES_DIR}")
#         return
    
#     print(f"Found {len(images)} images")
#     print("\nControls:")
#     print("  Left/Right Arrow: Navigate between images")
#     print("  'D': Delete bounding box for current image")
#     print("  'S': Save current bounding box")
#     print("  'Q': Quit and save all changes")
#     print("  Mouse: Click to create bbox | Drag center to move | Drag corners to resize")
#     print("\nPress any key to start...")
    
#     # Initialize
#     current_image_idx = 0
#     param = {'image': None, 'bbox_changed': False}
    
#     cv2.namedWindow('Bounding Box Review', cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback('Bounding Box Review', mouse_callback, param)
    
#     while True:
#         if current_image_idx < 0:
#             current_image_idx = 0
#         if current_image_idx >= len(images):
#             current_image_idx = len(images) - 1
        
#         image_name = images[current_image_idx]
#         image_path = IMAGES_DIR / image_name
        
#         # Load annotation
#         current_bbox = load_annotation(image_name)
#         param['bbox_changed'] = False
        
#         # Display image
#         display_img = display_image(image_path, current_bbox, current_image_idx, len(images))
#         if display_img is None:
#             current_image_idx += 1
#             continue
        
#         param['image'] = display_img.copy()  # Keep a copy for mouse callback
#         cv2.imshow('Bounding Box Review', display_img)
        
#         # Handle keyboard input
#         key = cv2.waitKey(0) & 0xFF
        
#         if key == ord('q') or key == 27:  # Q or ESC
#             print("\nSaving all changes and exiting...")
#             break
#         elif key == 81 or key == 2:  # Left arrow
#             # Save current bbox if changed
#             if param['bbox_changed'] and current_bbox is not None:
#                 save_annotation(image_name, current_bbox)
#             current_image_idx -= 1
#         elif key == 83 or key == 3:  # Right arrow
#             # Save current bbox if changed
#             if param['bbox_changed'] and current_bbox is not None:
#                 save_annotation(image_name, current_bbox)
#             current_image_idx += 1
#         elif key == ord('d'):  # Delete bbox
#             current_bbox = None
#             # Delete annotation file
#             txt_path = OUTPUT_DIR / image_name.replace('.jpg', '.txt').replace('.png', '.txt')
#             json_path = OUTPUT_DIR / image_name.replace('.jpg', '.json').replace('.png', '.json')
#             if txt_path.exists():
#                 txt_path.unlink()
#             if json_path.exists():
#                 json_path.unlink()
#             print(f"Deleted bounding box for {image_name}")
#         elif key == ord('s'):  # Save
#             if current_bbox is not None:
#                 save_annotation(image_name, current_bbox)
#                 print(f"Saved bounding box for {image_name}")
#             else:
#                 print(f"No bounding box to save for {image_name}")
    
#     # Save all remaining changes
#     if current_bbox is not None and param['bbox_changed']:
#         save_annotation(images[current_image_idx], current_bbox)
    
#     cv2.destroyAllWindows()
#     print("Review complete!")


# if __name__ == "__main__":
#     main()
"""
Visualize YOLOv8 bounding boxes on images to verify correctness.
Press:
  - 'n' → next image
  - 'q' → quit
"""

import cv2
from pathlib import Path

# Paths
IMAGES_DIR = Path("dataset/training/images/game_1")
LABELS_DIR = Path("output/game_1")

# Settings
WINDOW_NAME = "YOLO Bounding Box Check"
BOX_COLOR = (0, 255, 0)  # Green
THICKNESS = 2

def yolo_to_pixel_bbox(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized bbox to pixel bbox."""
    cx *= img_w
    cy *= img_h
    w *= img_w
    h *= img_h

    x1 = int(cx - w / 2)
    y1 = int(cy - h / 2)
    x2 = int(cx + w / 2)
    y2 = int(cy + h / 2)

    return x1, y1, x2, y2


def main():
    image_files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix in [".jpg", ".png"]])

    if not image_files:
        print("❌ No images found.")
        return

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        label_path = LABELS_DIR / img_path.with_suffix(".txt").name

        if not label_path.exists():
            print(f"⚠️ No label for {img_path.name}, skipping")
            continue

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Could not load {img_path}")
            continue

        img_h, img_w = img.shape[:2]

        # Read YOLO annotation
        with open(label_path, "r") as f:
            line = f.readline().strip()
            class_id, cx, cy, w, h = map(float, line.split())

        # Convert bbox
        x1, y1, x2, y2 = yolo_to_pixel_bbox(cx, cy, w, h, img_w, img_h)

        # Draw bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), BOX_COLOR, THICKNESS)
        cv2.putText(
            img,
            f"Ball",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            BOX_COLOR,
            2,
        )

        # Show
        cv2.imshow(WINDOW_NAME, img)
        key = cv2.waitKey(0)

        if key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
