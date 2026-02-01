# Table Tennis Ball Bounding Box Conversion and Review

This repository contains scripts to convert ball coordinates to YOLOv8 bounding boxes and review/correct them.

## Scripts

### 1. `convert_coords_to_bbox.py`
Converts ball coordinates from the Open TT Games dataset to YOLOv8 bounding box format.

**Configuration (at the top of the script):**
- `IMAGE_LIMIT`: Maximum number of images to process (default: 1000)
- `BALL_BBOX_SIZE`: Size of bounding box in pixels (default: 40x40)
- `GAME_NAME`: Which game to process (default: "game_1")
- `OUTPUT_FORMAT`: "txt" for YOLOv8 format or "json" (default: "txt")

**Usage:**
```bash
python convert_coords_to_bbox.py
```

**Output:**
- Creates `output/game_1/` directory
- Generates `.txt` files (YOLOv8 format) or `.json` files for each image with ball coordinates
- Format: `class_id center_x center_y width height` (all normalized 0-1)

### 2. `review_bboxes.py`
Interactive tool to review and correct bounding boxes.

**Configuration (at the top of the script):**
- `GAME_NAME`: Which game to review (default: "game_1")
- `ANNOTATIONS_DIR`: Directory with generated annotations (default: `output/game_1`)
- `BALL_BBOX_SIZE`: Default bounding box size for new boxes (default: 40)

**Usage:**
```bash
python review_bboxes.py
```

**Controls:**
- **Left/Right Arrow Keys**: Navigate between images
- **'D' Key**: Delete bounding box for current image
- **'S' Key**: Save current bounding box
- **'Q' Key**: Quit and save all changes
- **Mouse Click**: Create new bounding box (if none exists)
- **Mouse Drag (center)**: Move existing bounding box
- **Mouse Drag (corners)**: Resize bounding box

## Installation

Install required packages:
```bash
pip install -r requirements.txt
```

## Workflow

1. **Convert coordinates to bounding boxes:**
   ```bash
   python convert_coords_to_bbox.py
   ```

2. **Review and correct bounding boxes:**
   ```bash
   python review_bboxes.py
   ```

3. **Use the generated annotations for YOLOv8 training:**
   - The `.txt` files in `output/game_1/` are in YOLOv8 format
   - Each line: `class_id center_x center_y width height`
   - Class ID 0 = table tennis ball

## Notes

- The conversion script only processes images that have ball coordinates in the JSON file
- Images without coordinates are skipped
- The review script automatically saves changes when navigating or quitting
- Bounding boxes are normalized to [0, 1] range as required by YOLOv8
