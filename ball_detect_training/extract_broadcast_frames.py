"""
Extract frames from broadcast video for ball annotation.
- Samples at ~2 FPS to avoid near-duplicates
- Filters out non-gameplay frames using table color heuristic (HSV)
- Outputs numbered .jpg frames ready for annotation
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# Default HSV ranges for common ITTF table colors
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

# Minimum fraction of center region that must be table color to count as gameplay
MIN_TABLE_FRACTION = 0.05


def is_gameplay_frame(frame, table_color="blue"):
    """Check if a frame likely shows gameplay by looking for table color in center region."""
    h, w = frame.shape[:2]
    # Sample center 50% of the frame
    y1, y2 = h // 4, 3 * h // 4
    x1, x2 = w // 4, 3 * w // 4
    center = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    preset = TABLE_COLOR_PRESETS.get(table_color, TABLE_COLOR_PRESETS["blue"])
    mask = cv2.inRange(hsv, preset["lower"], preset["upper"])
    fraction = np.count_nonzero(mask) / mask.size
    return fraction >= MIN_TABLE_FRACTION


def extract_frames(video_path, output_dir, sample_fps=2.0, table_color="blue", max_frames=0):
    """Extract gameplay frames from a broadcast video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps / sample_fps))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    frame_idx = 0

    print(f"Video: {video_path}")
    print(f"  FPS: {fps:.1f}, Total frames: {total}")
    print(f"  Sampling every {frame_interval} frames (~{sample_fps} FPS)")
    print(f"  Table color filter: {table_color}")
    print(f"  Output: {output_dir}")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            if is_gameplay_frame(frame, table_color):
                fname = f"broadcast_{frame_idx:06d}.jpg"
                cv2.imwrite(str(output_dir / fname), frame)
                saved += 1
                if saved % 50 == 0:
                    print(f"  Saved {saved} frames (skipped {skipped} non-gameplay)...")
            else:
                skipped += 1

            if max_frames > 0 and saved >= max_frames:
                print(f"  Reached max_frames={max_frames}, stopping.")
                break

        frame_idx += 1

    cap.release()
    print(f"\nDone! Saved {saved} gameplay frames, skipped {skipped} non-gameplay frames.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract broadcast frames for ball annotation")
    parser.add_argument("video", help="Path to broadcast video file")
    parser.add_argument("--output", "-o", default="broadcast_frames", help="Output directory for extracted frames")
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frames per second to sample (default: 2.0)")
    parser.add_argument("--table-color", choices=["blue", "green"], default="blue",
                        help="Table color for gameplay detection (default: blue)")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to extract (0=unlimited)")
    args = parser.parse_args()

    extract_frames(args.video, args.output, args.sample_fps, args.table_color, args.max_frames)
