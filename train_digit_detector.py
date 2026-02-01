"""
Train a Robust Digit Detection Model using YOLOv8
==================================================
This script trains a YOLOv8 model for detecting digits 0-9 with:
- Data augmentation for robustness
- Multiple model sizes to choose from
- Proper hyperparameters for small object detection
- Comprehensive training monitoring
"""

import os
import yaml
import shutil
from pathlib import Path
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION
# =============================================================================

# Dataset paths
DATASET_DIR = Path("digits-detect.v1i.yolov8")
OUTPUT_DIR = Path("runs/detect/digits_v2")

# Model configuration
MODEL_SIZE = "n"  # Options: 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra-large)
                   # 'n' is fastest, 'x' is most accurate

# Training hyperparameters
EPOCHS = 150           # Number of training epochs
BATCH_SIZE = 16        # Batch size (reduce if OOM error)
IMG_SIZE = 640         # Image size for training
PATIENCE = 30          # Early stopping patience
WORKERS = 8            # Number of data loader workers

# Augmentation settings (optimized for digit detection)
AUGMENTATION_CONFIG = {
    # Spatial augmentations
    'degrees': 15.0,       # Rotation (+/- degrees) - digits can be slightly rotated
    'translate': 0.1,      # Translation (+/- fraction)
    'scale': 0.3,          # Scale (+/- gain) - handle different digit sizes
    'shear': 5.0,          # Shear (+/- degrees)
    'perspective': 0.0005, # Slight perspective changes
    'flipud': 0.0,         # No vertical flip (6 and 9 would be confused)
    'fliplr': 0.0,         # No horizontal flip (digits are not symmetric)
    
    # Color augmentations
    'hsv_h': 0.015,        # HSV-Hue augmentation
    'hsv_s': 0.5,          # HSV-Saturation augmentation
    'hsv_v': 0.4,          # HSV-Value (brightness) augmentation
    
    # Other augmentations
    'mosaic': 0.8,         # Mosaic augmentation probability
    'mixup': 0.1,          # Mixup augmentation probability
    'copy_paste': 0.0,     # Copy-paste augmentation
    'erasing': 0.2,        # Random erasing probability
}


def create_data_yaml():
    """Create a properly configured data.yaml file."""
    dataset_path = DATASET_DIR.absolute()
    
    # Check structure
    train_images = dataset_path / "train" / "images"
    valid_images = dataset_path / "valid" / "images"
    test_images = dataset_path / "test" / "images"
    
    if not train_images.exists():
        # Try alternative structure
        train_images = dataset_path / "train"
        valid_images = dataset_path / "valid"
        test_images = dataset_path / "test"
    
    data_config = {
        'path': str(dataset_path),
        'train': 'train/images' if (dataset_path / "train" / "images").exists() else 'train',
        'val': 'valid/images' if (dataset_path / "valid" / "images").exists() else 'valid',
        'test': 'test/images' if (dataset_path / "test" / "images").exists() else 'test',
        'nc': 10,
        'names': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    }
    
    # Save the corrected data.yaml
    output_yaml = dataset_path / "data_train.yaml"
    with open(output_yaml, 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"Created data config: {output_yaml}")
    print(f"  Train: {data_config['train']}")
    print(f"  Val: {data_config['val']}")
    print(f"  Test: {data_config['test']}")
    
    return output_yaml


def count_dataset_stats(dataset_path):
    """Count images and labels in the dataset."""
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        split_path = dataset_path / split
        
        # Check for images subfolder
        images_path = split_path / "images"
        if not images_path.exists():
            images_path = split_path
        
        labels_path = split_path / "labels"
        if not labels_path.exists():
            labels_path = split_path
        
        # Count files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = sum(1 for f in images_path.glob('*') if f.suffix.lower() in image_extensions)
        labels = sum(1 for f in labels_path.glob('*.txt'))
        
        stats[split] = {'images': images, 'labels': labels}
    
    return stats


def train_model():
    """Train the YOLOv8 digit detection model."""
    
    print("\n" + "="*70)
    print("DIGIT DETECTION MODEL TRAINING")
    print("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create proper data.yaml
    data_yaml = create_data_yaml()
    
    # Print dataset statistics
    print("\nDataset Statistics:")
    print("-" * 40)
    stats = count_dataset_stats(DATASET_DIR)
    for split, counts in stats.items():
        print(f"  {split:8s}: {counts['images']:5d} images, {counts['labels']:5d} labels")
    
    # Load pretrained YOLOv8 model
    model_name = f"yolov8{MODEL_SIZE}.pt"
    print(f"\nLoading pretrained model: {model_name}")
    model = YOLO(model_name)
    
    # Training configuration
    train_args = {
        # Data
        'data': str(data_yaml),
        'imgsz': IMG_SIZE,
        
        # Training
        'epochs': EPOCHS,
        'batch': BATCH_SIZE,
        'patience': PATIENCE,
        'workers': WORKERS,
        
        # Output
        'project': str(OUTPUT_DIR.parent),
        'name': OUTPUT_DIR.name,
        'exist_ok': True,
        
        # Device
        'device': 0,  # Use GPU 0
        
        # Optimizer
        'optimizer': 'AdamW',
        'lr0': 0.001,           # Initial learning rate
        'lrf': 0.01,            # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3.0,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        
        # Loss weights
        'box': 7.5,             # Box loss weight
        'cls': 0.5,             # Classification loss weight
        'dfl': 1.5,             # Distribution focal loss weight
        
        # Augmentations
        **AUGMENTATION_CONFIG,
        
        # Other settings
        'close_mosaic': 10,     # Disable mosaic for last N epochs
        'amp': True,            # Automatic mixed precision
        'cache': True,          # Cache images for faster training
        'save': True,
        'save_period': 10,      # Save checkpoint every N epochs
        'plots': True,
        'verbose': True,
    }
    
    print("\nTraining Configuration:")
    print("-" * 40)
    print(f"  Model: YOLOv8{MODEL_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Device: GPU")
    
    print("\nAugmentation Settings:")
    print("-" * 40)
    for key, value in AUGMENTATION_CONFIG.items():
        print(f"  {key}: {value}")
    
    print("\n" + "="*70)
    print("STARTING TRAINING...")
    print("="*70 + "\n")
    
    # Train the model
    results = model.train(**train_args)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    
    return model, results


def evaluate_model(model):
    """Evaluate the trained model on test set."""
    
    print("\n" + "="*70)
    print("EVALUATING MODEL ON TEST SET")
    print("="*70)
    
    data_yaml = DATASET_DIR / "data_train.yaml"
    
    # Validate on test set
    results = model.val(
        data=str(data_yaml),
        split='test',
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        verbose=True,
        plots=True,
    )
    
    print("\nTest Set Results:")
    print("-" * 40)
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")
    
    return results


def export_model(model):
    """Export the model to different formats."""
    
    print("\n" + "="*70)
    print("EXPORTING MODEL")
    print("="*70)
    
    # The best weights are automatically saved
    best_weights = OUTPUT_DIR / "weights" / "best.pt"
    
    if best_weights.exists():
        print(f"\nBest weights saved at: {best_weights}")
        
        # Copy to a convenient location
        final_model_path = Path("digit_detector_best.pt")
        shutil.copy(best_weights, final_model_path)
        print(f"Copied to: {final_model_path}")
        
        # Optionally export to ONNX for deployment
        try:
            print("\nExporting to ONNX...")
            model.export(format='onnx', imgsz=IMG_SIZE, simplify=True)
            print("ONNX export complete!")
        except Exception as e:
            print(f"ONNX export failed: {e}")
    
    return best_weights


def quick_test(model):
    """Run a quick inference test on a sample image."""
    
    print("\n" + "="*70)
    print("QUICK INFERENCE TEST")
    print("="*70)
    
    # Find a test image
    test_images_path = DATASET_DIR / "test" / "images"
    if not test_images_path.exists():
        test_images_path = DATASET_DIR / "test"
    
    test_images = list(test_images_path.glob("*.jpg"))[:5]
    
    if test_images:
        print(f"\nTesting on {len(test_images)} sample images...")
        
        for img_path in test_images:
            results = model(str(img_path), conf=0.25, verbose=False)
            
            detections = []
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = box.conf.item()
                detections.append(f"{cls}({conf:.2f})")
            
            print(f"  {img_path.name}: {', '.join(detections) if detections else 'No detections'}")
    else:
        print("No test images found for quick test.")


def main():
    """Main training pipeline."""
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"Error: Dataset not found at {DATASET_DIR}")
        print("Please ensure the dataset is downloaded and extracted.")
        return
    
    # Train the model
    model, train_results = train_model()
    
    # Evaluate on test set
    test_results = evaluate_model(model)
    
    # Export model
    best_weights = export_model(model)
    
    # Quick inference test
    quick_test(model)
    
    print("\n" + "="*70)
    print("ALL DONE!")
    print("="*70)
    print(f"\nBest model saved at: {best_weights}")
    print(f"\nTo use the model in your ball tracking script, update the path:")
    print(f'  DIGIT_MODEL_PATH = "{best_weights}"')
    print("\nOr copy 'digit_detector_best.pt' to your desired location.")


if __name__ == "__main__":
    main()
