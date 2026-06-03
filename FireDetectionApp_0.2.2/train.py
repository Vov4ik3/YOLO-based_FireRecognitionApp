import os
import sys
import torch
from ultralytics import YOLO

# ----------------------------------------------------------------
# Config
# ----------------------------------------------------------------

DATASET_YAML  = "C:/Stuff/main project diplom/training data/merged_dataset/data.yaml"
BASE_MODEL    = "C:/Stuff/main project diplom/FireDetectionApp/FireDetectionApp_0.2.2/runs/detect/fire_s_clean_v3/weights/last.pt"
OUTPUT_NAME   = "fire_s_clean_v4"

EPOCHS        = 50
IMAGE_SIZE    = 640
BATCH_SIZE    = 24
LEARNING_RATE = 0.0001   # moderate rate, starting from a clean well-trained checkpoint
FREEZE_LAYERS = 2        # protect early feature detection layers
CONFIDENCE    = 0.5


# ----------------------------------------------------------------
# Sanity checks
# ----------------------------------------------------------------

def check_setup() -> str:
    print("\n  Fire Detection Model - Clean Merged Dataset Training")
    print("=" * 52)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:      {gpu_name}")
        print(f"  VRAM:     {vram:.1f} GB")
        device = 0
    else:
        print("  GPU:      Not found, using CPU (will be slow)")
        device = "cpu"

    try:
        import albumentations
        print(f"  Albumentations: v{albumentations.__version__} found")
    except ImportError:
        print("  Albumentations: not installed (run: pip install albumentations)")
        print("  Training will continue with YOLO built-in augmentation only.")

    if not os.path.exists(DATASET_YAML):
        print(f"\n  ERROR: Dataset not found at: {DATASET_YAML}")
        print("  Update DATASET_YAML at the top of this file.")
        sys.exit(1)

    if not os.path.exists(BASE_MODEL):
        print(f"\n  ERROR: Base model not found: {BASE_MODEL}")
        print("  Check the path to last.pt is correct.")
        sys.exit(1)

    print(f"  Dataset:  {DATASET_YAML}")
    print(f"  Model:    {BASE_MODEL}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  LR:       {LEARNING_RATE}")
    print(f"  Freeze:   first {FREEZE_LAYERS} layers")
    print("=" * 52)
    print("  Starting training...\n")

    return device


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

def run_training(device):
    model = YOLO(BASE_MODEL)

    model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        freeze=FREEZE_LAYERS,
        device=device,
        name=OUTPUT_NAME,
        exist_ok=True,
        verbose=True,
        conf=CONFIDENCE,
        plots=True,
        workers=4,
        cache="disk",
        patience=0,
        cls = 1.0,

        # geometric augmentations
        degrees=10,         # slight rotation, fire can be at angles
        fliplr=0.5,         # horizontal flip, fire looks the same mirrored
        flipud=0.0,         # no vertical flip, fire always burns upward
        scale=0.4,          # random zoom to simulate different distances
        shear=2.0,          # slight shear, different camera angles
        perspective=0.0005, # subtle perspective warp
        translate=0.1,      # random position shift

        # color augmentations to handle different lighting conditions
        hsv_h=0.015,        # hue shift, fire ranges from orange to blue-white
        hsv_s=0.7,          # saturation shift, washed out vs vivid fire
        hsv_v=0.4,          # brightness shift, night fires vs daylight

        # advanced augmentations
        mosaic=1.0,         # combine 4 images, forces detection of smaller fires
        mixup=0.1,          # blend two images, helps with fire behind smoke
        copy_paste=0.1,     # paste fire onto new backgrounds
        erasing=0.3,        # randomly black out parts, stops over-relying on one area
    )

    best_weights = os.path.join("runs", "detect", OUTPUT_NAME, "weights", "best.pt")

    print("\n" + "=" * 52)
    print("  Training complete!")
    print("=" * 52)

    if os.path.exists(best_weights):
        print(f"\n  Your new model is at:")
        print(f"  {best_weights}")
        print(f"\n  Copy it to your project folder:")
        print(f"  copy \"{best_weights}\" fire.pt")
        print(f"\n  Set confidence in core/detector.py:")
        print(f"  conf=0.5")
    else:
        print("\n  Could not find best.pt, check runs/detect folder manually.")

    print("=" * 52 + "\n")


# ----------------------------------------------------------------
# Windows requires __main__ guard for multiprocessing workers
# ----------------------------------------------------------------

if __name__ == "__main__":
    device = check_setup()
    run_training(device)
