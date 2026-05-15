import os
import sys
import torch
from ultralytics import YOLO

# ----------------------------------------------------------------
# Config - change these to match your setup
# ----------------------------------------------------------------

DATASET_YAML   = "C:/Stuff/main project diplom/training data/first fine_tuning/data.yaml"  # path to your dataset yaml file
BASE_MODEL     = "fire.pt"            # the model we're fine-tuning from
OUTPUT_NAME    = "fire_finetuned"     # results go to runs/detect/<this name>

EPOCHS         = 50       # 50 is plenty for fine-tuning, crank to 100 if you have time
IMAGE_SIZE     = 640      # standard YOLO input size, don't change unless you know why
BATCH_SIZE     = 8        # lower this to 4 if you run out of VRAM
LEARNING_RATE  = 0.001    # lower than default since we're fine-tuning not training fresh
FREEZE_LAYERS  = 10       # freeze the first 10 layers so we don't undo what's already learned
CONFIDENCE     = 0.5      # confidence threshold used during validation

# ----------------------------------------------------------------
# Sanity checks before we start
# ----------------------------------------------------------------

def check_setup() -> str:
    """Check GPU, dataset and model are all present. Returns device string."""
    print("\n  Fire Detection Model - Fine Tuning")
    print("=" * 45)

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU:      {gpu_name}")
        print(f"  VRAM:     {vram:.1f} GB")
        device = 0
    else:
        print("  GPU:      Not found, using CPU (will be slow)")
        device = "cpu"

    if not os.path.exists(DATASET_YAML):
        print(f"\n  ERROR: Dataset not found at: {DATASET_YAML}")
        print("  Make sure you downloaded and extracted your dataset")
        print("  and update DATASET_YAML at the top of this file.")
        sys.exit(1)

    if not os.path.exists(BASE_MODEL):
        print(f"\n  ERROR: Base model not found: {BASE_MODEL}")
        print("  Make sure fire.pt is in the same folder as train.py")
        sys.exit(1)

    print(f"  Dataset:  {DATASET_YAML}")
    print(f"  Model:    {BASE_MODEL}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  Freeze:   first {FREEZE_LAYERS} layers")
    print("=" * 45)
    print("  Starting training...\n")

    return device


def run_training(device):
    """Run the actual training and print where to find the output."""
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
        exist_ok=True,   # overwrite previous run with same name instead of erroring
        verbose=True,
        conf=CONFIDENCE,
        plots=True,      # saves training graphs to the output folder
    )

    best_weights = os.path.join("runs", "detect", OUTPUT_NAME, "weights", "best.pt")

    print("\n" + "=" * 45)
    print("  Training complete!")
    print("=" * 45)

    if os.path.exists(best_weights):
        print(f"\n  Your new model is at:")
        print(f"  {best_weights}")
        print(f"\n  To use it, replace fire.pt with this file:")
        print(f"  copy \"{best_weights}\" fire.pt")
        print(f"\n  Then bump confidence back up in core/detector.py:")
        print(f"  conf=0.9")
    else:
        print("\n  Could not find best.pt, check the runs/detect folder manually.")

    print("=" * 45 + "\n")


# ----------------------------------------------------------------
# Windows requires the __main__ guard when using multiprocessing.
# Without it Python crashes trying to spawn dataloader workers
# before the main process has finished setting up.
# ----------------------------------------------------------------

if __name__ == "__main__":
    device = check_setup()
    run_training(device)
