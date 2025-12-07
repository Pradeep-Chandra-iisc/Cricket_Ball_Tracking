from ultralytics import YOLO
import os
import shutil
import yaml
from sklearn.model_selection import train_test_split
import torch

# ============================================================
#                   CONFIGURATION
# ============================================================
RAW_DATA_PATH = "training_data"               # Folder containing images/ and labels/
TEMP_YOLO_PATH = "yolo_dataset_initial"       # Temporary YOLO-formatted dataset
PROJECT_NAME = "cricket_ball_v11s_initial"    # Training output folder

IMG_SIZE = 1280
BATCH_SIZE = 4
EPOCHS = 100                                   # Updated to 100 epochs


# ============================================================
#                   BUILD YOLO DATASET
# ============================================================
def setup_data():
    """Organizes raw dataset into YOLO format + train/val split."""

    # --- Clean previous dataset ---
    if os.path.exists(TEMP_YOLO_PATH):
        shutil.rmtree(TEMP_YOLO_PATH)

    # --- Create new folders ---
    for d in ['images/train', 'images/val', 'labels/train', 'labels/val']:
        os.makedirs(f"{TEMP_YOLO_PATH}/{d}", exist_ok=True)

    # --- All images ---
    images = [f for f in os.listdir(f"{RAW_DATA_PATH}/images")
              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # --- Split dataset 80/20 ---
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)
    print(f"📦 Dataset Split → {len(train_imgs)} training, {len(val_imgs)} validation")

    # --- Move images + labels ---
    def move_files(file_list, split):
        for img in file_list:
            label = os.path.splitext(img)[0] + ".txt"

            # Copy image
            shutil.copy(f"{RAW_DATA_PATH}/images/{img}",
                        f"{TEMP_YOLO_PATH}/images/{split}/{img}")

            # Copy label
            src_lbl = f"{RAW_DATA_PATH}/labels/{label}"
            if os.path.exists(src_lbl):
                shutil.copy(src_lbl, f"{TEMP_YOLO_PATH}/labels/{split}/{label}")

    move_files(train_imgs, "train")
    move_files(val_imgs, "val")

    # --- Build data.yaml ---
    yaml_data = {
        'path': os.path.abspath(TEMP_YOLO_PATH),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['ball']
    }

    yaml_path = f"{TEMP_YOLO_PATH}/data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f)

    return yaml_path


# ============================================================
#                      TRAIN YOLO11s
# ============================================================
if __name__ == "__main__":

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Build dataset
    yaml_path = setup_data()

    # Load YOLO11s base model
    model = YOLO("yolo11s.pt")

    # Train with all modifications
    results = model.train(
        data=yaml_path,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        project="yolo_results",
        name=PROJECT_NAME,
        device=0,

        # --- Training Performance ---
        batch=BATCH_SIZE,
        workers=1,
        patience=25,

        # ============================================================
        #               SMALL OBJECT AUGMENTATIONS (IMPORTANT)
        # ============================================================
        mosaic=1.0,          # Huge boost for tiny cricket ball recall
        mixup=0.1,
        copy_paste=0.2,

        # ============================================================
        #               LIGHT COLOR & GEOMETRIC AUGS
        # ============================================================
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        exist_ok=True,
        plots=True
    )

    print("✅ Training Finished Successfully!")

