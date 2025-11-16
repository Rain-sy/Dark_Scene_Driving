# filter_dark_images.py
import cv2
import numpy as np
from pathlib import Path
from shutil import copy2
from tqdm import tqdm

# ---------------- CONFIG ----------------
# Root directory of BDD100K
DATA_ROOT = Path("data/bdd100K")

# Original image folders (10k split)
IMG_TRAIN_DIR = DATA_ROOT / "images_10k" / "train"
IMG_VAL_DIR   = DATA_ROOT / "images_10k" / "val"

# Output folders for filtered dark images
OUT_TRAIN_DIR = DATA_ROOT / "images_10k_dark" / "train"
OUT_VAL_DIR   = DATA_ROOT / "images_10k_dark" / "val"

# Valid image extensions
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Dark threshold in [0, 255] based on HSV V-channel mean.
# Lower threshold = only very dark images.
# Try values around 60–80 first.
DARK_THRESHOLD = 80

# Whether to actually copy dark images to new directories
COPY_FILES = True

# Whether to save a list of selected dark images + their brightness
SAVE_LIST = True
LIST_TRAIN_PATH = DATA_ROOT / "dark_train_list.txt"
LIST_VAL_PATH   = DATA_ROOT / "dark_val_list.txt"
# ----------------------------------------


def is_dark_image(img_path, threshold=DARK_THRESHOLD):
    """
    Decide whether an image is "dark" by checking the mean value of
    the V channel in HSV color space.

    Args:
        img_path (Path): path to image.
        threshold (float): threshold of mean V (0-255). If mean V < threshold,
                           the image is considered dark.

    Returns:
        (bool, float): (is_dark, mean_v)
    """
    img = cv2.imread(str(img_path))
    if img is None:
        # If the image cannot be read, treat it as non-dark and skip it.
        return False, 255.0

    # Convert BGR (OpenCV default) to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]  # V channel, range [0, 255]
    mean_v = float(v_channel.mean())

    is_dark = mean_v < threshold
    return is_dark, mean_v


def filter_split(split_name, in_dir, out_dir, list_path=None):
    """
    Filter dark images for a specific split (e.g., train or val).

    Args:
        split_name (str): name of the split ("train" / "val").
        in_dir (Path): input directory with original images.
        out_dir (Path): output directory where dark images will be copied.
        list_path (Path or None): path to txt file to save selected image names
                                  and their mean V values.

    Returns:
        list[Path]: list of dark image paths (original locations).
    """
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    # Create output directory if we want to copy files
    if COPY_FILES:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Collect all images with valid extensions
    img_paths = sorted([p for p in in_dir.glob("*") if p.suffix.lower() in IMG_EXTS])

    print(f"\n🔎 Filtering split = {split_name}")
    print(f"  Input dir : {in_dir}")
    print(f"  Found {len(img_paths)} images.\n")

    dark_images = []
    dark_stats  = []  # store (filename, mean_v)

    # Scan through all images and check brightness
    for p in tqdm(img_paths, desc=f"{split_name} scanning"):
        is_dark, mean_v = is_dark_image(p)
        if is_dark:
            dark_images.append(p)
            dark_stats.append((p.name, mean_v))

            # Optionally copy dark images to a separate directory
            if COPY_FILES:
                dst = out_dir / p.name
                copy2(p, dst)

    print(f"\n✅ {split_name}: Found {len(dark_images)} dark images (thr={DARK_THRESHOLD}).")

    # Optionally save selected file names and brightness to a text file
    if SAVE_LIST and list_path is not None:
        list_path = Path(list_path)
        with open(list_path, "w", encoding="utf-8") as f:
            for name, mean_v in dark_stats:
                f.write(f"{name}\t{mean_v:.2f}\n")
        print(f"📝 Saved dark image list to: {list_path}")

    return dark_images


def main():
    print("====== Dark Image Filtering (BDD100K images_10k) ======")
    print(f"Dark threshold (V mean): {DARK_THRESHOLD}")
    print(f"Copy files to new dirs: {COPY_FILES}")
    print("=======================================================\n")

    # Process train split
    filter_split(
        split_name="train",
        in_dir=IMG_TRAIN_DIR,
        out_dir=OUT_TRAIN_DIR,
        list_path=LIST_TRAIN_PATH,
    )

    # Process val split
    filter_split(
        split_name="val",
        in_dir=IMG_VAL_DIR,
        out_dir=OUT_VAL_DIR,
        list_path=LIST_VAL_PATH,
    )

    print("\n🎉 Done.")


if __name__ == "__main__":
    main()
