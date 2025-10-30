# image_enhancement.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ Enhancement Functions ------------------

def enhance_hist_eq(image):
    """CLAHE (adaptive histogram equalization)"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    enhanced = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    return enhanced

def enhance_retinex(image, sigma=66):
    """Single Scale Retinex (SSR)"""
    image = image.astype(np.float32) + 1.0  # 防止 log(0)
    img_log = np.log(image)
    blur = cv2.GaussianBlur(image, (0, 0), sigma)
    blur_log = np.log(blur)
    retinex = img_log - blur_log
    retinex = (retinex - retinex.min()) / (retinex.max() - retinex.min()) * 255.0
    return np.uint8(retinex)

def enhance_gamma(image, gamma=1.5):
    """Gamma correction"""
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# ------------------ Visualization ------------------

def visualize_all(original, clahe, retinex, gamma_img, save_path=None):
    """Create a 2x2 subplot: Original / CLAHE / Retinex / Gamma"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    imgs = [original, clahe, retinex, gamma_img]
    titles = ["Original", "CLAHE", "Retinex", "Gamma"]

    for ax, img, title in zip(axes.ravel(), imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

# ------------------ Main Script ------------------

if __name__ == "__main__":
    # Input & output directories
    input_dir = Path("data/Dark_Zurich_val_anon/rgb_anon/val/night/GOPR0356")
    output_root = Path("outputs/enhanced")
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Running image enhancement for 3 methods (CLAHE, Retinex, Gamma)...")

    for img_path in sorted(input_dir.glob("*.png"))[:5]:
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply all three methods
        clahe_img = enhance_hist_eq(image)
        retinex_img = enhance_retinex(image)
        gamma_img = enhance_gamma(image)

        # Save each enhanced result
        for method_name, enhanced_img in zip(["clahe", "retinex", "gamma"], [clahe_img, retinex_img, gamma_img]):
            out_dir = output_root / method_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / img_path.name
            cv2.imwrite(str(out_path), cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR))

        # Save a comparison grid (Original + 3 methods)
        compare_path = output_root / f"{img_path.stem}_comparison.png"
        visualize_all(image, clahe_img, retinex_img, gamma_img, compare_path)

        print(f" Processed {img_path.name} → saved comparison: {compare_path.name}")

    print(f" All enhanced images saved under: {output_root}/")
