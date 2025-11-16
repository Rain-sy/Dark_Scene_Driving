# evaluate_enhancements.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.segformer_baseline import SegFormerBaseline
from main_segformer import compute_iou  # 直接复用你的 IoU 函数

# ---------------- CONFIG ----------------
base_model = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
base_dir = Path("data/Dark_Zurich_val_anon/rgb_anon/val/night/GOPR0356")
gt_dir = Path("data/Dark_Zurich_val_anon/gt/val/night/GOPR0356")

methods = {
    "Original": base_dir,
    "CLAHE": Path("outputs/enhanced/clahe"),
    "Retinex": Path("outputs/enhanced/retinex"),
    "Gamma": Path("outputs/enhanced/gamma"),
}

num_images = 5  # 控制测试数量
output_plot = Path("outputs/enhanced/eval_comparison.png")

# -----------------------------------------

print(f"Loading model: {base_model}")
model = SegFormerBaseline(model_name=base_model)

def evaluate_dir(data_dir, gt_dir):
    mious = []
    for img_path in sorted(data_dir.glob("*.png"))[:num_images]:
        gt_path = gt_dir / img_path.name.replace("_rgb_anon", "_gt_labelTrainIds")
        if not gt_path.exists():
            print(f"⚠️ No GT found for {img_path.name}")
            continue

        gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
        image, seg = model.predict(str(img_path))
        seg_resized = cv2.resize(seg, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
        miou, _ = compute_iou(seg_resized, gt)*100
        mious.append(miou)
        print(f"{img_path.stem} → mIoU: {miou:.3f}")

    mean_miou = np.nanmean(mious) if mious else 0
    return mean_miou

# ---------------- EVALUATION ----------------
results = {}
for method, path in methods.items():
    if not path.exists():
        print(f"❌ Skipping {method} (no folder found)")
        continue
    print(f"\n🔹 Evaluating {method} images ...")
    results[method] = evaluate_dir(path, gt_dir)
    print(f"✅ {method} Mean mIoU: {results[method]:.3f}")

# ---------------- RESULTS SUMMARY ----------------
print("\n📊 Mean IoU Comparison:")
for k, v in results.items():
    print(f"{k:<10s} → {v:.3f}")

# ---------------- PLOT ----------------
plt.figure(figsize=(8,5))
plt.bar(results.keys(), results.values(), color=["gray","orange","lightblue","violet"])
plt.ylabel("Mean mIoU")
plt.title("Comparison of Image Enhancement Methods on Dark Zurich")
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(output_plot, dpi=200)
plt.close()

print(f"\n✅ Plot saved to {output_plot}")
