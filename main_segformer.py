# main_segformer_baseline_eval.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from models.segformer_baseline import SegFormerBaseline

# ---------------- CONFIG ----------------
model_name = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
data_dir = Path("data/Dark_Zurich_val_anon/rgb_anon/val/night/GOPR0356")
gt_dir = Path("data/Dark_Zurich_val_anon/gt/val/night/GOPR0356")
output_dir = Path("outputs/segformer_baseline_eval")
output_dir.mkdir(parents=True, exist_ok=True)

num_classes = 19
ignore_index = 255

# Color map for 19 classes (Cityscapes-like)
COLORMAP = np.array([
    [128, 64,128], [244, 35,232], [ 70, 70, 70], [102,102,156], [190,153,153],
    [153,153,153], [250,170, 30], [220,220,  0], [107,142, 35], [152,251,152],
    [ 70,130,180], [220, 20, 60], [255,  0,  0], [  0,  0,142], [  0,  0, 70],
    [  0, 60,100], [  0, 80,100], [  0,  0,230], [119, 11, 32]
], dtype=np.uint8)

# ---------------- UTILS ----------------
def colorize(segmentation):
    seg_color = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(COLORMAP):
        seg_color[segmentation == label] = color
    return seg_color

def compute_iou(pred, gt, num_classes=num_classes, ignore_index=ignore_index):
    """Compute mean IoU and per-class IoU"""
    ious = []
    mask = gt != ignore_index
    pred = pred[mask]
    gt = gt[mask]
    for cls in range(num_classes):
        pred_mask = (pred == cls)
        gt_mask = (gt == cls)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print(f"Loading model: {model_name}")
    model = SegFormerBaseline(model_name=model_name)

    all_mious = []
    all_class_ious = []

    for img_path in sorted(data_dir.glob("*.png")):
        image, seg = model.predict(str(img_path))
        seg_color = colorize(seg)
        seg_color = cv2.resize(seg_color, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        blended = cv2.addWeighted(image, 0.6, seg_color, 0.4, 0)
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        axes[0].imshow(image); axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(seg_color); axes[1].set_title("Segmentation"); axes[1].axis("off")
        axes[2].imshow(blended); axes[2].set_title("Overlay"); axes[2].axis("off")
        fig.tight_layout()
        save_fig_path = output_dir / f"{img_path.stem}_subplot.png"
        plt.savefig(save_fig_path, bbox_inches='tight', dpi=200)
        plt.close(fig)
        gt_path = gt_dir / img_path.name.replace("_rgb_anon", "_gt_labelTrainIds")
        if gt_path.exists():
            gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)
            seg_resized = cv2.resize(seg, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
            miou, class_ious = compute_iou(seg_resized, gt)
            miou *= 100 
            class_ious = np.array(class_ious) * 100
            all_mious.append(miou)
            all_class_ious.append(class_ious)

            print(f"{img_path.stem} → mIoU: {miou:.2f}")

        else:
            print(f"{img_path.stem} → GT not found: {gt_path.name}")

    # ---------------- FINAL SUMMARY ----------------
    if all_mious:
        print("\n===== Final Results =====")
        print(f"Mean mIoU across {len(all_mious)} images: {np.nanmean(all_mious):.2f}")
        mean_class_ious = np.nanmean(np.array(all_class_ious), axis=0)
        for cls_id, iou in enumerate(mean_class_ious):
            print(f"Class {cls_id} average IoU: {iou:.2f}")
    else:
        print("No ground truth masks found. Only visualizations saved.")
