import torch
import numpy as np
import random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
from main_segformer import compute_iou
# ---------------- CONFIG ----------------
MODEL_NAME = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
DATA_ROOT = Path("data/bdd100K")         # ← 你的根目录

IMG_TRAIN_DIR = DATA_ROOT / "images_10k" / "train"
IMG_VAL_DIR   = DATA_ROOT / "images_10k" / "val"

MASK_TRAIN_DIR = DATA_ROOT / "labels" / "train"
MASK_VAL_DIR   = DATA_ROOT / "labels" / "val"

OUTPUT_DIR = Path("outputs/segformer_bdd100k")
NUM_CLASSES = 19          # BDD100K seg 常用 19 类（Cityscapes 风格）
EPOCHS = 20
BATCH_SIZE = 1
LR = 5e-5
SEED = 42
# ----------------------------------------

# ---------------- SETUP ----------------
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🟢 Using device: {device}")

# ---------------- DATASET ----------------
class BDD100KSegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, processor):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.processor = processor
        print(f"Initializing dataset:")
        print(f"  img_dir  = {self.img_dir}")
        print(f"  mask_dir = {self.mask_dir}")

        img_exts = [".png", ".jpg", ".jpeg"]
        all_imgs = sorted([p for p in self.img_dir.glob("*") if p.suffix.lower() in img_exts])
        print(f"  Found {len(all_imgs)} image files.")

        self.img_paths = []
        self.mask_paths = []

        for ip in all_imgs:
            stem = ip.stem 
            mp = self.mask_dir / f"{stem}_train_id.png"

            if mp.exists():
                self.img_paths.append(ip)
                self.mask_paths.append(mp)
            else:
                if len(self.img_paths) < 5:
                    print(f"  ⚠️ Missing mask for {ip.name} (expected {mp.name})")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        image = Image.open(img_path).convert("RGB")
        mask  = Image.open(mask_path)
        if mask.mode != "L":
            mask = mask.convert("L")

        inputs = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt"
        )
        for k, v in inputs.items():
            inputs[k] = v.squeeze(0)
        return inputs

# ---------------- LOAD DATA ----------------
processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)

train_dataset = BDD100KSegDataset(IMG_TRAIN_DIR, MASK_TRAIN_DIR, processor)
val_dataset   = BDD100KSegDataset(IMG_VAL_DIR, MASK_VAL_DIR, processor)

print(f"✅ Dataset sizes - train: {len(train_dataset)}, val: {len(val_dataset)}")

# ---------------- MODEL ----------------
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(device)

# ---------------- METRICS ----------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.from_numpy(logits)
    labels = torch.from_numpy(labels)

    upsampled_logits = torch.nn.functional.interpolate(
        logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
    )
    preds = upsampled_logits.argmax(dim=1).cpu().numpy()

    mious = []
    for i in range(preds.shape[0]):
        miou, _ = compute_iou(preds[i], labels[i])
        mious.append(miou)
    mean_miou = float(np.nanmean(mious))
    return {"mean_iou": mean_miou}

# ---------------- TRAINING SETUP ----------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=LR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    save_total_limit=2,
    logging_dir=str(OUTPUT_DIR / "logs"),
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    eval_accumulation_steps=8, 
    save_strategy="epoch",
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="mean_iou",
    greater_is_better=True,
    fp16=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# ---------------- RUN ----------------
if __name__ == "__main__":
    print("🚀 Start training SegFormer on BDD100K...")
    trainer.train()

    print("🔍 Evaluating best model...")
    trainer.evaluate()

    best_dir = OUTPUT_DIR / "best_model"
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)

    print("✅ Training completed! Model + preprocessor saved to:", best_dir)
