# models/segformer_baseline.py
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
from PIL import Image
import numpy as np

class SegFormerBaseline:
    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024", device=None):
        self.device = "cuda"
        self.feature_extractor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits  # [1, num_classes, H, W]
        seg = logits.argmax(dim=1)[0].cpu().numpy()
        return np.array(image), seg
