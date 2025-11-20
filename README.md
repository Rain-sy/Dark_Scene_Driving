# Semantic Segmentation for Low-Light Driving Scenes via Image Enhancement

This repository contains the course project for **Robot Perception Systems**, focusing on **semantic segmentation in dark / low-light driving scenes**.

We mainly work with:

- **Dark Zurich** (nighttime driving scenes) – used for evaluation
- **BDD100K** (driving scenes, including nighttime) – used for fine-tuning SegFormer

We explain each file below:

1. `filter_dark_images.py` - this script is used to filter out the dark images from the BDD10k dataset. 

2. `image_enhancement.py` - this script is used to apply 3 image enhancement techniques (Gamma, CLAHE, and Retinex) to a specified directory of images.

3. `main_segformer.py` - this script is used to load the pre-trained SegFormer model, and it contains methods to evaluate the model on a given dataset.

4. `train_segformer_bdd100k.py` - this script is used to fine-tune the SegFormer model on bdd100k.

5. `evaluate_enhancements.py` - this script is used to evaluate SegFormer (via mIoU) on the original and enhanced images.

6. `unet_segmentation_model.ipynb` - this notebook contains the code for fine-tuning the pre-trained U-net model on the BDD10k training images, and evaluating on Dark Zurich and BDD10k validation images.
```text
Project/
│
├── data/
│   ├── Dark_Zurich_val_anon/
│   │   ├── rgb_anon/
│   │   │   └── val/night/GOPR0356/      # Dark Zurich validation night images
│   │   └── gt/
│   │       └── val/night/GOPR0356/      # corresponding ground-truth labelTrainIds
│   │
│   └── bdd100K/
│       ├── images_10k/
│       │   ├── train/                    # all BDD100K train images
│       │   └── val/                      # all BDD100K val images
│       ├── labels/
│       │   ├── train/                    # BDD100K *_train_id.png segmentation masks
│       │   └── val/
│       └── filtered_dark/
│           ├── images_train/             # filtered dark images (train)
│           ├── images_val/               # filtered dark images (val)
│           ├── labels_train/             # filtered dark labels (train)
│           └── labels_val/               # filtered dark labels (val)
│
├── models/
│   └── segformer_baseline.py             # Wrapper around SegFormer (HuggingFace)
│
├── main_segformer.py       # Baseline SegFormer on Dark Zurich (eval + vis)
├── image_enhancement.py                  # Image enhancement (CLAHE, Retinex, Gamma)
├── evaluate_enhancements.py              # Evaluate mIoU for enhanced images
│
├── filter_dark_images.py                # Filter BDD100K to nighttime / dark images
├── train_segformer_bdd100k.py            # Fine-tune SegFormer on (filtered) BDD100K
├── evaluate_finetuned_model.py   # Evaluate finetuned BDD100K model on Dark Zurich
│
└── outputs/
    ├── segformer_baseline_eval/          # Baseline predictions & overlays on Dark Zurich
    ├── enhanced/
    │   ├── clahe/                        # CLAHE-enhanced images
    │   ├── retinex/                      # Retinex-enhanced images
    │   ├── gamma/                        # Gamma-corrected images
    │   └── eval_comparison.png           # Bar chart comparing mIoU of methods
    └── segformer_bdd100k/
        └── best_model/                   # Fine-tuned SegFormer weights & processor
    └── eval_bdd100k_on_darkzurich/       # Finetuned model predictions & overlays
