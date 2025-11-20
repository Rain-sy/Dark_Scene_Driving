# Semantic Segmentation for Low-Light Driving Scenes via Image Enhancement

This repository contains the code base for our final project. We explain each file below:

1. `filter_dark_images.py` - this script is used to filter out the dark images from the BDD10k dataset. 

2. `image_enhancement.py` - this script is used to apply 3 image enhancement techniques (Gamma, CLAHE, and Retinex) to a specified directory of images.

3. `main_segformer.py` - this script is used to load the pre-trained SegFormer model, and it contains methods to evaluate the model on a given dataset.

4. `train_segformer_bdd100k.py` - this script is used to fine-tune the SegFormer model on bdd100k.

5. `evaluate_enhancements.py` - this script is used to evaluate SegFormer (via mIoU) on the original and enhanced images.

6. `unet_segmentation_model.ipynb` - this notebook contains the code for fine-tuning the pre-trained U-net model on the BDD10k training images, and evaluating on Dark Zurich and BDD10k validation images.

