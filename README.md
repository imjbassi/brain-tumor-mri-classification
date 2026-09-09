# Brain Tumor MRI Classification with PyTorch

This project implements a brain tumor classifier using a pretrained ResNet-18 model in PyTorch. It classifies T1-weighted MRI images into four categories: **glioma**, **meningioma**, **pituitary tumor**, and **no tumor**. The pipeline includes data loading, preprocessing, training, evaluation, and interpretability (Grad-CAM).

On the held-out test set of the Mendeley/Kaggle brain tumor dataset, the released configuration reaches **99.16% accuracy** (macro F1 0.991, macro one-vs-rest AUC 0.9999), and across a 5-seed sweep, **99.16% ± 0.14%**. Full methodology, per-class results, a seed-variance and augmentation/fine-tuning ablation, and a calibration analysis are in [`paper/main.pdf`](paper/main.pdf).

## Preprint

**Bassi, J.** (2025). *Brain Tumor Classification with Pretrained CNNs in PyTorch*.
[DOI: 10.13140/RG.2.2.21638.28484](https://doi.org/10.13140/RG.2.2.21638.28484)

---

## Dataset

Four classes: glioma, meningioma, pituitary tumor, no tumor, pre-split into training and testing sets by the dataset source.

**Download**: [https://data.mendeley.com/datasets/w4sw3s9f59/1](https://data.mendeley.com/datasets/w4sw3s9f59/1)

**Expected structure** (place this at `data/` in the project root, or point `--data_dir` at it):

```
data/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

Folder names inside `Training/`/`Testing/` may vary slightly depending on which mirror of the dataset you download; rename them to match if needed.

![Sample Brain Tumor MRI](paper/figures/Figure_1.png)

---

## Installation

```bash
git clone https://github.com/imjbassi/Brain-Tumor-MRI-Classification.git
cd Brain-Tumor-MRI-Classification
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build for your GPU/driver instead of the default CPU wheel (see [pytorch.org](https://pytorch.org/get-started/locally/)).

---

## Training

Run from the project root; `--data_dir` points at the dataset root:

```bash
python src/train.py \
    --data_dir ./data \
    --backbone resnet18 \
    --epochs 30 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --augment
```

* The `Training` split is stratified 80/20 into train and validation sets; `Testing` is kept fully held out for `evaluate.py`.
* `--augment` turns on random flip/rotation/color-jitter for training images.
* `--balance_classes` uses a class-weighted sampler if your class counts are uneven.
* Training stops early if validation loss hasn't improved for `--patience` epochs (default 5), and the learning rate is halved on a plateau.
* Saves the checkpoint to `checkpoints/tumor_model.pth` (bundled with `class_names` and `backbone`, so downstream scripts never hardcode the class order), and writes `history.json`, `training_curves.png`, and `confusion_matrix_val.png` to `paper/figures/`.

Other backbones: `--backbone resnet34` or `--backbone efficientnet_b0`. Add `--freeze_backbone` to only train the new classification head (faster, useful for quick experiments). Override output locations with `--checkpoint_dir` / `--fig_dir`.

---

## Evaluation

Run the held-out test set through a trained checkpoint to get precision/recall/F1 per class plus the figures used in the paper:

```bash
python src/evaluate.py --data_dir ./data --model_path checkpoints/tumor_model.pth
```

Saves `confusion_matrix.png`, `roc_curves.png`, and `misclassified_examples.png` to `paper/figures/`.

## Interpretability (Grad-CAM)

```bash
python src/gradcam.py --data_dir ./data --model_path checkpoints/tumor_model.pth
```

Saves `gradcam_examples.png` to `paper/figures/`, one heatmap overlay per class, showing which regions of the scan drove the prediction (Selvaraju et al., 2017).

## Dataset figures

```bash
python src/visualizer.py --data_dir ./data --save
```

Saves `sample_grid.png` and `class_distribution.png` to `paper/figures/`.

## Calibration

Fits temperature scaling on the validation set and reports expected calibration error (ECE) and negative log-likelihood before/after, plus how confident the model's test-set errors were before/after scaling:

```bash
python src/calibrate.py --data_dir ./data --model_path checkpoints/tumor_model.pth
```

## Reproducing the seed-variance and ablation results

The paper's seed-variance table (5 seeds, main config) and ablation table (augmentation on/off × full fine-tune/frozen backbone, seed 42) are each just `train.py` runs with different flags, e.g.:

```bash
python src/train.py --data_dir ./data --augment --seed 0 --checkpoint_dir runs/seed0 --fig_dir runs/seed0
python src/train.py --data_dir ./data --seed 42 --freeze_backbone --checkpoint_dir runs/frozen --fig_dir runs/frozen
```

then evaluating each resulting checkpoint with `evaluate.py`.

---

## Inference

```bash
python src/inference.py \
    --model_path checkpoints/tumor_model.pth \
    --image_path ./data/Testing/glioma/example1.jpg
```

**Output**:

```
Predicted class: glioma (91.4% confidence)
Runner-up predictions:
  meningioma: 5.2%
  pituitary: 2.1%
```

---

## Rebuilding the paper

Once all figures above have been generated, compile the LaTeX source from the `paper/` directory:

```bash
cd paper
pdflatex main.tex
pdflatex main.tex
```

(Two passes to resolve citations and figure/table references.)

---

## Project Structure

```
.
├── src/
│   ├── data_loader.py     # Dataset class and DataLoader utilities (stratified split, augmentation)
│   ├── model.py            # Backbone factory (ResNet-18/34, EfficientNet-B0) + Grad-CAM hook
│   ├── train.py              # Training loop: early stopping, LR scheduling, checkpoint metadata
│   ├── evaluate.py             # Test-set metrics, confusion matrix, ROC curves, misclassified grid
│   ├── gradcam.py                # Grad-CAM heatmap generation
│   ├── calibrate.py                # Temperature scaling + calibration metrics
│   ├── inference.py                  # Single-image inference
│   ├── visualizer.py                 # Sample grid + class distribution figures
│   └── utils.py                        # Seeding, checkpoint I/O, shared plotting helpers
├── paper/
│   ├── main.tex             # Paper source
│   ├── main.pdf             # Compiled paper
│   └── figures/              # All figures referenced by the paper (generated by src/ scripts)
├── checkpoints/            # Trained model weights (gitignored, created by train.py)
├── data/                   # Dataset (gitignored, download separately)
├── requirements.txt
└── README.md
```

---

## Citation

**Dataset**:

> Ghaffar, A. (2024). *Brain Tumor Classification (MRI)*. Mendeley Data, V1. [https://doi.org/10.17632/w4sw3s9f59.1](https://doi.org/10.17632/w4sw3s9f59.1)
