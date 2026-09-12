# Brain Tumor MRI Benchmark Audit

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22721031.svg)](https://doi.org/10.5281/zenodo.22721031)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An audit of a widely used four-class brain tumor MRI benchmark, plus the
pipeline used to evaluate it. The headline result of the audit is that the
benchmark's released split **cannot measure generalization to a new patient**,
and that the ~99% accuracies reported on it are substantially an artifact of
that.

## What the audit found

| Finding | Detail |
|---|---|
| **Complete patient-level leakage** | 206 of 233 patients appear in both released folders. Of the 906 tumor test images, 761 (84%) could be traced to a patient, and **every one of them** belongs to a patient the model trains on. |
| **Duplicate contamination** | 216 test images (16.5%) are pixel-identical to a training image; 103 are byte-identical files. |
| **Redundancy** | The 7,023 advertised files contain only 6,597 distinct images (5,994 after near-duplicate clustering). The `notumor` class is 44% redundant against itself. |
| **Deduplication does not fix it** | Removing duplicates leaves accuracy unchanged (99.22% → 99.42%; 95% CI on the difference [−0.03, +0.43] pts), because it leaves every repeated patient in place. |

## What it costs

| Split | Test n | Accuracy |
|---|---|---|
| Released | 1,311 | 99.22% ± 0.16 |
| Deduplicated | 1,092 | 99.42% ± 0.30 |
| **Patient-disjoint (5-fold CV)** | 6,292 | **95.19%** |
| **— tumor classes only** | 4,586 | **93.68%** |

In a paired comparison on the 761 images scored under both regimes, with
identical images, labels, architecture and hyperparameters: **99.12% when the
patient appeared in training, 93.60% when it did not — a 7.3× increase in
error rate.**

Partition choice contributes **2.6× the standard deviation** that random
seeding does (σ_partition = 2.18 pts vs σ_seed = 0.83 pts), and per-fold
accuracy ranges from 92.3% to 98.2%. Single-split accuracies on this
benchmark carry far wider error bars than are usually reported.

Full methodology, the corrected provenance matching, and the re-evaluation
are in [`paper/main.pdf`](paper/main.pdf).

## Using the corrected split

If you work with this dataset, the two artifacts worth taking are:

- **`figshare_patient_map.json`** — a recovered patient identifier for 4,586
  of the 5,023 tumor images.
- **`cv_folds.json`** — a fixed five-fold patient-disjoint partition, every
  patient in exactly one test fold.

Both are derived metadata, not redistributed images, so they can be used
without reference to the licensing of the underlying collections.

```bash
python src/build_cv_folds.py --data_dir ./data --map figshare_patient_map.json --n_folds 5
python src/build_patient_split.py --data_dir ./data --folds cv_folds.json --fold_idx 0 --out_dir ./data_cv_fold0
python src/run_seed_sweep.py --data_dir ./data_cv_fold0 --tag cv_fold0 --augment --seeds 42 0 1 --group_map group_map.json --fold 0
python src/analyze_cv.py --runs_dir runs --n_folds 5
```

**Group your validation split too.** This project originally did not, and on
the first patient-disjoint fold 80.3% of validation images turned out to
belong to a patient also in training — so checkpoint selection, the LR
schedule, early stopping and temperature calibration were all being chosen
against the same contamination. Pass `--group_map` to `train.py`.

## Citing this work

If you use the recovered patient map, the patient-disjoint split, or the audit
tooling, please cite the archived release:

> Bassi, J. (2026). *Brain Tumor MRI Benchmark Audit: patient-identifier
> recovery, leakage analysis, and a patient-disjoint split* (v1.0.0). Zenodo.
> https://doi.org/10.5281/zenodo.22721031

```bibtex
@software{bassi2026audit,
  author    = {Bassi, Jaiveer},
  title     = {Brain Tumor MRI Benchmark Audit: patient-identifier recovery,
               leakage analysis, and a patient-disjoint split},
  version   = {v1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.22721031},
  url       = {https://doi.org/10.5281/zenodo.22721031}
}
```

## Preprint

An earlier version of this work was posted as:

**Bassi, J.** (2025). *Brain Tumor Classification with Pretrained CNNs in PyTorch*.
[DOI: 10.13140/RG.2.2.21638.28484](https://doi.org/10.13140/RG.2.2.21638.28484)

**That version is superseded.** It predates the dataset audit and reports the
released-split accuracy as a result rather than as a measurement artifact.
Cite `paper/main.pdf` in this repository instead.

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

## Auditing the dataset

Reproduce the audit on your own copy. These are the scripts behind the paper's Section 3.

```bash
python src/audit_leakage.py --data_dir ./data
python src/count_unique.py --data_dir ./data
```

`audit_leakage.py` reports exact and near-duplicate contamination across the split and writes evidence figures. `count_unique.py` reports how many genuinely distinct images each class contains.

### Recovering patient identifiers

The aggregated dataset drops the patient IDs present in its figshare source. To recover them you need the original figshare release (~879 MB, CC BY 4.0):

```bash
# download the 4 .zip archives + cvind.mat from
# https://doi.org/10.6084/m9.figshare.1512427  ->  external/figshare/, then unzip to external/figshare/mat/
python src/match_figshare.py --data_dir ./data --mat_dir external/figshare/mat
```

This crop-normalizes both collections and searches all eight dihedral orientations (the aggregated images are rotated 90° relative to figshare). It writes `figshare_patient_map.json` and reports patient-level train/test overlap.

A naive version of this matching returns a **false negative**. `src/provenance_sensitivity.py` shows why: the similarity measures survive JPEG and resizing but collapse under brain cropping, so a genuine match scores no better than an unrelated image unless both sides are crop-normalized first.

```bash
python src/provenance_sensitivity.py --mat_dir external/figshare/mat
```

### Building a patient-disjoint split

```bash
python src/build_patient_split.py --data_dir ./data --map figshare_patient_map.json --out_dir ./data_patient
python src/run_seed_sweep.py --data_dir ./data_patient --tag patient --augment --seeds 42 0 1 2 3
```

`build_patient_split.py` assigns whole patients to one side of the boundary and **asserts** zero patient overlap before writing. Tumor images without a recoverable ID go to training only, so they can never contaminate the test set. The `notumor` class has no patient IDs and is deduplicated and split randomly, which is why results are also reported for the three tumor classes alone.

### Testing the memorization hypothesis

```bash
python src/prediction_agreement.py --data_dir ./data --runs_dir runs/leaky
```

Measures per-image prediction agreement across seeds and error rates, split by whether an image is duplicated in training.

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
