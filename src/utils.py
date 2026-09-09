# utils.py
"""Shared helpers: reproducibility, checkpointing, and figure generation."""
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed=42):
    """Fix RNG state across torch/numpy/random so runs are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path, model, class_names, model_name, extra=None):
    """Bundle weights with the metadata inference/evaluation need.

    Earlier versions of this repo saved a bare state_dict, which forced
    inference.py to hardcode class_names and silently break if the class
    order ever changed. Saving the metadata alongside the weights removes
    that failure mode.
    """
    payload = {
        "model_state": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state" not in ckpt:
        raise ValueError(
            f"'{path}' looks like an old-style checkpoint (state_dict only). "
            "Re-train with the current train.py to get class_names/model_name metadata."
        )
    return ckpt


def save_history(history, path="history.json"):
    with open(path, "w") as f:
        json.dump(history, f, indent=2)


def plot_training_curves(history, out_path="training_curves.png"):
    """Plot loss and accuracy curves side by side for quick convergence checks."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].legend()

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved training curves to {out_path}")


def plot_confusion_matrix(cm, class_names, out_path="confusion_matrix.png", normalize=True):
    """Render a confusion matrix without pulling in seaborn as a dependency."""
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt, vmax = ".2f", 1.0
    else:
        fmt, vmax = "d", cm.max()

    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix" + (" (normalized)" if normalize else ""))

    thresh = vmax / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            text = format(value, fmt)
            ax.text(j, i, text, ha="center", va="center",
                     color="white" if value > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


def plot_class_distribution(counts_by_split, out_path="class_distribution.png"):
    """counts_by_split: dict like {'Training': {cls: n, ...}, 'Testing': {...}}."""
    class_names = sorted({c for counts in counts_by_split.values() for c in counts})
    splits = list(counts_by_split.keys())
    x = np.arange(len(class_names))
    width = 0.8 / len(splits)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, split in enumerate(splits):
        values = [counts_by_split[split].get(c, 0) for c in class_names]
        ax.bar(x + i * width, values, width, label=split)

    ax.set_xticks(x + width * (len(splits) - 1) / 2)
    ax.set_xticklabels(class_names, rotation=20, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Class Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved class distribution plot to {out_path}")


def plot_image_grid(images, titles, out_path, ncols=4, suptitle=None):
    """Generic grid plotter reused by misclassification and Grad-CAM figures."""
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")

    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {out_path}")
