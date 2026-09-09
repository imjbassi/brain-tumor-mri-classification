# evaluate.py
"""Run a trained checkpoint against the held-out Testing set and produce the
figures referenced in the paper: confusion matrix, ROC curves, and a grid of
misclassified examples."""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

from data_loader import get_dataloaders
from model import get_model
from utils import load_checkpoint, plot_confusion_matrix, plot_image_grid


def collect_predictions(model, device, loader):
    all_probs, all_preds, all_labels, all_paths = [], [], [], []
    dataset = loader.dataset
    idx = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)

            all_probs.append(probs)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())
            all_paths.extend(dataset.image_paths[idx:idx + len(labels)])
            idx += len(labels)

    return np.concatenate(all_probs), np.array(all_preds), np.array(all_labels), all_paths


def plot_roc_curves(probs, labels, class_names, out_path="roc_curves.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, cls in enumerate(class_names):
        binary_labels = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary_labels, probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc(fpr, tpr):.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Per-Class ROC Curves (one-vs-rest)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved ROC curves to {out_path}")


def plot_misclassified(paths, preds, labels, class_names, out_path="misclassified_examples.png", max_examples=8):
    wrong = np.where(preds != labels)[0]
    if len(wrong) == 0:
        print("No misclassified examples to plot.")
        return
    chosen = wrong[:max_examples]

    images = [Image.open(paths[i]).convert("RGB") for i in chosen]
    titles = [f"true: {class_names[labels[i]]}\npred: {class_names[preds[i]]}" for i in chosen]
    plot_image_grid(images, titles, out_path, ncols=4, suptitle="Misclassified Examples")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the held-out test set.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--fig_dir", type=str, default="paper/figures",
                        help="Directory to save confusion_matrix.png / roc_curves.png / misclassified_examples.png in.")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model_path, map_location=device)

    _, _, test_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size)
    if test_loader is None:
        raise FileNotFoundError(f"No 'Testing' subfolder found in {args.data_dir}.")

    class_names = ckpt["class_names"]
    model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    probs, preds, labels, paths = collect_predictions(model, device, test_loader)

    print(classification_report(labels, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(labels, preds, labels=range(len(class_names)))
    plot_confusion_matrix(cm, class_names, os.path.join(args.fig_dir, "confusion_matrix.png"))
    plot_roc_curves(probs, labels, class_names, os.path.join(args.fig_dir, "roc_curves.png"))
    plot_misclassified(paths, preds, labels, class_names, os.path.join(args.fig_dir, "misclassified_examples.png"))


if __name__ == "__main__":
    main()
