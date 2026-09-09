# calibrate.py
"""Temperature scaling (Guo et al., 2017) for the trained checkpoint.

Softmax confidence out of the box is not the same thing as "probability of
being correct" -- a network can be systematically overconfident. Temperature
scaling fits a single scalar T > 1 on the held-out validation set (never the
test set) that divides the logits before softmax, which softens the
distribution without changing which class wins (argmax is invariant to a
positive rescaling of the logits, so predictions and errors are identical
before and after; only the reported confidence changes).

Usage:
    python src/calibrate.py --data_dir ./data --model_path checkpoints/tumor_model.pth
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import get_dataloaders
from model import get_model
from utils import load_checkpoint


def collect_logits(model, device, loader):
    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            all_logits.append(model(images).cpu())
            all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels, max_iter=200, lr=0.01):
    """Finds the scalar T minimizing NLL on (logits, labels) via LBFGS,
    following the original temperature scaling paper's setup."""
    temperature = torch.nn.Parameter(torch.ones(1) * 1.5)
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def expected_calibration_error(probs, preds, labels, n_bins=10):
    """Bins predictions by confidence and compares average confidence to
    average accuracy in each bin; ECE is the weighted average gap."""
    confidences = probs.max(axis=1)
    accuracies = (preds == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / len(confidences)) * abs(bin_acc - bin_conf)
    return ece


def main():
    parser = argparse.ArgumentParser(description="Fit temperature scaling and report calibration before/after.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42,
                        help="Must match the seed used for training, so the validation split is identical.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.model_path, map_location=device)
    class_names = ckpt["class_names"]

    model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, val_loader, test_loader, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size, seed=args.seed)
    if test_loader is None:
        raise FileNotFoundError(f"No 'Testing' subfolder found in {args.data_dir}.")

    val_logits, val_labels = collect_logits(model, device, val_loader)
    test_logits, test_labels = collect_logits(model, device, test_loader)

    T = fit_temperature(val_logits, val_labels)
    print(f"Fitted temperature: T = {T:.4f}")

    for split_name, logits, labels in [("Validation", val_logits, val_labels), ("Test", test_logits, test_labels)]:
        labels_np = labels.numpy()

        probs_before = F.softmax(logits, dim=1).numpy()
        preds_before = probs_before.argmax(axis=1)

        probs_after = F.softmax(logits / T, dim=1).numpy()
        preds_after = probs_after.argmax(axis=1)

        assert np.array_equal(preds_before, preds_after), \
            "Temperature scaling changed a prediction -- this should be mathematically impossible for T > 0."

        ece_before = expected_calibration_error(probs_before, preds_before, labels_np)
        ece_after = expected_calibration_error(probs_after, preds_after, labels_np)

        nll_before = F.cross_entropy(logits, labels).item()
        nll_after = F.cross_entropy(logits / T, labels).item()

        print(f"\n{split_name} (n={len(labels_np)}):")
        print(f"  ECE  before: {ece_before:.4f}   after: {ece_after:.4f}")
        print(f"  NLL  before: {nll_before:.4f}   after: {nll_after:.4f}")

        if split_name == "Test":
            wrong = np.where(preds_before != labels_np)[0]
            print(f"  {len(wrong)} test errors, confidence before -> after temperature scaling:")
            for i in wrong:
                c_before = probs_before[i].max()
                c_after = probs_after[i].max()
                print(f"    true={class_names[labels_np[i]]:>10s}  pred={class_names[preds_before[i]]:<10s}  "
                      f"conf {c_before:.3f} -> {c_after:.3f}")
            high_conf_before = int((probs_before[wrong].max(axis=1) > 0.9).sum())
            high_conf_after = int((probs_after[wrong].max(axis=1) > 0.9).sum())
            print(f"  errors with confidence > 0.90: {high_conf_before} before, {high_conf_after} after")


if __name__ == "__main__":
    main()
