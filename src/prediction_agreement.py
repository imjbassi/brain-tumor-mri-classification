# prediction_agreement.py
"""Are duplicated test images memorized, or are they just harder?

An earlier version of this work argued that duplicate contamination suppressed
run-to-run variance, on the grounds that the observed seed-to-seed spread sat
below a binomial "sampling floor". That argument was wrong: binomial sampling
error describes the uncertainty of an accuracy estimate under resampling of
the test set, but across seeds the test set is fixed. Two independently
trained models can agree on nearly every image, so there is no lower bound of
that kind to violate.

This script replaces the argument with a direct measurement. Across the ten
trained models in runs/leaky, it computes:

  * per-image prediction agreement (the fraction of seeds assigning the modal
    class), split by whether the test image is duplicated in training
  * the error rate on duplicated versus non-duplicated test images, with a
    Fisher exact test and confidence intervals

If duplicated images were memorized, they would be predicted identically on
every run and essentially never missed. The data say otherwise.

Usage:
    python src/prediction_agreement.py --data_dir ./data --runs_dir runs/leaky
"""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
from scipy import stats

from audit_leakage import pixel_md5
from data_loader import BrainMRIDataset, get_dataloaders
from evaluate import collect_predictions
from model import get_model
from utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Prediction agreement across seeds, by duplication status.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default="runs/leaky")
    parser.add_argument("--out", type=str, default="paper/figures/agreement_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- which test images are duplicated in training? ---------------------
    train_ds = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test_ds = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                               class_to_idx=train_ds.class_to_idx)
    train_hashes = {pixel_md5(p) for p in train_ds.image_paths}
    is_dup = np.array([pixel_md5(p) in train_hashes for p in test_ds.image_paths])
    print(f"Test images: {len(is_dup)}, duplicated in training: {is_dup.sum()}")

    # ---- collect predictions from every seed -------------------------------
    run_dirs = sorted(d for d in os.listdir(args.runs_dir)
                       if os.path.isdir(os.path.join(args.runs_dir, d)))
    preds_all, labels_ref = [], None
    for rd in run_dirs:
        ckpt_path = os.path.join(args.runs_dir, rd, "tumor_model.pth")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        class_names = ckpt["class_names"]
        model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
        model.load_state_dict(ckpt["model_state"])
        _, _, test_loader, _ = get_dataloaders(args.data_dir, batch_size=32, num_workers=0)
        _, preds, labels, _ = collect_predictions(model, device, test_loader)
        preds_all.append(preds)
        labels_ref = labels
        print(f"  {rd}: {(preds != labels).sum()} errors")

    P = np.stack(preds_all)          # (n_seeds, n_test)
    labels = labels_ref
    n_seeds = P.shape[0]
    print(f"\nModels loaded: {n_seeds}")

    # ---- per-image agreement ------------------------------------------------
    agreement = np.array([Counter(P[:, i]).most_common(1)[0][1] / n_seeds
                           for i in range(P.shape[1])])
    unanimous = np.array([len(set(P[:, i])) == 1 for i in range(P.shape[1])])

    print("\n--- prediction agreement across seeds ---")
    for name, m in (("duplicated", is_dup), ("not duplicated", ~is_dup)):
        print(f"  {name:>15s} (n={m.sum():4d}): mean agreement {agreement[m].mean():.4f}, "
              f"unanimous on {100*unanimous[m].mean():.1f}% of images")

    # ---- error rates --------------------------------------------------------
    errs = P != labels[None, :]
    print("\n--- error rate by duplication status (pooled over seeds) ---")
    res = {}
    for name, m in (("duplicated", is_dup), ("not duplicated", ~is_dup)):
        n_err = int(errs[:, m].sum())
        n_tot = int(errs[:, m].size)
        rate = n_err / n_tot
        lo, hi = stats.beta.interval(0.95, max(n_err, 0.5), max(n_tot - n_err, 0.5))
        print(f"  {name:>15s}: {n_err} / {n_tot} = {100*rate:.3f}%  95% CI [{100*lo:.3f}%, {100*hi:.3f}%]")
        res[name] = {"errors": n_err, "total": n_tot, "rate": rate}

    table = [[res["duplicated"]["errors"],
              res["duplicated"]["total"] - res["duplicated"]["errors"]],
             [res["not duplicated"]["errors"],
              res["not duplicated"]["total"] - res["not duplicated"]["errors"]]]
    odds, p = stats.fisher_exact(table)
    ratio = res["duplicated"]["rate"] / res["not duplicated"]["rate"]
    print(f"\n  duplicated / non-duplicated error-rate ratio: {ratio:.2f}x")
    print(f"  Fisher exact: odds ratio {odds:.3f}, p = {p:.4g}")
    if ratio > 1:
        print("  Duplicated images are MISSED MORE OFTEN than non-duplicated ones,")
        print("  which is the opposite of what memorization would produce.")

    out = {
        "n_seeds": n_seeds,
        "n_test": int(len(is_dup)),
        "n_duplicated": int(is_dup.sum()),
        "agreement_duplicated": float(agreement[is_dup].mean()),
        "agreement_not_duplicated": float(agreement[~is_dup].mean()),
        "unanimous_frac_duplicated": float(unanimous[is_dup].mean()),
        "unanimous_frac_not_duplicated": float(unanimous[~is_dup].mean()),
        "error_rates": res,
        "error_rate_ratio": float(ratio),
        "fisher_odds_ratio": float(odds),
        "fisher_p": float(p),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
