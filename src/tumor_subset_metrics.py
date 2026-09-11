# tumor_subset_metrics.py
"""Report metrics restricted to the three tumor classes.

The patient-disjoint split of build_patient_split.py is fully patient-disjoint
for glioma, meningioma and pituitary, but the notumor class has no recoverable
patient identifiers and is only deduplicated and split randomly. The 4-class
accuracy therefore carries a caveat that the tumor-only accuracy does not.

This script recomputes metrics over the tumor classes alone, for every seed in
a sweep directory, so the paper can report the number with no asterisk beside
it next to the headline 4-class number.

Usage:
    python src/tumor_subset_metrics.py --data_dir ./data_patient --runs_dir runs/patient
"""
import argparse
import json
import os

import numpy as np
import torch
from sklearn.metrics import classification_report

from data_loader import get_dataloaders
from evaluate import collect_predictions
from model import get_model
from utils import load_checkpoint

TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")


def main():
    parser = argparse.ArgumentParser(description="Metrics restricted to the tumor classes.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dirs = sorted(d for d in os.listdir(args.runs_dir)
                       if os.path.isdir(os.path.join(args.runs_dir, d)))

    rows = []
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

        tumor_idx = [i for i, c in enumerate(class_names) if c in TUMOR_CLASSES]
        m = np.isin(labels, tumor_idx)

        # Accuracy over true-tumor images: a tumor image predicted as notumor
        # counts as an error, which is the clinically meaningful convention.
        acc_all = float((preds[m] == labels[m]).mean())
        n_err = int((preds[m] != labels[m]).sum())

        rep = classification_report(labels[m], preds[m],
                                     labels=tumor_idx,
                                     target_names=[class_names[i] for i in tumor_idx],
                                     digits=4, output_dict=True, zero_division=0)
        rows.append({
            "run": rd,
            "tumor_accuracy": acc_all,
            "tumor_macro_f1": rep["macro avg"]["f1-score"],
            "n_tumor_test": int(m.sum()),
            "n_errors": n_err,
            "per_class_f1": {class_names[i]: rep[class_names[i]]["f1-score"] for i in tumor_idx},
        })
        print(f"  {rd}: tumor acc {acc_all*100:.2f}%  errors {n_err}/{int(m.sum())}")

    acc = np.array([r["tumor_accuracy"] for r in rows])
    f1 = np.array([r["tumor_macro_f1"] for r in rows])
    errs = np.array([r["n_errors"] for r in rows], dtype=float)
    print(f"\nTumor-only, {len(rows)} runs on {rows[0]['n_tumor_test']} test images:")
    print(f"  accuracy : {acc.mean()*100:.2f}% +/- {acc.std(ddof=1)*100:.2f}")
    print(f"  macro F1 : {f1.mean():.4f} +/- {f1.std(ddof=1):.4f}")
    print(f"  errors   : {errs.mean():.1f} +/- {errs.std(ddof=1):.1f}")

    for c in TUMOR_CLASSES:
        v = np.array([r["per_class_f1"].get(c, np.nan) for r in rows], dtype=float)
        if not np.isnan(v).all():
            print(f"    {c:>11s} F1: {np.nanmean(v):.4f} +/- {np.nanstd(v, ddof=1):.4f}")

    out = args.out or os.path.join(args.runs_dir, "tumor_subset_metrics.json")
    with open(out, "w") as f:
        json.dump({"runs": rows,
                    "mean_accuracy": float(acc.mean()), "sd_accuracy": float(acc.std(ddof=1)),
                    "mean_macro_f1": float(f1.mean()), "sd_macro_f1": float(f1.std(ddof=1))},
                   f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
