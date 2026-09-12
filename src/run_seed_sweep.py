# run_seed_sweep.py
"""Train the same configuration across several seeds and report mean +/- SD.

Used for both the seed-variance table and the leaky/clean split comparison:

    # baseline (original, leaky split)
    python src/run_seed_sweep.py --data_dir ./data --tag leaky --augment

    # after deduplication
    python src/run_seed_sweep.py --data_dir ./data_dedup --tag dedup --augment

Each run is a fresh `train.py` subprocess (clean GPU state per run), followed
by evaluation on that dataset's held-out test set. Results accumulate in
runs/<tag>/results.json and the script is resumable: rerunning skips any seed
whose checkpoint already exists.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score

from data_loader import get_dataloaders, load_group_map
from evaluate import collect_predictions
from model import get_model
from utils import load_checkpoint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_one(run_dir, data_dir, seed, augment, freeze, epochs, group_map=None):
    ckpt = os.path.join(run_dir, "tumor_model.pth")
    if os.path.exists(ckpt):
        print(f"  [{os.path.basename(run_dir)}] checkpoint exists, skipping training")
        return ckpt

    cmd = [sys.executable, os.path.join("src", "train.py"),
           "--data_dir", data_dir, "--seed", str(seed), "--epochs", str(epochs),
           "--checkpoint_dir", run_dir, "--fig_dir", run_dir]
    if augment:
        cmd.append("--augment")
    if freeze:
        cmd.append("--freeze_backbone")
    if group_map:
        cmd += ["--group_map", group_map]

    print(f"  running: {' '.join(cmd)}")
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "train_log.txt"), "w") as f:
        f.write(proc.stdout + "\n--- STDERR ---\n" + proc.stderr)
    print(f"  finished in {time.time() - t0:.0f}s (exit {proc.returncode})")
    if proc.returncode != 0 or not os.path.exists(ckpt):
        raise RuntimeError(f"training failed; see {run_dir}/train_log.txt")
    return ckpt


def evaluate_one(ckpt_path, data_dir, run_dir, group_map=None, fold=None, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    class_names = ckpt["class_names"]

    _, _, test_loader, _ = get_dataloaders(data_dir, batch_size=32, num_workers=0)
    model = get_model(num_classes=len(class_names), backbone=ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    probs, preds, labels, paths = collect_predictions(model, device, test_loader)

    # Per-image rows. Downstream analyses (pooled CV metrics, patient-level
    # bootstrap, paired leaky-vs-CV comparison, permutation tests) all need
    # these; recovering them later means re-running inference over every
    # checkpoint, so write them once here.
    gmap = load_group_map(group_map) if isinstance(group_map, str) else (group_map or {})
    pred_csv = os.path.join(run_dir, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "group", "true", "pred", "correct", "confidence",
                    "fold", "seed"] + [f"p_{c}" for c in class_names])
        for p, pr, tr, pb in zip(paths, preds, labels, probs):
            key = os.path.normpath(os.path.relpath(p))
            # the map is keyed by basename so it survives copies into derived
            # split directories; see data_loader.load_group_map
            w.writerow([key.replace("\\", "/"), gmap.get(os.path.basename(p), ""), class_names[tr],
                        class_names[pr], int(pr == tr), f"{float(pb.max()):.6f}",
                        "" if fold is None else fold,
                        "" if seed is None else seed]
                       + [f"{float(x):.6f}" for x in pb])

    rep = classification_report(labels, preds, target_names=class_names,
                                 digits=4, output_dict=True)
    out = {
        "test_accuracy": rep["accuracy"],
        "macro_f1": rep["macro avg"]["f1-score"],
        "macro_auc": float(roc_auc_score(labels, probs, multi_class="ovr", average="macro")),
        "n_test": int(len(labels)),
        "n_errors": int((preds != labels).sum()),
        "per_class_f1": {c: rep[c]["f1-score"] for c in class_names},
    }
    hist_path = os.path.join(run_dir, "history.json")
    if os.path.exists(hist_path):
        h = json.load(open(hist_path))
        out["n_epochs_run"] = len(h["val_loss"])
        out["best_val_acc"] = h["val_acc"][int(np.argmin(h["val_loss"]))]
    return out


def main():
    parser = argparse.ArgumentParser(description="Seed sweep with mean +/- SD reporting.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True,
                        help="Name for this sweep, e.g. 'leaky' or 'dedup'.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--group_map", type=str, default=None,
                        help="JSON image->group map, forwarded to train.py so the "
                             "train/validation split is patient-grouped, and used to "
                             "annotate the per-image predictions file.")
    parser.add_argument("--fold", type=int, default=None,
                        help="Fold index to record in the predictions file (CV bookkeeping only).")
    args = parser.parse_args()

    base = os.path.join("runs", args.tag)
    os.makedirs(base, exist_ok=True)
    results_path = os.path.join(base, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    for seed in args.seeds:
        key = f"seed{seed}"
        if key in results:
            print(f"[{args.tag}/{key}] already have results, skipping")
            continue
        print(f"\n[{args.tag}/{key}]")
        run_dir = os.path.join(base, key)
        ckpt = train_one(run_dir, args.data_dir, seed, args.augment,
                          args.freeze_backbone, args.epochs, group_map=args.group_map)
        results[key] = evaluate_one(ckpt, args.data_dir, run_dir,
                                     group_map=args.group_map, fold=args.fold, seed=seed)
        results[key]["seed"] = seed
        if args.fold is not None:
            results[key]["fold"] = args.fold
        results[key]["grouped_val"] = bool(args.group_map)
        print(f"  test_accuracy={results[key]['test_accuracy']:.4f} "
              f"macro_f1={results[key]['macro_f1']:.4f} "
              f"errors={results[key]['n_errors']}/{results[key]['n_test']}")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\n=== {args.tag}: {len(results)} runs ===")
    for metric in ["test_accuracy", "macro_f1", "macro_auc"]:
        v = np.array([results[k][metric] for k in results], dtype=float)
        sd = v.std(ddof=1) if len(v) > 1 else 0.0
        print(f"  {metric:>14s}: {v.mean():.4f} +/- {sd:.4f}   "
              f"(min {v.min():.4f}, max {v.max():.4f})")
    errs = np.array([results[k]["n_errors"] for k in results], dtype=float)
    ntest = results[list(results)[0]]["n_test"]
    print(f"  {'errors':>14s}: {errs.mean():.1f} +/- "
          f"{errs.std(ddof=1) if len(errs) > 1 else 0:.1f}  of {ntest}")
    print(f"\nResults: {results_path}")


if __name__ == "__main__":
    main()
