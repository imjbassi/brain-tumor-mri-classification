# analyze_cv.py
"""Analyze the patient-level cross-validation runs.

Reads the per-image predictions written by run_seed_sweep.py and produces the
numbers the paper reports for the patient-disjoint evaluation:

  * pooled accuracy per replicate (one complete pass over all 233 patients,
    every patient predicted by a model that never trained on them)
  * a patient-level bootstrap confidence interval on the pooled estimate,
    which captures the dominant uncertainty (which patients you happen to
    have) and respects the correlation between slices of one patient
  * a variance decomposition separating partition variance from seed variance
  * pooled per-class metrics and a pooled confusion matrix
  * the paired comparison against the released-split models on the identical
    set of test images

Usage:
    python src/analyze_cv.py --runs_dir runs --n_folds 5 --out paper/figures/cv_analysis.json
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_cv(runs_dir, n_folds):
    """Returns {seed: {basename: row}} across all folds, plus fold membership."""
    by_seed = defaultdict(dict)
    fold_of = {}
    for k in range(n_folds):
        for pred_file in sorted(glob.glob(os.path.join(runs_dir, f"cv_fold{k}", "seed*", "predictions.csv"))):
            seed = os.path.basename(os.path.dirname(pred_file)).replace("seed", "")
            for r in read_predictions(pred_file):
                base = os.path.basename(r["path"])
                by_seed[seed][base] = r
                fold_of[base] = k
    return by_seed, fold_of


def pooled_metrics(rows, class_names):
    y_true = [r["true"] for r in rows]
    y_pred = [r["pred"] for r in rows]
    rep = classification_report(y_true, y_pred, labels=class_names,
                                 target_names=class_names, digits=4,
                                 output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    return rep, cm


def patient_bootstrap(rows, n_boot=10000, seed=0):
    """Resample patients (groups) with replacement; recompute pooled accuracy.

    Captures the dominant uncertainty term, which patients are in the sample,
    and handles within-patient correlation between slices. It is conditional
    on the training procedure and does not propagate training variability.
    """
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group"] or os.path.basename(r["path"])].append(int(r["correct"]))
    groups = list(by_group)
    arrs = [np.array(by_group[g]) for g in groups]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    n = len(groups)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        tot = corr = 0
        for i in idx:
            a = arrs[i]
            tot += a.size
            corr += a.sum()
        out[b] = corr / tot
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def variance_components(acc_by_fold_seed, n_folds, seeds):
    """One-way random-effects ANOVA: acc[f,s] = mu + F_f + eps.

    sigma^2_seed = MS_within; sigma^2_partition = (MS_between - MS_within)/n_seeds.
    """
    M = np.array([[acc_by_fold_seed[(f, s)] for s in seeds] for f in range(n_folds)])
    n_s = M.shape[1]
    grand = M.mean()
    fold_means = M.mean(axis=1)
    ms_between = n_s * ((fold_means - grand) ** 2).sum() / (n_folds - 1)
    ms_within = ((M - fold_means[:, None]) ** 2).sum() / (n_folds * (n_s - 1))
    var_seed = ms_within
    var_part = max((ms_between - ms_within) / n_s, 0.0)
    return {
        "sigma_seed": float(np.sqrt(var_seed)), "df_seed": int(n_folds * (n_s - 1)),
        "sigma_partition": float(np.sqrt(var_part)), "df_partition": int(n_folds - 1),
        "sigma_single_run_single_split": float(np.sqrt(var_seed + var_part)),
        "fold_means": [float(x) for x in fold_means],
        "grand_mean": float(grand),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze patient-level CV runs.")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--leaky_dir", type=str, default="runs/leaky",
                        help="Released-split runs, for the paired comparison.")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--out", type=str, default="paper/figures/cv_analysis.json")
    args = parser.parse_args()

    by_seed, fold_of = load_cv(args.runs_dir, args.n_folds)
    if not by_seed:
        raise SystemExit("no CV predictions found; has the sweep finished?")
    seeds = sorted(by_seed, key=lambda s: int(s))
    class_names = sorted({r["true"] for rows in by_seed.values() for r in rows.values()})
    print(f"replicates (seeds): {seeds}; classes: {class_names}")

    result = {"seeds": seeds, "n_folds": args.n_folds, "classes": class_names}

    # ---- pooled per replicate ----------------------------------------------
    print("\n--- pooled over all folds, per replicate ---")
    per_rep = {}
    for s in seeds:
        rows = list(by_seed[s].values())
        rep, cm = pooled_metrics(rows, class_names)
        lo, hi = patient_bootstrap(rows, n_boot=args.n_boot)
        per_rep[s] = {
            "n": len(rows), "accuracy": rep["accuracy"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "per_class_f1": {c: rep[c]["f1-score"] for c in class_names},
            "bootstrap_ci95": [lo, hi],
            "confusion_matrix": cm.tolist(),
        }
        print(f"  seed {s}: n={len(rows)} acc={100*rep['accuracy']:.2f}% "
              f"macroF1={rep['macro avg']['f1-score']:.4f} "
              f"patient-bootstrap 95% CI [{100*lo:.2f}, {100*hi:.2f}]")
    result["per_replicate"] = per_rep

    accs = np.array([per_rep[s]["accuracy"] for s in seeds])
    print(f"  mean over replicates: {100*accs.mean():.2f}% "
          f"(spread {100*accs.min():.2f}-{100*accs.max():.2f})")
    result["pooled_mean_over_replicates"] = float(accs.mean())

    # ---- tumor-only ---------------------------------------------------------
    tumor = [c for c in class_names if c != "notumor"]
    print("\n--- tumor classes only (fully patient-disjoint) ---")
    tum = {}
    for s in seeds:
        rows = [r for r in by_seed[s].values() if r["true"] in tumor]
        rep, _ = pooled_metrics(rows, tumor)
        lo, hi = patient_bootstrap(rows, n_boot=args.n_boot)
        # classification_report omits "accuracy" when `labels` is a subset of
        # the classes present, so compute it directly over these rows.
        acc = float(np.mean([int(r["correct"]) for r in rows]))
        tum[s] = {"n": len(rows), "accuracy": acc,
                  "macro_f1": rep["macro avg"]["f1-score"],
                  "per_class_f1": {c: rep[c]["f1-score"] for c in tumor},
                  "bootstrap_ci95": [lo, hi]}
        rep = {"accuracy": acc, **rep}
        print(f"  seed {s}: n={len(rows)} acc={100*rep['accuracy']:.2f}% "
              f"CI [{100*lo:.2f}, {100*hi:.2f}]")
    result["tumor_only"] = tum

    # ---- variance decomposition --------------------------------------------
    acc_fs = {}
    for k in range(args.n_folds):
        rj = os.path.join(args.runs_dir, f"cv_fold{k}", "results.json")
        if not os.path.exists(rj):
            continue
        d = json.load(open(rj))
        for key, v in d.items():
            s = str(v.get("seed", key.replace("seed", "")))
            acc_fs[(k, s)] = v["test_accuracy"]
    if len(acc_fs) == args.n_folds * len(seeds):
        vc = variance_components(acc_fs, args.n_folds, seeds)
        result["variance_components"] = vc
        print("\n--- variance decomposition (per-fold test accuracy) ---")
        print(f"  sigma_seed        = {100*vc['sigma_seed']:.3f} pts (df {vc['df_seed']})")
        print(f"  sigma_partition   = {100*vc['sigma_partition']:.3f} pts (df {vc['df_partition']})")
        print(f"  one run, one split= {100*vc['sigma_single_run_single_split']:.3f} pts")
        print(f"  fold means: {[round(100*x,2) for x in vc['fold_means']]}")
    else:
        print(f"\n(variance decomposition skipped: {len(acc_fs)} of "
              f"{args.n_folds*len(seeds)} fold/seed cells present)")

    # ---- paired comparison against the released-split models ---------------
    leaky_files = sorted(glob.glob(os.path.join(args.leaky_dir, "seed*", "predictions.csv")))
    if leaky_files:
        print("\n--- paired comparison on identical images ---")
        leaky = {}
        for pf in leaky_files:
            s = os.path.basename(os.path.dirname(pf)).replace("seed", "")
            leaky[s] = {os.path.basename(r["path"]): r for r in read_predictions(pf)}
        common = set.intersection(
            set().union(*[set(v) for v in leaky.values()]),
            *[set(by_seed[s]) for s in seeds])
        common = sorted(common)
        print(f"  images evaluated under both regimes: {len(common)}")
        if common:
            leaky_acc = float(np.mean([
                np.mean([int(leaky[s][b]["correct"]) for b in common]) for s in leaky]))
            cv_acc = float(np.mean([
                np.mean([int(by_seed[s][b]["correct"]) for b in common]) for s in seeds]))
            print(f"  patient seen in training (released-split models): {100*leaky_acc:.2f}%")
            print(f"  patient absent from training (CV models):         {100*cv_acc:.2f}%")
            err_ratio = (1 - cv_acc) / max(1 - leaky_acc, 1e-12)
            print(f"  error-rate ratio: {err_ratio:.1f}x")
            result["paired"] = {"n_images": len(common), "acc_patient_seen": leaky_acc,
                                "acc_patient_unseen": cv_acc, "error_ratio": err_ratio}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
