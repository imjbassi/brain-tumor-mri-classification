# compare_random_control.py
"""Compare the patient-disjoint arm against the matched random-split control.

The two arms differ in exactly one respect. Both are five-fold cross-validation
over the same 6,292 eligible images, with the same per-fold per-class test
counts, the same treatment of untraced images, the same pipeline and the same
three seeds. In one, folds are formed by patient; in the other, test images are
assigned at random and a patient's slices may be separated. Because each arm is
a partition, every image is a test image exactly once per seed in both, so the
comparison is paired image by image.

That makes the interesting quantity a decomposition. The released split reports
99.22%; the patient-disjoint estimate is 95.19%. Part of that gap is the
grouping rule and part is the change in evaluation design, and the control
separates them: random CV holds the design fixed and varies only the grouping.

Uncertainty on the paired difference is computed by resampling patients rather
than images, since slices of one patient are not independent and the difference
is itself a within-patient quantity.

Usage:
    python src/compare_random_control.py --out paper/figures/random_control.json
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_arm(runs_dir, prefix, n_folds):
    """{seed: {basename: (correct, true, group)}} pooled over the folds."""
    by_seed = defaultdict(dict)
    for k in range(n_folds):
        pat = os.path.join(runs_dir, f"{prefix}{k}", "seed*", "predictions.csv")
        for pf in sorted(glob.glob(pat)):
            seed = os.path.basename(os.path.dirname(pf)).replace("seed", "")
            with open(pf, encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    base = os.path.basename(r["path"])
                    if base in by_seed[seed]:
                        raise SystemExit(
                            f"{prefix}: {base} is a test image in more than one fold "
                            f"for seed {seed}; the folds are not a partition")
                    by_seed[seed][base] = (int(r["correct"]), r["true"], r["group"])
    return by_seed


def paired_bootstrap(images, pat, rand, groups, n_boot=10000, seed=0):
    """Patient-clustered bootstrap on the paired difference (random - patient)."""
    by_group = defaultdict(list)
    for b in images:
        by_group[groups[b]].append(b)
    keys = list(by_group)
    # Per-patient totals, averaged over seeds, so one draw is one patient.
    p_hit = {g: np.mean([np.mean([pat[s][b][0] for b in by_group[g]])
                         for s in pat]) for g in keys}
    r_hit = {g: np.mean([np.mean([rand[s][b][0] for b in by_group[g]])
                         for s in rand]) for g in keys}
    n_img = {g: len(by_group[g]) for g in keys}
    P = np.array([p_hit[g] * n_img[g] for g in keys])
    R = np.array([r_hit[g] * n_img[g] for g in keys])
    N = np.array([n_img[g] for g in keys], dtype=float)
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        tot = N[idx].sum()
        out[i] = (R[idx].sum() - P[idx].sum()) / tot
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)), out


def acc(arm, images):
    return float(np.mean([np.mean([arm[s][b][0] for b in images]) for s in arm]))


def main():
    ap = argparse.ArgumentParser(description="Patient-disjoint vs random-split control.")
    ap.add_argument("--runs_dir", type=str, default="runs")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--released", type=float, default=0.9922,
                    help="Released-split accuracy, for the gap decomposition.")
    ap.add_argument("--out", type=str, default="paper/figures/random_control.json")
    args = ap.parse_args()

    pat = load_arm(args.runs_dir, "cv_fold", args.n_folds)
    rand = load_arm(args.runs_dir, "rand_fold", args.n_folds)
    if not pat or not rand:
        raise SystemExit("missing predictions for one of the two arms")
    seeds = sorted(set(pat) & set(rand), key=int)
    pat = {s: pat[s] for s in seeds}
    rand = {s: rand[s] for s in seeds}

    # The arms must cover the same images, or the comparison is not paired.
    sets = [set(pat[s]) for s in seeds] + [set(rand[s]) for s in seeds]
    images = sorted(set.intersection(*sets))
    union = set.union(*sets)
    if len(images) != len(union):
        raise SystemExit(f"arms cover different images: {len(images)} common, "
                         f"{len(union)} in union")
    print(f"seeds {seeds}; paired on {len(images)} images covered by both arms")

    groups = {b: pat[seeds[0]][b][2] or b for b in images}
    truth = {b: pat[seeds[0]][b][1] for b in images}
    classes = sorted(set(truth.values()))
    tumor = [b for b in images if truth[b] != "notumor"]

    res = {"seeds": seeds, "n_images": len(images), "n_tumor": len(tumor),
           "n_patients_or_clusters": len(set(groups.values()))}

    for name, subset in (("all", images), ("tumor_only", tumor)):
        a_p, a_r = acc(pat, subset), acc(rand, subset)
        lo, hi, draws = paired_bootstrap(subset, pat, rand, groups,
                                         args.n_boot, seed=0)
        err_ratio = (1 - a_p) / max(1 - a_r, 1e-12)
        # One-sided bootstrap evidence that the grouping rule costs accuracy.
        p_boot = float(np.mean(draws <= 0))
        print(f"\n--- {name} ({len(subset)} images) ---")
        print(f"  random-split CV    : {100*a_r:.2f}%")
        print(f"  patient-disjoint CV: {100*a_p:.2f}%")
        print(f"  paired difference  : {100*(a_r-a_p):.2f} pts "
              f"[95% CI {100*lo:.2f}, {100*hi:.2f}] (patient-clustered)")
        print(f"  error-rate ratio   : {err_ratio:.2f}x   "
              f"bootstrap P(diff<=0) = {p_boot:.4f}")
        res[name] = {"acc_random": a_r, "acc_patient": a_p,
                     "difference": a_r - a_p, "ci95": [lo, hi],
                     "error_ratio": err_ratio, "p_bootstrap_one_sided": p_boot,
                     "n": len(subset)}

    # ---- gap decomposition -------------------------------------------------
    a_p, a_r = res["all"]["acc_patient"], res["all"]["acc_random"]
    total = args.released - a_p
    design = args.released - a_r
    grouping = a_r - a_p
    print(f"\n--- decomposition of the {100*total:.2f} pt gap from the released split ---")
    print(f"  evaluation design (released -> random CV): {100*design:.2f} pts "
          f"({100*design/total:.0f}%)")
    print(f"  grouping rule (random CV -> patient CV)  : {100*grouping:.2f} pts "
          f"({100*grouping/total:.0f}%)")
    res["decomposition"] = {"released": args.released, "total_gap": total,
                            "design_component": design,
                            "grouping_component": grouping,
                            "grouping_share": grouping / total}

    # ---- per-class ---------------------------------------------------------
    print("\n--- per-class accuracy ---")
    res["per_class"] = {}
    for c in classes:
        sub = [b for b in images if truth[b] == c]
        a_pc, a_rc = acc(pat, sub), acc(rand, sub)
        print(f"  {c:11s} random {100*a_rc:6.2f}%  patient {100*a_pc:6.2f}%  "
              f"delta {100*(a_rc-a_pc):+.2f}")
        res["per_class"][c] = {"n": len(sub), "acc_random": a_rc,
                               "acc_patient": a_pc, "difference": a_rc - a_pc}

    # ---- fold spread, the second finding -----------------------------------
    print("\n--- fold-to-fold spread ---")
    res["fold_spread"] = {}
    for tag, prefix in (("patient", "cv_fold"), ("random", "rand_fold")):
        means = []
        for k in range(args.n_folds):
            rj = os.path.join(args.runs_dir, f"{prefix}{k}", "results.json")
            if not os.path.exists(rj):
                continue
            d = json.load(open(rj))
            means.append(float(np.mean([v["test_accuracy"] for v in d.values()])))
        if means:
            print(f"  {tag:8s} fold means {[round(100*m,2) for m in means]}  "
                  f"range {100*(max(means)-min(means)):.2f} pts")
            res["fold_spread"][tag] = {"fold_means": means,
                                       "range": max(means) - min(means)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
