"""Compare error rates on duplicated vs non-duplicated released test images.

Duplicated images are the released test images whose decoded pixels are identical
to a training image. The comparison is run two ways, because the ten released-split
models score the same fixed images and their predictions are therefore correlated:

  * model-level: a paired two-tailed test over the ten per-seed error-rate pairs
  * image-level: a two-tailed permutation test that shuffles the duplicate label
    across images, holding the pooled predictions fixed

Usage:
    python src/duplicate_error_test.py --runs runs/leaky --n_perm 10000
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats

DUP_REASON = "pixel-identical to a training image"


def duplicated_test_images(manifest_path):
    """Return the set of released test images that duplicate a training image."""
    manifest = json.loads(Path(manifest_path).read_text())
    return {p for p, reason in manifest["removed"].items() if reason == DUP_REASON}


def load_seed_predictions(runs_dir):
    """Return {seed: {path: correct}} for every seed under runs_dir."""
    by_seed = {}
    for pred_file in sorted(Path(runs_dir).glob("seed*/predictions.csv")):
        seed = pred_file.parent.name
        with pred_file.open(newline="") as fh:
            by_seed[seed] = {row["path"]: int(row["correct"]) for row in csv.DictReader(fh)}
    return by_seed


def permutation_test(is_dup, errors, n_perm, seed=0):
    """Two-tailed permutation test on the duplicated-minus-other error-rate gap."""
    rng = np.random.default_rng(seed)
    observed = errors[is_dup].mean() - errors[~is_dup].mean()
    n_dup = int(is_dup.sum())
    idx = np.arange(errors.size)
    count = 0
    for _ in range(n_perm):
        pick = rng.permutation(idx)
        gap = errors[pick[:n_dup]].mean() - errors[pick[n_dup:]].mean()
        if abs(gap) >= abs(observed) - 1e-12:
            count += 1
    # add-one correction keeps the p value strictly positive
    return observed, (count + 1) / (n_perm + 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="runs/leaky")
    parser.add_argument("--manifest", default="dedup_manifest.json")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    dup = duplicated_test_images(args.manifest)
    by_seed = load_seed_predictions(args.runs)
    paths = sorted(next(iter(by_seed.values())))
    is_dup = np.array([p in dup for p in paths])

    # image-level: pool the ten models, so each image carries its mean error rate
    errors = np.array([
        1.0 - np.mean([by_seed[s][p] for s in by_seed]) for p in paths
    ])
    observed, p_perm = permutation_test(is_dup, errors, args.n_perm, args.seed)

    # model-level: one paired observation per seed
    dup_rates, other_rates = [], []
    for s in by_seed:
        correct = np.array([by_seed[s][p] for p in paths])
        dup_rates.append(1.0 - correct[is_dup].mean())
        other_rates.append(1.0 - correct[~is_dup].mean())
    paired = stats.ttest_rel(dup_rates, other_rates)
    wilcoxon = stats.wilcoxon(dup_rates, other_rates)

    print(f"seeds: {len(by_seed)}  images: {len(paths)}  duplicated: {int(is_dup.sum())}")
    print(f"pooled error rate, duplicated:     {errors[is_dup].mean() * 100:.2f}%")
    print(f"pooled error rate, non-duplicated: {errors[~is_dup].mean() * 100:.2f}%")
    print(f"observed gap: {observed * 100:.2f} pt")
    print(f"image-level permutation ({args.n_perm} resamples): p = {p_perm:.3f}")
    print(f"model-level paired t-test: t = {paired.statistic:.3f}, p = {paired.pvalue:.3f}")
    print(f"model-level Wilcoxon: W = {wilcoxon.statistic:.1f}, p = {wilcoxon.pvalue:.3f}")


if __name__ == "__main__":
    main()
