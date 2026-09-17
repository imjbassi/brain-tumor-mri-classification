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
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

DUP_REASON = "pixel-identical to a training image"


def pixel_md5(path, size=(128, 128)):
    """Hash of the decoded grayscale pixels, matching src/audit_leakage.py."""
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return hashlib.md5(np.asarray(img).tobytes()).hexdigest()


def duplicated_by_hash(data_dir):
    """Released test images whose decoded pixels match some training image.

    This is the definition used for the 216 exact pixel duplicates reported in
    the audit, and it is not recoverable from the dedup manifest alone: images
    removed for another reason first (for example, duplicates within the test
    folder) carry that other reason there.
    """
    data_dir = Path(data_dir)
    train_hashes = {
        pixel_md5(p) for p in sorted((data_dir / "Training").rglob("*"))
        if p.is_file()
    }
    dup = set()
    for p in sorted((data_dir / "Testing").rglob("*")):
        if p.is_file() and pixel_md5(p) in train_hashes:
            dup.add("data/Testing/" + p.parent.name + "/" + p.name)
    return dup


def duplicated_from_manifest(manifest_path):
    """Test images the dedup manifest labels as pixel-identical to training."""
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
    parser.add_argument("--data_dir", default="data",
                        help="released folders; duplicates are recomputed by pixel hash")
    parser.add_argument("--manifest", default="dedup_manifest.json",
                        help="fallback when the images are not available locally")
    parser.add_argument("--n_perm", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if Path(args.data_dir).is_dir():
        dup = duplicated_by_hash(args.data_dir)
        source = f"pixel hashes under {args.data_dir}"
    else:
        dup = duplicated_from_manifest(args.manifest)
        source = f"{args.manifest} (images unavailable; undercounts)"
    print(f"duplicate definition: {source}")
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
