# build_random_split.py
"""Matched random-split control for the patient-disjoint cross-validation.

The patient-disjoint estimate (95.19%) and the released-split estimate
(99.22%) differ in two ways at once: the grouping rule, and the evaluation
design (five-fold cross-validation over a pooled test population against a
single 5,712/1,311 holdout). A reviewer is entitled to ask how much of the
gap is the design change rather than the removal of patient overlap.

This script builds the control that isolates it: five-fold cross-validation
with test images assigned at random, ignoring patient identity, while holding
everything else fixed against `build_patient_split.py --folds`:

  * the same image pool (traced tumor images plus deduplicated notumor);
  * the same per-fold, per-class test counts, read from the patient folds;
  * the same treatment of untraced tumor images, which stay in training;
  * the same training pipeline, including the patient-grouped validation
    split, so only the test-assignment rule varies.

If random-split cross-validation lands near the released-split number while
patient-disjoint cross-validation lands near 95%, the gap is attributable to
patient overlap and not to the evaluation design.

The random assignment is computed once, as a genuine five-fold partition of
the eligible pool, and cached in `rand_folds.json`. Every image is therefore
a test image in exactly one fold, exactly as in the patient-disjoint arm, so
pooling over the five folds covers the identical 6,292 images in both arms and
the comparison is paired image by image. Drawing each fold independently would
not have this property: some images would be tested twice and others never,
and the two pooled populations would not be the same set.

Usage:
    python src/build_random_split.py --data_dir ./data --folds cv_folds.json \\
        --fold_idx 0 --out_dir ./data_rand_fold0
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from audit_leakage import pixel_md5
from data_loader import BrainMRIDataset

TUMOR = ("glioma", "meningioma", "pituitary")


def scan_pool(folds, data_dir, mapping, n_folds):
    """Walk the dataset once, mirroring build_patient_split's eligibility rules.

    Returns (eligible, forced_train, target) where `eligible` maps class to the
    list of images that may appear in a test fold, `forced_train` are untraced
    tumor images that never can, and `target[k][cls]` is the per-class test
    count of patient fold k, which the control reproduces exactly.
    """
    pid_to_fold = folds["pid_to_fold"]
    notumor_cluster = folds["notumor_cluster"]
    cluster_to_fold = folds["cluster_to_fold"]
    unmatched = set(folds["unmatched_tumor"])

    train_ds = BrainMRIDataset(os.path.join(data_dir, "Training"))
    test_ds = BrainMRIDataset(os.path.join(data_dir, "Testing"),
                               class_to_idx=train_ds.class_to_idx)
    idx_to_class = {i: c for c, i in train_ds.class_to_idx.items()}

    eligible = defaultdict(list)
    forced_train = []
    target = {k: defaultdict(int) for k in range(n_folds)}
    seen_nt = set()
    for ds in (train_ds, test_ds):
        for p, lab in zip(ds.image_paths, ds.labels):
            key = os.path.relpath(p).replace("\\", "/")
            cls = idx_to_class[lab]
            if cls == "notumor":
                cid = notumor_cluster.get(key)
                if cid is None:
                    continue
                h = pixel_md5(p)
                if h in seen_nt:          # same dedup rule as the patient split
                    continue
                seen_nt.add(h)
                eligible[cls].append(p)
                target[cluster_to_fold[cid]][cls] += 1
            elif key in unmatched:
                forced_train.append((p, cls))
            else:
                eligible[cls].append(p)
                target[pid_to_fold[mapping[key]["pid"]]][cls] += 1
    return eligible, forced_train, {k: dict(v) for k, v in target.items()}


def assign_partition(eligible, target, n_folds, seed):
    """Partition the eligible pool into n_folds test sets at random.

    Two constraints make this more than a shuffle. First, the per-class, per-
    fold sizes must match the patient folds, so that nothing but the grouping
    rule differs between the arms. Second, pixel-duplicate images are assigned
    as a unit: in the patient split duplicated slices cannot straddle the
    train/test boundary because they belong to one patient and patients do not
    cross, so letting them cross here would stack duplicate leakage on top of
    patient leakage and confound the very comparison this control exists to
    make. Groups are therefore shuffled and then placed by largest remaining
    deficit, which is exact whenever the duplicate groups are small relative to
    the fold sizes, as they are here.
    """
    rng = np.random.default_rng(seed)
    assign = {}                      # path -> fold
    actual = {k: defaultdict(int) for k in range(n_folds)}
    for cls, paths in sorted(eligible.items()):
        groups = defaultdict(list)
        for p in sorted(paths):
            groups[pixel_md5(p)].append(p)
        keys = sorted(groups)
        rng.shuffle(keys)
        # Largest groups first, so the awkward ones are placed while every
        # fold still has room; singletons then fill the remainders exactly.
        keys.sort(key=lambda h: -len(groups[h]))
        need = {k: target[k].get(cls, 0) for k in range(n_folds)}
        for h in keys:
            k = max(range(n_folds), key=lambda j: (need[j] - actual[j][cls],
                                                   rng.random()))
            for p in groups[h]:
                assign[p] = k
            actual[k][cls] += len(groups[h])
    return assign, {k: dict(v) for k, v in actual.items()}


def main():
    parser = argparse.ArgumentParser(description="Matched random-split CV control.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--folds", type=str, default="cv_folds.json")
    parser.add_argument("--map", type=str, default="figshare_patient_map.json")
    parser.add_argument("--fold_idx", type=int, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Controls the random test assignment only; independent of "
                             "the training seed so the partition is fixed across runs.")
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--assignment", type=str, default="rand_folds.json",
                        help="Cache for the partition, so every fold is materialized "
                             "from one consistent assignment.")
    parser.add_argument("--n_folds", type=int, default=5)
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} already exists")

    folds = json.load(open(args.folds))
    mapping = json.load(open(args.map))
    k = args.fold_idx

    eligible, forced_train, target = scan_pool(folds, args.data_dir, mapping,
                                               args.n_folds)

    # Compute the partition once and reuse it. Recomputing per fold would give
    # five independent draws rather than a partition: some images would be
    # tested twice and others never, and the pooled population would no longer
    # match the patient arm's.
    if os.path.exists(args.assignment):
        cached = json.load(open(args.assignment))
        if cached.get("seed") != args.seed:
            raise SystemExit(f"{args.assignment} was built with seed "
                             f"{cached.get('seed')}, not {args.seed}")
        assign = {p: int(v) for p, v in cached["assign"].items()}
        actual = {int(j): v for j, v in cached["counts"].items()}
        print(f"loaded partition from {args.assignment}")
    else:
        assign, actual = assign_partition(eligible, target, args.n_folds, args.seed)
        with open(args.assignment, "w") as f:
            json.dump({"seed": args.seed, "n_folds": args.n_folds,
                       "target": {str(j): target[j] for j in target},
                       "counts": {str(j): actual[j] for j in actual},
                       "assign": {p.replace("\\", "/"): v
                                  for p, v in assign.items()}}, f, indent=2)
        print(f"built partition, wrote {args.assignment}")
    assign = {p.replace("\\", "/"): v for p, v in assign.items()}

    for j in range(args.n_folds):
        flag = "" if actual[j] == target[j] else "   <-- MISMATCH"
        print(f"  fold {j}: target {target[j]} actual {actual[j]}{flag}")
    if actual[k] != target[k]:
        raise SystemExit(f"fold {k} composition does not match the patient fold")
    test_set = {p for p in (q for cls in eligible for q in eligible[cls])
                if assign[p.replace("\\", "/")] == k}

    def dest(split, cls, path):
        d = os.path.join(args.out_dir, split, cls)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, os.path.basename(path))

    counts = defaultdict(lambda: defaultdict(int))
    for cls, paths in eligible.items():
        for p in paths:
            split = "Testing" if p in test_set else "Training"
            shutil.copy2(p, dest(split, cls, p))
            counts[split][cls] += 1
    for p, cls in forced_train:
        shutil.copy2(p, dest("Training", cls, p))
        counts["Training"][cls] += 1

    # Sanity: no pixel-identical image across the boundary (the dedup rule
    # already guarantees this, but the patient split asserts it too).
    tr_hashes = set()
    for cls_dir in os.listdir(os.path.join(args.out_dir, "Training")):
        d = os.path.join(args.out_dir, "Training", cls_dir)
        for fn in os.listdir(d):
            tr_hashes.add(pixel_md5(os.path.join(d, fn)))
    dup = 0
    for cls_dir in os.listdir(os.path.join(args.out_dir, "Testing")):
        d = os.path.join(args.out_dir, "Testing", cls_dir)
        for fn in os.listdir(d):
            if pixel_md5(os.path.join(d, fn)) in tr_hashes:
                dup += 1
    assert dup == 0, f"{dup} pixel-identical images across the boundary"

    n_tr = sum(counts["Training"].values())
    n_te = sum(counts["Testing"].values())
    print(f"random fold {k}: {n_tr} train, {n_te} test")
    print(f"  Testing:  {dict(counts['Testing'])}")
    print(f"  duplicates across boundary: 0")

    man = args.manifest or f"rand_fold{k}_manifest.json"
    with open(man, "w") as f:
        json.dump({"mode": "random_cv_control", "fold_idx": k,
                   "assignment_seed": args.seed,
                   "matched_test_counts": target[k],
                   "counts": {s: dict(c) for s, c in counts.items()},
                   "pixel_duplicates_across_boundary": 0}, f, indent=2)
    print(f"Wrote {man}")


if __name__ == "__main__":
    main()
