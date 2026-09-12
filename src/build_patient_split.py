# build_patient_split.py
"""Build a patient-disjoint version of the dataset.

match_figshare.py recovers a figshare patient ID for 4,586 of the 5,023 tumor
images and shows that the released split puts 100% of matched test images
into patients the model also trains on. No evaluation on that split can
measure generalization to an unseen patient.

This script re-splits from scratch at the patient level. Whole patients go
to one side of the boundary or the other, never both. The released
Training/Testing folders are ignored entirely: an image's original split
says nothing about which patient it belongs to, so it carries no information
worth preserving here.

Handling of the parts without patient IDs:

  * Tumor images that did not match figshare (437) go to TRAIN only. They can
    never contaminate the test set, and excluding them from training as well
    would discard usable data for no benefit.
  * The notumor class has no figshare provenance and therefore no patient
    IDs at all. It is deduplicated (it is 44% redundant against itself) and
    split randomly. This is an acknowledged limitation: the notumor class
    cannot be patient-separated, which is why results are also reported on
    the three tumor classes alone, where the split is fully patient-disjoint.

Usage:
    python src/build_patient_split.py --data_dir ./data --map figshare_patient_map.json \\
        --out_dir ./data_patient
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np

from audit_leakage import pixel_md5
from data_loader import BrainMRIDataset

TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")


def build_fold(args):
    """Materialize one cross-validation fold from a precomputed cv_folds.json.

    The fold assignment is read, never recomputed, so the partition on disk is
    exactly the one recorded in the manifest and is independent of any
    training seed.
    """
    folds = json.load(open(args.folds))
    k = args.fold_idx
    if not 0 <= k < folds["n_folds"]:
        raise SystemExit(f"--fold_idx must be in [0, {folds['n_folds']})")

    pid_to_fold = folds["pid_to_fold"]
    notumor_cluster = folds["notumor_cluster"]
    cluster_to_fold = folds["cluster_to_fold"]
    unmatched = set(folds["unmatched_tumor"])
    mapping = json.load(open(args.map))

    train_ds = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test_ds = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                               class_to_idx=train_ds.class_to_idx)
    idx_to_class = {i: c for c, i in train_ds.class_to_idx.items()}

    all_images = []
    for ds in (train_ds, test_ds):
        for p, lab in zip(ds.image_paths, ds.labels):
            all_images.append((p, os.path.relpath(p).replace("\\", "/"), idx_to_class[lab]))

    def dest(split, cls, path):
        d = os.path.join(args.out_dir, split, cls)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, os.path.basename(path))

    counts = defaultdict(lambda: defaultdict(int))
    train_groups, test_groups = set(), set()
    seen_notumor = set()
    n_skipped_dup = 0

    for path, key, cls in all_images:
        if cls == "notumor":
            cid = notumor_cluster.get(key)
            if cid is None:
                continue
            # collapse exact duplicates: keep one representative per pixel hash
            h = pixel_md5(path)
            if h in seen_notumor:
                n_skipped_dup += 1
                continue
            seen_notumor.add(h)
            split = "Testing" if cluster_to_fold[cid] == k else "Training"
            group = cid
        elif key in unmatched:
            split, group = "Training", None      # never eligible for test
        else:
            pid = mapping[key]["pid"]
            split = "Testing" if pid_to_fold[pid] == k else "Training"
            group = pid

        if group is not None:
            (test_groups if split == "Testing" else train_groups).add(group)
        shutil.copy2(path, dest(split, cls, path))
        counts[split][cls] += 1

    overlap = train_groups & test_groups
    assert not overlap, f"GROUP LEAKAGE in fold {k}: {len(overlap)} groups on both sides"

    # independent verification: no identical pixels across the boundary
    tr_hashes = set()
    for cls_dir in os.listdir(os.path.join(args.out_dir, "Training")):
        d = os.path.join(args.out_dir, "Training", cls_dir)
        for fn in os.listdir(d):
            tr_hashes.add(pixel_md5(os.path.join(d, fn)))
    dup_across = 0
    for cls_dir in os.listdir(os.path.join(args.out_dir, "Testing")):
        d = os.path.join(args.out_dir, "Testing", cls_dir)
        for fn in os.listdir(d):
            if pixel_md5(os.path.join(d, fn)) in tr_hashes:
                dup_across += 1
    assert dup_across == 0, f"{dup_across} pixel-identical images across the fold boundary"

    print(f"Fold {k}: {sum(counts['Training'].values())} train, "
          f"{sum(counts['Testing'].values())} test")
    for split in ("Training", "Testing"):
        print(f"  {split}: {dict(counts[split])}")
    print(f"  groups: {len(train_groups)} train, {len(test_groups)} test, 0 shared")
    print(f"  notumor exact duplicates dropped: {n_skipped_dup}")

    manifest = {
        "mode": "cv_fold", "fold_idx": k, "n_folds": folds["n_folds"],
        "counts": {s: dict(c) for s, c in counts.items()},
        "n_train_groups": len(train_groups), "n_test_groups": len(test_groups),
        "group_overlap": 0, "pixel_duplicates_across_boundary": 0,
        "notumor_exact_duplicates_dropped": n_skipped_dup,
        "folds_file": args.folds,
    }
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {args.manifest}")


def main():
    parser = argparse.ArgumentParser(description="Build a patient-disjoint dataset split.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--map", type=str, default="figshare_patient_map.json")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=str, default="patient_split_manifest.json")
    parser.add_argument("--folds", type=str, default=None,
                        help="cv_folds.json from build_cv_folds.py. With --fold_idx, "
                             "materializes that fold instead of a fresh random partition.")
    parser.add_argument("--fold_idx", type=int, default=None)
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} already exists; delete it or pick another --out_dir")

    if args.folds is not None or args.fold_idx is not None:
        if args.folds is None or args.fold_idx is None:
            raise SystemExit("--folds and --fold_idx must be given together")
        return build_fold(args)

    rng = np.random.default_rng(args.seed)
    mapping = json.load(open(args.map))

    # ---- gather every image in the dataset, regardless of original split ----
    train_ds = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test_ds = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                               class_to_idx=train_ds.class_to_idx)
    idx_to_class = {i: c for c, i in train_ds.class_to_idx.items()}

    all_images = []  # (abspath, relpath_key, class)
    for ds in (train_ds, test_ds):
        for p, lab in zip(ds.image_paths, ds.labels):
            key = os.path.relpath(p).replace("\\", "/")
            all_images.append((p, key, idx_to_class[lab]))
    print(f"Total images: {len(all_images)}")

    # ---- tumor images: group by patient -------------------------------------
    pid_slices = defaultdict(list)      # pid -> [(path, class)]
    pid_class = {}                      # pid -> dominant class
    unmatched_tumor = []
    notumor = []

    for path, key, cls in all_images:
        if cls == "notumor":
            notumor.append(path)
            continue
        m = mapping.get(key)
        if m is None:
            unmatched_tumor.append(path)
        else:
            pid_slices[m["pid"]].append((path, cls))
            pid_class.setdefault(m["pid"], cls)

    print(f"Tumor images with a patient ID: {sum(len(v) for v in pid_slices.values())} "
          f"across {len(pid_slices)} patients")
    print(f"Tumor images without a patient ID: {len(unmatched_tumor)} (all -> train)")
    print(f"notumor images: {len(notumor)} (no patient IDs; deduplicated and split randomly)")

    # ---- assign whole patients to train/test, stratified by class -----------
    by_class = defaultdict(list)
    for pid, cls in pid_class.items():
        by_class[cls].append(pid)

    test_pids = set()
    for cls in TUMOR_CLASSES:
        pids = sorted(by_class.get(cls, []))
        rng.shuffle(pids)
        total = sum(len(pid_slices[p]) for p in pids)
        target = args.test_frac * total
        acc = 0
        for pid in pids:
            if acc >= target:
                break
            test_pids.add(pid)
            acc += len(pid_slices[pid])
        print(f"  {cls:>11s}: {len(pids)} patients, {total} slices -> "
              f"{acc} slices ({100*acc/total:.1f}%) from "
              f"{sum(1 for p in pids if p in test_pids)} patients in test")

    # ---- notumor: deduplicate, then random split ---------------------------
    seen, notumor_unique = set(), []
    for p in notumor:
        h = pixel_md5(p)
        if h not in seen:
            seen.add(h)
            notumor_unique.append(p)
    print(f"notumor after deduplication: {len(notumor_unique)} of {len(notumor)}")
    rng.shuffle(notumor_unique)
    n_test = int(round(args.test_frac * len(notumor_unique)))
    notumor_test = set(notumor_unique[:n_test])

    # ---- materialize --------------------------------------------------------
    def dest(split, cls, path):
        d = os.path.join(args.out_dir, split, cls)
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, os.path.basename(path))

    counts = defaultdict(lambda: defaultdict(int))
    train_pid_check, test_pid_check = set(), set()

    for pid, slices in pid_slices.items():
        split = "Testing" if pid in test_pids else "Training"
        (test_pid_check if pid in test_pids else train_pid_check).add(pid)
        for path, cls in slices:
            shutil.copy2(path, dest(split, cls, path))
            counts[split][cls] += 1

    for path in unmatched_tumor:
        cls = next(c for p, k, c in all_images if p == path)
        shutil.copy2(path, dest("Training", cls, path))
        counts["Training"][cls] += 1

    for path in notumor_unique:
        split = "Testing" if path in notumor_test else "Training"
        shutil.copy2(path, dest(split, "notumor", path))
        counts[split]["notumor"] += 1

    # ---- the guarantee this whole script exists to provide ------------------
    overlap = train_pid_check & test_pid_check
    assert not overlap, f"PATIENT LEAKAGE: {len(overlap)} patients in both splits: {sorted(overlap)[:5]}"
    print(f"\nPatient disjointness verified: {len(train_pid_check)} train patients, "
          f"{len(test_pid_check)} test patients, 0 shared")

    print("\nResulting dataset:")
    for split in ("Training", "Testing"):
        tot = sum(counts[split].values())
        print(f"  {split}: {tot}  {dict(counts[split])}")

    with open(args.manifest, "w") as f:
        json.dump({
            "test_frac": args.test_frac, "seed": args.seed,
            "n_train_patients": len(train_pid_check),
            "n_test_patients": len(test_pid_check),
            "test_pids": sorted(test_pid_check),
            "counts": {k: dict(v) for k, v in counts.items()},
            "n_unmatched_tumor_to_train": len(unmatched_tumor),
            "notumor_unique": len(notumor_unique),
            "notumor_original": len(notumor),
        }, f, indent=2)
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
