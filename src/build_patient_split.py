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


def main():
    parser = argparse.ArgumentParser(description="Build a patient-disjoint dataset split.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--map", type=str, default="figshare_patient_map.json")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=str, default="patient_split_manifest.json")
    args = parser.parse_args()

    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} already exists; delete it or pick another --out_dir")

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
