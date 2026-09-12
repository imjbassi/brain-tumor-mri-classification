# build_cv_folds.py
"""Compute the patient-level cross-validation fold assignment, once.

Splitting decisions are made here and written to `cv_folds.json`; nothing
downstream re-derives them. That matters for a paper about split integrity:
the partition must be auditable, reproducible, and independent of any
training seed, so that "which patients were in fold 3" is a fact on disk
rather than a function of whatever RNG state a training script happened to
have.

Grouping units:

  * Tumor images -> figshare patient ID, recovered by match_figshare.py.
    Two patients contribute slices to more than one class; each is assigned
    to its dominant class and the exception is recorded in the manifest.
  * notumor -> near-duplicate cluster. This class has no patient IDs, and
    deduplicating it is not enough: count_unique.py shows 2,000 files contain
    1,706 unique images but only ~1,113 near-duplicate clusters, so a
    deduplicated-but-randomly-split notumor class still puts near-identical
    scans on both sides. Clusters are the closest available proxy for subject
    identity here. It is a lower bound on correct grouping, not a guarantee.
  * Tumor images with no recovered patient ID are never placed in a test
    fold; they go to training in every fold.

Fold balancing uses greedy longest-processing-time within each class:
patients range from 2 to 46 slices, so assigning them at random produces
badly unequal folds. Sorting by slice count descending and always placing
the next patient into the currently-smallest fold gives near-equal slice
counts and near-equal patient counts, and stratifies by class for free.

Usage:
    python src/build_cv_folds.py --data_dir ./data --map figshare_patient_map.json
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from audit_leakage import embed_images, pixel_md5
from count_unique import cluster_greedy
from data_loader import BrainMRIDataset

TUMOR_CLASSES = ("glioma", "meningioma", "pituitary")


def greedy_lpt(items, n_folds):
    """Assign items (id, weight) to folds, largest first, always into the
    currently-lightest fold. Deterministic given the input order."""
    loads = [0] * n_folds
    counts = [0] * n_folds
    assignment = {}
    for item_id, weight in sorted(items, key=lambda t: (-t[1], str(t[0]))):
        f = min(range(n_folds), key=lambda i: (loads[i], counts[i], i))
        assignment[item_id] = f
        loads[f] += weight
        counts[f] += 1
    return assignment, loads, counts


def main():
    parser = argparse.ArgumentParser(description="Compute patient-level CV folds.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--map", type=str, default="figshare_patient_map.json")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--cos_threshold", type=float, default=0.99,
                        help="Near-duplicate threshold for notumor clustering.")
    parser.add_argument("--out", type=str, default="cv_folds.json")
    parser.add_argument("--group_map_out", type=str, default="group_map.json")
    args = parser.parse_args()

    mapping = json.load(open(args.map))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test_ds = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                               class_to_idx=train_ds.class_to_idx)
    idx_to_class = {i: c for c, i in train_ds.class_to_idx.items()}

    all_images = []
    for ds in (train_ds, test_ds):
        for p, lab in zip(ds.image_paths, ds.labels):
            key = os.path.relpath(p).replace("\\", "/")
            all_images.append((p, key, idx_to_class[lab]))
    print(f"Total images: {len(all_images)}")

    # ---- tumor: group by patient -------------------------------------------
    pid_slices = defaultdict(list)
    pid_class_counts = defaultdict(lambda: defaultdict(int))
    unmatched_tumor = []
    notumor_paths = []

    for path, key, cls in all_images:
        if cls == "notumor":
            notumor_paths.append((path, key))
            continue
        m = mapping.get(key)
        if m is None:
            unmatched_tumor.append(key)
        else:
            pid_slices[m["pid"]].append(key)
            pid_class_counts[m["pid"]][cls] += 1

    pid_class, multiclass_pids = {}, []
    for pid, counts in pid_class_counts.items():
        if len(counts) > 1:
            multiclass_pids.append({"pid": pid, "classes": dict(counts)})
        pid_class[pid] = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    print(f"Patients: {len(pid_slices)}; tumor slices with a PID: "
          f"{sum(len(v) for v in pid_slices.values())}")
    print(f"Tumor images without a PID: {len(unmatched_tumor)} (train-only in every fold)")
    if multiclass_pids:
        print(f"Patients spanning >1 class (assigned by dominant class): {len(multiclass_pids)}")

    # ---- assign patients to folds, balanced within class --------------------
    pid_to_fold = {}
    per_class_report = {}
    for cls in TUMOR_CLASSES:
        pids = [(pid, len(pid_slices[pid])) for pid in pid_slices if pid_class[pid] == cls]
        assign, loads, counts = greedy_lpt(pids, args.n_folds)
        pid_to_fold.update(assign)
        per_class_report[cls] = {"n_patients": len(pids),
                                 "slices_per_fold": loads,
                                 "patients_per_fold": counts}
        print(f"  {cls:>11s}: {len(pids)} patients -> slices/fold {loads}, patients/fold {counts}")

    # ---- notumor: cluster near-duplicates, then assign whole clusters -------
    print(f"\nClustering {len(notumor_paths)} notumor images "
          f"(cosine >= {args.cos_threshold})...")
    # collapse exact duplicates first so the embedding pass is smaller
    seen, unique_notumor = {}, []
    for path, key in notumor_paths:
        h = pixel_md5(path)
        if h not in seen:
            seen[h] = key
            unique_notumor.append((path, key))
    print(f"  {len(unique_notumor)} pixel-unique of {len(notumor_paths)}")

    emb = embed_images([p for p, _ in unique_notumor], device)
    n_clusters, assign_idx = cluster_greedy(emb, args.cos_threshold, device)
    print(f"  {n_clusters} near-duplicate clusters")

    notumor_cluster = {}                       # image key -> cluster id
    cluster_size = defaultdict(int)
    for (path, key), c in zip(unique_notumor, assign_idx):
        cid = f"nt{int(c)}"
        notumor_cluster[key] = cid
        cluster_size[cid] += 1
    # exact duplicates inherit their representative's cluster
    hash_to_cluster = {h: notumor_cluster[k] for h, k in seen.items()}
    for path, key in notumor_paths:
        if key not in notumor_cluster:
            notumor_cluster[key] = hash_to_cluster[pixel_md5(path)]

    cluster_to_fold, nt_loads, nt_counts = greedy_lpt(
        list(cluster_size.items()), args.n_folds)
    print(f"  notumor unique images/fold {nt_loads}, clusters/fold {nt_counts}")

    # ---- group map for the grouped train/validation split -------------------
    group_map = {}
    for path, key, cls in all_images:
        if cls == "notumor":
            group_map[key] = notumor_cluster[key]
        else:
            m = mapping.get(key)
            group_map[key] = m["pid"] if m else f"unmatched::{key}"

    with open(args.group_map_out, "w") as f:
        json.dump(group_map, f, indent=2)
    print(f"\nWrote {args.group_map_out} ({len(group_map)} images)")

    # ---- checks -------------------------------------------------------------
    assert set(pid_to_fold) == set(pid_slices), "every patient must get a fold"
    assert all(0 <= f < args.n_folds for f in pid_to_fold.values())
    fold_pid_sets = [ {p for p, f in pid_to_fold.items() if f == k}
                      for k in range(args.n_folds) ]
    for a in range(args.n_folds):
        for b in range(a + 1, args.n_folds):
            assert not (fold_pid_sets[a] & fold_pid_sets[b]), "patient in two folds"
    assert sum(len(s) for s in fold_pid_sets) == len(pid_slices), \
        "every patient must appear in exactly one fold"

    out = {
        "n_folds": args.n_folds,
        "cos_threshold": args.cos_threshold,
        "pid_to_fold": pid_to_fold,
        "pid_class": pid_class,
        "pid_slice_counts": {p: len(v) for p, v in pid_slices.items()},
        "notumor_cluster": notumor_cluster,
        "cluster_to_fold": cluster_to_fold,
        "unmatched_tumor": unmatched_tumor,
        "per_class_fold_balance": per_class_report,
        "notumor_fold_balance": {"images_per_fold": nt_loads, "clusters_per_fold": nt_counts},
        "multiclass_patients": multiclass_pids,
        "n_patients": len(pid_slices),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.out}: {len(pid_slices)} patients over {args.n_folds} folds, "
          f"each tested exactly once")


if __name__ == "__main__":
    main()
