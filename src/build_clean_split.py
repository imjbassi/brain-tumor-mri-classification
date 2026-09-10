# build_clean_split.py
"""Build a duplicate-disjoint copy of the dataset.

audit_leakage.py shows that a substantial fraction of the test set is
duplicated in the training set. This script writes a new dataset directory
with those duplicates removed, keeping the folder layout identical so every
other script in this repo runs against it unchanged:

    python src/build_clean_split.py --data_dir ./data --out_dir ./data_dedup --mode exact
    python src/train.py --data_dir ./data_dedup --augment ...

Removal policy:
  * A test image that duplicates any training image is dropped from TEST.
    (Dropping from test, not train, keeps the training data intact so the
    comparison isolates the effect of a clean test set.)
  * Duplicate groups WITHIN training collapse to a single representative,
    so the train/validation split can't be contaminated either.
  * Duplicate groups WITHIN test also collapse to one representative, so no
    single scan is counted twice in the metrics.

Modes:
  exact  - pixel-identical only (decoded-pixel hash). Conservative.
  near   - pixel-identical, plus cosine similarity >= --cos_threshold against
           an ImageNet-pretrained ResNet-18 embedding. Aggressive.

Writes dedup_manifest.json recording every removed file and the reason.
"""
import argparse
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import torch

from audit_leakage import embed_images, pixel_md5
from data_loader import BrainMRIDataset


def group_by_hash(paths):
    groups = defaultdict(list)
    for p in paths:
        groups[pixel_md5(p)].append(p)
    return groups


def main():
    parser = argparse.ArgumentParser(description="Build a duplicate-disjoint dataset copy.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["exact", "near"], default="exact")
    parser.add_argument("--cos_threshold", type=float, default=0.99,
                        help="Only used in --mode near.")
    parser.add_argument("--manifest", type=str, default="dedup_manifest.json")
    args = parser.parse_args()

    train = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                            class_to_idx=train.class_to_idx)
    idx_to_class = {i: c for c, i in train.class_to_idx.items()}
    print(f"Input: {len(train)} train, {len(test)} test")

    removed = {}          # path -> reason
    print("Hashing pixels...")
    train_groups = group_by_hash(train.image_paths)
    test_groups = group_by_hash(test.image_paths)

    # 1. collapse duplicate groups within each split
    for groups, split in [(train_groups, "train"), (test_groups, "test")]:
        for h, members in groups.items():
            for extra in sorted(members)[1:]:
                removed[extra] = f"duplicate within {split}"

    # 2. drop test images duplicating a training image
    for h, members in test_groups.items():
        if h in train_groups:
            for p in members:
                removed.setdefault(p, "pixel-identical to a training image")

    n_exact = sum(1 for r in removed.values() if "training image" in r)
    print(f"  exact cross-split test removals: {n_exact}")
    print(f"  within-split duplicate removals: {len(removed) - n_exact}")

    # 3. optional near-duplicate pass
    if args.mode == "near":
        print(f"Embedding for near-duplicate pass (threshold {args.cos_threshold})...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        keep_train = [p for p in train.image_paths if p not in removed]
        keep_test = [p for p in test.image_paths if p not in removed]
        tr_emb = embed_images(keep_train, device).to(device)
        te_emb = embed_images(keep_test, device).to(device)
        sims = (te_emb @ tr_emb.T).max(dim=1).values.cpu().numpy()
        n_near = 0
        for p, s in zip(keep_test, sims):
            if s >= args.cos_threshold:
                removed[p] = f"cosine {s:.4f} >= {args.cos_threshold} to a training image"
                n_near += 1
        print(f"  near-duplicate test removals: {n_near}")

    # 4. write the deduplicated copy
    if os.path.exists(args.out_dir):
        raise SystemExit(f"{args.out_dir} already exists; delete it or pick another --out_dir")
    print(f"Copying survivors to {args.out_dir}...")
    kept = defaultdict(lambda: defaultdict(int))
    for split, ds in [("Training", train), ("Testing", test)]:
        for p, lab in zip(ds.image_paths, ds.labels):
            if p in removed:
                continue
            cls = idx_to_class[lab]
            dst_dir = os.path.join(args.out_dir, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(p, os.path.join(dst_dir, os.path.basename(p)))
            kept[split][cls] += 1

    print("\nResulting dataset:")
    for split in ("Training", "Testing"):
        total = sum(kept[split].values())
        print(f"  {split}: {total}  {dict(kept[split])}")

    dropped_test = sum(1 for p in removed if os.sep + "Testing" + os.sep in p)
    dropped_train = len(removed) - dropped_test
    print(f"\nRemoved {len(removed)} images total "
          f"({dropped_test} from test, {dropped_train} from train)")

    with open(args.manifest, "w") as f:
        json.dump({
            "mode": args.mode,
            "cos_threshold": args.cos_threshold if args.mode == "near" else None,
            "n_removed": len(removed),
            "n_removed_test": dropped_test,
            "n_removed_train": dropped_train,
            "kept": {k: dict(v) for k, v in kept.items()},
            "removed": {os.path.relpath(p).replace("\\", "/"): r for p, r in sorted(removed.items())},
        }, f, indent=2)
    print(f"Wrote {args.manifest}")


if __name__ == "__main__":
    main()
