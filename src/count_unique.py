# count_unique.py
"""How many distinct images does the dataset actually contain?

The released collection advertises a certain number of images per class, but
Section 3.2 of the paper shows many of those files are duplicates of each
other. This script reports, per class and overall:

  * file count            (what the folder listing claims)
  * unique pixel content  (distinct decoded-pixel hashes)
  * near-duplicate clusters at a cosine threshold (an effective sample size)

The gap between the first and the last is the number of images that carry no
information the dataset does not already contain elsewhere.

Usage:
    python src/count_unique.py --data_dir ./data
"""
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from audit_leakage import embed_images, pixel_md5
from data_loader import BrainMRIDataset


def cluster_greedy(emb, threshold, device, block=512):
    """Single-link-ish greedy clustering: walk images in order, assign each to
    an existing cluster if it exceeds `threshold` against that cluster's
    representative, else start a new cluster. Returns the number of clusters.
    Deliberately conservative: it never merges two representatives, so it
    over-counts clusters rather than under-counting them."""
    emb = emb.to(device)
    reps = []           # indices of cluster representatives
    assign = np.full(len(emb), -1)
    for i in range(len(emb)):
        if reps:
            sims = emb[reps] @ emb[i]
            j = int(torch.argmax(sims))
            if float(sims[j]) >= threshold:
                assign[i] = j
                continue
        assign[i] = len(reps)
        reps.append(i)
    return len(reps), assign


def main():
    parser = argparse.ArgumentParser(description="Count unique images per class.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--cos_threshold", type=float, default=0.99)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                            class_to_idx=train.class_to_idx)
    idx_to_class = {i: c for c, i in train.class_to_idx.items()}

    # pool both splits: the question is how much distinct data exists overall
    paths, labels = [], []
    for ds in (train, test):
        paths.extend(ds.image_paths)
        labels.extend(ds.labels)
    labels = np.array(labels)

    print(f"Total files: {len(paths)}")
    print("Hashing pixels...")
    hashes = [pixel_md5(p) for p in paths]

    print("Embedding (for near-duplicate clustering)...")
    emb = embed_images(paths, device)

    print(f"\n{'class':>11s}  {'files':>6s}  {'unique':>7s}  {'clusters':>9s}  {'redundant':>10s}")
    rows = {}
    for ci, cname in idx_to_class.items():
        m = labels == ci
        idx = np.where(m)[0]
        uniq = len({hashes[i] for i in idx})
        nclus, _ = cluster_greedy(emb[idx], args.cos_threshold, device)
        rows[cname] = (int(m.sum()), uniq, nclus)
        print(f"{cname:>11s}  {int(m.sum()):6d}  {uniq:7d}  {nclus:9d}  "
              f"{int(m.sum()) - nclus:10d} ({100*(1-nclus/m.sum()):.1f}%)")

    tot_files = len(paths)
    tot_uniq = len(set(hashes))
    nclus_all, _ = cluster_greedy(emb, args.cos_threshold, device)
    print(f"{'TOTAL':>11s}  {tot_files:6d}  {tot_uniq:7d}  {nclus_all:9d}  "
          f"{tot_files - nclus_all:10d} ({100*(1-nclus_all/tot_files):.1f}%)")

    print(f"\nThe dataset ships {tot_files} files containing {tot_uniq} distinct images "
          f"({tot_files - tot_uniq} exact duplicates).")
    print(f"At cosine >= {args.cos_threshold} it contains roughly {nclus_all} "
          f"distinct scans.")


if __name__ == "__main__":
    main()
