# audit_leakage.py
"""Audit the train/test split for near-duplicate leakage.

This dataset is an aggregation of several public sources (figshare/Cheng et
al., SARTAJ, Br35H). The figshare component contains multiple 2D slices per
patient. If the aggregated train/test split was made by shuffling slices
rather than by patient, slices from the same patient can land on both sides
of the boundary, and the held-out test set stops measuring generalization to
new patients.

This script checks for that directly, at three levels of strictness:

  1. Exact file duplicates      (MD5 of the raw file bytes)
  2. Exact pixel duplicates     (MD5 of the decoded, resized pixel array)
  3. Near-duplicates            (perceptual hash Hamming distance, and
                                 cosine similarity of ImageNet features)

Embeddings come from an ImageNet-pretrained ResNet-18 with its classifier
head removed. It is deliberately NOT the fine-tuned checkpoint from this
repo: using a model trained on this split to audit this split would be
circular.

Nothing here assumes leakage exists. The distributions and the example grid
are the evidence; read them before concluding anything.

Usage:
    python src/audit_leakage.py --data_dir ./data
"""
import argparse
import hashlib
import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from scipy.fftpack import dct

from data_loader import BrainMRIDataset, build_transforms
from utils import plot_image_grid


def file_md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def pixel_md5(path, size=(128, 128)):
    """Hash of the decoded grayscale pixels, so that the same image saved
    twice with different JPEG settings still collides."""
    img = Image.open(path).convert("L").resize(size, Image.BILINEAR)
    return hashlib.md5(np.asarray(img).tobytes()).hexdigest()


def phash(path, hash_size=8, highfreq_factor=4):
    """DCT-based perceptual hash, returned as a 64-bit integer."""
    img_size = hash_size * highfreq_factor
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=float)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size].flatten()
    # Drop the DC term before taking the median, it dominates overall brightness.
    med = np.median(low[1:])
    bits = low > med
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hamming(a, b):
    return bin(a ^ b).count("1")


def load_split(data_dir, split, class_to_idx=None):
    ds = BrainMRIDataset(os.path.join(data_dir, split), class_to_idx=class_to_idx)
    return ds


def embed_images(paths, device, batch_size=64):
    """ImageNet-pretrained ResNet-18 penultimate features, L2-normalized."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net = net.to(device).eval()

    _, tf = build_transforms(augment=False)
    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            batch = [tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]
            x = torch.stack(batch).to(device)
            f = net(x)
            feats.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(feats)


def main():
    parser = argparse.ArgumentParser(description="Audit train/test split for duplicate leakage.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--fig_dir", type=str, default="paper/figures")
    parser.add_argument("--cos_threshold", type=float, default=0.95,
                        help="Cosine similarity at or above which a pair is called a near-duplicate.")
    parser.add_argument("--phash_threshold", type=int, default=8,
                        help="pHash Hamming distance at or below which a pair is called a near-duplicate.")
    parser.add_argument("--n_examples", type=int, default=8,
                        help="How many top cross-split pairs to render as visual evidence.")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = load_split(args.data_dir, "Training")
    test_ds = load_split(args.data_dir, "Testing", class_to_idx=train_ds.class_to_idx)
    class_names = train_ds.classes
    idx_to_class = {i: c for c, i in train_ds.class_to_idx.items()}

    print(f"Train: {len(train_ds)} images, Test: {len(test_ds)} images")
    print(f"Classes: {class_names}\n")

    # --- Level 1 and 2: exact duplicates -------------------------------------
    print("Hashing files (exact-duplicate check)...")
    train_fmd5 = [file_md5(p) for p in train_ds.image_paths]
    test_fmd5 = [file_md5(p) for p in test_ds.image_paths]
    train_pmd5 = [pixel_md5(p) for p in train_ds.image_paths]
    test_pmd5 = [pixel_md5(p) for p in test_ds.image_paths]

    train_fset = {}
    for h, p in zip(train_fmd5, train_ds.image_paths):
        train_fset.setdefault(h, []).append(p)
    train_pset = {}
    for h, p in zip(train_pmd5, train_ds.image_paths):
        train_pset.setdefault(h, []).append(p)

    exact_file = [i for i, h in enumerate(test_fmd5) if h in train_fset]
    exact_pixel = [i for i, h in enumerate(test_pmd5) if h in train_pset]

    print(f"  Exact file duplicates across train/test:  {len(exact_file)} / {len(test_ds)}")
    print(f"  Exact pixel duplicates across train/test: {len(exact_pixel)} / {len(test_ds)}")

    # Duplicates inside the training split matter too: they inflate the
    # train/validation split used for model selection.
    dup_in_train = sum(len(v) - 1 for v in train_pset.values() if len(v) > 1)
    print(f"  Exact pixel duplicates within training split: {dup_in_train}\n")

    # --- Level 3: near-duplicates --------------------------------------------
    print("Computing perceptual hashes...")
    train_ph = np.array([phash(p) for p in train_ds.image_paths], dtype=object)
    test_ph = np.array([phash(p) for p in test_ds.image_paths], dtype=object)

    print("Computing ImageNet embeddings (this is the slow part)...")
    train_emb = embed_images(train_ds.image_paths, device)
    test_emb = embed_images(test_ds.image_paths, device)

    print("Matching test images against training images...\n")
    sims = test_emb.to(device) @ train_emb.to(device).T  # (n_test, n_train)
    best_sim, best_idx = sims.max(dim=1)
    best_sim = best_sim.cpu().numpy()
    best_idx = best_idx.cpu().numpy()

    # pHash distance to the embedding-nearest training image.
    phash_dist = np.array([hamming(test_ph[i], train_ph[best_idx[i]]) for i in range(len(test_ph))])

    test_labels = np.array(test_ds.labels)
    results = {}
    print(f"{'class':>12s}  {'n':>5s}  {'mean sim':>9s}  {'>=thr':>7s}  {'pHash<=thr':>11s}  {'both':>6s}")
    for ci, cname in enumerate(class_names):
        mask = test_labels == ci
        s = best_sim[mask]
        d = phash_dist[mask]
        near_cos = int((s >= args.cos_threshold).sum())
        near_ph = int((d <= args.phash_threshold).sum())
        both = int(((s >= args.cos_threshold) & (d <= args.phash_threshold)).sum())
        results[cname] = dict(n=int(mask.sum()), mean_sim=float(s.mean()),
                               near_cos=near_cos, near_phash=near_ph, both=both)
        print(f"{cname:>12s}  {int(mask.sum()):5d}  {s.mean():9.4f}  "
              f"{near_cos:4d} ({100*near_cos/mask.sum():4.1f}%)  "
              f"{near_ph:4d} ({100*near_ph/mask.sum():4.1f}%)  {both:5d}")

    overall_cos = int((best_sim >= args.cos_threshold).sum())
    print(f"\nOverall: {overall_cos} / {len(test_ds)} test images "
          f"({100*overall_cos/len(test_ds):.1f}%) have a training image at "
          f"cosine >= {args.cos_threshold}")

    # --- Evidence: distributions and example pairs ---------------------------
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, cname in enumerate(class_names):
        ax.hist(best_sim[test_labels == ci], bins=50, range=(0.4, 1.0),
                 alpha=0.55, label=cname)
    ax.axvline(args.cos_threshold, color="k", linestyle="--", linewidth=1,
                label=f"threshold {args.cos_threshold}")
    ax.set_xlabel("Cosine similarity to nearest training image")
    ax.set_ylabel("Number of test images")
    ax.set_title("Test-to-train nearest-neighbour similarity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.fig_dir, "leakage_similarity_hist.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out}")

    order = np.argsort(-best_sim)[:args.n_examples]
    images, titles = [], []
    for i in order:
        images.append(Image.open(test_ds.image_paths[i]).convert("RGB").resize((224, 224)))
        images.append(Image.open(train_ds.image_paths[best_idx[i]]).convert("RGB").resize((224, 224)))
        titles.append(f"TEST {idx_to_class[test_labels[i]]}\ncos={best_sim[i]:.4f}")
        titles.append(f"TRAIN {idx_to_class[train_ds.labels[best_idx[i]]]}\npHash d={phash_dist[i]}")
    out = os.path.join(args.fig_dir, "leakage_examples.png")
    plot_image_grid(images, titles, out, ncols=4,
                     suptitle="Most similar test/train pairs (each test image beside its nearest training image)")

    import json
    summary = {
        "n_train": len(train_ds), "n_test": len(test_ds),
        "exact_file_duplicates": len(exact_file),
        "exact_pixel_duplicates": len(exact_pixel),
        "exact_pixel_duplicates_within_train": dup_in_train,
        "cos_threshold": args.cos_threshold,
        "phash_threshold": args.phash_threshold,
        "overall_near_duplicate_cos": overall_cos,
        "per_class": results,
        "best_sim_percentiles": {p: float(np.percentile(best_sim, p)) for p in [5, 25, 50, 75, 95, 99]},
    }
    out = os.path.join(args.fig_dir, "leakage_audit.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
