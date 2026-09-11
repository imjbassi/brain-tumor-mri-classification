# match_figshare_cropnorm.py
"""Crop-normalized provenance matching.

provenance_sensitivity.py shows that our original measures survive JPEG
re-encoding and resizing but collapse under cropping: a genuine figshare
slice, cropped to its brain region and resized, scores only ~0.60 equalized
correlation and ~0.963 embedding cosine against its own source. Since 0.963
is exactly the best dataset-to-figshare score we observed, the original
negative result could not distinguish "not derived" from "derived and
cropped".

This script removes that confound by applying the same brain-region crop to
BOTH sides before comparing, so a difference in framing cannot suppress the
similarity. It first validates the approach on a synthetic control (a figshare
slice against a cropped, resized, JPEG-compressed copy of itself), then runs
the full dataset-against-figshare search.

Usage:
    python src/match_figshare_cropnorm.py --data_dir ./data --mat_dir external/figshare/mat
"""
import argparse
import os
import tempfile

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps

from audit_leakage import embed_images
from data_loader import BrainMRIDataset

CANON = 256
S = 64


def normalize_like_readme(img):
    im = img.astype(np.float64)
    lo, hi = im.min(), im.max()
    if hi <= lo:
        return np.zeros_like(im, dtype=np.uint8)
    return (255.0 / (hi - lo) * (im - lo)).astype(np.uint8)


def brain_crop(arr, thr=10, margin=2):
    mask = arr > thr
    if not mask.any():
        return arr
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, arr.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def canonicalize(arr):
    """Brain-crop then resize to a fixed square, so framing differences
    between the two collections cannot depress the similarity."""
    c = brain_crop(arr)
    return np.asarray(Image.fromarray(c).convert("L").resize((CANON, CANON), Image.BILINEAR))


def eqvec(arr):
    img = ImageOps.equalize(Image.fromarray(arr).convert("L").resize((S, S), Image.BILINEAR))
    v = np.asarray(img, float).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def main():
    parser = argparse.ArgumentParser(description="Crop-normalized figshare provenance matching.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mat_dir", type=str, required=True)
    parser.add_argument("--control_n", type=int, default=60)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmpdir = tempfile.mkdtemp()
    work = os.path.join(tmpdir, "canon")
    os.makedirs(work, exist_ok=True)

    files = sorted(f for f in os.listdir(args.mat_dir) if f.endswith(".mat"))

    # ---------------- control: does crop-normalization recover a match? ------
    print("Control: figshare slice vs cropped/resized/JPEG copy of itself,")
    print("comparing after crop-normalizing BOTH sides.\n")
    ctl_src, ctl_dst = [], []
    for i, fn in enumerate(files[:args.control_n]):
        with h5py.File(os.path.join(args.mat_dir, fn), "r") as f:
            img = np.array(f["cjdata"]["image"])
        src = normalize_like_readme(img)

        # simulate a release pipeline: crop, resize, JPEG
        c = brain_crop(src)
        jp = os.path.join(tmpdir, f"c{i}.jpg")
        Image.fromarray(c).convert("L").resize((512, 512), Image.BILINEAR).save(jp, "JPEG", quality=90)
        transformed = np.asarray(Image.open(jp).convert("L"))

        a = os.path.join(work, f"ctl{i}_a.png")
        b = os.path.join(work, f"ctl{i}_b.png")
        Image.fromarray(canonicalize(src)).save(a)
        Image.fromarray(canonicalize(transformed)).save(b)
        ctl_src.append(a)
        ctl_dst.append(b)

    ea = embed_images(ctl_src, device)
    eb = embed_images(ctl_dst, device)
    cos = (ea * eb).sum(dim=1).numpy()
    eqc = np.array([eqvec(np.asarray(Image.open(a))) @ eqvec(np.asarray(Image.open(b)))
                     for a, b in zip(ctl_src, ctl_dst)])
    print(f"  embedding cosine : median {np.median(cos):.4f}  min {cos.min():.4f}")
    print(f"  equalized corr   : median {np.median(eqc):.4f}  min {eqc.min():.4f}")
    if np.median(cos) < 0.98:
        print("\n  WARNING: crop-normalization does not recover the match either.")
        print("  A negative result from the search below would remain uninformative.")
    else:
        print("\n  Crop-normalization recovers the match. The search below is informative.")

    # ---------------- render canonical figshare set --------------------------
    print(f"\nCanonicalizing {len(files)} figshare slices...")
    fs_paths, fs_meta = [], []
    for i, fn in enumerate(files):
        with h5py.File(os.path.join(args.mat_dir, fn), "r") as f:
            cj = f["cjdata"]
            lab = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel()).strip()
            img = np.array(cj["image"])
        p = os.path.join(work, f"fs{i}.png")
        Image.fromarray(canonicalize(normalize_like_readme(img))).save(p)
        fs_paths.append(p)
        fs_meta.append({"file": fn, "label": lab, "pid": pid})
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}")

    fs_emb = embed_images(fs_paths, device).to(device)

    # ---------------- canonicalize dataset tumor images ----------------------
    train = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test = BrainMRIDataset(os.path.join(args.data_dir, "Testing"), class_to_idx=train.class_to_idx)
    idx_to_class = {i: c for c, i in train.class_to_idx.items()}
    TUMOR = {"glioma", "meningioma", "pituitary"}

    for split, ds in (("Training", train), ("Testing", test)):
        sel = [(p, idx_to_class[l]) for p, l in zip(ds.image_paths, ds.labels)
               if idx_to_class[l] in TUMOR]
        print(f"\nCanonicalizing {len(sel)} {split} tumor images...")
        paths = []
        for i, (p, c) in enumerate(sel):
            arr = np.asarray(Image.open(p).convert("L"))
            q = os.path.join(work, f"{split}{i}.png")
            Image.fromarray(canonicalize(arr)).save(q)
            paths.append(q)
        emb = embed_images(paths, device).to(device)

        sims = emb @ fs_emb.T
        mx, am = sims.max(dim=1)
        mx = mx.cpu().numpy()
        print(f"  crop-normalized nearest-figshare cosine:")
        print(f"    median {np.median(mx):.4f}  p95 {np.percentile(mx,95):.4f}  max {mx.max():.4f}")
        for thr in (0.999, 0.99, 0.98, 0.95):
            print(f"    >= {thr}: {(mx >= thr).sum()} / {len(mx)}")


if __name__ == "__main__":
    main()
