# match_figshare.py
"""Recover figshare patient IDs for the aggregated dataset's tumor images.

The figshare release (Cheng et al., 3064 slices from 233 patients) stores a
patient ID with every slice. The aggregated four-class dataset dropped that
metadata during conversion to JPEG. Without it there is no way to check
whether a train/test split keeps each patient's slices on one side of the
boundary, which is the question that actually matters for evaluation.

Matching the two collections by content requires care. An earlier version of
this script compared the images directly and found nothing, but that was a
measurement failure, not a real negative:

  * The aggregated images are cropped to the brain region relative to the
    figshare originals. provenance_sensitivity.py shows that cropping alone
    drops equalized correlation to ~0.60 and embedding cosine to ~0.963 for
    a genuine match, which is indistinguishable from an unrelated image.
  * The aggregated images are also rotated 90 degrees relative to figshare.

This version corrects both. It crop-normalizes both sides (brain bounding box,
then resize to a canonical square) and searches all eight dihedral
orientations of each dataset image. A synthetic control establishes the score
a genuine match achieves under this procedure, so the acceptance threshold is
calibrated rather than guessed.

Writes figshare_patient_map.json and reports patient-level train/test overlap.

Usage:
    python src/match_figshare.py --data_dir ./data --mat_dir external/figshare/mat
"""
import argparse
import json
import os
import tempfile
from collections import Counter

import h5py
import numpy as np
import torch
from PIL import Image

from audit_leakage import embed_images
from data_loader import BrainMRIDataset

CANON = 256
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}

DIHEDRAL = [
    ("identity",       lambda a: a),
    ("rot90",          lambda a: np.rot90(a, 1)),
    ("rot180",         lambda a: np.rot90(a, 2)),
    ("rot270",         lambda a: np.rot90(a, 3)),
    ("mirror",         lambda a: np.fliplr(a)),
    ("mirror_rot90",   lambda a: np.rot90(np.fliplr(a), 1)),
    ("mirror_rot180",  lambda a: np.rot90(np.fliplr(a), 2)),
    ("mirror_rot270",  lambda a: np.rot90(np.fliplr(a), 3)),
]


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
    c = brain_crop(np.ascontiguousarray(arr))
    return np.asarray(Image.fromarray(c).convert("L").resize((CANON, CANON), Image.BILINEAR))


def load_figshare(mat_dir, work, device):
    files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))
    paths, meta = [], []
    print(f"Canonicalizing {len(files)} figshare slices...")
    for i, fn in enumerate(files):
        with h5py.File(os.path.join(mat_dir, fn), "r") as f:
            cj = f["cjdata"]
            lab = int(np.array(cj["label"]).ravel()[0])
            pid = "".join(chr(int(c)) for c in np.array(cj["PID"]).ravel()).strip()
            img = np.array(cj["image"])
        p = os.path.join(work, f"fs{i}.png")
        Image.fromarray(canonicalize(normalize_like_readme(img))).save(p)
        paths.append(p)
        meta.append({"file": fn, "label": lab, "class": FIGSHARE_LABELS.get(lab), "pid": pid})
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}")
    return embed_images(paths, device).to(device), meta


def run_control(mat_dir, work, device, n=60):
    """What does a genuine match score under this procedure? Simulates a
    release pipeline (crop, resize, JPEG) and matches the result back."""
    files = sorted(f for f in os.listdir(mat_dir) if f.endswith(".mat"))[:n]
    a_paths, b_paths = [], []
    for i, fn in enumerate(files):
        with h5py.File(os.path.join(mat_dir, fn), "r") as f:
            img = np.array(f["cjdata"]["image"])
        src = normalize_like_readme(img)
        jp = os.path.join(work, f"ctl{i}.jpg")
        Image.fromarray(brain_crop(src)).convert("L").resize((512, 512), Image.BILINEAR) \
             .save(jp, "JPEG", quality=90)
        transformed = np.asarray(Image.open(jp).convert("L"))
        a = os.path.join(work, f"ctla{i}.png")
        b = os.path.join(work, f"ctlb{i}.png")
        Image.fromarray(canonicalize(src)).save(a)
        Image.fromarray(canonicalize(transformed)).save(b)
        a_paths.append(a)
        b_paths.append(b)
    ea = embed_images(a_paths, device)
    eb = embed_images(b_paths, device)
    cos = (ea * eb).sum(dim=1).numpy()
    return float(np.median(cos)), float(cos.min())


def main():
    parser = argparse.ArgumentParser(description="Recover figshare patient IDs by content matching.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mat_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="figshare_patient_map.json")
    parser.add_argument("--threshold", type=float, default=0.98,
                        help="Minimum crop-normalized cosine to accept a match.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work = tempfile.mkdtemp()

    ctl_med, ctl_min = run_control(args.mat_dir, work, device)
    print(f"Control (genuine match through a crop/resize/JPEG pipeline): "
          f"median {ctl_med:.4f}, min {ctl_min:.4f}")
    print(f"Acceptance threshold: {args.threshold}\n")

    fs_emb, fs_meta = load_figshare(args.mat_dir, work, device)

    train = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test = BrainMRIDataset(os.path.join(args.data_dir, "Testing"),
                            class_to_idx=train.class_to_idx)
    idx_to_class = {i: c for c, i in train.class_to_idx.items()}

    mapping = {}
    orientation_counts = Counter()

    for split, ds in (("Training", train), ("Testing", test)):
        sel = [(p, idx_to_class[l]) for p, l in zip(ds.image_paths, ds.labels)
               if idx_to_class[l] in TUMOR_CLASSES]
        print(f"\nMatching {len(sel)} {split} tumor images across 8 orientations...")

        best_sim = np.full(len(sel), -1.0)
        best_idx = np.zeros(len(sel), dtype=int)
        best_ori = [""] * len(sel)

        for name, op in DIHEDRAL:
            paths = []
            for i, (p, _) in enumerate(sel):
                arr = np.asarray(Image.open(p).convert("L"))
                q = os.path.join(work, f"q{i}.png")
                Image.fromarray(canonicalize(op(arr))).save(q)
                paths.append(q)
            emb = embed_images(paths, device).to(device)
            sims = emb @ fs_emb.T
            mx, am = sims.max(dim=1)
            mx = mx.cpu().numpy()
            am = am.cpu().numpy()
            upd = mx > best_sim
            best_sim[upd] = mx[upd]
            best_idx[upd] = am[upd]
            for i in np.where(upd)[0]:
                best_ori[i] = name
            print(f"  {name:>14s}: median {np.median(mx):.4f}  >= thr: {(mx >= args.threshold).sum()}")

        n_matched = 0
        for i, (p, cls) in enumerate(sel):
            if best_sim[i] >= args.threshold:
                m = fs_meta[best_idx[i]]
                mapping[os.path.relpath(p).replace("\\", "/")] = {
                    "split": split, "class": cls, "pid": m["pid"],
                    "figshare_file": m["file"], "figshare_class": m["class"],
                    "orientation": best_ori[i], "cosine": round(float(best_sim[i]), 4),
                }
                orientation_counts[best_ori[i]] += 1
                n_matched += 1
        print(f"  matched {n_matched} / {len(sel)} "
              f"({100*n_matched/len(sel):.1f}%) at cosine >= {args.threshold}")

    print(f"\nOrientation of accepted matches: {dict(orientation_counts)}")

    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote {args.out} ({len(mapping)} matched images)")

    # ---- the question this was all for: do patients cross the boundary? ----
    by_pid = {}
    for path, m in mapping.items():
        v = by_pid.setdefault(m["pid"], {"Training": 0, "Testing": 0})
        v[m["split"]] += 1

    crossing = {p: v for p, v in by_pid.items() if v["Training"] and v["Testing"]}
    print(f"\nPatients matched: {len(by_pid)} of 233 in figshare")
    print(f"Patients with slices in BOTH splits: {len(crossing)}"
          + (f" ({100*len(crossing)/len(by_pid):.1f}%)" if by_pid else ""))

    leaked = sum(v["Testing"] for v in crossing.values())
    tot_test = sum(1 for m in mapping.values() if m["split"] == "Testing")
    if tot_test:
        print(f"Matched test images whose patient also appears in training: "
              f"{leaked} / {tot_test} ({100*leaked/tot_test:.1f}%)")

    per_class = {}
    for m in mapping.values():
        if m["split"] != "Testing":
            continue
        d = per_class.setdefault(m["class"], {"total": 0, "leaked": 0})
        d["total"] += 1
        if m["pid"] in crossing:
            d["leaked"] += 1
    print("\nPer-class patient-level test leakage:")
    for c, v in sorted(per_class.items()):
        pct = 100 * v["leaked"] / v["total"] if v["total"] else 0
        print(f"  {c:>11s}: {v['leaked']:4d} / {v['total']:4d}  ({pct:5.1f}%)")

    mismatch = sum(1 for m in mapping.values() if m["class"] != m["figshare_class"])
    print(f"\nLabel disagreements figshare vs dataset: {mismatch}")
    if mismatch:
        agg = Counter((m["figshare_class"], m["class"]) for m in mapping.values()
                      if m["class"] != m["figshare_class"])
        for (fs, ag), n in agg.most_common():
            print(f"  figshare {fs} -> dataset {ag}: {n}")


if __name__ == "__main__":
    main()
