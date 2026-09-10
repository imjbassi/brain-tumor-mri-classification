# match_figshare.py
"""Map the aggregated dataset's JPEGs back to the original figshare slices,
so every figshare-derived image can be tagged with its patient ID.

The figshare release (Cheng et al., 3064 slices from 233 patients) stores a
patient ID per slice. The aggregated four-class dataset used in this repo
dropped that metadata when the slices were converted to JPEG. Without it,
there is no way to tell whether a train/test split keeps a patient's slices
on one side of the boundary.

We recover it by content: each .mat slice is normalized exactly the way the
figshare README specifies the JPEG conversion was done, then matched to the
JPEGs by perceptual hash (with an embedding-similarity fallback and a
pixel-correlation confirmation step).

Writes figshare_patient_map.json mapping each matched JPEG path to its
patient ID, source slice, and match confidence.

Usage:
    python src/match_figshare.py --data_dir ./data --mat_dir external/figshare/mat
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.fftpack import dct

from data_loader import BrainMRIDataset

# figshare label codes -> the class folder names used in the aggregated dataset
FIGSHARE_LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}


def normalize_like_readme(img):
    """uint8 conversion exactly as the figshare README documents it."""
    im = img.astype(np.float64)
    lo, hi = im.min(), im.max()
    if hi <= lo:
        return np.zeros_like(im, dtype=np.uint8)
    return (255.0 / (hi - lo) * (im - lo)).astype(np.uint8)


def phash_array(arr, hash_size=8, highfreq_factor=4):
    size = hash_size * highfreq_factor
    img = Image.fromarray(arr).convert("L").resize((size, size), Image.BILINEAR)
    pixels = np.asarray(img, dtype=float)
    coeffs = dct(dct(pixels, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = coeffs[:hash_size, :hash_size].flatten()
    med = np.median(low[1:])
    out = 0
    for bit in low > med:
        out = (out << 1) | int(bit)
    return out


def phash_bits(h, nbits=64):
    return np.array([(h >> i) & 1 for i in range(nbits)], dtype=np.uint8)


def thumb(arr, size=64):
    """Small normalized thumbnail for pixel-correlation confirmation."""
    img = Image.fromarray(arr).convert("L").resize((size, size), Image.BILINEAR)
    v = np.asarray(img, dtype=np.float64).ravel()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def load_mat_slices(mat_dir):
    """Returns list of dicts with pid, label, phash, thumb for each slice."""
    out = []
    files = sorted(os.listdir(mat_dir))
    for i, fn in enumerate(files):
        if not fn.endswith(".mat"):
            continue
        path = os.path.join(mat_dir, fn)
        with h5py.File(path, "r") as f:
            cj = f["cjdata"]
            label = int(np.array(cj["label"]).ravel()[0])
            # PID is stored as a char array
            pid_raw = np.array(cj["PID"]).ravel()
            pid = "".join(chr(int(c)) for c in pid_raw).strip()
            image = np.array(cj["image"])
        arr = normalize_like_readme(image)
        out.append({
            "file": fn,
            "pid": pid,
            "label": label,
            "class": FIGSHARE_LABELS.get(label, f"unknown{label}"),
            "phash": phash_array(arr),
            "thumb": thumb(arr),
        })
        if (i + 1) % 500 == 0:
            print(f"  read {i + 1} .mat files")
    return out


def main():
    parser = argparse.ArgumentParser(description="Recover figshare patient IDs for the aggregated dataset.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--mat_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="figshare_patient_map.json")
    parser.add_argument("--phash_max", type=int, default=10,
                        help="Max pHash Hamming distance to accept as a match.")
    parser.add_argument("--corr_min", type=float, default=0.90,
                        help="Min pixel correlation to confirm a match.")
    args = parser.parse_args()

    print("Reading figshare .mat slices...")
    slices = load_mat_slices(args.mat_dir)
    print(f"  {len(slices)} slices, {len({s['pid'] for s in slices})} unique patient IDs")
    from collections import Counter
    print("  slices per class:", dict(Counter(s["class"] for s in slices)))

    mat_bits = np.stack([phash_bits(s["phash"]) for s in slices])
    mat_thumbs = np.stack([s["thumb"] for s in slices])

    train = BrainMRIDataset(os.path.join(args.data_dir, "Training"))
    test = BrainMRIDataset(os.path.join(args.data_dir, "Testing"), class_to_idx=train.class_to_idx)
    idx_to_class = {i: c for c, i in train.class_to_idx.items()}

    mapping = {}
    stats = {"matched": 0, "unmatched": 0}
    for split, ds in [("Training", train), ("Testing", test)]:
        print(f"\nMatching {split} ({len(ds)} images)...")
        for n, (p, lab) in enumerate(zip(ds.image_paths, ds.labels)):
            cls = idx_to_class[lab]
            arr = np.asarray(Image.open(p).convert("L"))
            ph = phash_bits(phash_array(arr))
            th = thumb(arr)

            dists = (mat_bits ^ ph).sum(axis=1)
            j = int(np.argmin(dists))
            d = int(dists[j])
            corr = float(mat_thumbs[j] @ th)

            # Confirm with pixel correlation; among close pHash candidates,
            # pick the one with the best correlation.
            cands = np.where(dists <= max(d, args.phash_max))[0]
            if len(cands) > 1:
                corrs = mat_thumbs[cands] @ th
                k = int(cands[int(np.argmax(corrs))])
                if corrs.max() > corr:
                    j, d, corr = k, int(dists[k]), float(corrs.max())

            if d <= args.phash_max and corr >= args.corr_min:
                mapping[os.path.relpath(p).replace("\\", "/")] = {
                    "split": split,
                    "class": cls,
                    "pid": slices[j]["pid"],
                    "figshare_file": slices[j]["file"],
                    "figshare_class": slices[j]["class"],
                    "phash_dist": d,
                    "corr": round(corr, 4),
                }
                stats["matched"] += 1
            else:
                stats["unmatched"] += 1
            if (n + 1) % 1000 == 0:
                print(f"  {n + 1} done")

    print(f"\nMatched {stats['matched']} images to figshare slices, {stats['unmatched']} unmatched")

    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Wrote {args.out}")

    # ---- the actual question: do patients cross the train/test boundary? ----
    by_pid = {}
    for path, m in mapping.items():
        by_pid.setdefault(m["pid"], {"Training": 0, "Testing": 0, "classes": set()})
        by_pid[m["pid"]][m["split"]] += 1
        by_pid[m["pid"]]["classes"].add(m["class"])

    crossing = {pid: v for pid, v in by_pid.items() if v["Training"] > 0 and v["Testing"] > 0}
    print(f"\nPatients matched: {len(by_pid)}")
    print(f"Patients with slices in BOTH train and test: {len(crossing)}")
    if by_pid:
        print(f"  ({100 * len(crossing) / len(by_pid):.1f}% of matched patients)")

    leaked_test = sum(v["Testing"] for v in crossing.values())
    total_test_matched = sum(1 for m in mapping.values() if m["split"] == "Testing")
    print(f"Test images belonging to a patient also seen in training: "
          f"{leaked_test} / {total_test_matched} matched test images")

    per_class = {}
    for path, m in mapping.items():
        if m["split"] != "Testing":
            continue
        c = m["class"]
        per_class.setdefault(c, {"total": 0, "leaked": 0})
        per_class[c]["total"] += 1
        if m["pid"] in crossing:
            per_class[c]["leaked"] += 1
    print("\nper-class test leakage (patient seen in training):")
    for c, v in sorted(per_class.items()):
        pct = 100 * v["leaked"] / v["total"] if v["total"] else 0
        print(f"  {c:>11s}: {v['leaked']:4d} / {v['total']:4d}  ({pct:5.1f}%)")

    # label agreement between figshare and the aggregated dataset
    mismatch = [(p, m) for p, m in mapping.items() if m["class"] != m["figshare_class"]]
    print(f"\nlabel disagreements between figshare and aggregated dataset: {len(mismatch)}")
    if mismatch:
        agg = {}
        for p, m in mismatch:
            agg.setdefault((m["figshare_class"], m["class"]), 0)
            agg[(m["figshare_class"], m["class"])] += 1
        for (fs, ag), n in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  figshare says {fs:>11s} -> dataset says {ag:<11s}: {n}")


if __name__ == "__main__":
    main()
