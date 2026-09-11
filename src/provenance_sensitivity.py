# provenance_sensitivity.py
"""Can our provenance measures detect a match that has been transformed?

match_figshare.py reports that no image in the aggregated dataset matches any
figshare slice above ~0.87 on three measures. That negative result is only
meaningful if those measures would actually *survive* the transformations a
release pipeline plausibly applies. A positive control built from two
byte-identical files does not establish this: it confirms the measures return
1.0 on identical inputs, which was never in question.

This script builds the control that matters. It takes figshare slices,
pushes them through candidate release pipelines (8-bit conversion, brain
cropping, resizing, JPEG re-encoding at several qualities), and measures
each transformed image against its own source. If the measures stay near 1.0,
a true derivation would have been detected and the negative result stands.
If they collapse toward the ~0.87 that match_figshare.py reports as its best
match, the negative result is uninformative and cannot distinguish "not
derived" from "derived but transformed beyond our ability to detect".

Usage:
    python src/provenance_sensitivity.py --mat_dir external/figshare/mat --n 60
"""
import argparse
import os
import tempfile

import h5py
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy.fftpack import dct

from audit_leakage import embed_images

S = 64


def normalize_like_readme(img):
    im = img.astype(np.float64)
    lo, hi = im.min(), im.max()
    if hi <= lo:
        return np.zeros_like(im, dtype=np.uint8)
    return (255.0 / (hi - lo) * (im - lo)).astype(np.uint8)


def eqvec(arr):
    img = Image.fromarray(arr).convert("L")
    img = ImageOps.equalize(img.resize((S, S), Image.BILINEAR))
    v = np.asarray(img, float).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n else v


def maskvec(arr, thr=25):
    img = Image.fromarray(arr).convert("L").resize((S, S), Image.BILINEAR)
    a = np.asarray(img, float)
    a = 255 * (a - a.min()) / max(a.max() - a.min(), 1)
    return (a > thr).ravel()


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def brain_crop(arr, thr=10, margin=2):
    """Crop to the bounding box of non-background pixels, as a release
    pipeline that trims empty border would do."""
    mask = arr > thr
    if not mask.any():
        return arr
    rows, cols = np.where(mask)
    r0, r1 = max(rows.min() - margin, 0), min(rows.max() + margin + 1, arr.shape[0])
    c0, c1 = max(cols.min() - margin, 0), min(cols.max() + margin + 1, arr.shape[1])
    return arr[r0:r1, c0:c1]


def jpeg_roundtrip(arr, quality, tmpdir):
    p = os.path.join(tmpdir, f"tmp_q{quality}.jpg")
    Image.fromarray(arr).convert("L").save(p, "JPEG", quality=quality)
    return np.asarray(Image.open(p).convert("L")), p


def variants(arr, tmpdir):
    """Candidate release pipelines, from mildest to most aggressive."""
    out = {}
    out["identity"] = arr

    for q in (95, 90, 85):
        v, _ = jpeg_roundtrip(arr, q, tmpdir)
        out[f"jpeg_q{q}"] = v

    r = np.asarray(Image.fromarray(arr).resize((512, 512), Image.BILINEAR))
    out["resize_512"] = r

    c = brain_crop(arr)
    out["crop"] = np.asarray(Image.fromarray(c).resize((512, 512), Image.BILINEAR))

    # the full plausible pipeline: crop, resize, then JPEG
    full, _ = jpeg_roundtrip(out["crop"], 90, tmpdir)
    out["crop+resize+jpeg90"] = full

    # a harsher but still realistic variant
    small = np.asarray(Image.fromarray(c).resize((224, 224), Image.BILINEAR))
    harsh, _ = jpeg_roundtrip(small, 85, tmpdir)
    out["crop+resize224+jpeg85"] = harsh

    return out


def main():
    parser = argparse.ArgumentParser(description="Sensitivity of provenance measures to transformation.")
    parser.add_argument("--mat_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=60)
    args = parser.parse_args()

    files = sorted(f for f in os.listdir(args.mat_dir) if f.endswith(".mat"))[:args.n]
    print(f"Testing {len(files)} figshare slices through candidate release pipelines.\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tmpdir = tempfile.mkdtemp()

    results = defaultdict_list = {}
    embed_pairs = {}

    for fn in files:
        with h5py.File(os.path.join(args.mat_dir, fn), "r") as f:
            img = np.array(f["cjdata"]["image"])
        src = normalize_like_readme(img)
        vs = variants(src, tmpdir)

        src_eq, src_mask = eqvec(src), maskvec(src)
        for name, v in vs.items():
            results.setdefault(name, {"eq": [], "iou": []})
            results[name]["eq"].append(float(src_eq @ eqvec(v)))
            results[name]["iou"].append(iou(src_mask, maskvec(v)))

        # stash paths for the embedding measure (needs files on disk)
        base = os.path.join(tmpdir, f"{fn}_src.png")
        Image.fromarray(src).convert("L").save(base)
        for name, v in vs.items():
            p = os.path.join(tmpdir, f"{fn}_{name.replace('+','_')}.png")
            Image.fromarray(v).convert("L").save(p)
            embed_pairs.setdefault(name, []).append((base, p))

    # embedding cosine, batched
    print("Computing embedding cosine for each variant...")
    emb_results = {}
    for name, pairs in embed_pairs.items():
        a = embed_images([p[0] for p in pairs], device)
        b = embed_images([p[1] for p in pairs], device)
        emb_results[name] = (a * b).sum(dim=1).numpy()

    print(f"\n{'variant':>24s}  {'eq-corr':>18s}  {'mask-IoU':>18s}  {'emb-cos':>18s}")
    print(f"{'':>24s}  {'median (min)':>18s}  {'median (min)':>18s}  {'median (min)':>18s}")
    for name in results:
        eq = np.array(results[name]["eq"])
        io = np.array(results[name]["iou"])
        em = emb_results[name]
        print(f"{name:>24s}  {np.median(eq):8.4f} ({eq.min():.4f})  "
              f"{np.median(io):8.4f} ({io.min():.4f})  "
              f"{np.median(em):8.4f} ({em.min():.4f})")

    print("\nReference points from match_figshare.py / the paper:")
    print("  best observed dataset->figshare match : eq-corr 0.872, IoU 0.938, emb-cos 0.963")
    print("  median unrelated pair within dataset  : eq-corr 0.838")
    print("\nInterpretation: if a variant's median sits near or below the best")
    print("observed match, the measures cannot distinguish a transformed true")
    print("match from an unrelated image, and the negative provenance result")
    print("is uninformative for that transformation.")


if __name__ == "__main__":
    main()
