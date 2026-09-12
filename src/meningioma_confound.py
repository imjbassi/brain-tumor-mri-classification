# meningioma_confound.py
"""Is the meningioma collapse a leakage effect or a coverage artifact?

Only 58% of meningioma test images could be traced to a figshare patient,
against 96% and 99% for glioma and pituitary. The 437 untraced tumor images
are assigned to training in every cross-validation fold, and meningioma is
also the class that degrades most on the patient-disjoint split. Those facts
admit two explanations:

  (a) removing patient leakage genuinely hurts meningioma most, or
  (b) the traced and untraced meningioma subpopulations differ, so the
      patient-disjoint evaluation is scored on a harder subset than the
      released-split evaluation was.

Two checks that separate them, neither requiring any training:

  1. Recompute released-split per-class metrics restricted to the images
     that WERE traced. If meningioma still scores ~0.99 on the released split
     over exactly the population the patient-disjoint result is computed on,
     then the drop is not a coverage artifact.
  2. Test whether traced and untraced images are separable by file-level
     properties (dimensions, file size, JPEG quantization tables, intensity
     statistics). If they are, that is a provenance signal and a limitation
     worth stating; if they are not, explanation (b) loses its mechanism.

Usage:
    python src/meningioma_confound.py --data_dir ./data --runs_dir runs/leaky
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
from PIL import Image
from sklearn.metrics import classification_report

TUMOR = ("glioma", "meningioma", "pituitary")


def read_predictions(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_matched_only_metrics(runs_dir, traced):
    """Released-split per-class F1, full test set vs traced-only subset."""
    files = sorted(glob.glob(os.path.join(runs_dir, "seed*", "predictions.csv")))
    if not files:
        print("  no predictions found; run the sweep first")
        return None

    full, sub = defaultdict(list), defaultdict(list)
    for pf in files:
        rows = read_predictions(pf)
        classes = sorted({r["true"] for r in rows})

        rep = classification_report([r["true"] for r in rows], [r["pred"] for r in rows],
                                     labels=classes, target_names=classes,
                                     output_dict=True, zero_division=0)
        for c in classes:
            full[c].append(rep[c]["f1-score"])

        keep = [r for r in rows if os.path.basename(r["path"]) in traced]
        kc = sorted({r["true"] for r in keep})
        rep2 = classification_report([r["true"] for r in keep], [r["pred"] for r in keep],
                                      labels=kc, target_names=kc,
                                      output_dict=True, zero_division=0)
        for c in kc:
            sub[c].append(rep2[c]["f1-score"])

    n_keep = len([r for r in read_predictions(files[0])
                  if os.path.basename(r["path"]) in traced])
    print(f"  released-split F1 over {len(files)} seeds "
          f"(full test set vs the {n_keep} traced images only):\n")
    print(f"    {'class':>11s}  {'full':>16s}  {'traced only':>16s}")
    out = {}
    for c in sorted(full):
        f = np.array(full[c])
        s = np.array(sub.get(c, []))
        s_txt = f"{s.mean():.4f} +/- {s.std(ddof=1):.4f}" if len(s) > 1 else "n/a"
        print(f"    {c:>11s}  {f.mean():.4f} +/- {f.std(ddof=1):.4f}  {s_txt:>16s}")
        out[c] = {"full": float(f.mean()),
                  "traced_only": float(s.mean()) if len(s) else None}
    return out


def file_level_properties(data_dir, traced):
    """Are traced and untraced tumor images separable by file-level artifacts?"""
    rows = []
    for split in ("Training", "Testing"):
        for cls in TUMOR:
            d = os.path.join(data_dir, split, cls)
            if not os.path.isdir(d):
                continue
            for fn in os.listdir(d):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                p = os.path.join(d, fn)
                try:
                    with Image.open(p) as im:
                        w, h = im.size
                        qt = getattr(im, "quantization", None)
                        qkey = None
                        if qt:
                            qkey = int(np.sum(list(qt.values())[0][:16]))
                        g = np.asarray(im.convert("L"), dtype=np.float32)
                except Exception:
                    continue
                rows.append({
                    "base": fn, "cls": cls, "traced": fn in traced,
                    "w": w, "h": h, "bytes": os.path.getsize(p),
                    "qt": qkey, "mean": float(g.mean()), "std": float(g.std()),
                })

    print(f"\n  inspected {len(rows)} tumor images")
    res = {}
    for cls in TUMOR:
        sub = [r for r in rows if r["cls"] == cls]
        tr = [r for r in sub if r["traced"]]
        un = [r for r in sub if not r["traced"]]
        if not un:
            print(f"    {cls:>11s}: all {len(tr)} traced, nothing to compare")
            res[cls] = {"n_traced": len(tr), "n_untraced": 0}
            continue

        def summarize(g, k):
            v = np.array([x[k] for x in g], dtype=float)
            return v.mean(), v.std()

        dims_tr = Counter((r["w"], r["h"]) for r in tr).most_common(2)
        dims_un = Counter((r["w"], r["h"]) for r in un).most_common(2)
        qt_tr = Counter(r["qt"] for r in tr).most_common(2)
        qt_un = Counter(r["qt"] for r in un).most_common(2)
        m_tr, s_tr = summarize(tr, "mean")
        m_un, s_un = summarize(un, "mean")

        print(f"\n    {cls}: {len(tr)} traced, {len(un)} untraced")
        print(f"      dimensions  traced {dims_tr}")
        print(f"                untraced {dims_un}")
        print(f"      JPEG qtable traced {qt_tr}")
        print(f"                untraced {qt_un}")
        print(f"      mean intensity  traced {m_tr:.1f}+/-{s_tr:.1f}   "
              f"untraced {m_un:.1f}+/-{s_un:.1f}")

        # a crude separability signal: do the dominant dimension/qtable differ?
        sep = (dims_tr[0][0] != dims_un[0][0]) or (qt_tr[0][0] != qt_un[0][0])
        print(f"      dominant file signature differs: {sep}")
        res[cls] = {
            "n_traced": len(tr), "n_untraced": len(un),
            "dims_traced": [list(k) + [v] for k, v in dims_tr],
            "dims_untraced": [list(k) + [v] for k, v in dims_un],
            "qtable_traced": qt_tr, "qtable_untraced": qt_un,
            "mean_intensity_traced": m_tr, "mean_intensity_untraced": m_un,
            "dominant_signature_differs": bool(sep),
        }
    return res


def main():
    parser = argparse.ArgumentParser(description="Test the meningioma coverage confound.")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--runs_dir", type=str, default="runs/leaky")
    parser.add_argument("--map", type=str, default="figshare_patient_map.json")
    parser.add_argument("--out", type=str, default="paper/figures/meningioma_confound.json")
    args = parser.parse_args()

    traced = {os.path.basename(k) for k in json.load(open(args.map))}
    print(f"traced images: {len(traced)}\n")

    print("=== 1. released-split metrics on the traced population only ===")
    metrics = check_matched_only_metrics(args.runs_dir, traced)

    print("\n=== 2. file-level separability of traced vs untraced ===")
    props = file_level_properties(args.data_dir, traced)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"released_split_f1": metrics, "file_properties": props}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
