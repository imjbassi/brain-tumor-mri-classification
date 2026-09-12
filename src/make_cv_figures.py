# make_cv_figures.py
"""Figures for the patient-disjoint evaluation.

The released-split baseline had five figures and the paper's main result had
none. This produces the three that carry the argument:

  cv_per_class_f1.png  per-class F1 across released -> deduplicated ->
                       patient-disjoint, which is where the meningioma
                       collapse becomes visible
  cv_fold_spread.png   per-fold accuracy against the pooled estimate, the
                       visual case against single-holdout reporting
  cv_paired.png        the paired contrast on the 761 traceable images

Usage:
    python src/make_cv_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = "paper/figures"
BLUE, ORANGE, RED, GREY = "#3b6ea5", "#e08214", "#b2182b", "#888888"


def load():
    cv = json.load(open(os.path.join(FIG, "cv_analysis.json")))
    return cv


def fig_per_class(cv):
    """Per-class F1 across the three splits."""
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    seeds = cv["seeds"]

    # released split (10 seeds) and deduplicated (10 seeds)
    released = {"glioma": 0.9913, "meningioma": 0.9870, "notumor": 0.9951, "pituitary": 0.9967}
    dedup = {"glioma": 0.9929, "meningioma": 0.9845, "notumor": 0.9992, "pituitary": 0.9971}
    patient = {c: float(np.mean([cv["per_replicate"][s]["per_class_f1"][c] for s in seeds]))
               for c in classes}

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = [0, 1, 2]
    labels = ["Released", "Deduplicated", "Patient-disjoint"]
    colors = {"glioma": BLUE, "meningioma": RED, "notumor": GREY, "pituitary": ORANGE}
    for c in classes:
        ys = [released[c], dedup[c], patient[c]]
        ax.plot(x, ys, marker="o", linewidth=2.2, markersize=7,
                color=colors[c], label=c)
        ax.annotate(f"{ys[-1]:.3f}", (x[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, -3), fontsize=9, color=colors[c])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1")
    ax.set_title("Per-class F1 collapses only when patient overlap is removed")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.25, 2.55)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG, "cv_per_class_f1.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_fold_spread(cv):
    """Per-fold accuracy vs the pooled estimate."""
    vc = cv["variance_components"]
    folds = np.array(vc["fold_means"]) * 100
    seeds = cv["seeds"]
    pooled = np.mean([cv["per_replicate"][s]["accuracy"] for s in seeds]) * 100
    cis = [cv["per_replicate"][s]["bootstrap_ci95"] for s in seeds]
    lo = np.mean([c[0] for c in cis]) * 100
    hi = np.mean([c[1] for c in cis]) * 100

    # per-fold, per-seed points
    pts = []
    for k in range(cv["n_folds"]):
        p = os.path.join("runs", f"cv_fold{k}", "results.json")
        if os.path.exists(p):
            d = json.load(open(p))
            for v in d.values():
                pts.append((k, v["test_accuracy"] * 100))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=38, color=BLUE, alpha=0.75, zorder=3, label="individual runs (3 seeds/fold)")
    ax.scatter(range(len(folds)), folds, marker="_", s=700, linewidths=2.6,
               color=RED, zorder=4, label="fold mean")
    ax.axhspan(lo, hi, color=ORANGE, alpha=0.18, zorder=1,
               label="pooled 95% CI (patient bootstrap)")
    ax.axhline(pooled, color=ORANGE, linewidth=2, zorder=2,
               label=f"pooled estimate {pooled:.2f}%")
    ax.axhline(99.22, color=GREY, linestyle="--", linewidth=1.6, zorder=2,
               label="released split 99.22%")
    ax.set_xticks(range(len(folds)))
    ax.set_xticklabels([f"fold {k}" for k in range(len(folds))])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Which patients land in the test fold matters more than the seed")
    ax.grid(axis="y", alpha=0.3)
    # keep the legend clear of fold 4, which sits lowest
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.set_ylim(91.5, 100.2)
    fig.tight_layout()
    out = os.path.join(FIG, "cv_fold_spread.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


def fig_paired(cv):
    """The paired contrast on the traceable images."""
    p = cv.get("paired")
    if not p:
        print("no paired result in cv_analysis.json; skipping")
        return
    seen = p["acc_patient_seen"] * 100
    unseen = p["acc_patient_unseen"] * 100
    n = p["n_images"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 4.0),
                                    gridspec_kw={"width_ratios": [1, 1]})
    bars = ax1.bar(["Patient in\ntraining", "Patient not\nin training"],
                   [seen, unseen], color=[GREY, RED], width=0.6)
    for b, v in zip(bars, [seen, unseen]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax1.set_ylim(90, 100.6)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Same {n:,} images, same pipeline")
    ax1.grid(axis="y", alpha=0.3)

    err_seen, err_unseen = 100 - seen, 100 - unseen
    bars2 = ax2.bar(["Patient in\ntraining", "Patient not\nin training"],
                    [err_seen, err_unseen], color=[GREY, RED], width=0.6)
    for b, v in zip(bars2, [err_seen, err_unseen]):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Error rate (%)")
    ax2.set_title(f"{p['error_ratio']:.1f}$\\times$ the error rate")
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG, "cv_paired.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    cv = load()
    fig_per_class(cv)
    fig_fold_spread(cv)
    fig_paired(cv)
