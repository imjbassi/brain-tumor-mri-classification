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


def fig_rand_control():
    """The matched random-split control, per class and by fold spread.

    Left panel is the point of the control: the classes that have patients all
    move, and notumor, which has none, does not. Right panel is the second
    finding, that fold-to-fold variance is a property of patient grouping and
    not of cross-validation.
    """
    path = os.path.join(FIG, "random_control.json")
    if not os.path.exists(path):
        print(f"skipping rand_control.png: {path} not found")
        return
    rc = json.load(open(path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.4),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    order = ["glioma", "meningioma", "pituitary", "notumor"]
    x = np.arange(len(order))
    w = 0.38
    rand = [100 * rc["per_class"][c]["acc_random"] for c in order]
    pat = [100 * rc["per_class"][c]["acc_patient"] for c in order]
    ax1.bar(x - w / 2, rand, w, label="Random folds", color=GREY)
    ax1.bar(x + w / 2, pat, w, label="Patient folds", color=BLUE)
    for i, c in enumerate(order):
        d = 100 * rc["per_class"][c]["difference"]
        ax1.text(i, max(rand[i], pat[i]) + 0.45, f"{d:+.2f}", ha="center",
                 fontsize=9.5, fontweight="bold",
                 color=RED if abs(d) > 1 else GREY)
    # notumor has no patient identity, so it is the null.
    ax1.axvspan(len(order) - 1.5, len(order) - 0.5, color=ORANGE, alpha=0.09,
                zorder=0)
    ax1.text(len(order) - 1, 87.4, "no patients\nto separate", ha="center",
             fontsize=8.5, color=ORANGE, style="italic")
    ax1.set_xticks(x)
    ax1.set_xticklabels([c if c != "notumor" else "notumor" for c in order])
    ax1.set_ylim(86, 101.4)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Only the classes with patients move")
    ax1.legend(loc="lower left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    fs = rc.get("fold_spread", {})
    if fs:
        for j, (tag, color, label) in enumerate(
                [("random", GREY, "Random folds"), ("patient", BLUE, "Patient folds")]):
            means = [100 * m for m in fs[tag]["fold_means"]]
            ax2.scatter([j] * len(means), means, s=62, color=color, zorder=3,
                        label=label)
            ax2.plot([j - 0.16, j + 0.16], [np.mean(means)] * 2, color=color, lw=2)
            # Axis-fraction y so the label cannot fall outside the view.
            ax2.text(j, 0.035, f"range {max(means)-min(means):.2f} pts",
                     transform=ax2.get_xaxis_transform(), ha="center",
                     fontsize=9, color=color, fontweight="bold")
        allm = [100 * m for t in ("random", "patient") for m in fs[t]["fold_means"]]
        ax2.set_ylim(min(allm) - 1.3, max(allm) + 0.5)   # room for the labels
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Random", "By patient"])
        ax2.set_xlim(-0.5, 1.5)
        ax2.set_ylabel("Per-fold accuracy (%)")
        ax2.set_title("Fold spread is about patients")
        ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = os.path.join(FIG, "rand_control.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    cv = load()
    fig_per_class(cv)
    fig_fold_spread(cv)
    fig_paired(cv)
    fig_rand_control()
