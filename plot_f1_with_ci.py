"""
WHAT THIS SCRIPT DOES
----------------------
Draws each ablation model's F1 score as a single dot, with its 95%
bootstrap confidence interval as an error bar, along an x-axis grouped
into baseline / single adapter / pair / full-model categories — a
publication-style companion to summarize_ablation_results.py's bar chart,
showing the uncertainty behind each number instead of just the point
estimate.

It doesn't train or evaluate anything itself. It reuses each model's
saved per-example test predictions (from evaluate_test.py run with
--save-predictions) and the same MODELS list summarize_ablation_results.py
uses, so the two plots stay consistent with each other.

The confidence interval here only captures test-set sampling variability
(how much F1 would wobble if the test set had been sampled slightly
differently) — not variability from different random training seeds, since
each model here was only trained once. See bootstrap_compare.py for the
same underlying method applied to pairwise F1 differences instead of
single-model scores.

Menu of what happens when you run this file, in order:
  1. Load every model's saved test-set predictions.
  2. Resample the test set thousands of times to build a distribution of
     each model's F1 score, and take its 95% interval.
  3. Plot one dot + error bar per model, grouped and color-coded by
     category, and save it as PNG (for slides/drafts) and PDF (vector,
     for a paper/LaTeX).

Usage:
    python plot_f1_with_ci.py --drive-root /content/drive/MyDrive
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from summarize_ablation_results import CATEGORY_COLORS, MODELS


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def bootstrap_f1(labels_boot, preds_boot):
    tp = ((preds_boot == 1) & (labels_boot == 1)).sum(axis=1)
    fp = ((preds_boot == 1) & (labels_boot == 0)).sum(axis=1)
    fn = ((preds_boot == 0) & (labels_boot == 1)).sum(axis=1)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-plot", default="ablation_f1_ci_plot")
    args = parser.parse_args()

    # ---- Load each model's saved predictions ----
    rows = []
    reference_labels = None
    for name, label, category, output_dir in MODELS:
        path = f"{args.drive_root}/{output_dir}/test_predictions.npz"
        try:
            data = np.load(path)
        except FileNotFoundError:
            print(f"Skipping '{name}': no test_predictions.npz at {path} yet.")
            continue
        if reference_labels is None:
            reference_labels = data["labels"]
        rows.append({"name": name, "label": label, "category": category, "preds": data["preds"]})

    if not rows:
        print("No predictions found yet — run evaluate_test.py with --save-predictions first.")
        return

    labels = reference_labels
    n = len(labels)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    labels_boot = labels[idx]

    # ---- Bootstrap each model's own F1 distribution ----
    for row in rows:
        preds = row["preds"]
        row["f1"] = bootstrap_f1(labels[None, :], preds[None, :])[0]
        f1_boot = bootstrap_f1(labels_boot, preds[idx])
        row["ci_low"], row["ci_high"] = np.percentile(f1_boot, [2.5, 97.5])

    # ---- Plot ----
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Leave a visual gap between categories by spacing x positions instead
    # of using consecutive integers.
    x_positions = []
    x = 0
    prev_category = None
    for row in rows:
        if prev_category is not None and row["category"] != prev_category:
            x += 1.5  # extra gap when the category changes
        else:
            x += 1
        x_positions.append(x)
        prev_category = row["category"]

    for x_pos, row in zip(x_positions, rows):
        color = CATEGORY_COLORS[row["category"]]
        lower_err = row["f1"] - row["ci_low"]
        upper_err = row["ci_high"] - row["f1"]
        ax.errorbar(
            x_pos, row["f1"], yerr=[[lower_err], [upper_err]],
            fmt="o", color=color, ecolor=color, elinewidth=1.6, capsize=5,
            markersize=9, markeredgecolor="#222222", markeredgewidth=0.8, zorder=3,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([row["label"] for row in rows], fontsize=9.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score (test set)", fontsize=12)
    ax.set_title("Polarization detection: F1 by adapter combination (95% bootstrap CI)", fontsize=13, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=10)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=CATEGORY_COLORS[c],
                   markeredgecolor="#222222", markersize=9)
        for c in ["baseline", "single", "pair", "full"]
    ]
    ax.legend(
        legend_handles, ["Baseline", "Single adapter", "Pair", "Full (three adapters)"],
        loc="upper left", frameon=False, fontsize=9.5,
    )

    fig.tight_layout()
    fig.savefig(f"{args.output_plot}.png", dpi=300)
    fig.savefig(f"{args.output_plot}.pdf")
    print(f"Saved plot to {args.output_plot}.png and {args.output_plot}.pdf")


if __name__ == "__main__":
    main()
