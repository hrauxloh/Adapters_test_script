"""
WHAT THIS SCRIPT DOES
----------------------
Pulls together the results from every ablation model you've trained (the
baseline, all three single-adapter models, all three pairs, and the full
three-adapter model — everything except the unrelated-adapter confound
control, which isn't built yet) into one results table and one chart,
ready to drop into a report or paper.

It doesn't train or evaluate anything itself — it only reads the
summary.md file each model's evaluate_test.py run already produced.

Menu of what happens when you run this file, in order:
  1. Read the final test metrics out of each model's summary.md.
  2. Print a results table (and save it as a CSV).
  3. Draw a bar chart of F1 score across all models and save it as both a
     PNG (for slides/drafts) and a PDF (vector, for a paper/LaTeX).

Usage:
    python summarize_ablation_results.py --drive-root /content/drive/MyDrive
"""

import argparse
import re

import pandas as pd
import matplotlib.pyplot as plt

# name, human-readable label, category (for coloring), output-dir suffix
MODELS = [
    ("baseline", "Baseline\n(no adapters)", "baseline", "output_baseline"),
    ("group_only", "Group\nonly", "single", "output_group_only"),
    ("valence_only", "Valence\nonly", "single", "output_valence_only"),
    ("emotion_only", "Emotion\nonly", "single", "output_emotion_only"),
    ("valence_emotion", "Valence +\nEmotion", "pair", "output_custom"),
    ("valence_group", "Valence +\nGroup", "pair", "output_valence_group"),
    ("emotion_group", "Emotion +\nGroup", "pair", "output_emotion_group"),
    ("full", "All three\n(full model)", "full", "output_full"),
]

CATEGORY_COLORS = {
    "baseline": "#B5BAB0",
    "single": "#7C93A8",
    "pair": "#3C6E85",
    "full": "#1B3A5C",
}


def extract(pattern, text):
    match = re.search(pattern, text)
    return float(match.group(1)) if match else None


def load_metrics(summary_path):
    with open(summary_path) as f:
        text = f.read()
    return {
        "accuracy": extract(r"- accuracy: ([\d.]+)", text),
        "precision": extract(r"- precision: ([\d.]+)", text),
        "recall": extract(r"- recall: ([\d.]+)", text),
        "f1": extract(r"- f1: ([\d.]+)", text),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--output-csv", default="ablation_results.csv")
    parser.add_argument("--output-plot", default="ablation_f1_plot")  # extension added per format
    args = parser.parse_args()

    rows = []
    for name, label, category, output_dir in MODELS:
        summary_path = f"{args.drive_root}/{output_dir}/summary.md"
        try:
            metrics = load_metrics(summary_path)
        except FileNotFoundError:
            print(f"Skipping '{name}': no summary.md found at {summary_path} yet.")
            continue
        rows.append({"model": name, "label": label, "category": category, **metrics})

    if not rows:
        print("No results found yet — run the training scripts first.")
        return

    results = pd.DataFrame(rows)
    pd.set_option("display.width", 100)
    print(results[["model", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
    results.to_csv(args.output_csv, index=False)
    print(f"\nSaved results table to {args.output_csv}")

    # ---- Plot ----
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["axes.edgecolor"] = "#444444"
    plt.rcParams["axes.linewidth"] = 0.8

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = [CATEGORY_COLORS[c] for c in results["category"]]
    bars = ax.bar(results["label"], results["f1"], color=colors, width=0.62, zorder=3)

    for bar, value in zip(bars, results["f1"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}",
            ha="center", va="bottom", fontsize=10.5, color="#222222",
        )

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1 score (test set)", fontsize=12)
    ax.set_title("Polarization detection: F1 by adapter combination", fontsize=13.5, pad=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#E0E0E0", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.tick_params(axis="y", labelsize=10)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c])
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
