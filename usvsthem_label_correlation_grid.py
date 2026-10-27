"""Full pairwise correlation grid across every label the us-vs-them model
predicts: usvsthem_score plus all 13 emotion_prob_* columns (classify_
usvsthem.py's output) -- 14 labels, 14x14 associations. This looks only at
the us-vs-them model's own outputs against each other (e.g. does Anger
co-occur with Contempt, is Gratitude the mirror image of Anger); it's not
a comparison against the polarization classifier -- see
compare_usvsthem_polarization.py for that.

Menu of what happens when you run this file, in order:
  1. Load --input (debate_usvsthem_scores.csv).
  2. Compute the Pearson correlation matrix across usvsthem_score and
     every emotion_prob_* column.
  3. Print the full matrix.
  4. Print the single strongest pair (excluding the diagonal) for a quick
     headline, plus the top --top-n pairs by |r|.
  5. Save a heatmap: diverging color scale (blue = positive, red =
     negative, centered at 0), every cell annotated with its r value.

Usage:
    python usvsthem_label_correlation_grid.py --input debate_usvsthem_scores.csv
"""

import argparse
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="debate_usvsthem_scores.csv")
    parser.add_argument("--top-n", type=int, default=10, help="How many of the strongest pairs to list")
    parser.add_argument("--output-plot", default="usvsthem_label_correlation_grid.png")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    label_columns = ["usvsthem_score"] + [c for c in df.columns if c.startswith("emotion_prob_")]
    missing = [c for c in label_columns if c not in df.columns]
    if missing:
        raise SystemExit(f"--input is missing expected column(s): {missing} -- was it produced by classify_usvsthem.py?")

    print(f"Loaded {len(df)} row(s) from {args.input}. Correlating {len(label_columns)} label(s): {label_columns}")

    # Short, readable labels for the plot/table (strip the emotion_prob_ prefix)
    short_names = {c: c.removeprefix("emotion_prob_") if c != "usvsthem_score" else "usvsthem_score" for c in label_columns}

    corr = df[label_columns].corr(method="pearson")
    corr_display = corr.rename(index=short_names, columns=short_names)

    print("\nFull correlation matrix:")
    print(corr_display.round(3).to_string())

    pairs = []
    for a, b in itertools.combinations(label_columns, 2):
        pairs.append({"label_a": short_names[a], "label_b": short_names[b], "r": corr.loc[a, b]})
    pairs_df = pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False)

    strongest = pairs_df.iloc[0]
    print(f"\nStrongest pair overall: {strongest['label_a']} <-> {strongest['label_b']}, r={strongest['r']:.4f}")

    print(f"\nTop {args.top_n} pair(s) by |r|:")
    print(pairs_df.head(args.top_n).to_string(index=False, formatters={"r": "{:.4f}".format}))

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr_display.values, vmin=-1, vmax=1, cmap="RdBu")
    labels = list(corr_display.columns)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = corr_display.values[i, j]
            text_color = "white" if abs(value) > 0.6 else "black"
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=7)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title(f"Correlation grid: usvsthem_score + 13 emotions (n={len(df)})")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"\nSaved heatmap to {args.output_plot}")


if __name__ == "__main__":
    main()
