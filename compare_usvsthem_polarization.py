"""Compares the standalone us-vs-them RoBERTa model's predictions
(classify_usvsthem.py's debate_usvsthem_scores.csv) against the
polarization classifier's predictions (classify_debates.py's /
classify_edited_dataset.py's debate_predictions*.csv), at the item level:
does "us vs. them" framing overlap with what the polarization model
already picks up, or is it capturing something different?

Both files are expected to come from the SAME --input run through both
classify_usvsthem.py and classify_debates.py/classify_edited_dataset.py,
so they share DebateId + ItemId per row -- merged on those two columns, not
assumed to already be in the same row order.

Menu of what happens when you run this file, in order:
  1. Load --usvsthem-file and --predictions-file, merge on
     (DebateId, ItemId).
  2. Report how many rows matched (and how many didn't, from either side
     -- a mismatch usually means the two files came from different
     --input runs).
  3. Pearson correlation between usvsthem_score and
     polarization_probability (item level), with its p-value and a plain-
     language read of the correlation's size.
  4. Mean usvsthem_score split by predicted_polarization (0 vs 1) -- a
     simpler, more interpretable view of the same relationship.
  5. Save a scatter plot of usvsthem_score vs polarization_probability.

Usage:
    python compare_usvsthem_polarization.py --usvsthem-file debate_usvsthem_scores.csv --predictions-file debate_predictions_values_edited.csv
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usvsthem-file", default="debate_usvsthem_scores.csv")
    parser.add_argument("--predictions-file", required=True, help="e.g. debate_predictions_values_edited.csv")
    parser.add_argument("--output-plot", default="usvsthem_vs_polarization.png")
    args = parser.parse_args()

    usvsthem = pd.read_csv(args.usvsthem_file)
    predictions = pd.read_csv(args.predictions_file)

    for col in ("DebateId", "ItemId"):
        if col not in usvsthem.columns:
            raise SystemExit(f"--usvsthem-file is missing a '{col}' column -- was it produced by classify_usvsthem.py?")
        if col not in predictions.columns:
            raise SystemExit(f"--predictions-file is missing a '{col}' column -- was it produced by classify_debates.py/classify_edited_dataset.py?")

    merged = usvsthem[["DebateId", "ItemId", "usvsthem_score"]].merge(
        predictions[["DebateId", "ItemId", "polarization_probability", "predicted_polarization"]],
        on=["DebateId", "ItemId"], how="inner",
    )
    print(
        f"{len(usvsthem)} row(s) in {args.usvsthem_file}, {len(predictions)} row(s) in {args.predictions_file}, "
        f"{len(merged)} row(s) matched on (DebateId, ItemId)."
    )
    if len(merged) < len(usvsthem) or len(merged) < len(predictions):
        print(
            f"WARNING: {len(usvsthem) - len(merged)} row(s) from --usvsthem-file and "
            f"{len(predictions) - len(merged)} row(s) from --predictions-file didn't match -- "
            "make sure both came from the same --input run."
        )

    if merged.empty:
        raise SystemExit("No matching rows -- nothing to compare.")

    r, p = pearsonr(merged["usvsthem_score"], merged["polarization_probability"])
    print(f"\nPearson correlation (usvsthem_score vs polarization_probability): r={r:.4f}, p={p:.4g}")
    abs_r = abs(r)
    if abs_r < 0.1:
        print("-> Negligible relationship: the two signals look largely independent.")
    elif abs_r < 0.3:
        print("-> Weak relationship: some overlap, but mostly capturing different things.")
    elif abs_r < 0.5:
        print("-> Moderate relationship: meaningful overlap.")
    else:
        print("-> Strong relationship: largely redundant signals.")

    by_group = merged.groupby("predicted_polarization")["usvsthem_score"].agg(["mean", "std", "size"])
    print("\nMean usvsthem_score by predicted_polarization:")
    print(by_group.to_string())

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(merged["polarization_probability"], merged["usvsthem_score"], alpha=0.3, s=10)
    ax.set_xlabel("polarization_probability")
    ax.set_ylabel("usvsthem_score")
    ax.set_title(f"usvsthem_score vs polarization_probability (n={len(merged)}, r={r:.3f})")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"\nSaved scatter plot to {args.output_plot}")


if __name__ == "__main__":
    main()
