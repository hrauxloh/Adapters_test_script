"""Adds 3 composite "emotion cluster" indicator columns to a
classify_usvsthem.py output (debate_usvsthem_scores.csv), based on the
hierarchical clustering in usvsthem_emotion_clustering.py. Each indicator
is just the mean of its cluster's emotion_prob_* columns per row -- a
cheap, interpretable way to carry a validated cluster's combined signal
forward instead of several separate, correlated emotion columns.

The cluster membership below is NOT re-derived from the data each run --
it's the specific 3-cluster result already chosen and confirmed against
real debate data (see the README): Cluster 1 ("hostility": Anger,
Contempt, Disgust, Fear) individually outperforms usvsthem_score itself
at predicting polarization_probability (r~0.50-0.56 each vs usvsthem_
score's r=0.32), which is why it's the one worth carrying forward as its
own feature. Cluster 2 and Cluster 3 are included for completeness but
Cluster 2 in particular has no coherent shared theme (see the README) --
treat its indicator as a convenience column, not a validated construct.

If you re-run usvsthem_emotion_clustering.py on different/more data and
get a different grouping, update CLUSTER_DEFINITIONS below to match --
this file intentionally doesn't re-cluster on every run, so the indicator
columns stay stable and comparable across re-deployments rather than
silently shifting if a later clustering run gives a slightly different
split.

Menu of what happens when you run this file, in order:
  1. Load --input (debate_usvsthem_scores.csv).
  2. For each cluster in CLUSTER_DEFINITIONS, add a column (the mean of
     that cluster's emotion_prob_* columns per row).
  3. Print each new column's mean/std for a quick sanity check.
  4. Save everything (original columns + the 3 new ones) to --output.

Usage:
    python build_emotion_cluster_indicators.py --input debate_usvsthem_scores.csv --output debate_usvsthem_scores_with_clusters.csv
"""

import argparse

import pandas as pd

CLUSTER_DEFINITIONS = {
    "emotion_cluster1_hostility": ["Anger", "Contempt", "Disgust", "Fear"],
    "emotion_cluster2_mixed": ["Gratitude", "Guilt", "Happiness", "Pride", "Relief", "Sadness"],
    "emotion_cluster3_other_positive": ["Hope", "Sympathy"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="debate_usvsthem_scores.csv")
    parser.add_argument("--output", default=None, help="Defaults to <input> with _with_clusters suffix")
    args = parser.parse_args()
    output = args.output or args.input.rsplit(".csv", 1)[0] + "_with_clusters.csv"

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} row(s) from {args.input}.")

    for indicator_name, emotions in CLUSTER_DEFINITIONS.items():
        columns = [f"emotion_prob_{e}" for e in emotions]
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise SystemExit(f"--input is missing column(s) {missing} needed for {indicator_name} -- was it produced by classify_usvsthem.py?")
        df[indicator_name] = df[columns].mean(axis=1)
        print(f"{indicator_name} = mean({', '.join(emotions)}): mean={df[indicator_name].mean():.4f}, std={df[indicator_name].std():.4f}")

    df.to_csv(output, index=False)
    print(f"\nSaved {len(df)} row(s) (original columns + {len(CLUSTER_DEFINITIONS)} cluster indicator(s)) to {output}")


if __name__ == "__main__":
    main()
