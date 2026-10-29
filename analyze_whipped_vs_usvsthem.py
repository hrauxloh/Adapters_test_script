"""Flips the relationship analyze_item_level_logit.py tests: instead of
asking "does Whipped predict predicted_polarization", this asks "does
Whipped predict the us-vs-them score / emotion cluster indicators" --
Whipped stays the single X, but the Y is each of usvsthem_score,
emotion_cluster1_hostility, emotion_cluster2_mixed, and
emotion_cluster3_other_positive in turn (classify_usvsthem.py's output,
run through build_emotion_cluster_indicators.py first).

Since these Y's are continuous (0..1 averages of model probabilities),
not binary, this fits cluster-robust OLS (Y ~ Whipped), not logit --
same cluster-by-DebateId robust covariance as analyze_item_level_logit.py,
just a different model family because the outcome is a different kind of
variable. The coefficient on Whipped is directly the gap in that Y's mean
between whipped and non-whipped items (OLS on a single binary predictor
is exactly a difference-in-means test with cluster-robust SEs).

Reuses load_item_level from analyze_item_level_logit.py to build Whipped
the same way (so this stays consistent if that logic ever changes), then
merges in the usvsthem/emotion-cluster columns on (DebateId, ItemId).

Menu of what happens when you run this file, in order:
  1. Load --predictions/--features (build Whipped, same as
     analyze_item_level_logit.py) and --usvsthem-file (usvsthem_score +
     the 3 cluster indicators), merge on (DebateId, ItemId).
  2. Fit cluster-robust OLS Y ~ Whipped for each of the 4 Y's.
  3. Print one comparison table (coefficient / std err / p-value / R²),
     sorted by R2 descending.
  4. Save a 2x2 grid of jittered strip plots (Y vs Whipped, one panel per
     Y) so the group differences are visible, not just numeric.

Usage:
    python analyze_whipped_vs_usvsthem.py --predictions debate_predictions_values_edited.csv --features debate_features_edited.csv --usvsthem-file debate_usvsthem_scores_with_clusters.csv
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analyze_item_level_logit import load_item_level

Y_VARIABLES = [
    "usvsthem_score",
    "emotion_cluster1_hostility",
    "emotion_cluster2_mixed",
    "emotion_cluster3_other_positive",
]


def fit_clustered_ols(data, formula, group_col="DebateId"):
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data[group_col]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="e.g. debate_predictions_values_edited.csv (item-level)")
    parser.add_argument("--features", required=True, help="e.g. debate_features_edited.csv (debate-level, for Gov_Cohesion/Opp_Cohension -> Whipped)")
    parser.add_argument("--usvsthem-file", required=True, help="build_emotion_cluster_indicators.py's output")
    parser.add_argument("--output-plot", default="whipped_vs_usvsthem.png")
    args = parser.parse_args()

    data, _ = load_item_level(args.predictions, args.features)
    if "Whipped" not in data.columns:
        raise SystemExit("Whipped couldn't be built -- --features is missing Gov_Cohesion/Opp_Cohension.")

    usvsthem = pd.read_csv(args.usvsthem_file)
    missing_y = [y for y in Y_VARIABLES if y not in usvsthem.columns]
    if missing_y:
        raise SystemExit(
            f"--usvsthem-file is missing {missing_y} -- run build_emotion_cluster_indicators.py on it first."
        )

    merged = data[["DebateId", "ItemId", "Whipped"]].merge(
        usvsthem[["DebateId", "ItemId"] + Y_VARIABLES], on=["DebateId", "ItemId"], how="inner"
    )
    print(
        f"{len(data)} row(s) with Whipped, {len(usvsthem)} row(s) in --usvsthem-file, "
        f"{len(merged)} row(s) matched on (DebateId, ItemId)."
    )
    if merged.empty:
        raise SystemExit("No matching rows -- check --predictions/--usvsthem-file came from the same --input run.")

    rows = []
    for y in Y_VARIABLES:
        subset = merged[["DebateId", "Whipped", y]].dropna()
        result = fit_clustered_ols(subset, f"{y} ~ Whipped")
        coef = result.params["Whipped[T.True]"]
        se = result.bse["Whipped[T.True]"]
        p = result.pvalues["Whipped[T.True]"]
        rows.append({
            "y_variable": y,
            "n_items": int(result.nobs),
            "n_debates": subset["DebateId"].nunique(),
            "whipped_coef": coef,
            "std_err": se,
            "p_value": p,
            "r_squared": result.rsquared,
        })

    comparison = pd.DataFrame(rows).sort_values("r_squared", ascending=False)
    print("\n" + "=" * 70)
    print("Whipped -> Y (cluster-robust OLS by DebateId), sorted by R2 descending")
    print("=" * 70)
    print(comparison.to_string(index=False, formatters={
        "whipped_coef": "{:.4f}".format, "std_err": "{:.4f}".format,
        "p_value": "{:.4g}".format, "r_squared": "{:.4f}".format,
    }))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    rng = np.random.default_rng(0)
    for ax, y in zip(axes.flat, Y_VARIABLES):
        subset = merged[["Whipped", y]].dropna()
        jitter = rng.uniform(-0.08, 0.08, size=len(subset))
        x = subset["Whipped"].astype(int) + jitter
        ax.scatter(x, subset[y], alpha=0.15, s=8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Not whipped", "Whipped"])
        ax.set_title(y)
        ax.set_ylabel(y)
    fig.suptitle("Whipped vs. each Y variable (jittered)")
    fig.tight_layout()
    fig.savefig(args.output_plot, dpi=150)
    print(f"\nSaved plot to {args.output_plot}")


if __name__ == "__main__":
    main()
