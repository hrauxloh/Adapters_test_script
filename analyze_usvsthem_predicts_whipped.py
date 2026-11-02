"""The other direction from analyze_whipped_vs_usvsthem.py: there, Whipped
was the X and usvsthem_score/the 3 emotion cluster indicators were each a
Y, fit with OLS (continuous outcome). Here, Whipped is the Y and each of
those same 4 variables is tried as the sole X, one at a time, fit with
logistic regression (Whipped is a yes/no label) -- the same single-
feature-comparison approach and cluster-robust-by-DebateId logit
analyze_item_level_logit.py uses for predicted_polarization, just with a
different binary outcome.

Reuses load_item_level and fit_clustered_logit from
analyze_item_level_logit.py directly, so this stays consistent with that
file if its logic ever changes.

Menu of what happens when you run this file, in order:
  1. Load --predictions/--features (build Whipped) and --usvsthem-file
     (usvsthem_score + the 3 cluster indicators), merge on
     (DebateId, ItemId).
  2. Fit a cluster-robust univariate logit, Whipped ~ X, for each of the
     4 X's in turn.
  3. Print one comparison table (pseudo-R2 / LLR p-value / AIC / odds
     ratio + 95% CI), sorted by pseudo-R2 descending -- same ranking
     logic as analyze_item_level_logit.py's single-feature comparison.

Usage:
    python analyze_usvsthem_predicts_whipped.py --predictions debate_predictions_values_edited.csv --features debate_features_edited.csv --usvsthem-file debate_usvsthem_scores_with_clusters.csv
"""

import argparse

import numpy as np
import pandas as pd

from analyze_item_level_logit import fit_clustered_logit, load_item_level

X_VARIABLES = [
    "usvsthem_score",
    "emotion_cluster1_hostility",
    "emotion_cluster2_mixed",
    "emotion_cluster3_other_positive",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="e.g. debate_predictions_values_edited.csv (item-level)")
    parser.add_argument("--features", required=True, help="e.g. debate_features_edited.csv (debate-level, for Gov_Cohesion/Opp_Cohension -> Whipped)")
    parser.add_argument("--usvsthem-file", required=True, help="build_emotion_cluster_indicators.py's output")
    args = parser.parse_args()

    data, _ = load_item_level(args.predictions, args.features)
    if "Whipped" not in data.columns:
        raise SystemExit("Whipped couldn't be built -- --features is missing Gov_Cohesion/Opp_Cohension.")

    usvsthem = pd.read_csv(args.usvsthem_file)
    missing_x = [x for x in X_VARIABLES if x not in usvsthem.columns]
    if missing_x:
        raise SystemExit(
            f"--usvsthem-file is missing {missing_x} -- run build_emotion_cluster_indicators.py on it first."
        )

    merged = data[["DebateId", "ItemId", "Whipped"]].merge(
        usvsthem[["DebateId", "ItemId"] + X_VARIABLES], on=["DebateId", "ItemId"], how="inner"
    )
    print(
        f"{len(data)} row(s) with Whipped, {len(usvsthem)} row(s) in --usvsthem-file, "
        f"{len(merged)} row(s) matched on (DebateId, ItemId)."
    )
    if merged.empty:
        raise SystemExit("No matching rows -- check --predictions/--usvsthem-file came from the same --input run.")

    # Logit needs a numeric 0/1 response, not a bool, for a reliably clean fit/summary.
    merged["Whipped"] = merged["Whipped"].astype(int)

    rows = []
    for x in X_VARIABLES:
        subset = merged[["DebateId", "Whipped", x]].dropna()
        try:
            result = fit_clustered_logit(subset, f"Whipped ~ {x}")
        except Exception as exc:
            print(f"  {x}: failed to fit ({exc})")
            continue

        coef = result.params[x]
        ci_low, ci_high = result.conf_int().loc[x]
        rows.append({
            "x_variable": x,
            "n_items": int(result.nobs),
            "n_debates": subset["DebateId"].nunique(),
            "pseudo_r2": result.prsquared,
            "llr_p_value": result.llr_pvalue,
            "aic": result.aic,
            "odds_ratio": np.exp(coef),
            "odds_ratio_2.5%": np.exp(ci_low),
            "odds_ratio_97.5%": np.exp(ci_high),
        })

    comparison = pd.DataFrame(rows).sort_values("pseudo_r2", ascending=False)
    print("\n" + "=" * 70)
    print("Whipped ~ X (cluster-robust logit by DebateId), sorted by pseudo-R2 descending")
    print("=" * 70)
    print(comparison.to_string(index=False, formatters={
        "pseudo_r2": "{:.4f}".format, "llr_p_value": "{:.4g}".format, "aic": "{:.1f}".format,
        "odds_ratio": "{:.4f}".format, "odds_ratio_2.5%": "{:.4f}".format, "odds_ratio_97.5%": "{:.4f}".format,
    }))
    print(
        "\n(odds_ratio here means: for a one-unit increase in X -- i.e. going from 0 to 1 on these "
        "0..1-scaled scores -- how much do the odds of a debate being Whipped change. An odds_ratio "
        "far from 1 with a 95% CI that excludes 1 is the sign of a real relationship; pseudo_r2 is "
        "still small on an absolute 0-1 scale even for a real effect, same as elsewhere in this project.)"
    )

    # Whipped is constant within a debate, and these X's are item-level but
    # often vary little within a debate too (they're driven by debate-level
    # tone/topic) -- if an X happens to line up closely with whether a
    # debate is Whipped, the fit can (quasi-)separate, the same failure mode
    # documented in analyze_item_level_logit.py's own module docstring: a
    # real fit that's formally "converged" but reports a nonsense odds
    # ratio and a CI spanning many orders of magnitude. Flag that plainly
    # rather than let an enormous number look like a strong, trustworthy
    # effect.
    extreme = comparison[(comparison["odds_ratio"] > 20) | (comparison["odds_ratio"] < 0.05)]
    if not extreme.empty:
        print(
            f"\nWARNING: {', '.join(extreme['x_variable'])} show implausibly extreme odds ratio(s) "
            "(>20x or <0.05x) -- likely quasi-separation (Whipped is debate-level constant; if this X "
            "barely varies within a debate too, the two can nearly perfectly align). Trust pseudo_r2 "
            "and the fact that a real relationship exists, not the literal odds ratio magnitude here."
        )


if __name__ == "__main__":
    main()
