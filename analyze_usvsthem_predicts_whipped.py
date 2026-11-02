"""The other direction from analyze_whipped_vs_usvsthem.py: there, Whipped
was the X and usvsthem_score/the 3 emotion cluster indicators were each a
Y, fit with OLS (continuous outcome). Here, Whipped is the Y and each of
those same 4 variables is tried as the sole X, one at a time, fit with
logistic regression (Whipped is a yes/no label).

THIS IS FIT AT THE DEBATE LEVEL, NOT THE ITEM LEVEL, and that's not a
stylistic choice -- Whipped has zero variation within a debate (every item
in a debate shares the same Whipped value), so item-level DebateId-clustered
logit was the wrong unit of analysis for this direction specifically: it
treats ~125 copies of the same debate-level label as semi-independent rows,
and if an item-level X happens to track debate-level tone closely (which
these emotion/us-vs-them scores do), the fit can (quasi-)separate --
"converged" but reporting an odds ratio in the billions with a confidence
interval spanning many orders of magnitude, confirmed happening on this
project's real data. Aggregating each debate's items down to one row (mean
X per debate, Whipped unchanged) and fitting an ordinary (non-clustered --
there's nothing left to cluster, every row already IS one debate) logit
fixes this: confirmed by testing both versions side by side on synthetic
data with a realistic (weak, noisy) effect size -- the item-level version
still gave a nonsense odds ratio, the debate-level version converged
cleanly to a plausible, if appropriately wide-CI'd (only ~157 debates to
work with), estimate.

THAT FIX ISN'T COMPLETE ON ITS OWN, THOUGH: with likely well under 20
debates actually Whipped out of ~157, logistic regression is fundamentally
fragile here regardless of unit of analysis -- a rare-events problem, the
same family as the sparse-cell issue this project already hit once before
(analyze_item_level_logit.py's near-zero-event thresholds). Confirmed by
testing: even a variable deliberately constructed to have NO relationship
to Whipped can occasionally produce an extreme odds ratio purely because so
few positive cases exist to constrain the fit. So each X is ALSO tested
with a Mann-Whitney U test (do whipped debates' X values rank higher/lower
than non-whipped debates', with a rank-biserial effect size in [-1, 1]) --
a non-parametric comparison that cannot blow up numerically no matter how
few debates are Whipped, serving as the trustworthy fallback whenever the
logit's odds ratio gets flagged as implausible.

Reuses load_item_level from analyze_item_level_logit.py to build Whipped
the same way that file does.

Menu of what happens when you run this file, in order:
  1. Load --predictions/--features (build Whipped) and --usvsthem-file
     (usvsthem_score + the 3 cluster indicators), merge on
     (DebateId, ItemId).
  2. Collapse to one row per debate: Whipped (unchanged, it's already
     constant within a debate) and the mean of each X across that
     debate's items.
  3. For each of the 4 X's: fit an ordinary univariate logit (Whipped ~ X)
     AND a Mann-Whitney U test (X's values, whipped debates vs not), on
     this debate-level table.
  4. Print one comparison table (pseudo-R2 / LLR p-value / odds ratio +
     95% CI from the logit, plus the Mann-Whitney p-value and
     rank-biserial effect size), sorted by pseudo-R2 descending, plus a
     warning for any odds ratio that looks implausible -- when that fires,
     read the Mann-Whitney columns instead, not the odds ratio.

Usage:
    python analyze_usvsthem_predicts_whipped.py --predictions debate_predictions_values_edited.csv --features debate_features_edited.csv --usvsthem-file debate_usvsthem_scores_with_clusters.csv
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import mannwhitneyu

from analyze_item_level_logit import load_item_level

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

    # Collapse to one row per debate -- Whipped is already constant within a
    # debate (take the first value), each X becomes that debate's mean.
    agg = {"Whipped": "first", **{x: "mean" for x in X_VARIABLES}}
    debate_level = merged.groupby("DebateId").agg(agg).reset_index()
    debate_level["Whipped"] = debate_level["Whipped"].astype(int)
    n_whipped = debate_level["Whipped"].sum()
    print(
        f"Collapsed to {len(debate_level)} debate(s) ({n_whipped} whipped, "
        f"{len(debate_level) - n_whipped} not) -- this is the actual sample size for this question, "
        "since Whipped doesn't vary within a debate."
    )

    rows = []
    for x in X_VARIABLES:
        subset = debate_level[["Whipped", x]].dropna()

        row = {"x_variable": x, "n_debates": len(subset)}

        try:
            result = smf.logit(f"Whipped ~ {x}", data=subset).fit(disp=0)
            if not result.mle_retvals.get("converged", True):
                print(f"  {x}: logit did NOT converge -- likely still separated even at debate level.")
            coef = result.params[x]
            ci_low, ci_high = result.conf_int().loc[x]
            row.update({
                "pseudo_r2": result.prsquared,
                "llr_p_value": result.llr_pvalue,
                "odds_ratio": np.exp(coef),
                "odds_ratio_2.5%": np.exp(ci_low),
                "odds_ratio_97.5%": np.exp(ci_high),
            })
        except Exception as exc:
            print(f"  {x}: logit failed to fit ({exc})")
            row.update({"pseudo_r2": np.nan, "llr_p_value": np.nan, "odds_ratio": np.nan, "odds_ratio_2.5%": np.nan, "odds_ratio_97.5%": np.nan})

        # Non-parametric companion: can't blow up numerically regardless of
        # how few debates are Whipped, since it only compares ranks rather
        # than fitting a parametric curve through the 0/1 boundary.
        whipped_vals = subset.loc[subset["Whipped"] == 1, x]
        not_whipped_vals = subset.loc[subset["Whipped"] == 0, x]
        u_stat, mw_p = mannwhitneyu(whipped_vals, not_whipped_vals, alternative="two-sided")
        row["mannwhitney_p"] = mw_p
        row["rank_biserial"] = 1 - (2 * u_stat) / (len(whipped_vals) * len(not_whipped_vals))

        rows.append(row)

    comparison = pd.DataFrame(rows).sort_values("pseudo_r2", ascending=False)
    print("\n" + "=" * 70)
    print("Whipped ~ X, one row per debate, sorted by pseudo-R2 descending")
    print("=" * 70)
    print(comparison.to_string(index=False, formatters={
        "pseudo_r2": "{:.4f}".format, "llr_p_value": "{:.4g}".format,
        "odds_ratio": "{:.4f}".format, "odds_ratio_2.5%": "{:.4f}".format, "odds_ratio_97.5%": "{:.4f}".format,
        "mannwhitney_p": "{:.4g}".format, "rank_biserial": "{:.4f}".format,
    }))
    print(
        "\n(odds_ratio: for a one-unit increase in a debate's mean X -- i.e. going from 0 to 1 on these "
        "0..1-scaled scores -- how much do the odds of that debate being Whipped change. n_debates is "
        "the real sample size for this question (~157, not ~19,627), so expect wide confidence "
        "intervals, especially since few debates are actually Whipped.\n"
        "rank_biserial: a non-parametric effect size in [-1, 1] -- 0 means whipped and non-whipped "
        "debates' X values are completely interleaved (no relationship), +-1 means every whipped "
        "debate's X is higher/lower than every non-whipped debate's. This CANNOT blow up numerically "
        "the way the odds ratio can, so trust it over the odds ratio whenever the warning below fires.)"
    )

    extreme = comparison[(comparison["odds_ratio"] > 20) | (comparison["odds_ratio"] < 0.05)]
    if not extreme.empty:
        print(
            f"\nWARNING: {', '.join(extreme['x_variable'])} show an implausibly extreme odds ratio "
            "(>20x or <0.05x) even at the debate level -- with likely well under 20 debates actually "
            "Whipped, near-separation can happen even for a variable with no real relationship, purely "
            "from how few positive cases exist. For these, read mannwhitney_p/rank_biserial instead of "
            "the odds ratio -- they answer the same question (is there a relationship) without the risk "
            "of this numerical blow-up."
        )


if __name__ == "__main__":
    main()
