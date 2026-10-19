"""Checks which debate-level metadata correlates with, and predicts,
polarization -- for MeanPolarizationProb and PolarizedItemShare
separately, since they capture different things (average tone across a
debate vs. the share of spans that crossed the polarization threshold)
and may not answer the same way.

Feature set is deliberately pruned to avoid definitional redundancy:
TotalRebels/TotalVotesCast/TotalAye/TotalNo are left out because
OverallRebellionRate (= TotalRebels / TotalVotesCast) and Gov_Share/
Opp_Share are already derived directly from them -- including both the
ratio and its own components in one regression would be comparing a
feature against itself. The remaining features (Gov_Share, Opp_Share,
Party_Distance, Gov_Cohesion, Opp_Cohension, All_Party_Distance) are
still mutually correlated by construction (all drawn from the same vote
matrix) -- expected, not a bug, and part of why this uses Lasso/multiple
regression rather than one-at-a-time correlation as the real answer.

For each outcome:
  1. Univariate correlation (Pearson, for a roughly linear relationship,
     and Spearman, for a monotonic one) against every feature, one at a
     time, sorted by strength. Easy to read, but can't tell you whether
     a feature matters on its own or only because it's correlated with
     a different feature that actually matters.
  2. A multiple linear regression using every feature together
     (standardized first, so coefficients are comparable across features
     on different scales) -- answers "net of the other features, does
     this one still matter."
  3. A cross-validated Lasso regression -- the "prediction rate" check:
     how well do these features predict the outcome on data they
     weren't fit on (not just how well they fit the rows they were
     trained on), plus which features survive Lasso's shrinkage.

With well under 200 debates, treat all of this as exploratory -- small
differences in R^2 or which feature "wins" are not strongly reliable.

Usage:
    python analyze_debate_features.py --input /content/drive/MyDrive/debate_features.csv
"""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler

OUTCOMES = ["MeanPolarizationProb", "PolarizedItemShare"]

FEATURE_COLUMNS = [
    "OverallRebellionRate",
    "Turnout",
    "Labour_whip",
    "Cons_whip",
    "Gov_Share",
    "Opp_Share",
    "Party_Distance",
    "Gov_Cohesion",
    "Opp_Cohension",
    "All_Party_Distance",
    "NumDivisions",
    "FreeVoteDivisionShare",
]


def univariate_correlations(df, outcome, features):
    rows = []
    for feature in features:
        pair = df[[feature, outcome]].dropna()
        if len(pair) < 3 or pair[feature].nunique() < 2:
            continue
        pear_r, pear_p = pearsonr(pair[feature], pair[outcome])
        spear_r, spear_p = spearmanr(pair[feature], pair[outcome])
        rows.append({
            "feature": feature, "n": len(pair),
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_r": spear_r, "spearman_p": spear_p,
        })
    result = pd.DataFrame(rows).sort_values("pearson_r", key=lambda s: s.abs(), ascending=False)
    return result.reset_index(drop=True)


def multivariate_regression(df, outcome, features):
    data = df[features + [outcome]].dropna()
    X_scaled = StandardScaler().fit_transform(data[features].values)
    y = data[outcome].values

    model = LinearRegression().fit(X_scaled, y)
    r2 = model.score(X_scaled, y)
    coefs = pd.DataFrame({"feature": features, "standardized_coef": model.coef_}).sort_values(
        "standardized_coef", key=lambda s: s.abs(), ascending=False
    )
    return r2, coefs.reset_index(drop=True), len(data)


def cross_validated_lasso(df, outcome, features):
    data = df[features + [outcome]].dropna()
    X_scaled = StandardScaler().fit_transform(data[features].values)
    y = data[outcome].values

    n_splits = min(5, len(data))
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    lasso = LassoCV(cv=cv, random_state=42, max_iter=100_000).fit(X_scaled, y)
    cv_r2 = cross_val_score(lasso, X_scaled, y, cv=cv, scoring="r2")

    coefs = pd.DataFrame({"feature": features, "lasso_coef": lasso.coef_}).sort_values(
        "lasso_coef", key=lambda s: s.abs(), ascending=False
    )
    kept = coefs[coefs["lasso_coef"] != 0]
    return cv_r2, kept.reset_index(drop=True), len(data)


def analyze_outcome(df, outcome, features):
    print("=" * 70)
    print(f"Outcome: {outcome}")
    print("=" * 70)

    print("\n-- Univariate correlations (one feature at a time) --")
    corr = univariate_correlations(df, outcome, features)
    print(corr.to_string(index=False))

    print("\n-- Multiple regression (all features together, standardized) --")
    r2, coefs, n = multivariate_regression(df, outcome, features)
    print(f"n={n}, in-sample R^2={r2:.3f} (optimistic -- fit and evaluated on the same rows)")
    print(coefs.to_string(index=False))

    print("\n-- Cross-validated Lasso (out-of-sample prediction rate) --")
    cv_r2, kept, n = cross_validated_lasso(df, outcome, features)
    print(f"n={n}, cross-validated R^2 per fold: {np.round(cv_r2, 3)}, mean={cv_r2.mean():.3f}")
    if kept.empty:
        print("No features survived Lasso's shrinkage -- none predict this outcome well enough on their own to be kept.")
    else:
        print("Features Lasso kept (nonzero coefficient):")
        print(kept.to_string(index=False))
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="debate_features.csv from build_debate_features.py")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    available = [f for f in FEATURE_COLUMNS if f in df.columns]
    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
        print(f"Note: {missing} not found in {args.input} -- skipped.")

    for outcome in OUTCOMES:
        if outcome not in df.columns:
            print(f"Skipping {outcome}: not found in {args.input}")
            continue
        analyze_outcome(df, outcome, available)


if __name__ == "__main__":
    main()
