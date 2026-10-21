"""Extends a single-predictor model like:

    predicted_polarization ~ VoteType

to a multi-feature comparison, so other debate-level metadata can be
tested against (and alongside) VoteType, using the same statsmodels
Logit reporting style (coef/std err/z/p/CI, odds ratios).

TWO THINGS THIS FIXES vs. a plain one-off VoteType-only model:

1. CLUSTERING. predicted_polarization is measured per ITEM (span), but
   VoteType and the other features here are debate-level -- every item
   in the same debate carries an identical copy of them. Treating all
   items as independent observations (statsmodels' default "nonrobust"
   covariance) overstates how much evidence there actually is. Every fit
   here clusters standard errors by DebateId: it keeps every item (so
   you don't lose the power of the full item-level data), but calculates
   uncertainty as if the debate, not the item, were the unit of
   observation for anything measured at the debate level.

2. COLLINEARITY. Several of these "different" features are near-
   duplicates of each other, confirmed by direct correlation (checked
   against this project's own debate_features.csv): Party_Distance and
   All_Party_Distance correlate at 1.00 (exact duplicates); Gov_Cohesion
   correlates -0.98 with Cons_whip and Opp_Cohension correlates -0.95
   with Labour_whip (cohesion is just ~1 minus that party's own whip
   rate); Gov_Share and Opp_Share sum to ~1 almost always (near-perfect
   complements). Throwing all of them into one plain (unregularized)
   logit at once is numerically unstable -- coefficients and standard
   errors can blow up to nonsense (we saw std errors in the hundreds of
   thousands and odds ratios of infinity on a test run). So:
     - the single-feature comparison avoids this entirely (each feature
       is fit alone, so collinearity between features never enters);
     - the "readable" multivariable model below uses a deliberately
       pruned, less-collinear subset (drops All_Party_Distance,
       Gov_Cohesion, Opp_Cohension, Opp_Share, and FreeVoteDivisionShare
       -- the last is redundant with VoteType itself, both derived from
       the same label);
     - the cross-validated L1 (Lasso-style) logistic regression at the
       end uses every feature and handles the collinearity by shrinking
       redundant ones toward zero automatically, instead of needing them
       hand-pruned -- the more rigorous "which features actually survive"
       answer, at the cost of not giving classical p-values.

Menu of what happens when you run this file, in order:
  1. Load the item-level predictions (debate_predictions.csv) and the
     debate-level metadata (debate_features.csv), and broadcast each
     debate's metadata onto every one of its items via DebateId.
  2. Fit a cluster-robust univariate Logit for every candidate feature
     alone (including VoteType) and report each one's pseudo-R^2 and LLR
     p-value side by side, sorted best first -- a direct answer to "what
     else predicts as well as, or better than, VoteType."
  3. Fit one multivariable Logit on the pruned, non-redundant feature
     subset (cluster-robust) -- the coefficient table + odds ratios, in
     the same format as a single-predictor model, showing each feature's
     effect net of the others.
  4. Fit a cross-validated L1-penalized logistic regression on every
     feature together (standardized, VoteType one-hot encoded) -- the
     "prediction rate" check: cross-validated accuracy/ROC-AUC, and
     which features keep a nonzero coefficient once collinear ones are
     allowed to shrink.

Usage:
    python analyze_item_level_logit.py --predictions /content/drive/MyDrive/debate_predictions.csv --features /content/drive/MyDrive/debate_features.csv
"""

import argparse

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

NUMERIC_CANDIDATES = [
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
CATEGORICAL_CANDIDATES = ["VoteType"]

# Deliberately non-redundant subset for the readable multivariable model --
# see the module docstring for the measured correlations behind each drop.
PRUNED_NUMERIC_FEATURES = [
    "OverallRebellionRate", "Turnout", "Labour_whip", "Cons_whip", "Gov_Share",
    "Party_Distance", "NumDivisions",
]


def load_item_level(predictions_path, features_path):
    predictions = pd.read_csv(predictions_path)
    features = pd.read_csv(features_path)

    predictions["DebateId"] = predictions["DebateId"].astype(str).str.strip().str.upper()
    features["DebateId"] = features["DebateId"].astype(str).str.strip().str.upper()

    metadata_cols = [c for c in NUMERIC_CANDIDATES + CATEGORICAL_CANDIDATES if c in features.columns]
    merged = predictions.merge(features[["DebateId"] + metadata_cols], on="DebateId", how="inner")
    print(
        f"{len(merged)} item(s) across {merged['DebateId'].nunique()} debate(s) "
        f"matched to debate-level metadata (of {predictions['DebateId'].nunique()} debates in predictions)."
    )
    return merged, metadata_cols


def fit_clustered_logit(data, formula):
    model = smf.logit(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["DebateId"]}, disp=0)


def compare_single_predictors(data, candidates):
    rows = []
    for feature in candidates:
        subset = data[["predicted_polarization", "DebateId", feature]].dropna()
        if subset[feature].nunique() < 2:
            continue
        term = f"C({feature})" if data[feature].dtype == object else feature
        try:
            result = fit_clustered_logit(subset, f"predicted_polarization ~ {term}")
        except Exception as exc:
            print(f"  {feature}: failed to fit ({exc})")
            continue
        rows.append({
            "feature": feature,
            "n_items": int(result.nobs),
            "n_debates": subset["DebateId"].nunique(),
            "pseudo_r2": result.prsquared,
            "llr_p_value": result.llr_pvalue,
            "aic": result.aic,
        })
    return pd.DataFrame(rows).sort_values("pseudo_r2", ascending=False).reset_index(drop=True)


def print_pruned_multivariable_model(data, numeric_features=PRUNED_NUMERIC_FEATURES):
    all_features = numeric_features + CATEGORICAL_CANDIDATES
    available = [f for f in all_features if f in data.columns]
    subset = data[["predicted_polarization", "DebateId"] + available].dropna()

    terms = [f for f in available if f in numeric_features] + [
        f"C({c})" for c in available if c in CATEGORICAL_CANDIDATES
    ]
    formula = "predicted_polarization ~ " + " + ".join(terms)
    print(f"Formula: {formula}")

    result = fit_clustered_logit(subset, formula)
    print(result.summary())

    conf = result.conf_int()
    conf["odds_ratio"] = result.params
    conf.columns = ["2.5%", "97.5%", "odds_ratio"]
    print("\nOdds ratios:")
    print(np.exp(conf).to_string())


def cross_validated_l1_logit(data, numeric_features, categorical_features):
    all_features = numeric_features + categorical_features
    subset = data[["predicted_polarization"] + all_features].dropna()

    X_parts = [StandardScaler().fit_transform(subset[numeric_features].values)]
    feature_names = list(numeric_features)
    for cat in categorical_features:
        dummies = pd.get_dummies(subset[cat], prefix=cat, drop_first=True)
        X_parts.append(dummies.values.astype(float))
        feature_names += list(dummies.columns)
    X = np.hstack(X_parts)
    y = subset["predicted_polarization"].values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    model = LogisticRegressionCV(
        penalty="l1", solver="liblinear", cv=cv, scoring="roc_auc", max_iter=10_000, random_state=42
    ).fit(X, y)
    cv_auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")

    coefs = pd.DataFrame({"feature": feature_names, "coef": model.coef_[0]}).sort_values(
        "coef", key=lambda s: s.abs(), ascending=False
    )
    kept = coefs[coefs["coef"] != 0]
    return cv_auc, kept.reset_index(drop=True), len(subset)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="classify_debates.py's debate_predictions.csv (item-level)")
    parser.add_argument("--features", required=True, help="build_debate_features.py's debate_features.csv (debate-level)")
    parser.add_argument(
        "--only-features",
        default=None,
        help="Comma-separated numeric feature names to use instead of the full candidate list, "
        "e.g. --only-features Gov_Share,Opp_Share,Party_Distance,Gov_Cohesion,Opp_Cohension. "
        "Used for all three steps below -- note some feature combinations are known to be "
        "collinear (see module docstring), which can make the readable multivariable model unstable.",
    )
    args = parser.parse_args()

    data, available = load_item_level(args.predictions, args.features)
    if args.only_features:
        requested = [f.strip() for f in args.only_features.split(",")]
        missing = [f for f in requested if f not in available]
        if missing:
            print(f"WARNING: requested feature(s) not found in the data, skipping: {missing}")
        numeric_features = [f for f in requested if f in available]
    else:
        numeric_features = [c for c in NUMERIC_CANDIDATES if c in available]
    categorical_features = [c for c in CATEGORICAL_CANDIDATES if c in available]

    print("\n" + "=" * 70)
    print("Single-feature comparison (cluster-robust by DebateId)")
    print("=" * 70)
    comparison = compare_single_predictors(data, numeric_features + categorical_features)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("Readable multivariable model (cluster-robust)")
    print("=" * 70)
    print_pruned_multivariable_model(data, numeric_features=numeric_features if args.only_features else PRUNED_NUMERIC_FEATURES)

    print("\n" + "=" * 70)
    print("Cross-validated L1 logistic regression, every feature (prediction rate)")
    print("=" * 70)
    cv_auc, kept, n = cross_validated_l1_logit(data, numeric_features, categorical_features)
    print(f"n={n} items, cross-validated ROC-AUC per fold: {np.round(cv_auc, 3)}, mean={cv_auc.mean():.3f}")
    print("(0.5 = no better than chance, 1.0 = perfect separation)")
    if kept.empty:
        print("No features survived L1 shrinkage.")
    else:
        print("Features kept (nonzero coefficient):")
        print(kept.to_string(index=False))


if __name__ == "__main__":
    main()
