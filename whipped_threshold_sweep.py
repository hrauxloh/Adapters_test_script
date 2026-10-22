"""Sweeps the cohesion threshold used to define "Whipped" and plots how well
each resulting definition predicts predicted_polarization alone (pseudo-R^2).

analyze_item_level_logit.py's Whipped column is hardcoded to the strictest
possible definition: Gov_Cohesion == 1 AND Opp_Cohension == 1 (every single
member on both sides voted with their party). This script answers "how much
does that strictness actually matter?" by relaxing it step by step and
re-fitting the same single-feature cluster-robust logit
(predicted_polarization ~ Whipped) at each threshold:

    ==1 (today's definition), >0.98, >0.97, >0.96, >0.95, >0.94, >0.93

-- i.e. "both sides' cohesion rate is strictly above this threshold," run in
decreasing order of strictness. Reuses load_item_level and fit_clustered_logit
from analyze_item_level_logit.py, so this file must sit next to it.

Usage:
    python whipped_threshold_sweep.py --predictions /content/drive/MyDrive/debate_predictions.csv --features /content/drive/MyDrive/debate_features.csv
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from analyze_item_level_logit import fit_clustered_logit, load_item_level

THRESHOLDS = [1.0, 0.98, 0.97, 0.96, 0.95, 0.94, 0.93]


def whipped_for_threshold(data, threshold):
    if threshold >= 1.0:
        return (data["Gov_Cohesion"] == 1) & (data["Opp_Cohension"] == 1)
    return (data["Gov_Cohesion"] > threshold) & (data["Opp_Cohension"] > threshold)


def threshold_label(threshold):
    return "==1" if threshold >= 1.0 else f">{threshold}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="classify_debates.py's debate_predictions.csv (item-level)")
    parser.add_argument("--features", required=True, help="build_debate_features.py's debate_features.csv (debate-level)")
    parser.add_argument("--output", default="whipped_threshold_sweep.png", help="Plot image path; a same-named .csv is also saved")
    args = parser.parse_args()

    data, _ = load_item_level(args.predictions, args.features)
    if "Gov_Cohesion" not in data.columns or "Opp_Cohension" not in data.columns:
        raise SystemExit("Gov_Cohesion/Opp_Cohension not found in the merged data -- can't sweep Whipped's threshold.")

    rows = []
    for threshold in THRESHOLDS:
        data["Whipped"] = whipped_for_threshold(data, threshold)
        subset = data[["predicted_polarization", "DebateId", "Whipped"]].dropna()
        if subset["Whipped"].nunique() < 2:
            print(f"threshold {threshold_label(threshold)}: Whipped has no variation at this threshold -- skipping.")
            continue

        result = fit_clustered_logit(subset, "predicted_polarization ~ Whipped")
        share_whipped = subset["Whipped"].mean()
        rows.append({
            "threshold": threshold,
            "pseudo_r2": result.prsquared,
            "aic": result.aic,
            "llr_p_value": result.llr_pvalue,
            "share_whipped": share_whipped,
            "n_items": int(result.nobs),
        })
        print(
            f"threshold {threshold_label(threshold)}: pseudo_r2={result.prsquared:.5f}, aic={result.aic:.1f}, "
            f"p={result.llr_pvalue:.2e}, share_whipped={share_whipped:.1%}, n_items={int(result.nobs)}"
        )

    if not rows:
        raise SystemExit("No threshold produced a fittable Whipped column -- nothing to plot.")

    summary = pd.DataFrame(rows)
    csv_path = args.output.rsplit(".", 1)[0] + ".csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved per-threshold results to {csv_path}")

    labels = [threshold_label(t) for t in summary["threshold"]]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(labels, summary["pseudo_r2"], marker="o")
    ax.set_xlabel('Cohesion threshold defining "Whipped" (decreasing strictness, left to right)')
    ax.set_ylabel("Pseudo R² (predicted_polarization ~ Whipped)")
    ax.set_title('How "Whipped"\'s predictive power changes with the cohesion threshold')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
