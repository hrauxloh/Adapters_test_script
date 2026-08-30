"""
WHAT THIS SCRIPT DOES
----------------------
Answers the question summarize_ablation_results.py's table can't: are these
F1 differences real, or could they just as easily have come from which
particular test examples happened to be sampled?

It doesn't train or evaluate anything itself. It reuses the per-example
predictions each model already made on the test set (saved by
evaluate_test.py when run with --save-predictions), and repeatedly
resamples the test set — with replacement, thousands of times — to see how
much each model's F1, and the *difference* between any two models' F1,
wobbles around just from test-set sampling luck. That spread becomes a 95%
confidence interval: if a difference's interval doesn't include zero, the
two models are behaving distinguishably on this test set.

This only captures test-set sampling variability — not variability from
different random training seeds (that would need retraining each model
several times, which this project's default single-run-per-model setup
doesn't do).

Menu of what happens when you run this file, in order:
  1. Load every model's saved test-set predictions.
  2. Resample the test set thousands of times, recomputing F1 for every
     model each time.
  3. For every pair of models, turn those resampled scores into a
     95% confidence interval on the F1 difference, and print/save the
     full comparison table.
  4. Print a short "highlights" section calling out the comparisons your
     research question actually cares about (each adapter vs. baseline,
     each pair vs. its best single component, full model vs. best pair).

Usage:
    python evaluate_test.py --output-dir output_baseline --test-file eng_test.csv --save-predictions
    (repeat --save-predictions for every model directory, then:)
    python bootstrap_compare.py --drive-root /content/drive/MyDrive
"""

import argparse
from itertools import combinations

import numpy as np
import pandas as pd

from summarize_ablation_results import MODELS

# Which two single-adapter models make up each pair — used for the
# "pair vs. its best single component" highlight below.
PAIR_COMPONENTS = {
    "valence_emotion": ("valence_only", "emotion_only"),
    "valence_group": ("valence_only", "group_only"),
    "emotion_group": ("emotion_only", "group_only"),
}


def safe_div(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def bootstrap_f1(labels_boot, preds_boot):
    # Vectorized F1 across every resample at once: labels_boot/preds_boot
    # are [n_boot, n_examples] arrays sharing the same resampled indices.
    tp = ((preds_boot == 1) & (labels_boot == 1)).sum(axis=1)
    fp = ((preds_boot == 1) & (labels_boot == 0)).sum(axis=1)
    fn = ((preds_boot == 0) & (labels_boot == 1)).sum(axis=1)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return safe_div(2 * precision * recall, precision + recall)


def load_predictions(drive_root):
    loaded = {}
    reference_labels = None
    for name, _label, _category, output_dir in MODELS:
        path = f"{drive_root}/{output_dir}/test_predictions.npz"
        try:
            data = np.load(path)
        except FileNotFoundError:
            print(f"Skipping '{name}': no test_predictions.npz at {path} yet "
                  f"(re-run evaluate_test.py with --save-predictions for this model).")
            continue
        if reference_labels is None:
            reference_labels = data["labels"]
        elif not np.array_equal(data["labels"], reference_labels):
            print(f"Skipping '{name}': its saved labels don't match the other "
                  f"models' — was it evaluated against a different --test-file?")
            continue
        loaded[name] = data["preds"]
    return reference_labels, loaded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive")
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-csv", default="bootstrap_comparison.csv")
    args = parser.parse_args()

    labels, preds_by_model = load_predictions(args.drive_root)
    if len(preds_by_model) < 2:
        print("\nNeed at least two models' predictions to compare. Nothing to do yet.")
        return

    n = len(labels)
    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n, size=(args.n_boot, n))
    labels_boot = labels[idx]

    f1_boot = {}
    f1_observed = {}
    for name, preds in preds_by_model.items():
        preds_boot = preds[idx]
        f1_boot[name] = bootstrap_f1(labels_boot, preds_boot)
        f1_observed[name] = bootstrap_f1(labels[None, :], preds[None, :])[0]

    rows = []
    for a, b in combinations(preds_by_model.keys(), 2):
        diff_boot = f1_boot[a] - f1_boot[b]
        ci_low, ci_high = np.percentile(diff_boot, [2.5, 97.5])
        rows.append({
            "model_a": a, "model_b": b,
            "f1_a": f1_observed[a], "f1_b": f1_observed[b],
            "diff (a-b)": f1_observed[a] - f1_observed[b],
            "ci_low": ci_low, "ci_high": ci_high,
            "distinguishable": not (ci_low <= 0 <= ci_high),
        })

    results = pd.DataFrame(rows).sort_values("diff (a-b)", key=abs, ascending=False)
    pd.set_option("display.width", 120)
    print(f"\nAll pairwise comparisons ({len(rows)} pairs, {args.n_boot} resamples, 95% CI):")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    results.to_csv(args.output_csv, index=False)
    print(f"\nSaved full comparison table to {args.output_csv}")
    print("\nNote: with this many pairwise comparisons, some 95% intervals will exclude zero by "
          "chance alone even if nothing is really different — treat the highlights below, which "
          "are the specific comparisons the research question cares about, as more meaningful "
          "than scanning the full table for any significant-looking row.")

    # ---- Highlights ----
    print(f"\n{'=' * 60}\nHighlights\n{'=' * 60}")

    def report(a, b, note):
        if a not in preds_by_model or b not in preds_by_model:
            return
        row = results[
            ((results["model_a"] == a) & (results["model_b"] == b))
            | ((results["model_a"] == b) & (results["model_b"] == a))
        ].iloc[0]
        sign = 1 if row["model_a"] == a else -1
        diff = sign * row["diff (a-b)"]
        ci_low, ci_high = sorted([sign * row["ci_low"], sign * row["ci_high"]])
        verdict = "distinguishable from zero" if row["distinguishable"] else "NOT distinguishable from zero"
        print(f"{note}: F1 diff ({a} - {b}) = {diff:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}] -> {verdict}")

    if "baseline" in preds_by_model:
        for name, _label, category, _output_dir in MODELS:
            if category == "single":
                report(name, "baseline", f"{name} vs. baseline")

    for pair_name, (comp_a, comp_b) in PAIR_COMPONENTS.items():
        if pair_name not in preds_by_model:
            continue
        best_component = comp_a if f1_observed.get(comp_a, -1) >= f1_observed.get(comp_b, -1) else comp_b
        report(pair_name, best_component, f"{pair_name} vs. its best single component ({best_component})")

    if "full" in preds_by_model:
        pair_names = [n for n in PAIR_COMPONENTS if n in preds_by_model]
        if pair_names:
            best_pair = max(pair_names, key=lambda n: f1_observed[n])
            report("full", best_pair, f"full model vs. best pair ({best_pair})")


if __name__ == "__main__":
    main()
