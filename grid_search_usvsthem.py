"""Grid search over train_usvsthem_classifier.py's settings: tries every
combination of learning rate x auxiliary-emotion-loss weight (a full
Cartesian grid, not the one-setting-at-a-time staged search sweep.py uses
for the adapter-fusion models), since full RoBERTa fine-tuning is expensive
enough per run that it's worth comparing every combination directly rather
than assuming the settings don't interact.

Every trial is checked against the validation split only -- UsVsThem_test_
public.csv is never touched here, so the search can't quietly overfit its
choice to the data the final model gets judged on. Once the grid is done,
this prints the winning combination and the exact command to retrain it
with --evaluate-test for the one, final test-set check.

Menu of what happens when you run this file, in order:
  1. Train once per (learning rate, loss-aux-weight) combination in the
     grid below, each a fresh model from scratch.
  2. Record each trial's validation MAE and correlation.
  3. Print a results table sorted best (lowest validation MAE) first.
  4. Save the winning combination to best_config.json, and print the exact
     command to retrain it with --evaluate-test for the final check.

Usage:
    python grid_search_usvsthem.py --train-file UsVsThem_train_public.csv --val-file UsVsThem_valid_public.csv
"""

import argparse
import itertools
import json

from train_usvsthem_classifier import build_arg_parser, train_and_evaluate

LEARNING_RATES = [1e-5, 2e-5, 3e-5]
LOSS_AUX_WEIGHTS = [0.0, 0.5]


def run_trial(base_args, lr, loss_aux_weight):
    args = argparse.Namespace(**vars(base_args))
    args.lr = lr
    args.loss_aux_weight = loss_aux_weight
    args.save_artifacts = False  # throwaway trial runs -- only the final retrain (outside this script) saves a model
    args.evaluate_test = False  # grid search never touches the test set
    # Each trial gets its own checkpoint subdirectory -- otherwise every
    # trial writes into the same --output-dir and overwrites the previous
    # trial's (still-needed-mid-training) checkpoint.
    args.output_dir = f"{base_args.output_dir}/trial_lr{lr}_aux{loss_aux_weight}"
    return train_and_evaluate(args)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    results = []
    combinations = list(itertools.product(LEARNING_RATES, LOSS_AUX_WEIGHTS))
    print(f"Running {len(combinations)} combination(s): {len(LEARNING_RATES)} learning rate(s) x {len(LOSS_AUX_WEIGHTS)} loss-aux-weight(s).")

    for i, (lr, loss_aux_weight) in enumerate(combinations, 1):
        print("\n" + "=" * 70)
        print(f"Trial {i}/{len(combinations)}: lr={lr}, loss_aux_weight={loss_aux_weight}")
        print("=" * 70)
        trial_result = run_trial(args, lr, loss_aux_weight)
        val_metrics = trial_result["val"]
        results.append({
            "lr": lr,
            "loss_aux_weight": loss_aux_weight,
            "mae": val_metrics["eval_mae"],
            "correlation": val_metrics["eval_correlation"],
        })

    results.sort(key=lambda r: r["mae"])

    print("\n" + "=" * 70)
    print("Grid search results, best (lowest validation MAE) first:")
    print("=" * 70)
    print(f"{'lr':>10} {'loss_aux_weight':>16} {'mae':>10} {'correlation':>12}")
    for r in results:
        print(f"{r['lr']:>10} {r['loss_aux_weight']:>16} {r['mae']:>10.4f} {r['correlation']:>12.4f}")

    best = results[0]
    with open("best_config.json", "w") as f:
        json.dump(best, f, indent=2)
    print(f"\nBest combination saved to best_config.json: {best}")
    print(
        "\nTo retrain on the winning config with the one, final test-set check:\n"
        f"    python train_usvsthem_classifier.py --train-file {args.train_file} --val-file {args.val_file} "
        f"--test-file {args.test_file} --lr {best['lr']} --loss-aux-weight {best['loss_aux_weight']} "
        f"--evaluate-test --save-artifacts"
    )


if __name__ == "__main__":
    main()
