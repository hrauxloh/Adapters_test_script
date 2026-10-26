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

RESUMABLE: each trial's result is appended to --results-file immediately
after that trial finishes, not just collected in memory and written at the
very end. On Colab in particular, a runtime disconnect mid-grid-search used
to throw away every trial that had already finished. Re-running the exact
same command now skips any (lr, loss_aux_weight) combination already
present in --results-file and only runs what's missing -- including the
case where every combination is already done, which just reprints the
summary without training anything.

Menu of what happens when you run this file, in order:
  1. Read --results-file if it exists, and skip any grid combination
     already recorded in it.
  2. Train once per remaining (learning rate, loss-aux-weight) combination,
     each a fresh model from scratch, appending its result to
     --results-file as soon as it finishes.
  3. Once every combination in the grid is accounted for (this run or a
     previous one), print a results table sorted best (lowest validation
     MAE) first.
  4. Save the winning combination to best_config.json, and print the exact
     command to retrain it with --evaluate-test for the final check.

Usage:
    python grid_search_usvsthem.py --train-file UsVsThem_train_public.csv --val-file UsVsThem_valid_public.csv
    (if interrupted, e.g. a Colab disconnect, just re-run the same command)
"""

import argparse
import itertools
import json
import os

import pandas as pd

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


def load_completed(results_file):
    if not os.path.exists(results_file):
        return pd.DataFrame(columns=["lr", "loss_aux_weight", "mae", "correlation"])
    return pd.read_csv(results_file)


def append_result(results_file, row):
    header_needed = not os.path.exists(results_file)
    pd.DataFrame([row]).to_csv(results_file, mode="a", header=header_needed, index=False)


def main():
    parser = build_arg_parser()
    parser.add_argument("--results-file", default="grid_search_results.csv", help="Appended to after every trial -- re-running the same command resumes from here instead of starting over.")
    args = parser.parse_args()

    all_combinations = list(itertools.product(LEARNING_RATES, LOSS_AUX_WEIGHTS))
    completed = load_completed(args.results_file)
    done = {(round(r.lr, 10), round(r.loss_aux_weight, 10)) for r in completed.itertuples()}
    remaining = [(lr, w) for lr, w in all_combinations if (round(lr, 10), round(w, 10)) not in done]

    if done:
        print(f"Found {len(done)} already-completed combination(s) in {args.results_file} -- skipping those.")
    print(f"Running {len(remaining)} of {len(all_combinations)} combination(s) this invocation.")

    for i, (lr, loss_aux_weight) in enumerate(remaining, 1):
        print("\n" + "=" * 70)
        print(f"Trial {i}/{len(remaining)} (remaining): lr={lr}, loss_aux_weight={loss_aux_weight}")
        print("=" * 70)
        trial_result = run_trial(args, lr, loss_aux_weight)
        val_metrics = trial_result["val"]
        row = {
            "lr": lr,
            "loss_aux_weight": loss_aux_weight,
            "mae": val_metrics["eval_mae"],
            "correlation": val_metrics["eval_correlation"],
        }
        append_result(args.results_file, row)
        print(f"Saved this trial's result to {args.results_file} (survives a disconnect from here on).")

    results = load_completed(args.results_file).sort_values("mae").to_dict("records")

    if len(results) < len(all_combinations):
        print(
            f"\nOnly {len(results)} of {len(all_combinations)} combination(s) are done -- "
            "re-run this same command to pick up where it left off. Not printing a final summary yet."
        )
        return

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
