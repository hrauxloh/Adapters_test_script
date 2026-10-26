"""Multi-seed stability check for train_usvsthem_classifier.py: reruns a
handful of named configs (built on top of the grid search's winning
lr=2e-5/loss_aux_weight=0.5) across several seeds each, since a single run
per config isn't enough to tell whether one setting is actually better than
another or just got a lucky/unlucky draw -- see the real example from this
project's own grid search, where the exact same (lr, loss_aux_weight) scored
MAE 0.172 in the search and 0.195-0.204 on a same-settings retrain.

Each config changes exactly one or two regularization/stability knobs
relative to the baseline (which matches the grid search's winning run
exactly), so any difference in the mean can be attributed to that specific
change rather than several at once:
  - baseline:    freeze_layers=0, weight_decay=0.0,  warmup_ratio=0.0   (the grid search's winner as-is)
  - frozen:      freeze_layers=6, weight_decay=0.0,  warmup_ratio=0.0   (train only the top half of roberta-base)
  - regularized: freeze_layers=0, weight_decay=0.01, warmup_ratio=0.06  (standard transformer fine-tuning regularization)
  - combined:    freeze_layers=6, weight_decay=0.01, warmup_ratio=0.06

RESUMABLE the same way grid_search_usvsthem.py is: each (config, seed)
result is appended to --results-file immediately, and re-running the exact
same command skips whatever's already there -- a Colab disconnect partway
through doesn't lose completed runs.

Menu of what happens when you run this file, in order:
  1. Read --results-file if it exists, skip any (config, seed) pair
     already recorded.
  2. Train once per remaining (config, seed) pair, appending its result
     to --results-file as soon as it finishes.
  3. Once every (config, seed) pair is accounted for, print each config's
     mean +/- standard deviation of validation MAE and correlation across
     its seeds, sorted best (lowest mean MAE) first -- and flag whether
     the apparent best config's advantage is bigger than the noise its
     own seeds show, so a config doesn't get called "better" over a gap
     smaller than what seed-to-seed luck alone can produce.

Usage:
    python multi_seed_usvsthem.py --train-file UsVsThem_train_public.csv --val-file UsVsThem_valid_public.csv
    (if interrupted, e.g. a Colab disconnect, just re-run the same command)
"""

import argparse
import itertools
import os

import pandas as pd

from train_usvsthem_classifier import build_arg_parser, train_and_evaluate

# Fixed at the grid search's winner; each config below isolates one or two
# stability/regularization knobs on top of it.
BASE_LR = 2e-5
BASE_LOSS_AUX_WEIGHT = 0.5

CONFIGS = {
    "baseline":    {"freeze_layers": 0, "weight_decay": 0.0,  "warmup_ratio": 0.0},
    "frozen":      {"freeze_layers": 6, "weight_decay": 0.0,  "warmup_ratio": 0.0},
    "regularized": {"freeze_layers": 0, "weight_decay": 0.01, "warmup_ratio": 0.06},
    "combined":    {"freeze_layers": 6, "weight_decay": 0.01, "warmup_ratio": 0.06},
}
SEEDS = [42, 1, 7]


def run_trial(base_args, config_name, overrides, seed):
    args = argparse.Namespace(**vars(base_args))
    args.lr = BASE_LR
    args.loss_aux_weight = BASE_LOSS_AUX_WEIGHT
    for key, value in overrides.items():
        setattr(args, key, value)
    args.seed = seed
    args.save_artifacts = False  # throwaway stability-check runs
    args.evaluate_test = False  # this script never touches the test set
    args.output_dir = f"{base_args.output_dir}/{config_name}_seed{seed}"
    return train_and_evaluate(args)


def load_completed(results_file):
    if not os.path.exists(results_file):
        return pd.DataFrame(columns=["config", "seed", "mae", "correlation"])
    return pd.read_csv(results_file)


def append_result(results_file, row):
    header_needed = not os.path.exists(results_file)
    pd.DataFrame([row]).to_csv(results_file, mode="a", header=header_needed, index=False)


def check_results_dir(results_file):
    """Fails fast, before any training, if --results-file's directory
    doesn't exist -- otherwise a full training run can complete and only
    then crash trying to save its result (e.g. a Google Drive path when
    Drive isn't actually mounted in this runtime), losing that run."""
    results_dir = os.path.dirname(os.path.abspath(results_file))
    if not os.path.isdir(results_dir):
        raise SystemExit(
            f"\n--results-file's directory does not exist: {results_dir}\n"
            "If this is a Google Drive path, Drive isn't mounted in this runtime -- run\n"
            "    from google.colab import drive; drive.mount('/content/drive')\n"
            "first, then re-run this exact command. Failing now, before any training, "
            "so a GPU run isn't wasted on a result that can't be saved."
        )


def main():
    parser = build_arg_parser()
    parser.add_argument("--results-file", default="multi_seed_results.csv", help="Appended to after every (config, seed) run -- re-running the same command resumes from here instead of starting over.")
    args = parser.parse_args()
    check_results_dir(args.results_file)

    all_pairs = list(itertools.product(CONFIGS.keys(), SEEDS))
    completed = load_completed(args.results_file)
    done = {(r.config, int(r.seed)) for r in completed.itertuples()}
    remaining = [(name, seed) for name, seed in all_pairs if (name, seed) not in done]

    if done:
        print(f"Found {len(done)} already-completed (config, seed) run(s) in {args.results_file} -- skipping those.")
    print(f"Running {len(remaining)} of {len(all_pairs)} run(s) this invocation ({len(CONFIGS)} configs x {len(SEEDS)} seeds).")

    for i, (name, seed) in enumerate(remaining, 1):
        print("\n" + "=" * 70)
        print(f"Run {i}/{len(remaining)} (remaining): config={name} ({CONFIGS[name]}), seed={seed}")
        print("=" * 70)
        trial_result = run_trial(args, name, CONFIGS[name], seed)
        val_metrics = trial_result["val"]
        row = {
            "config": name,
            "seed": seed,
            "mae": val_metrics["eval_mae"],
            "correlation": val_metrics["eval_correlation"],
        }
        append_result(args.results_file, row)
        print(f"Saved this run's result to {args.results_file} (survives a disconnect from here on).")

    results = load_completed(args.results_file)
    if len(results) < len(all_pairs):
        print(
            f"\nOnly {len(results)} of {len(all_pairs)} run(s) are done -- "
            "re-run this same command to pick up where it left off. Not printing a final summary yet."
        )
        return

    print("\n" + "=" * 70)
    print("Stability summary, best (lowest mean MAE) first:")
    print("=" * 70)
    summary = results.groupby("config").agg(
        mean_mae=("mae", "mean"), std_mae=("mae", "std"),
        mean_correlation=("correlation", "mean"), std_correlation=("correlation", "std"),
        n=("mae", "size"),
    ).sort_values("mean_mae")
    print(summary.to_string())

    best_name = summary.index[0]
    best_row = summary.loc[best_name]
    print(f"\nBest by mean MAE: {best_name} ({best_row['mean_mae']:.4f} +/- {best_row['std_mae']:.4f} over {int(best_row['n'])} seed(s))")

    if "baseline" in summary.index and best_name != "baseline":
        baseline_row = summary.loc["baseline"]
        gap = baseline_row["mean_mae"] - best_row["mean_mae"]
        noise = max(best_row["std_mae"], baseline_row["std_mae"])
        print(f"Improvement over baseline: {gap:.4f} (baseline's own run-to-run std is {baseline_row['std_mae']:.4f})")
        if gap < noise:
            print(
                "CAUTION: this improvement is smaller than the seed-to-seed noise in at least one of the "
                "two configs -- not confidently better than baseline yet, could just be a lucky seed draw."
            )
        else:
            print("This improvement is larger than the seed-to-seed noise in both configs -- a more trustworthy gain.")
    elif best_name == "baseline":
        print("Baseline itself came out on top -- none of the tested changes clearly helped.")


if __name__ == "__main__":
    main()
