"""Greedy hyperparameter search for train_fusion.py: tunes learning rate,
then class-weighted loss, then label smoothing, one axis at a time (each
stage keeps the previous stage's winner fixed) — far cheaper than a full
grid search. Runs on a subsample of the training data with early stopping,
evaluated against a validation split (never eng_test.csv), so it's fast and
doesn't tune to the test set.

Usage:
    python sweep.py --train-file eng_train.csv

Writes the winning settings to --config-out (default best_config.json),
which train_fusion.py's final full run should then be pointed at.
"""

import argparse
import json

from train_fusion import build_arg_parser, train_and_evaluate


def run_trial(overrides):
    args = build_arg_parser().parse_args([])
    args.save_artifacts = False
    args.output_dir = "sweep_tmp"
    for key, value in overrides.items():
        setattr(args, key, value)
    return train_and_evaluate(args)


def print_row(label, metrics):
    print(
        f"{label:>16} {metrics['eval_f1']:>8.3f} {metrics['eval_accuracy']:>10.3f} "
        f"{metrics['eval_precision']:>10.3f} {metrics['eval_recall']:>10.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", default="eng_train.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.5,
        help="Use a subset of the training portion to keep the sweep cheap; "
        "the winning config gets retrained on full data separately.",
    )
    parser.add_argument("--lr-candidates", default="1e-5,5e-5,1e-4,5e-4") #different learning rates
    parser.add_argument("--label-smoothing-candidates", default="0.0,0.1") #testing no smoothing vs smoothing 
    parser.add_argument("--epochs", type=int, default=10) # not tested
    parser.add_argument("--patience", type=int, default=2) # not tested
    parser.add_argument("--batch-size", type=int, default=32) # not tested
    parser.add_argument("--max-length", type=int, default=96) # not tested
    parser.add_argument("--seed", type=int, default=42) # not tested
    parser.add_argument("--config-out", default="best_config.json")
    args = parser.parse_args()

    common = dict(
        train_file=args.train_file,
        val_fraction=args.val_fraction,
        train_fraction=args.train_fraction,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_length=args.max_length,
        seed=args.seed,
    )

    header = f"{'setting':>16} {'f1':>8} {'accuracy':>10} {'precision':>10} {'recall':>10}"

    print("=== Stage 1: learning rate ===")
    print(header)
    lr_candidates = [float(x) for x in args.lr_candidates.split(",")]
    lr_results = []
    for lr in lr_candidates:
        metrics = run_trial({**common, "lr": lr, "class_weighted": False, "label_smoothing": 0.0})
        lr_results.append((lr, metrics))
        print_row(f"{lr:.0e}", metrics)
    best_lr, best_lr_metrics = max(lr_results, key=lambda r: r[1]["eval_f1"])
    print(f"-> best learning rate: {best_lr:.0e} (val f1={best_lr_metrics['eval_f1']:.3f})\n")

    print("=== Stage 2: class-weighted loss ===")
    print(header)
    cw_results = []
    for class_weighted in (False, True):
        metrics = run_trial(
            {**common, "lr": best_lr, "class_weighted": class_weighted, "label_smoothing": 0.0}
        )
        cw_results.append((class_weighted, metrics))
        print_row(str(class_weighted), metrics)
    best_cw, best_cw_metrics = max(cw_results, key=lambda r: r[1]["eval_f1"])
    print(f"-> best class_weighted: {best_cw} (val f1={best_cw_metrics['eval_f1']:.3f})\n")

    print("=== Stage 3: label smoothing ===")
    print(header)
    ls_candidates = [float(x) for x in args.label_smoothing_candidates.split(",")]
    ls_results = []
    for label_smoothing in ls_candidates:
        metrics = run_trial(
            {**common, "lr": best_lr, "class_weighted": best_cw, "label_smoothing": label_smoothing}
        )
        ls_results.append((label_smoothing, metrics))
        print_row(str(label_smoothing), metrics)
    best_ls, best_ls_metrics = max(ls_results, key=lambda r: r[1]["eval_f1"])
    print(f"-> best label_smoothing: {best_ls} (val f1={best_ls_metrics['eval_f1']:.3f})\n")

    best_config = {
        "lr": best_lr,
        "class_weighted": best_cw,
        "label_smoothing": best_ls,
        "val_fraction": args.val_fraction,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "val_f1_on_subset": best_ls_metrics["eval_f1"],
    }
    with open(args.config_out, "w") as f:
        json.dump(best_config, f, indent=2)

    print(f"Best config (saved to {args.config_out}):")
    print(json.dumps(best_config, indent=2))
    print(
        "\nNext: retrain on full data with these settings, e.g.\n"
        f"  python train_fusion.py --train-file {args.train_file} "
        f"--lr {best_lr} "
        f"{'--class-weighted' if best_cw else '--no-class-weighted'} "
        f"--label-smoothing {best_ls}"
    )


if __name__ == "__main__":
    main()
