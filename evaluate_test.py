"""
WHAT THIS SCRIPT DOES
----------------------
Produces the one number that actually counts: how well the trained,
calibrated model does on data it has never been checked against before
(eng_test.csv). Every other script in this project deliberately avoids
this file so nothing gets tuned to it — this is the only place it gets used.

Run this once, after every tuning decision (sweep.py, calibrate.py) has
already been made. Re-running it and then going back to change settings
based on what it shows defeats the entire point of keeping the test set
held out — if changes are needed, make them before this step, using the
validation split, not after seeing this file's output.

Menu of what happens when you run this file, in order:
  1. Load the calibration numbers (temperature, decision threshold) that
     calibrate.py already worked out.
  2. Reload the trained model and run it over every row of the test set.
  3. Apply the calibration and turn its output into final yes/no predictions.
  4. Print accuracy / precision / recall / F1, and how trustworthy its
     confidence actually was.
  5. Write everything to a single summary.md file — a permanent record
     of this run's settings and result.

Usage:
    python evaluate_test.py --output-dir output --test-file eng_test.csv
"""

import argparse
import json
from datetime import datetime, timezone

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from calibrate import get_logits
from data import load_split
from inspect_fusion import MODEL_NAME, load_trained_model


def write_summary(path, args, best_config, calibration, metrics, confidences, correct):
    lines = [
        "# Polarization fusion model — final test evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Test file: `{args.test_file}` (n={metrics['n']})",
        "",
        "## Training configuration",
        "",
    ]
    if best_config is not None:
        lines += [f"- `{key}`: {value}" for key, value in best_config.items()]
    else:
        lines.append("(no best_config.json found alongside the model — settings used are unknown)")

    lines += [
        "",
        "## Calibration",
        "",
        f"- temperature: {calibration['temperature']:.4f}",
        f"- decision threshold: {calibration['threshold']:.2f}",
        "",
        "## Final test metrics",
        "",
        f"- accuracy: {metrics['accuracy']:.3f}",
        f"- precision: {metrics['precision']:.3f}",
        f"- recall: {metrics['recall']:.3f}",
        f"- f1: {metrics['f1']:.3f}",
        "",
        "## Confidence calibration on the test set",
        "",
    ]
    if correct.any():
        lines.append(f"- avg confidence when correct: {confidences[correct].mean():.3f}")
    if (~correct).any():
        lines.append(f"- avg confidence when incorrect: {confidences[~correct].mean():.3f}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--test-file", default="eng_test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--summary-out",
        default=None,
        help="Where to write a markdown summary of this run. Defaults to <output-dir>/summary.md.",
    )
    parser.add_argument(
        "--best-config",
        default="best_config.json",
        help="Path to the config sweep.py wrote out, included in the summary if it exists.",
    )
    args = parser.parse_args()

    with open(f"{args.output_dir}/calibration.json") as f:
        calibration = json.load(f)
    temperature = calibration["temperature"]
    threshold = calibration["threshold"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir)

    dataset = load_split(args.test_file)
    texts = dataset["text"]
    labels = np.array(dataset["polarization"])

    logits = get_logits(model, tokenizer, texts, args.batch_size, args.max_length)
    calibrated_probs = torch.softmax(logits / temperature, dim=-1).numpy()[:, 1]
    preds = (calibrated_probs >= threshold).astype(int)
    confidences = np.where(preds == 1, calibrated_probs, 1 - calibrated_probs)
    correct = preds == labels

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)

    print(f"Final test set metrics (n={len(labels)}, temperature={temperature:.3f}, threshold={threshold:.2f}):")
    print(f"  accuracy:  {accuracy:.3f}")
    print(f"  precision: {precision:.3f}")
    print(f"  recall:    {recall:.3f}")
    print(f"  f1:        {f1:.3f}")
    print()
    if correct.any():
        print(f"  avg confidence when correct:   {confidences[correct].mean():.3f}")
    if (~correct).any():
        print(f"  avg confidence when incorrect: {confidences[~correct].mean():.3f}")

    best_config = None
    try:
        with open(args.best_config) as f:
            best_config = json.load(f)
    except FileNotFoundError:
        pass

    summary_path = args.summary_out or f"{args.output_dir}/summary.md"
    metrics = {
        "n": len(labels),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    write_summary(summary_path, args, best_config, calibration, metrics, confidences, correct)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
