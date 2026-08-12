"""Final, one-time evaluation on eng_test.csv, using the temperature scaling
and decision threshold fit by calibrate.py on the validation set.

Run this once, after every tuning decision (sweep.py, calibrate.py) has
already been made. Re-running it and then going back to change settings
based on what it shows defeats the point of keeping the test set held out.

Usage:
    python evaluate_test.py --output-dir output --test-file eng_test.csv
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from calibrate import get_logits
from data import load_split
from inspect_fusion import MODEL_NAME, load_trained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--test-file", default="eng_test.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
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


if __name__ == "__main__":
    main()
