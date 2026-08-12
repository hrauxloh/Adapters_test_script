"""Post-hoc calibration for a trained fusion model: fits a temperature
scaling factor and a decision threshold on the validation split (the same
one held out by train_fusion.py), then saves both to
<output-dir>/calibration.json for evaluate_test.py to use.

This doesn't retrain anything — it only rescales the model's existing output
probabilities so its confidence better reflects how often it's actually
right, and picks the probability cutoff that maximizes F1 instead of
assuming 0.5.

Usage:
    python calibrate.py --output-dir output --train-file eng_train.csv
"""

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer

from data import load_train_val_split
from inspect_fusion import MODEL_NAME, load_trained_model


def get_logits(model, tokenizer, texts, batch_size, max_length):
    all_logits = []
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        encodings = tokenizer(
            batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = model(**encodings)
        all_logits.append(outputs.logits)
    return torch.cat(all_logits, dim=0)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Find the scalar T that minimizes NLL of softmax(logits / T) against labels."""
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=100)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-2).item())


def find_best_threshold(probs_class1: np.ndarray, labels: np.ndarray):
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.01, 0.99, 99):
        preds = (probs_class1 >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold
    return float(best_threshold), float(best_f1)


def summarize(name, probs_class1, threshold, labels):
    preds = (probs_class1 >= threshold).astype(int)
    confidences = np.where(preds == 1, probs_class1, 1 - probs_class1)
    correct = preds == labels
    accuracy = accuracy_score(labels, preds)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    print(f"{name} (threshold={threshold:.2f}): accuracy={accuracy:.3f} f1={f1:.3f}")
    if correct.any():
        print(f"  avg confidence when correct:   {confidences[correct].mean():.3f}")
    if (~correct).any():
        print(f"  avg confidence when incorrect: {confidences[~correct].mean():.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--train-file", default="eng_train.csv")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir)

    # Same seed/val-fraction as training reproduces the identical held-out split.
    _, val_split = load_train_val_split(args.train_file, args.val_fraction, seed=args.seed)
    texts = val_split["text"]
    labels = np.array(val_split["polarization"])

    logits = get_logits(model, tokenizer, texts, args.batch_size, args.max_length)
    labels_t = torch.tensor(labels, dtype=torch.long)

    raw_probs = torch.softmax(logits, dim=-1).numpy()[:, 1]
    print("Before calibration:")
    summarize("  uncalibrated, threshold=0.5", raw_probs, 0.5, labels)

    temperature = fit_temperature(logits, labels_t)
    calibrated_probs = torch.softmax(logits / temperature, dim=-1).numpy()[:, 1]
    best_threshold, _ = find_best_threshold(calibrated_probs, labels)

    print(f"\nFitted temperature: {temperature:.3f} (>1 means the model was overconfident)")
    print("After calibration:")
    summarize("  calibrated, tuned threshold", calibrated_probs, best_threshold, labels)

    calibration = {"temperature": temperature, "threshold": best_threshold}
    with open(f"{args.output_dir}/calibration.json", "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"\nSaved calibration to {args.output_dir}/calibration.json: {calibration}")


if __name__ == "__main__":
    main()
