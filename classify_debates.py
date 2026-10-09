"""Apply the trained polarization fusion classifier to unlabeled debate
chunks (the output of hansard/chunk_debate_text.py), using the temperature
scaling and decision threshold already fit by calibrate.py.

This is inference only — there are no ground-truth labels for debate
speech, so no accuracy/F1 is computed here, just calibrated probabilities
and predictions per chunk, plus a per-debate rollup.

Menu of what happens when you run this file, in order:
  1. Load output-dir/calibration.json (written by calibrate.py).
  2. Load the trained fusion model (same as inspect_fusion.py).
  3. Load the chunked debates CSV (needs a clean_value text column and a
     DebateId column, as produced by hansard/chunk_debate_text.py).
  4. Run inference over every chunk's clean_value text.
  5. Save one row per chunk (with polarization_prob/polarization_pred
     columns added) and a debate-level rollup (mean probability and
     share of polarized chunks per DebateId).

Usage:
    python classify_debates.py --input chunked_debates.csv --output-dir output
    # if --output-dir's fusion was trained with the group adapter included:
    python classify_debates.py --input chunked_debates.csv --output-dir output_full --group-adapter-dir output_group/group
"""

import argparse
import json

import pandas as pd
import torch
from transformers import AutoTokenizer

from calibrate import get_logits
from inspect_fusion import MODEL_NAME, load_trained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV from hansard/chunk_debate_text.py")
    parser.add_argument("--output-dir", default="output", help="Trained model dir (fusion/, head/, calibration.json)")
    parser.add_argument(
        "--group-adapter-dir",
        default=None,
        help="Path to a standalone adapter saved by train_group_adapter.py (e.g. output_group/group). "
        "Only needed if --output-dir's fusion was trained with the group adapter included.",
    )
    parser.add_argument("--output", default="debate_predictions.csv")
    parser.add_argument("--debate-output", default=None, help="Defaults to <output> with _by_debate suffix")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    with open(f"{args.output_dir}/calibration.json") as f:
        calibration = json.load(f)
    temperature = calibration["temperature"]
    threshold = calibration["threshold"]
    print(f"Loaded calibration: temperature={temperature:.3f}, threshold={threshold:.2f}")

    df = pd.read_csv(args.input)
    texts = df["clean_value"].astype(str).tolist()
    print(f"Loaded {len(df)} chunk(s) from {args.input}.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir, group_adapter_dir=args.group_adapter_dir)

    logits = get_logits(model, tokenizer, texts, args.batch_size, args.max_length)
    calibrated_probs = torch.softmax(logits / temperature, dim=-1).numpy()[:, 1]
    preds = (calibrated_probs >= threshold).astype(int)

    df["polarization_prob"] = calibrated_probs
    df["polarization_pred"] = preds
    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} prediction(s) to {args.output}")

    debate_output = args.debate_output or args.output.rsplit(".csv", 1)[0] + "_by_debate.csv"
    rollup = (
        df.groupby("DebateId")
        .agg(
            NumChunks=("polarization_pred", "size"),
            MeanPolarizationProb=("polarization_prob", "mean"),
            PolarizedChunkShare=("polarization_pred", "mean"),
        )
        .reset_index()
    )
    rollup.to_csv(debate_output, index=False)
    print(f"Saved {len(rollup)} debate-level rollup row(s) to {debate_output}")


if __name__ == "__main__":
    main()
