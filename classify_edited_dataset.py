"""Applies the trained polarization fusion classifier to
working_dataset_values_edited.csv (build_working_dataset_values_edited.py's
output), the same way classify_debates.py applies it to raw debate
transcripts -- except the input here is already a flat, cleaned table
with a Value column, not raw debate_*.json files, so there's no
load_debates()/strip_html step: the text is classified exactly as it
already stands (HTML stripped, turns merged, paragraph breaks intact).

This is inference only -- there are no ground-truth labels, so no
accuracy/F1 is computed, just calibrated probabilities and predictions
per row, plus a per-debate rollup.

Results are written to --output incrementally (every --save-every rows,
not just once at the end), with a progress line printed each time. If
the process dies partway through and you run the exact same command
again, it picks up where --output left off instead of reprocessing
everything -- this assumes --input hasn't changed between runs.

Menu of what happens when you run this file, in order:
  1. Load output-dir/calibration.json and the trained model
     (load_trained_model -- reads output-dir/adapters.json to know which
     adapters to load).
  2. Load --input (working_dataset_values_edited.csv).
  3. Run inference in batches of --batch-size on the Value column as-is,
     appending to --output and printing a progress line every
     --save-every rows.
  4. Once every row is done, save a debate-level rollup (mean
     probability and share of polarized rows per DebateId) from the
     full --output.

Usage:
    python classify_edited_dataset.py --input working_dataset_values_edited.csv --output-dir /content/drive/MyDrive/output_full
"""

import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer

from calibrate import get_logits
from inspect_fusion import MODEL_NAME, load_trained_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="working_dataset_values_edited.csv")
    parser.add_argument("--output-dir", default="output", help="Trained model dir (adapters.json, fusion/, head/, calibration.json)")
    parser.add_argument("--output", default="debate_predictions_edited.csv")
    parser.add_argument("--debate-output", default=None, help="Defaults to <output> with _by_debate suffix")
    parser.add_argument("--batch-size", type=int, default=32, help="Forward-pass batch size")
    parser.add_argument("--save-every", type=int, default=500, help="Flush results to --output and print progress every N rows")
    parser.add_argument("--max-length", type=int, default=96, help="Tokens are truncated at this length -- merged rows are longer on average, so more gets cut off than before")
    args = parser.parse_args()

    with open(f"{args.output_dir}/calibration.json") as f:
        calibration = json.load(f)
    temperature = calibration["temperature"]
    threshold = calibration["threshold"]
    print(f"Loaded calibration: temperature={temperature:.3f}, threshold={threshold:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir)

    df = pd.read_csv(args.input)
    total = len(df)
    print(f"Loaded {total} row(s) across {df['DebateId'].nunique() if total else 0} debate(s) from {args.input}.")

    texts = df["Value"].fillna("").astype(str).tolist()

    done = 0
    if os.path.exists(args.output):
        done = len(pd.read_csv(args.output))
        print(f"Found existing {args.output} with {done} row(s) already processed — resuming from there.")
    header_needed = done == 0

    pending = []
    for batch_start in range(done, total, args.batch_size):
        batch_end = min(batch_start + args.batch_size, total)
        batch_df = df.iloc[batch_start:batch_end].copy()
        batch_texts = texts[batch_start:batch_end]

        logits = get_logits(model, tokenizer, batch_texts, args.batch_size, args.max_length)
        probs = torch.softmax(logits / temperature, dim=-1).numpy()[:, 1]
        batch_df["polarization_probability"] = probs
        batch_df["predicted_polarization"] = (probs >= threshold).astype(int)
        pending.append(batch_df)

        pending_count = sum(len(b) for b in pending)
        if pending_count >= args.save_every or batch_end == total:
            pd.concat(pending, ignore_index=True).to_csv(args.output, mode="a", header=header_needed, index=False)
            header_needed = False
            pending = []
            print(f"Processed {batch_end}/{total} row(s) ({batch_end / total:.1%}) — saved to {args.output}")

    print("Classification complete.")

    full_df = pd.read_csv(args.output)
    debate_output = args.debate_output or args.output.rsplit(".csv", 1)[0] + "_by_debate.csv"
    rollup = (
        full_df.groupby("DebateId")
        .agg(
            NumItems=("predicted_polarization", "size"),
            MeanPolarizationProb=("polarization_probability", "mean"),
            PolarizedItemShare=("predicted_polarization", "mean"),
        )
        .reset_index()
    )
    rollup.to_csv(debate_output, index=False)
    print(f"Saved {len(rollup)} debate-level rollup row(s) to {debate_output}")


if __name__ == "__main__":
    main()
