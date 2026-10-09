"""Apply the trained polarization fusion classifier to debate transcripts
fetched by hansard/fetch_debate_transcripts.py, using the same approach
already proven to work in earlier ad hoc runs (assisted dying, immigration,
EU debates): raw item text, HTML-stripped, truncated at --max-length tokens
by the tokenizer — no chunking.

This is inference only — there are no ground-truth labels for debate
speech, so no accuracy/F1 is computed here, just calibrated probabilities
and predictions per item, plus a per-debate rollup.

Results are written to --output incrementally (every --save-every items,
not just once at the end), with a progress line printed each time. If the
process dies partway through (a Colab disconnect, a crash) and you run the
exact same command again, it picks up where --output left off instead of
reprocessing everything — this assumes the same --input-dir contents in
the same order, which holds as long as the debate_*.json files haven't
changed between runs.

Menu of what happens when you run this file, in order:
  1. Load output-dir/calibration.json (written by calibrate.py) and the
     trained model (load_trained_model, same as calibrate.py/evaluate_test.py
     use — reads output-dir/adapters.json to know which adapters to load).
  2. Load every debate_*.json file in --input-dir and flatten their Items
     into one table (DebateId, ItemId, ItemType, MemberId, AttributedTo,
     OrderInSection, Value).
  3. Strip HTML from each item's Value.
  4. Run inference in batches of --batch-size, appending to --output and
     printing a progress line every --save-every items.
  5. Once every item is done, save a debate-level rollup (mean probability
     and share of polarized items per DebateId) from the full --output.

Usage:
    python classify_debates.py --input-dir debate_transcripts --output-dir /content/drive/MyDrive/output_full
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
import torch
from transformers import AutoTokenizer

from calibrate import get_logits
from inspect_fusion import MODEL_NAME, load_trained_model


def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_debates(input_dir):
    rows = []
    debate_files = sorted(glob.glob(os.path.join(input_dir, "debate_*.json")))
    print(f"Found {len(debate_files)} debate file(s).")
    for path in debate_files:
        debate_id = os.path.basename(path)[len("debate_"):-len(".json")]
        with open(path) as f:
            data = json.load(f)
        for item in data.get("Items", []):
            rows.append({
                "DebateId": debate_id,
                "ItemId": item.get("ItemId"),
                "ItemType": item.get("ItemType"),
                "MemberId": item.get("MemberId"),
                "AttributedTo": item.get("AttributedTo"),
                "OrderInSection": item.get("OrderInSection"),
                "Value": item.get("Value"),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of debate_*.json files from hansard/fetch_debate_transcripts.py")
    parser.add_argument("--output-dir", default="output", help="Trained model dir (adapters.json, fusion/, head/, calibration.json)")
    parser.add_argument("--output", default="debate_predictions.csv")
    parser.add_argument("--debate-output", default=None, help="Defaults to <output> with _by_debate suffix")
    parser.add_argument("--batch-size", type=int, default=32, help="Forward-pass batch size")
    parser.add_argument("--save-every", type=int, default=500, help="Flush results to --output and print progress every N items")
    parser.add_argument("--max-length", type=int, default=96)
    args = parser.parse_args()

    with open(f"{args.output_dir}/calibration.json") as f:
        calibration = json.load(f)
    temperature = calibration["temperature"]
    threshold = calibration["threshold"]
    print(f"Loaded calibration: temperature={temperature:.3f}, threshold={threshold:.2f}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = load_trained_model(args.output_dir)

    df = load_debates(args.input_dir)
    total = len(df)
    print(f"Loaded {total} item(s) across {df['DebateId'].nunique() if total else 0} debate(s).")

    texts = df["Value"].apply(strip_html).tolist()

    done = 0
    if os.path.exists(args.output):
        done = len(pd.read_csv(args.output))
        print(f"Found existing {args.output} with {done} item(s) already processed — resuming from there.")
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
            print(f"Processed {batch_end}/{total} item(s) ({batch_end / total:.1%}) — saved to {args.output}")

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
