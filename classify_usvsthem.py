"""Applies the trained "us vs. them" / populist-attitude RoBERTa model
(train_usvsthem_classifier.py) to the same item-level text the main
polarization pipeline classifies -- working_dataset_values_edited.csv
(build_working_dataset_values_edited.py's output) by default, or
debate_predictions.csv/debate_predictions_values_edited.csv, anything with
a Value column.

This is a SEPARATE, standalone signal, not a replacement for or a merge
into classify_debates.py's / classify_edited_dataset.py's polarization
output: it writes its own --output file (default
debate_usvsthem_scores.csv) with its own score column (usvsthem_score,
0..1) plus per-emotion probabilities, keyed by the same DebateId/ItemId as
the polarization predictions so the two can be joined and compared later
-- nothing here touches debate_predictions*.csv.

This is inference only -- there's no ground-truth usVSthem_scale for debate
speech, so no MAE/correlation is computed here, just predicted scores.

Results are written to --output incrementally (every --save-every rows,
not just once at the end), with a progress line printed each time. If the
process dies partway through and you run the exact same command again, it
picks up where --output left off instead of reprocessing everything --
this assumes --input hasn't changed between runs.

Menu of what happens when you run this file, in order:
  1. Load --model-dir (a train_usvsthem_classifier.py --save-artifacts run
     -- model_state_dict.pt, usvsthem_model_config.json, the tokenizer).
  2. Load --input.
  3. Run inference in batches of --batch-size on the --text-column (Value
     by default) as-is, appending to --output and printing a progress line
     every --save-every rows.
  4. Once every row is done, save a debate-level rollup (mean
     usvsthem_score per DebateId) from the full --output.

Usage:
    python classify_usvsthem.py --input working_dataset_values_edited.csv --model-dir /content/drive/MyDrive/usvsthem_final_seed42/final
"""

import argparse
import os

import pandas as pd
import torch

from train_usvsthem_classifier import load_usvsthem_model


def get_predictions(model, tokenizer, texts, batch_size, max_length, device):
    scores = []
    emotion_probs = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            encoded = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt")
            encoded = {k: v.to(device) for k, v in encoded.items()}
            outputs = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
            scores.append(outputs["score"].cpu())
            emotion_probs.append(torch.sigmoid(outputs["emotions"]).cpu())
    return torch.cat(scores).numpy(), torch.cat(emotion_probs).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="working_dataset_values_edited.csv, or any CSV with a text column")
    parser.add_argument("--model-dir", required=True, help="A train_usvsthem_classifier.py --save-artifacts run, e.g. <output-dir>/final")
    parser.add_argument("--text-column", default="Value", help="Column holding the item text (Value in this repo's main pipeline; body in the UsVsThem_*_public.csv training data)")
    parser.add_argument("--output", default="debate_usvsthem_scores.csv")
    parser.add_argument("--debate-output", default=None, help="Defaults to <output> with _by_debate suffix")
    parser.add_argument("--batch-size", type=int, default=32, help="Forward-pass batch size")
    parser.add_argument("--save-every", type=int, default=500, help="Flush results to --output and print progress every N rows")
    parser.add_argument("--max-length", type=int, default=128, help="Matches train_usvsthem_classifier.py's default -- shorter than this repo's polarization classifier's 96, since full fine-tuning was trained at 128")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, emotion_columns = load_usvsthem_model(args.model_dir, device=device)
    print(f"Loaded us-vs-them model from {args.model_dir} onto {device}.")

    df = pd.read_csv(args.input)
    total = len(df)
    print(f"Loaded {total} row(s) across {df['DebateId'].nunique() if total and 'DebateId' in df.columns else '?'} debate(s) from {args.input}.")

    texts = df[args.text_column].fillna("").astype(str).tolist()

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

        scores, emotion_probs = get_predictions(model, tokenizer, batch_texts, args.batch_size, args.max_length, device)
        batch_df["usvsthem_score"] = scores
        for i, emotion in enumerate(emotion_columns):
            batch_df[f"emotion_prob_{emotion}"] = emotion_probs[:, i]
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
    if "DebateId" in full_df.columns:
        rollup = (
            full_df.groupby("DebateId")
            .agg(NumItems=("usvsthem_score", "size"), MeanUsVsThemScore=("usvsthem_score", "mean"))
            .reset_index()
        )
        rollup.to_csv(debate_output, index=False)
        print(f"Saved {len(rollup)} debate-level rollup row(s) to {debate_output}")
    else:
        print(f"No DebateId column in {args.input} -- skipping the by-debate rollup.")


if __name__ == "__main__":
    main()
