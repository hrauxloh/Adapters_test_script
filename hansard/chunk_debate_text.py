"""
WHAT THIS SCRIPT DOES
----------------------
Turns the raw debate JSON files (from fetch_debate_transcripts.py) into
chunked, classifier-ready text — same method used earlier in this
project for the two-bill comparison, now applied across every fetched
debate.

Menu of what happens when you run this file, in order:
  1. Load every debate_*.json file in --input-dir.
  2. Drop any item whose raw Value contains "span id" — these are pure
     column-number markers (e.g. <span id="73" class="column-number"
     data-column-number="73"></span>), not real debate content.
  3. Strip HTML from the remaining items' Value, and drop anything left
     empty or under 10 characters (same "nonsense data" cleanup as
     before).
  4. Split each remaining item's text into <=94-token chunks (leaving
     room for [CLS]/[SEP] within the classifier's 96-token limit), the
     same chunk_text approach used previously.
  5. Save one combined table: one row per chunk, with the debate ID,
     item metadata (ItemId with a _chunk<n> suffix, ItemType, MemberId,
     AttributedTo), and the chunked text in a "clean_value" column --
     matching the column names used in the earlier two-bill comparison,
     so the next step (running the polarization classifier) can reuse
     that same code directly.

Usage:
    python chunk_debate_text.py --input-dir /content/drive/MyDrive/debate_transcripts --output chunked_debates.csv
"""

import argparse
import glob
import json
import os
import re

import pandas as pd
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MAX_CHUNK_TOKENS = 94  # leaves room for [CLS]/[SEP] within the classifier's 96-token limit


def strip_html(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text, tokenizer, max_tokens=MAX_CHUNK_TOKENS):
    ids = tokenizer.encode(text, add_special_tokens=False)
    id_chunks = [ids[i:i + max_tokens] for i in range(0, len(ids), max_tokens)]
    return [tokenizer.decode(c) for c in id_chunks]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="Folder of debate_*.json files from fetch_debate_transcripts.py")
    parser.add_argument("--output", default="chunked_debates.csv")
    parser.add_argument("--min-chars", type=int, default=10, help="Drop cleaned text shorter than this.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    debate_files = sorted(glob.glob(os.path.join(args.input_dir, "debate_*.json")))
    print(f"Found {len(debate_files)} debate file(s).")

    rows = []
    span_id_dropped = 0
    too_short_dropped = 0

    for file_i, path in enumerate(debate_files, start=1):
        debate_id = os.path.basename(path)[len("debate_"):-len(".json")]
        with open(path) as f:
            data = json.load(f)
        items = data.get("Items", [])

        kept_this_debate = 0
        for item in items:
            raw_value = item.get("Value") or ""
            if "span id" in raw_value.lower():
                span_id_dropped += 1
                continue

            clean_value = strip_html(raw_value)
            if len(clean_value) < args.min_chars:
                too_short_dropped += 1
                continue

            item_id = item.get("ItemId")
            for chunk_num, chunk_value in enumerate(chunk_text(clean_value, tokenizer), start=1):
                rows.append({
                    "DebateId": debate_id,
                    "ItemId": f"{item_id}_chunk{chunk_num}",
                    "ItemType": item.get("ItemType"),
                    "MemberId": item.get("MemberId"),
                    "AttributedTo": item.get("AttributedTo"),
                    "OrderInSection": item.get("OrderInSection"),
                    "clean_value": chunk_value,
                })
                kept_this_debate += 1

        print(f"  [{file_i}/{len(debate_files)}] {debate_id}: {len(items)} item(s) -> {kept_this_debate} chunk(s)")

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    print(f"\nDropped {span_id_dropped} 'span id' item(s), {too_short_dropped} too-short item(s).")
    print(f"Saved {len(df)} chunk(s) across {df['DebateId'].nunique() if not df.empty else 0} debate(s) to {args.output}")


if __name__ == "__main__":
    main()
