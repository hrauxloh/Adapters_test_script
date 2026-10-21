"""Builds an alternative, edit-based cleaning pipeline (as opposed to
build_working_dataset.py's drop-only rules): working_dataset_values_edited.csv.

This one EDITS the Value text itself (stripping HTML, merging split-up
speaking turns), so the existing polarization_probability/predicted_polarization
columns are removed -- they won't match the edited text, and the output
needs to be re-classified from scratch.

Menu of what happens when you run this file, in order:
  1. Load --input and drop polarization_probability/predicted_polarization.
  2. Add a row_hash_id column (a short hash of DebateId+ItemId), computed
     before any content changes, so every original row keeps a traceable
     identity even after merging below.
  3. Strip HTML tags (markup only -- inner text is kept) from every Value.
  4. Drop rows where to_remove (matched regardless of case/whitespace in
     the header) is "x".
  5. Drop rows with a blank AttributedTo (e.g. Timestamp marker rows).
     NOT explicitly requested, but necessary for step 6 to do anything
     useful: a Timestamp sitting between two fragments of the same
     speaker's turn has no AttributedTo of its own, so it would otherwise
     break the "consecutive identical" chain and leave the fragments
     split apart -- exactly the bug fix_missing_attributions.py exists to
     fix in the first place.
  6. Merge consecutive rows that share an identical DebateId, ItemType,
     MemberId, and AttributedTo -- these are speaking-turn fragments.
     Their Value text is concatenated (space-joined) in OrderInSection
     order into one row; the merged row keeps the first fragment's
     OrderInSection/ItemId, and a new merged_row_ids column records every
     contributing row_hash_id (space-separated) for traceability, plus
     n_merged_rows (how many original rows went into it).

Usage:
    python build_working_dataset_values_edited.py --input debate_predictions_fixed.csv --output working_dataset_values_edited.csv
"""

import argparse
import hashlib
import re

import pandas as pd

TAG = re.compile(r"<[^>]+>")


def strip_tags(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", TAG.sub(" ", text)).strip()


def find_to_remove_column(df):
    for col in df.columns:
        if col.strip().lower() == "to_remove":
            return col
    return None


def is_blank(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def compute_row_hash(row):
    key = f"{row['DebateId']}|{row['ItemId']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def merge_consecutive_groups(df):
    group_cols = ["DebateId", "ItemType", "MemberId", "AttributedTo"]
    changed = (df[group_cols] != df[group_cols].shift()).any(axis=1)
    group_id = changed.cumsum()

    rows = []
    for _, group in df.groupby(group_id, sort=False):
        first = group.iloc[0]
        merged_value = " ".join(v for v in group["Value"] if isinstance(v, str) and v)
        rows.append({
            "DebateId": first["DebateId"],
            "ItemId": first["ItemId"],
            "ItemType": first["ItemType"],
            "MemberId": first["MemberId"],
            "AttributedTo": first["AttributedTo"],
            "OrderInSection": first["OrderInSection"],
            "Value": merged_value,
            "merged_row_ids": " ".join(group["row_hash_id"]),
            "n_merged_rows": len(group),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="debate_predictions_fixed.csv")
    parser.add_argument("--output", default="working_dataset_values_edited.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} row(s) from {args.input}.")

    dropped_cols = [c for c in ("polarization_probability", "predicted_polarization") if c in df.columns]
    df = df.drop(columns=dropped_cols)
    print(f"Dropped stale prediction column(s): {dropped_cols}")

    df["row_hash_id"] = df.apply(compute_row_hash, axis=1)

    df["Value"] = df["Value"].apply(strip_tags)
    print("Stripped HTML tags from Value.")

    to_remove_col = find_to_remove_column(df)
    if to_remove_col is not None:
        drop_mask = df[to_remove_col].fillna("").astype(str).str.strip().str.lower().eq("x")
        print(f"Found manual-review column '{to_remove_col}': dropping {drop_mask.sum()} row(s) marked x.")
        df = df.loc[~drop_mask].reset_index(drop=True)
    else:
        print("No to_remove column found in --input -- skipping the manual-review step.")

    blank_attribution = is_blank(df["AttributedTo"])
    print(f"Dropping {blank_attribution.sum()} row(s) with blank AttributedTo (marker rows, e.g. Timestamp) before merging.")
    df = df.loc[~blank_attribution].reset_index(drop=True)

    df["OrderInSection"] = pd.to_numeric(df["OrderInSection"], errors="coerce")
    df = df.sort_values(["DebateId", "OrderInSection"], kind="stable").reset_index(drop=True)

    before = len(df)
    merged = merge_consecutive_groups(df)
    print(f"Merged {before} row(s) into {len(merged)} row(s) by consecutive (DebateId, ItemType, MemberId, AttributedTo).")

    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
