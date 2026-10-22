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
  3. Strip HTML tags (markup only -- inner text is kept) from every Value,
     preserving existing paragraph breaks (blank lines) as a distinct
     "\n\n" -- only ordinary whitespace (spaces, tabs, single newlines
     within a paragraph) gets collapsed.
  4. Drop rows with a blank AttributedTo (e.g. Timestamp marker rows),
     where AttributedTo contains "chair" or "speaker" (case-insensitive --
     catches Chairman/Deputy Chair/Madam Speaker/Deputy Speaker variants
     too), or with a blank MemberId. The blank-AttributedTo drop isn't
     explicitly requested, but is necessary for step 5 to do anything
     useful: a marker row sitting between two fragments of the same
     speaker's turn has no AttributedTo of its own, so it would otherwise
     break the "consecutive identical" chain and leave the fragments
     split apart -- exactly the bug fix_missing_attributions.py exists to
     fix in the first place. Same reasoning extends to the chair/speaker
     and blank-MemberId drops here too: none of these should block two
     fragments of a real speaker's turn from merging across them.
  5. Merge consecutive rows that share an identical DebateId, ItemType,
     MemberId, and AttributedTo -- these are speaking-turn fragments.
     Their Value text is concatenated (space-joined) in OrderInSection
     order into one row; the merged row keeps the first fragment's
     OrderInSection/ItemId, and a new merged_row_ids column records every
     contributing row_hash_id (space-separated) for traceability, plus
     n_merged_rows (how many original rows went into it).
  6. In each merged Value, truncate everything from the first occurrence
     of "Question put" onward (that substring and everything after it is
     dropped, keeping only what came before it) -- run after merging, so
     it also catches a "Question put" that only shows up once fragments
     split across a Timestamp are back together.
  7. Drop rows whose entire Value (after the truncation above, stripped
     of surrounding whitespace) is EXACTLY one of a fixed list of stock
     "will X give way?"/"rose"/"indicated assent" interjection phrases --
     not a substring match, so real speech that happens to mention
     giving way isn't touched, only a row whose whole content is nothing
     but one of these stock lines.

Usage:
    python build_working_dataset_values_edited.py --input debate_predictions_fixed.csv --output working_dataset_values_edited.csv
"""

import argparse
import hashlib
import re

import pandas as pd

TAG = re.compile(r"<[^>]+>")
CHAIR_OR_SPEAKER = re.compile(r"chair|speaker", re.IGNORECASE)

# Stock "will X give way?" / assent-dissent interjections -- a row is
# dropped only when its ENTIRE value matches one of these exactly, not
# when one just appears inside otherwise-substantive speech.
GIVE_WAY_PHRASES = {
    "Will the Minister give way?",
    "Will my hon. Friend give way?",
    "Will the hon. Gentleman give way?",
    "Will my right hon. Friend give way?",
    "Will the hon. Lady give way?",
    "Will the right hon. Gentleman give way?",
    "Will the hon. Lady give way? I want to be helpful.",
    "Will the right hon. Lady give way?",
    "Will the right hon. and learned Gentleman give way?",
    "Will my right hon. and learned Friend give way?",
    "Will the Secretary of State give way?",
    "Will the Lord Chancellor give way?",
    "Will the hon. Member give way?",
    "On that point, will the hon. Gentleman give way?",
    "Will the hon. and learned Lady give way?",
    "Will the Chancellor give way?",
    "Will my hon. Friend give way? [ Interruption. ]",
    "Will the Attorney General give way?",
    "Will the Prime Minister give way?",
    "Can I beseech the right hon. Gentleman to give way?",
    "Well, would my hon. Friend give way? [ Laughter. ]",
    "Will the hon. and learned Gentleman give way?",
    "Will my hon. and learned Friend give way?",
    "Will the Solicitor General give way?",
    "Will my hon. Friend give way? [Hon. Members: “ Oh.”] I will be very quick. Is my hon. Friend also thankful for the £21 million—",
    "On that point, will the hon. Lady give way?",
    "Will the right hon. Member give way?",
    "On that point, will the Minister give way?",
    "Will the Minister give way? I did give way to him.",
    "On that point, will my hon. Friend give way?",
    "Wil my right hon. Friend give way?",
    "Will the Leader of the Opposition give way?",
    "Can I not tempt my hon. Friend to give way?",
    "On the issue of potholes, will the Minister give way?",
    "Will the Leader of the House give way?",
    "On that point, will the Prime Minister give way?",
    "Is my hon. Friend asking me to give way?",
    "Will the Home Secretary give way?",
    "indicated assent.",
    "indicated dissent.",
    "rose —",
    "rose—",
    "I beg to move, That the clause be read a Second time.",
}


def strip_tags(text):
    """Removes HTML tags (markup only -- inner text is kept), preserving
    existing paragraph breaks as a single "\n\n" while collapsing any
    other run of whitespace (spaces, tabs, single newlines) to one space."""
    if not isinstance(text, str):
        return ""
    text = TAG.sub(" ", text)
    paragraphs = re.split(r"\n\s*\n", text)
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def truncate_at_question_put(text):
    """Drops "Question put" and everything after it in a value, keeping
    only what came before -- the procedural voting announcement that
    typically follows the end of substantive speech."""
    if not isinstance(text, str):
        return text
    idx = text.find("Question put")
    if idx == -1:
        return text
    return text[:idx].rstrip()


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

    blank_attribution = is_blank(df["AttributedTo"])
    print(f"Dropping {blank_attribution.sum()} row(s) with blank AttributedTo (marker rows, e.g. Timestamp) before merging.")

    chair_or_speaker = df["AttributedTo"].fillna("").astype(str).str.contains(CHAIR_OR_SPEAKER, na=False)
    print(f"Dropping {chair_or_speaker.sum()} row(s) where AttributedTo contains chair/speaker before merging.")

    blank_member_id = is_blank(df["MemberId"])
    print(f"Dropping {blank_member_id.sum()} row(s) with blank MemberId before merging.")

    df = df.loc[~(blank_attribution | chair_or_speaker | blank_member_id)].reset_index(drop=True)

    df["OrderInSection"] = pd.to_numeric(df["OrderInSection"], errors="coerce")
    df = df.sort_values(["DebateId", "OrderInSection"], kind="stable").reset_index(drop=True)

    before = len(df)
    merged = merge_consecutive_groups(df)
    print(f"Merged {before} row(s) into {len(merged)} row(s) by consecutive (DebateId, ItemType, MemberId, AttributedTo).")

    has_question_put = merged["Value"].str.contains("Question put", regex=False, na=False)
    print(f"Truncating {has_question_put.sum()} row(s) at 'Question put'.")
    merged["Value"] = merged["Value"].apply(truncate_at_question_put)

    give_way_only = merged["Value"].fillna("").astype(str).str.strip().isin(GIVE_WAY_PHRASES)
    print(f"Dropping {give_way_only.sum()} row(s) whose entire value is a stock give-way/interjection phrase.")
    merged = merged.loc[~give_way_only].reset_index(drop=True)

    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
