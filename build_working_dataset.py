"""Filters a debate_predictions.csv-style item table down to a clean
working analysis dataset, dropping rows that aren't real, attributable
speech content.

A row is dropped if ANY of these hold:

AttributedTo:
  - blank/missing
  - contains "chair" or "speaker" (case-insensitive) -- the Speaker/Chair
    (and Deputy/Madam variants) aren't debate participants whose speech
    should count toward a member's/party's polarization. This subsumes
    the narrower "Mr Speaker"-only check an earlier version of this
    script used.

Value (checked on the raw column, HTML included, since several of these
rules are specifically about the raw markup):
  - blank/missing
  - contains no letter character at all once surrounding whitespace is
    stripped (catches bare timestamps like "15:15:00", stray punctuation,
    column-number-only fragments, etc.)
  - contains the literal "<em>rose</em><em>—</em>" (a member standing
    without making a speech)
  - contains the literal "<em>Brought up, and read the First time.</em>"
    (a standard procedural marker for a new clause)
  - starts with "<" and ends with ">" once stripped (a value that's
    nothing but an HTML tag, e.g. a lone column-number <span>)
  - starts with a bracketed list marker like "(a)", "(iii)", "(1)"
    (optionally preceded by a straight or curly quote/apostrophe, e.g.
    '(a) or "(a)) once stripped -- a sub-paragraph label with no real
    content of its own attached. These are saved separately, to
    --removed-bracketed-marker (defaults to <output>_bracketed_marker_removed.csv),
    so they can be reviewed later without being in the main dataset.
  - under --min-length characters (default 100), checked BOTH on the raw
    value and on the value with HTML tags removed (tags only -- inner
    text is kept) -- dropped if EITHER version is too short, since a
    value padded mostly by markup shouldn't survive just because the raw
    count looks long enough.

Manual review:
  - the "to_remove" column (if present in --input, matched regardless of
    case/surrounding whitespace in the header) is "x" -- for rows
    flagged by hand, e.g. procedural/legislative-text blocks too long or
    structurally different to catch with the rules above. Anything else
    in that column (blank, "y", or any other value) is kept. Prints
    whether a to_remove-like column was found at all, so a header-naming
    mismatch shows up instead of silently skipping the step.

Prints a breakdown of how many rows each individual rule caught (rules
can overlap on the same row -- a row counted under several rules is only
dropped once) before saving the filtered dataset.

Usage:
    python build_working_dataset.py --input debate_predictions_fixed.csv --output working_dataset.csv
    python build_working_dataset.py --input debate_predictions_fixed.csv --output working_dataset.csv --min-length 100
"""

import argparse
import re

import pandas as pd

CHAIR_OR_SPEAKER = re.compile(r"chair|speaker|chairman", re.IGNORECASE)
ROSE_MARKER = "<em>rose</em><em>—</em>"
BROUGHT_UP_MARKER = "<em>Brought up, and read the First time.</em>"
BRACKETED_START = re.compile(r"^[\'\"‘’“”]?\s*\(\s*[A-Za-z0-9]+\s*\)")
HAS_LETTER = re.compile(r"[A-Za-z]")
TAG = re.compile(r"<[^>]+>")


def is_blank(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def strip_tags(text):
    """Removes HTML tags only -- keeps whatever text was inside them."""
    return re.sub(r"\s+", " ", TAG.sub(" ", text)).strip()


def compute_bracketed_flag(df):
    value_stripped = df["Value"].fillna("").astype(str).str.strip()
    return value_stripped.apply(lambda v: bool(BRACKETED_START.match(v)))


def find_to_remove_column(df):
    """Matches a 'to_remove' column regardless of case or surrounding
    whitespace in the header, so a naming mismatch doesn't silently skip
    the manual-review step."""
    for col in df.columns:
        if col.strip().lower() == "to_remove":
            return col
    return None


def build_filter_report(df, min_length):
    attributed_to = df["AttributedTo"].fillna("").astype(str)
    value = df["Value"].fillna("").astype(str)
    value_stripped = value.str.strip()
    tag_stripped_length = value.apply(strip_tags).str.len()

    reasons = {
        "AttributedTo blank": is_blank(df["AttributedTo"]),
        "AttributedTo contains chair/speaker": attributed_to.str.contains(CHAIR_OR_SPEAKER, na=False),
        "Value blank": is_blank(df["Value"]),
        "Value has no letters": ~value_stripped.apply(lambda v: bool(HAS_LETTER.search(v))),
        "Value is a 'rose—' marker": value.str.contains(re.escape(ROSE_MARKER), na=False, regex=True),
        "Value is a 'brought up' marker": value.str.contains(re.escape(BROUGHT_UP_MARKER), na=False, regex=True),
        "Value is a bare HTML tag": value_stripped.str.startswith("<") & value_stripped.str.endswith(">"),
        "Value starts with a bracketed marker": compute_bracketed_flag(df),
        f"Value under {min_length} chars (raw or tag-stripped)": (
            (value.str.len() < min_length) | (tag_stripped_length < min_length)
        ),
    }

    to_remove_col = find_to_remove_column(df)
    if to_remove_col is not None:
        reasons["Manually flagged (to_remove)"] = (
            df[to_remove_col].fillna("").astype(str).str.strip().str.lower().eq("x")
        )
        print(f"Found manual-review column '{to_remove_col}'.")
    else:
        print("No to_remove column found in --input -- skipping the manual-review step.")

    return pd.DataFrame(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="debate_predictions.csv (ideally after fix_missing_attributions.py)")
    parser.add_argument("--output", default="working_dataset.csv")
    parser.add_argument("--min-length", type=int, default=100, help="Minimum character length, raw and tag-stripped, to keep a row")
    parser.add_argument(
        "--removed-bracketed-marker",
        default=None,
        help="Where to save the rows dropped for starting with a bracketed marker, for future reference. Defaults to <output>_bracketed_marker_removed.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} row(s) from {args.input}.")

    reasons = build_filter_report(df, args.min_length)
    drop_mask = reasons.any(axis=1)

    print("\nRows matching each exclusion rule (rules can overlap):")
    print(reasons.sum().to_string())
    print(f"\nTotal row(s) dropped (any rule): {drop_mask.sum()}")

    bracketed_marker_mask = reasons["Value starts with a bracketed marker"]
    removed_bracketed_marker_path = args.removed_bracketed_marker or args.output.rsplit(".csv", 1)[0] + "_bracketed_marker_removed.csv"
    df.loc[bracketed_marker_mask].to_csv(removed_bracketed_marker_path, index=False)
    print(f"Saved {bracketed_marker_mask.sum()} bracketed-marker row(s) to {removed_bracketed_marker_path} for future reference.")

    kept = df.loc[~drop_mask].reset_index(drop=True)
    kept.to_csv(args.output, index=False)
    print(f"Saved {len(kept)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
