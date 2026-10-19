"""Filters a debate_predictions.csv-style item table down to a clean
working analysis dataset, dropping rows that aren't real, attributable
speech content.

A row is dropped if ANY of these hold:

AttributedTo:
  - blank/missing
  - contains "Mr Speaker" (the Speaker isn't a debate participant whose
    speech should count toward a member's/party's polarization)

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
  - starts with a bracketed list marker like "(a)", "(iii)", "(1)" once
    stripped (a sub-paragraph label with no real content of its own
    attached, from how these get split out of a numbered list)

Prints a breakdown of how many rows each individual rule caught (rules
can overlap on the same row -- a row counted under several rules is only
dropped once) before saving the filtered dataset.

Usage:
    python build_working_dataset.py --input debate_predictions_fixed.csv --output working_dataset.csv
"""

import argparse
import re

import pandas as pd

MR_SPEAKER = "Mr Speaker"
ROSE_MARKER = "<em>rose</em><em>—</em>"
BROUGHT_UP_MARKER = "<em>Brought up, and read the First time.</em>"
BRACKETED_START = re.compile(r"^\(\s*[A-Za-z0-9]+\s*\)")
HAS_LETTER = re.compile(r"[A-Za-z]")


def is_blank(series):
    return series.isna() | series.astype(str).str.strip().eq("")


def build_filter_report(df):
    attributed_to = df["AttributedTo"].fillna("").astype(str)
    value = df["Value"].fillna("").astype(str)
    value_stripped = value.str.strip()

    reasons = {
        "AttributedTo blank": is_blank(df["AttributedTo"]),
        "AttributedTo is Mr Speaker": attributed_to.str.contains(MR_SPEAKER, na=False),
        "Value blank": is_blank(df["Value"]),
        "Value has no letters": ~value_stripped.apply(lambda v: bool(HAS_LETTER.search(v))),
        "Value is a 'rose—' marker": value.str.contains(re.escape(ROSE_MARKER), na=False, regex=True),
        "Value is a 'brought up' marker": value.str.contains(re.escape(BROUGHT_UP_MARKER), na=False, regex=True),
        "Value is a bare HTML tag": value_stripped.str.startswith("<") & value_stripped.str.endswith(">"),
        "Value starts with a bracketed marker": value_stripped.apply(lambda v: bool(BRACKETED_START.match(v))),
    }
    return pd.DataFrame(reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="debate_predictions.csv (ideally after fix_missing_attributions.py)")
    parser.add_argument("--output", default="working_dataset.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    reasons = build_filter_report(df)
    drop_mask = reasons.any(axis=1)

    print(f"Loaded {len(df)} row(s) from {args.input}.")
    print("Rows matching each exclusion rule (rules can overlap):")
    print(reasons.sum().to_string())
    print(f"\nTotal row(s) dropped (any rule): {drop_mask.sum()}")

    kept = df.loc[~drop_mask].reset_index(drop=True)
    kept.to_csv(args.output, index=False)
    print(f"Saved {len(kept)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
