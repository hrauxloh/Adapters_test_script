"""Experimental addition to the edit-based cleaning pipeline: tests whether
merging a colon-introduced quote back into the same paragraph changes
anything downstream, without touching build_working_dataset_values_edited.py
itself.

strip_tags() in build_working_dataset_values_edited.py preserves original
paragraph breaks as "\\n\\n" within a Value. When a speaker's sentence ends
with ":" right before a paragraph break, and the next paragraph opens with a
quotation mark (" or “ -- a quote is always OPENED with one of these,
even though it isn't always closed with one, so the opening mark is the only
reliable signal to key off), that's almost always one continuous sentence
that got split onto its own paragraph by the original formatting -- e.g.

    He said:

    "This is unacceptable."

becomes one paragraph:

    He said: "This is unacceptable."

This only collapses the specific paragraph break between a trailing ":" and
an opening quote; every other paragraph break in the Value is left alone.

Takes working_dataset_values_edited.csv (or any CSV with a Value column) and
writes a new CSV with this transformation applied, so it can be re-classified
separately (via classify_edited_dataset.py) and compared against the
unmodified version -- this script makes no classification calls itself.

Usage:
    python merge_colon_quote_paragraphs.py --input working_dataset_values_edited.csv --output working_dataset_values_edited_test.csv
"""

import argparse
import re

import pandas as pd

OPENING_QUOTES = "\"“‘"
MERGE_PATTERN = re.compile(r":\s*\n\s*\n\s*(?=[" + re.escape(OPENING_QUOTES) + "])")


def merge_colon_quote_paragraphs(text):
    if not isinstance(text, str):
        return text, 0
    merged, n = MERGE_PATTERN.subn(": ", text)
    return merged, n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="working_dataset_values_edited.csv")
    parser.add_argument("--output", default="working_dataset_values_edited_test.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} row(s) from {args.input}.")

    results = df["Value"].apply(merge_colon_quote_paragraphs)
    df["Value"] = results.apply(lambda r: r[0])
    merge_counts = results.apply(lambda r: r[1])

    n_rows_changed = (merge_counts > 0).sum()
    n_merges = int(merge_counts.sum())
    print(f"Merged {n_merges} colon-then-quote paragraph break(s) across {n_rows_changed} row(s).")

    df.to_csv(args.output, index=False)
    print(f"Saved {len(df)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
