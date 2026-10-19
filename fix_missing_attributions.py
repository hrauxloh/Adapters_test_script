"""Fixes a data bug in debate_predictions.csv (and the raw chunked/item
tables it's built from): when a Timestamp item (or any other non-speech
marker row) is inserted mid-speech, the Contribution rows that follow it
lose their MemberId/AttributedTo, even though they're still the same
speaker talking -- the attribution only reappears once a genuinely new
speaker starts. Confirmed against a real example: a timestamp inserted
mid-paragraph left five subsequent Contribution rows blank, all actually
still spoken by the same MP, until the next real speaker change.

Fix: within each debate, in OrderInSection order, forward-fill
MemberId/AttributedTo from the most recent row that had them -- but only
onto Contribution rows that are missing them. Timestamp (and any other
non-Contribution) rows are left exactly as they were; they correctly have
no speaker.

A Contribution row that's STILL blank after this (i.e. no earlier row in
its debate had an attribution to carry forward -- the very first row of
a debate, or a genuinely broken source file) is reported separately, since
that's a real gap this method can't fix, not a false positive.

Usage:
    python fix_missing_attributions.py --input debate_predictions.csv --output debate_predictions_fixed.csv
"""

import argparse

import pandas as pd

CONTRIBUTION_TYPE = "Contribution"


def fix_attributions(df):
    df = df.copy()
    df["OrderInSection"] = pd.to_numeric(df["OrderInSection"], errors="coerce")
    df = df.sort_values(["DebateId", "OrderInSection"], kind="stable")

    was_blank = df["ItemType"].eq(CONTRIBUTION_TYPE) & df["MemberId"].isna() & df["AttributedTo"].isna()

    filled = df.groupby("DebateId")[["MemberId", "AttributedTo"]].ffill()
    df.loc[was_blank, ["MemberId", "AttributedTo"]] = filled.loc[was_blank]

    still_blank = df["ItemType"].eq(CONTRIBUTION_TYPE) & df["MemberId"].isna() & df["AttributedTo"].isna()

    df["MemberId"] = pd.to_numeric(df["MemberId"], errors="coerce").astype("Int64")

    print(f"Fixed {was_blank.sum()} row(s) that were missing attribution across a marker row.")
    if still_blank.any():
        print(
            f"WARNING: {still_blank.sum()} Contribution row(s) still have no attribution "
            f"(likely the first row of their debate, or a genuinely broken source) -- left as-is:"
        )
        print(df.loc[still_blank, ["DebateId", "ItemId", "OrderInSection"]].to_string(index=False))

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=None, help="Defaults to <input> with _fixed suffix")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    fixed = fix_attributions(df)

    output = args.output or args.input.rsplit(".csv", 1)[0] + "_fixed.csv"
    fixed.to_csv(output, index=False)
    print(f"Saved {len(fixed)} row(s) to {output}")


if __name__ == "__main__":
    main()
