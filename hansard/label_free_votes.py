"""
WHAT THIS SCRIPT DOES
----------------------
Labels every division in all_divisions_combined.csv as `Whipped` (the
default) or `Possible Free Vote - needs verification`, based on keywords
in the division's Title.

THIS IS A HEURISTIC, NOT A VERIFIED ANSWER. There is no reliable record
built into this script (or in the person/model writing it) of which
specific historical votes were actually free — only a general awareness
of the small, well-established set of topics Parliament has traditionally
allowed free votes on (assisted dying, abortion, embryology/fertility
research, capital punishment, hunting, and a few similar conscience
issues). Anything this flags is a CANDIDATE worth checking against a real
source (e.g. the Commons Library's free-votes briefing, once accessible),
not a confirmed label. Everything else defaults to Whipped, which is
correct as a prior — free votes are the exception, not the norm — but
that also means a genuine free vote on a topic outside this keyword list
would be silently mislabeled Whipped. This is a starting point, not a
finished dataset.

Menu of what happens when you run this file, in order:
  1. Load all_divisions_combined.csv.
  2. Check each division's Title against a short list of conscience-issue
     keywords.
  3. Label matches "Possible Free Vote - needs verification", everything
     else "Whipped".
  4. Save the labeled table, and print the flagged rows so they can be
     manually checked.

Usage:
    python label_free_votes.py
    python label_free_votes.py --input all_divisions_combined.csv --output all_divisions_labelled.csv
"""

import argparse
import re

import pandas as pd

FREE_VOTE_KEYWORDS = [
    r"assisted dying", r"end of life", r"euthanasia",
    r"abortion", r"termination of pregnancy",
    r"embryo", r"fertilisation", r"stem cell",
    r"capital punishment", r"death penalty",
    r"\bhunting\b", r"fox hunting",
    r"sunday trading",
    r"gambling",
    r"same.sex marriage", r"equal marriage",
]
PATTERN = re.compile("|".join(FREE_VOTE_KEYWORDS), re.IGNORECASE)

WHIPPED = "Whipped"
POSSIBLE_FREE = "Possible Free Vote - needs verification"


def label_vote(title):
    if isinstance(title, str) and PATTERN.search(title):
        return POSSIBLE_FREE
    return WHIPPED


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="all_divisions_combined.csv")
    parser.add_argument("--output", default="all_divisions_labelled.csv")
    parser.add_argument("--sep", default=";", help="CSV delimiter (the source file is semicolon-separated).")
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep=args.sep)
    df["VoteType"] = df["Title"].apply(label_vote)

    print(df["VoteType"].value_counts())
    print()

    flagged = df[df["VoteType"] != WHIPPED]
    print(f"{len(flagged)} flagged row(s) - CHECK THESE AGAINST A REAL SOURCE, they are candidates, not confirmed:")
    print(flagged[["Date", "Title", "AyeCount", "NoCount"]].to_string())

    df.to_csv(args.output, index=False)
    print(f"\nSaved labeled table to {args.output}")


if __name__ == "__main__":
    main()
