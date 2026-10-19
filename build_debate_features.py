"""Compile a one-row-per-debate dataframe combining the classifier's
debate-level polarization rollup with debate-level vote/bill metadata,
for feature-vs-polarization analysis (e.g. which metadata best predicts
polarization).

Metadata sources, both keyed by DivisionId (a debate can span several
divisions -- Report Stage amendments, Third Reading, etc. -- so metadata
is aggregated up to one row per DebateId before joining):
  - consolidated_votes_table_simplified.xlsx (Sheet1): the real vote
    breakdown per division -- rebellion rate, turnout, government/
    opposition party, Gov/Opp cohesion and distance, and Labour_whip/
    Cons_whip (each major party's OWN rebellion rate on that division --
    despite the name, these are continuous rates, not a binary flag).
  - all_divisions_labelled.csv: VoteType, a HEURISTIC "Possible Free Vote
    - needs verification" vs "Whipped" label from hansard/label_free_votes.py,
    based on keyword matches in the division's Title. This is a candidate
    flag, not a verified one -- see that script's own caveat.

Debate IDs are extracted from the xlsx's Debate column with the same
logic hansard/fetch_debate_transcripts.py uses (handles both a full
hansard.parliament.uk URL and a bare ID), and only rows with Sample == "x"
are used, matching exactly which debates were actually fetched/classified.

Menu of what happens when you run this file, in order:
  1. Load consolidated_votes_table_simplified.xlsx, keep Sample == "x"
     rows, extract each row's DebateId.
  2. Load all_divisions_labelled.csv and join its VoteType onto those
     rows by DivisionId.
  3. Aggregate from division-level to one row per DebateId: numeric vote
     metrics averaged across the debate's divisions; BillOrDebateName/
     Government/Party/Opposition/Date taken from the first division (see
     NumDistinctBillNames for how often a debate's divisions actually
     disagree on BillOrDebateName -- different amendments within the same
     bill often have slightly different titles); VoteType turned into
     AnyPossibleFreeVote / FreeVoteDivisionShare.
  4. Load the classifier's per-debate rollup (--predictions, the
     debate_predictions_by_debate.csv that classify_debates.py produces).
  5. Merge the two on DebateId (inner join -- only debates with both a
     classifier prediction and vote metadata are kept) and save one CSV,
     ready for feature-vs-polarization analysis.

Usage:
    python build_debate_features.py --predictions /content/drive/MyDrive/debate_predictions_by_debate.csv --output debate_features.csv
"""

import argparse
import re

import pandas as pd

VOTES_XLSX = "consolidated_votes_table_simplified.xlsx"
LABELLED_DIVISIONS = "all_divisions_labelled.csv"

NUMERIC_AGG_COLUMNS = [
    "OverallRebellionRate", "TotalRebels", "TotalVotesCast", "Turnout",
    "Labour_whip", "Cons_whip", "Gov_Share", "Opp_Share",
    "Party_Distance", "Gov_Cohesion", "Opp_Cohension", "All_Party_Distance",
    "TotalAye", "TotalNo",
]
FIRST_AGG_COLUMNS = ["Date", "BillOrDebateName", "Government", "Party", "Opposition", "VoteType"]

POSSIBLE_FREE_VOTE = "Possible Free Vote - needs verification"


def extract_debate_id(raw_value):
    """Same logic as hansard/fetch_debate_transcripts.py -- handles both a
    full hansard.parliament.uk URL and a bare ID."""
    if not isinstance(raw_value, str):
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    match = re.search(r"/debates/([^/]+)", raw_value)
    if match:
        return match.group(1)
    return raw_value


def load_division_metadata(votes_xlsx, labelled_divisions):
    votes = pd.read_excel(votes_xlsx, sheet_name="Sheet1")
    votes.columns = [c.strip() for c in votes.columns]

    sampled = votes[votes["Sample"].astype(str).str.strip().str.lower() == "x"].copy()
    sampled["DebateId"] = sampled["Debate"].apply(extract_debate_id)
    missing = sampled["DebateId"].isna().sum()
    if missing:
        print(f"WARNING: {missing} sampled division(s) have no usable Debate value -- skipped.")
    sampled = sampled.dropna(subset=["DebateId"]).copy()
    sampled["DebateId"] = sampled["DebateId"].str.strip().str.upper()

    labelled = pd.read_csv(labelled_divisions)
    labelled["DivisionId"] = pd.to_numeric(labelled["DivisionId"], errors="coerce").astype("Int64")
    sampled["DivisionId"] = pd.to_numeric(sampled["DivisionId"], errors="coerce").astype("Int64")
    sampled = sampled.merge(labelled[["DivisionId", "VoteType"]], on="DivisionId", how="left")

    print(f"{len(sampled)} sampled division(s) across {sampled['DebateId'].nunique()} debate(s).")
    return sampled


def aggregate_to_debate_level(sampled):
    agg = {col: "mean" for col in NUMERIC_AGG_COLUMNS if col in sampled.columns}
    agg.update({col: "first" for col in FIRST_AGG_COLUMNS if col in sampled.columns})

    debate_level = sampled.groupby("DebateId").agg(agg)
    debate_level["NumDivisions"] = sampled.groupby("DebateId").size()
    debate_level["NumDistinctBillNames"] = sampled.groupby("DebateId")["BillOrDebateName"].nunique()
    debate_level["AnyPossibleFreeVote"] = sampled.groupby("DebateId")["VoteType"].apply(
        lambda s: (s == POSSIBLE_FREE_VOTE).any()
    )
    debate_level["FreeVoteDivisionShare"] = sampled.groupby("DebateId")["VoteType"].apply(
        lambda s: (s == POSSIBLE_FREE_VOTE).mean()
    )
    return debate_level.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="classify_debates.py's debate_predictions_by_debate.csv")
    parser.add_argument("--votes-xlsx", default=VOTES_XLSX)
    parser.add_argument("--labelled-divisions", default=LABELLED_DIVISIONS)
    parser.add_argument("--output", default="debate_features.csv")
    args = parser.parse_args()

    sampled = load_division_metadata(args.votes_xlsx, args.labelled_divisions)
    debate_metadata = aggregate_to_debate_level(sampled)

    predictions = pd.read_csv(args.predictions)
    predictions["DebateId"] = predictions["DebateId"].astype(str).str.strip().str.upper()
    print(f"Loaded {len(predictions)} debate(s) with classifier predictions from {args.predictions}.")

    combined = predictions.merge(debate_metadata, on="DebateId", how="inner")
    print(
        f"{len(combined)} debate(s) matched between predictions and vote metadata "
        f"({len(predictions) - len(combined)} prediction-only, "
        f"{len(debate_metadata) - len(combined)} metadata-only -- not merged)."
    )

    combined.to_csv(args.output, index=False)
    print(f"Saved {len(combined)} row(s), {len(combined.columns)} column(s) to {args.output}")


if __name__ == "__main__":
    main()
