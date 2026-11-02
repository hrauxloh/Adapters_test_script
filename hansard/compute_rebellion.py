"""
WHAT THIS SCRIPT DOES
----------------------
Computes a rebellion rate for every party in every division, from the
votes already collected by collect_divisions.py.

Rebellion is defined here as: for a given party in a given division, find
which way most of that party's MPs voted (Aye or No) — treat that as the
party's position — then count how many of that party's MPs voted the
OTHER way. Those are the rebels. This only looks at actual Aye/No votes;
tellers and no-vote-recorded members aren't counted as "voting" for this
purpose.

This is a proxy, not a record of what the whips actually instructed —
there's no independent source here confirming the real party line, just
an assumption that "what most of the party did" approximates it. It's a
reasonable, transparent assumption (roughly what Public Whip itself
computes), but it can't tell you about a division where a party's whip
position and its majority vote genuinely differed.

Ties (a party split exactly 50/50) are treated as the party's position
being Aye, arbitrarily — flagged in the output via IsTied so these can be
handled differently if needed.

Menu of what happens when you run this file, in order:
  1. Load every non-empty divisions_*.csv and votes_*.csv, skipping any
     broken/empty files rather than crashing on them.
  2. For each (division, party), count Aye/No votes, find the majority
     direction, and count rebels (votes against that direction).
  3. Save a party-level table (one row per division+party) and a
     division-level table (one row per division, rebels/votes summed
     across all parties).

Usage:
    python compute_rebellion.py --input-dir /content/drive/MyDrive/hansard_divisions
"""

import argparse
import glob

import pandas as pd


def read_csvs_safely(pattern):
    dfs = []
    skipped = 0
    for f in glob.glob(pattern):
        try:
            df = pd.read_csv(f)
            if not df.empty:
                dfs.append(df)
        except pd.errors.EmptyDataError:
            skipped += 1
    print(f"{pattern}: loaded {len(dfs)} file(s), skipped {skipped} empty/broken file(s)")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def compute_party_rebellion(votes):
    cast_votes = votes[votes["Role"].isin(["Aye", "No"])]

    party_counts = (
        cast_votes.groupby(["DivisionId", "Party", "Role"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("Aye", "No"):
        if col not in party_counts.columns:
            party_counts[col] = 0
    party_counts = party_counts.rename(columns={"Aye": "PartyAyeCount", "No": "PartyNoCount"}).reset_index()

    party_counts["PartyTotalVotes"] = party_counts["PartyAyeCount"] + party_counts["PartyNoCount"]
    party_counts["IsTied"] = party_counts["PartyAyeCount"] == party_counts["PartyNoCount"]
    # Ties default to "Aye" as the party's position — arbitrary, flagged via IsTied.
    party_counts["MajorityDirection"] = party_counts.apply(
        lambda r: "Aye" if r["PartyAyeCount"] >= r["PartyNoCount"] else "No", axis=1
    )
    party_counts["RebelCount"] = party_counts.apply(
        lambda r: r["PartyNoCount"] if r["MajorityDirection"] == "Aye" else r["PartyAyeCount"], axis=1
    )
    party_counts["RebellionRate"] = party_counts["RebelCount"] / party_counts["PartyTotalVotes"]

    return party_counts


def compute_division_rebellion(party_rebellion):
    return (
        party_rebellion.groupby("DivisionId")
        .agg(TotalRebels=("RebelCount", "sum"), TotalVotesCast=("PartyTotalVotes", "sum"))
        .assign(OverallRebellionRate=lambda d: d["TotalRebels"] / d["TotalVotesCast"])
        .reset_index()
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--party-output", default="party_rebellion.csv")
    parser.add_argument("--division-output", default="division_rebellion.csv")
    args = parser.parse_args()

    votes = read_csvs_safely(f"{args.input_dir}/votes_*.csv")
    if votes.empty:
        print("No votes found — nothing to compute.")
        return

    party_rebellion = compute_party_rebellion(votes)
    division_rebellion = compute_division_rebellion(party_rebellion)

    print(f"\n{len(party_rebellion)} (division, party) row(s).")
    print(party_rebellion.sort_values("RebellionRate", ascending=False).head(10).to_string(index=False))

    print(f"\n{len(division_rebellion)} division(s).")
    print(division_rebellion.sort_values("OverallRebellionRate", ascending=False).head(10).to_string(index=False))

    party_rebellion.to_csv(args.party_output, index=False)
    division_rebellion.to_csv(args.division_output, index=False)
    print(f"\nSaved {args.party_output} and {args.division_output}")


if __name__ == "__main__":
    main()
