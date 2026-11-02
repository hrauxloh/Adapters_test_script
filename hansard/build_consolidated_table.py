"""
WHAT THIS SCRIPT DOES
----------------------
Builds the one-row-per-division consolidated table: bill/debate name,
division reference, overall Aye/No, every party's Aye/No split, every
party's rebel count, and an overall rebellion rate — all in a single CSV.

Reuses compute_rebellion.py's tested rebellion logic (rebellion = votes
against a party's own majority direction in that division) rather than
recomputing it separately, so the two stay consistent.

Note on "DivisionRef": there's no real Hansard volume+column citation in
this data (that approach was tried and abandoned — see the git history
for compute_rebellion.py's siblings). This uses the division's own
Number field and Date instead, the closest identifying information
actually available.

Menu of what happens when you run this file, in order:
  1. Load every non-empty divisions_*.csv and votes_*.csv.
  2. Compute each party's Aye/No counts and rebel count per division
     (via compute_rebellion.py's functions).
  3. Pivot those into wide columns: {Party}_Aye, {Party}_No,
     {Party}_Rebels, one pair+one per party.
  4. Merge with each division's own Date/Number/Title/AyeCount/NoCount,
     plus the overall TotalRebels/OverallRebellionRate.
  5. Save the single consolidated CSV.

Usage:
    python build_consolidated_table.py --input-dir /content/drive/MyDrive/hansard_divisions
"""

import argparse

from compute_rebellion import compute_division_rebellion, compute_party_rebellion, read_csvs_safely


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", default="consolidated_votes_table.csv")
    args = parser.parse_args()

    divisions = read_csvs_safely(f"{args.input_dir}/divisions_*.csv")
    votes = read_csvs_safely(f"{args.input_dir}/votes_*.csv")
    if divisions.empty or votes.empty:
        print("No data found — nothing to build.")
        return

    party_rebellion = compute_party_rebellion(votes)
    division_rebellion = compute_division_rebellion(party_rebellion)

    # Wide: one triple of columns per party (Aye count, No count, rebel count)
    party_wide = party_rebellion.pivot(
        index="DivisionId", columns="Party", values=["PartyAyeCount", "PartyNoCount", "RebelCount"]
    )
    party_wide.columns = [
        f"{party}_{'Aye' if field == 'PartyAyeCount' else 'No' if field == 'PartyNoCount' else 'Rebels'}"
        for field, party in party_wide.columns
    ]
    party_wide = party_wide.fillna(0)
    party_wide[[c for c in party_wide.columns]] = party_wide[[c for c in party_wide.columns]].astype(int)
    party_wide = party_wide.reset_index()

    consolidated = (
        divisions[["DivisionId", "Date", "Number", "Title", "AyeCount", "NoCount"]]
        .merge(party_wide, on="DivisionId", how="left")
        .merge(division_rebellion, on="DivisionId", how="left")
        .rename(columns={
            "Number": "DivisionRef",  # closest available reference -- no true Hansard volume/column ref
            "Title": "BillOrDebateName",
            "AyeCount": "TotalAye",
            "NoCount": "TotalNo",
        })
    )

    print(consolidated.head())
    consolidated.to_csv(args.output, index=False)
    print(f"\nSaved {len(consolidated)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
