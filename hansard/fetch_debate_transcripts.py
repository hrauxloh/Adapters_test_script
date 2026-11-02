"""
WHAT THIS SCRIPT DOES
----------------------
Step 1 of the polarization-vs-metadata analysis: just retrieve the raw
debate transcripts for the sampled divisions, and save them. No text
processing, no classification, no joining to the vote metadata yet —
that all comes later, once the raw data is actually in hand and looks
right.

Reads consolidated_votes_table_simplified.xlsx (Sheet1), keeps only rows
where the Sample column is "x" (case-insensitive, so both "x" and "X"
count), and looks at their Debate column for the Hansard debate ID.

The Debate column is a MIX of two formats, confirmed by inspecting the
real file:
  - a full URL, e.g. https://hansard.parliament.uk/Commons/2016-03-22/
    debates/16032232000001/BudgetResolutionsAndEconomicSituation
    (the ID is the path segment right after "/debates/")
  - a bare ID directly, e.g. 6982211D-4150-48BD-8986-41F8886216FB

Both are handled here. The same debate ID often repeats across several
rows (one debate can have multiple divisions) — this fetches each UNIQUE
id only once, not once per row.

RESUMABLE BY DESIGN, same as collect_divisions.py: each debate's raw JSON
is saved as its own file (debate_<id>.json) in --output-dir, the moment
it's fetched. Before fetching, it checks whether that file already
exists and skips it if so — safe to interrupt and re-run the same
command, already-fetched debates won't be re-downloaded.

Menu of what happens when you run this file, in order:
  1. Load the spreadsheet, filter to Sample == "x"/"X" rows.
  2. Extract a clean debate ID from each row's Debate value (URL or bare
     ID), and build the de-duplicated set of IDs actually needing a
     fetch.
  3. For each ID not already saved: call hansard-api's
     /debates/debate/{id}.json, save the raw response.
  4. Save a manifest CSV mapping every sampled row to its extracted ID
     and whether that debate was fetched successfully, for joining back
     to the vote metadata later.
  5. Print a summary: how many unique debates, how many fetched OK, how
     many failed (and why).

Usage:
    python fetch_debate_transcripts.py --input consolidated_votes_table_simplified.xlsx --output-dir debate_transcripts
"""

import argparse
import json
import os
import re
import time

import pandas as pd
import requests

BASE_URL = "https://hansard-api.parliament.uk"


def fetch_with_retry(url, max_attempts=3):
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt
            print(f"    request failed ({exc}); retrying in {wait}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)


def extract_debate_id(raw_value):
    """Handle both a full hansard.parliament.uk URL and a bare ID."""
    if not isinstance(raw_value, str):
        return None
    raw_value = raw_value.strip()
    if not raw_value:
        return None
    match = re.search(r"/debates/([^/]+)", raw_value)
    if match:
        return match.group(1)
    return raw_value  # already a bare ID


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path or URL to consolidated_votes_table_simplified.xlsx")
    parser.add_argument("--sheet", default="Sheet1")
    parser.add_argument("--sample-column", default="Sample")
    parser.add_argument("--debate-column", default="Debate")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", default="debate_fetch_manifest.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    print(f"Loaded {len(df)} row(s) from {args.sheet!r}.")

    sampled = df[df[args.sample_column].astype(str).str.strip().str.lower() == "x"].copy()
    print(f"{len(sampled)} row(s) with {args.sample_column} == 'x' (case-insensitive).")

    sampled["ExtractedDebateId"] = sampled[args.debate_column].apply(extract_debate_id)
    missing = sampled["ExtractedDebateId"].isna().sum()
    if missing:
        print(f"WARNING: {missing} sampled row(s) have no usable {args.debate_column} value — skipped.")

    unique_ids = sorted(sampled["ExtractedDebateId"].dropna().unique())
    print(f"{len(unique_ids)} unique debate ID(s) to fetch "
          f"(from {len(sampled) - missing} sampled rows — some debates repeat across rows).")

    fetch_status = {}
    for i, debate_id in enumerate(unique_ids, start=1):
        out_path = os.path.join(args.output_dir, f"debate_{debate_id}.json")
        if os.path.exists(out_path):
            print(f"  [{i}/{len(unique_ids)}] {debate_id}: already fetched, skipping")
            fetch_status[debate_id] = "already_done"
            continue

        url = f"{BASE_URL}/debates/debate/{debate_id}.json"
        try:
            data = fetch_with_retry(url)
        except requests.RequestException as exc:
            print(f"  [{i}/{len(unique_ids)}] {debate_id}: FAILED ({exc})")
            fetch_status[debate_id] = f"failed: {exc}"
            continue

        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        n_items = len(data.get("Items", [])) if isinstance(data, dict) else 0
        print(f"  [{i}/{len(unique_ids)}] {debate_id}: saved ({n_items} item(s))")
        fetch_status[debate_id] = "fetched"
        time.sleep(0.2)  # be polite to a public API

    sampled["FetchStatus"] = sampled["ExtractedDebateId"].map(fetch_status).fillna("no_id")
    manifest_path = os.path.join(args.output_dir, args.manifest)
    sampled.to_csv(manifest_path, index=False)

    print(f"\n{'=' * 60}")
    status_counts = sampled["FetchStatus"].value_counts()
    print(status_counts.to_string())
    print(f"\nManifest saved to {manifest_path}")
    print(f"Debate JSON files saved to {args.output_dir}/debate_<id>.json")


if __name__ == "__main__":
    main()
