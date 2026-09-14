"""
WHAT THIS SCRIPT DOES
----------------------
Collects every House of Commons debate section that had a recorded
division (an actual vote) within a date range, and saves them as one
summary table — one row per debate section, no per-item text.

queryParameters.withDivision=true on /search/debates.json was expected to
filter results down to division-bearing debates server-side, but running
it confirmed it doesn't actually filter anything — it returns every
debate regardless. (An earlier check had already hinted at this: a debate
withDivision=true claimed had a division came back with zero divisions
when checked directly.) So this script doesn't trust that parameter at
all — it fetches EVERY debate in the date range, then checks each one
directly against /debates/divisions/{id}.json, keeping only the ones
that genuinely come back with at least one division.

This means more API calls than the filtered approach would have needed,
but it's checking ground truth instead of trusting a parameter that's now
failed twice.

Important: TotalResultCount (a field this API returns) was confirmed
earlier not to reflect an accurate filtered count either, so this never
trusts it. Instead it pages with skip/take and stops the moment a page
comes back empty — that's the only reliable way to know when everything's
been collected.

Menu of what happens when you run this file, in order:
  1. Request page after page of ALL debates for the given date range,
     stopping when a page comes back empty.
  2. For each debate found, call /debates/divisions/{id}.json and keep it
     only if that comes back with at least one division.
  3. Save the kept debates to one CSV, one row per debate section.
  4. Print a short summary (how many checked, how many kept).

Usage:
    python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-01
    python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-22 --output may_divisions.csv
"""

import argparse
import time

import pandas as pd
import requests

BASE_URL = "https://hansard-api.parliament.uk"
SEARCH_PATH = "/search/debates.json"


def fetch_with_retry(url, params=None, max_attempts=3):
    # A long, many-call pull is more exposed to one transient network
    # hiccup than a single request would be — a short retry with backoff
    # means one dropped connection doesn't sink a whole date-range run.
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt
            print(f"    request failed ({exc}); retrying in {wait}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)


def fetch_all_debates(start_date, end_date, house="Commons", take=20, sleep_seconds=0.2):
    """Page through every debate section in the date range (no division
    filter — see module docstring for why that parameter isn't trusted).

    Stops on the first empty page — never trusts TotalResultCount.
    """
    all_results = []
    skip = 0
    page = 1

    while True:
        params = {
            "queryParameters.house": house,
            "queryParameters.startDate": start_date,
            "queryParameters.endDate": end_date,
            "queryParameters.skip": skip,
            "queryParameters.take": take,
        }
        try:
            data = fetch_with_retry(f"{BASE_URL}{SEARCH_PATH}", params)
        except requests.RequestException as exc:
            print(f"  Giving up on page {page} after retries: {exc}")
            print(f"  Returning the {len(all_results)} debate(s) collected so far.")
            break

        results = data.get("Results", [])
        print(f"  page {page} (skip={skip}): {len(results)} debate(s)")
        if not results:
            break

        all_results.extend(results)
        skip += take
        page += 1
        time.sleep(sleep_seconds)  # be polite to a public API, don't hammer it

    return all_results


def has_division(debate_section_ext_id, sleep_seconds=0.2):
    url = f"{BASE_URL}/debates/divisions/{debate_section_ext_id}.json"
    try:
        divisions = fetch_with_retry(url)
    except requests.RequestException as exc:
        print(f"    could not check {debate_section_ext_id}: {exc} — treating as no division")
        return False
    finally:
        time.sleep(sleep_seconds)
    return isinstance(divisions, list) and len(divisions) > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--house", default="Commons")
    parser.add_argument("--take", type=int, default=20, help="Results per page.")
    parser.add_argument(
        "--debate-section",
        default="Commons Chamber",
        help="Only keep debates whose DebateSection matches this exactly (e.g. excludes "
        "Westminster Hall, Public Bill Committees, etc.). Pass an empty string to keep all.",
    )
    parser.add_argument("--output", default="divisions_summary.csv")
    args = parser.parse_args()

    print(f"Collecting all {args.house} debates, {args.start_date} to {args.end_date}...")
    all_debates = fetch_all_debates(args.start_date, args.end_date, args.house, args.take)
    print(f"Found {len(all_debates)} debate(s) total.")

    if args.debate_section:
        before = len(all_debates)
        all_debates = [d for d in all_debates if d.get("DebateSection") == args.debate_section]
        print(f"Kept {len(all_debates)} of {before} matching DebateSection == {args.debate_section!r}.")

    print("Checking each remaining debate for an actual division...")

    kept = []
    for i, debate in enumerate(all_debates, start=1):
        ext_id = debate["DebateSectionExtId"]
        if has_division(ext_id):
            kept.append(debate)
            print(f"  [{i}/{len(all_debates)}] {ext_id}  HAS a division  ({debate.get('Title')})")
        else:
            print(f"  [{i}/{len(all_debates)}] {ext_id}  no division")

    if not kept:
        print("\nNo division-bearing debates found in this date range.")
        return

    df = pd.DataFrame(kept)
    df.to_csv(args.output, index=False)

    print(f"\n{len(kept)} of {len(all_debates)} debate(s) actually had a division.")
    print(f"Saved to {args.output}")
    print(f"\nColumns: {list(df.columns)}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
