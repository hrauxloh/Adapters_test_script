"""
WHAT THIS SCRIPT DOES
----------------------
Collects every House of Commons debate section that had a recorded
division (an actual vote) within a date range, and saves them as one
summary table — one row per debate section, no per-item text.

Uses queryParameters.withDivision=true on /search/debates.json, which
filters the results down to only debate sections with a division, so
there's no need to fetch every debate and cross-reference — the API does
the filtering for us.

Important: TotalResultCount (a field this API returns) was confirmed
earlier not to reflect an accurate filtered count, so this never trusts
it. Instead it pages with skip/take and stops the moment a page comes
back empty — that's the only reliable way to know when everything's been
collected.

Menu of what happens when you run this file, in order:
  1. Request page after page of divisions-only debate search results for
     the given date range, stopping when a page comes back empty.
  2. Save every result to one CSV, one row per debate section.
  3. Print a short summary (how many found, date range covered).

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


def fetch_page(params, max_attempts=3):
    # A long, many-page pull is more exposed to one transient network
    # hiccup than a single request would be — a short retry with backoff
    # means one dropped connection doesn't sink a whole day/date-range run.
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{BASE_URL}{SEARCH_PATH}", params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise
            wait = 2 ** attempt
            print(f"    request failed ({exc}); retrying in {wait}s (attempt {attempt}/{max_attempts})")
            time.sleep(wait)


def fetch_all_divisions(start_date, end_date, house="Commons", take=20, sleep_seconds=0.2):
    """Page through every division-bearing debate section in the date range.

    Stops on the first empty page — never trusts TotalResultCount, which
    was confirmed unreliable during earlier exploration of this API.

    If a page ultimately fails after retries, returns whatever was already
    collected instead of losing it — the caller can still save a partial
    result rather than getting nothing at all from a long pull.
    """
    all_results = []
    skip = 0
    page = 1

    while True:
        params = {
            "queryParameters.house": house,
            "queryParameters.startDate": start_date,
            "queryParameters.endDate": end_date,
            "queryParameters.withDivision": "true",
            "queryParameters.skip": skip,
            "queryParameters.take": take,
        }
        try:
            data = fetch_page(params)
        except requests.RequestException as exc:
            print(f"  Giving up on page {page} after retries: {exc}")
            print(f"  Returning the {len(all_results)} result(s) collected so far.")
            break

        results = data.get("Results", [])
        print(f"  page {page} (skip={skip}): {len(results)} result(s)")
        if not results:
            break

        all_results.extend(results)
        skip += take
        page += 1
        time.sleep(sleep_seconds)  # be polite to a public API, don't hammer it

    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--house", default="Commons")
    parser.add_argument("--take", type=int, default=20, help="Results per page.")
    parser.add_argument("--output", default="divisions_summary.csv")
    args = parser.parse_args()

    print(f"Collecting {args.house} debates with a division, "
          f"{args.start_date} to {args.end_date}...")
    results = fetch_all_divisions(args.start_date, args.end_date, args.house, args.take)

    if not results:
        print("\nNo division-bearing debates found in this date range.")
        return

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)

    print(f"\nFound {len(df)} debate section(s) with a division.")
    print(f"Saved to {args.output}")
    print(f"\nColumns: {list(df.columns)}")
    print(df.head(10).to_string())


if __name__ == "__main__":
    main()
