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

A long date range (multiple years) also ran into a THIRD API quirk:
requesting 2010-01-01 to 2024-12-31 in one go only ever returned results
from the last couple of months of the range. Results appear to come back
newest-first, and the search endpoint seems to silently cap how far
pagination actually reaches for one query — so a big range just returns
its most recent slice and then empty pages, long before covering
everything. To work around this, this script now queries one calendar
month at a time and stitches the results together, which also means
results get written to the output CSV after each month instead of only
at the very end — a long multi-year run surviving a crash/disconnect
partway through won't lose everything collected so far.

Menu of what happens when you run this file, in order:
  1. Split the requested date range into calendar-month chunks.
  2. For each month: request page after page of ALL debates in that
     month, stopping when a page comes back empty.
  3. For each debate found, call /debates/divisions/{id}.json and keep it
     only if that comes back with at least one division.
  4. Append that month's kept debates to the output CSV immediately.
  5. Print a short summary (how many checked, how many kept, in total).

Usage:
    python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-01
    python collect_divisions.py --start-date 2020-01-01 --end-date 2024-12-31 --output divisions_2020_2024.csv
"""

import argparse
import calendar
import os
import time
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://hansard-api.parliament.uk"
SEARCH_PATH = "/search/debates.json"


def month_chunks(start_date, end_date):
    """Yield (chunk_start, chunk_end) ISO date strings, one per calendar
    month, covering [start_date, end_date] inclusive.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = date(start.year, start.month, 1)

    while current <= end:
        last_day = calendar.monthrange(current.year, current.month)[1]
        chunk_start = max(current, start)
        chunk_end = min(date(current.year, current.month, last_day), end)
        yield chunk_start.isoformat(), chunk_end.isoformat()

        if current.month == 12:
            current = date(current.year + 1, 1, 1)
        else:
            current = date(current.year, current.month + 1, 1)


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

    chunks = list(month_chunks(args.start_date, args.end_date))
    print(f"Splitting {args.start_date} to {args.end_date} into {len(chunks)} month-chunk(s).")

    total_checked = 0
    total_kept = 0

    for chunk_i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"\n{'=' * 60}\nMonth {chunk_i}/{len(chunks)}: {chunk_start} to {chunk_end}\n{'=' * 60}")

        debates = fetch_all_debates(chunk_start, chunk_end, args.house, args.take)
        print(f"Found {len(debates)} debate(s) this month.")

        if args.debate_section:
            before = len(debates)
            debates = [d for d in debates if d.get("DebateSection") == args.debate_section]
            print(f"Kept {len(debates)} of {before} matching DebateSection == {args.debate_section!r}.")

        kept = []
        for i, debate in enumerate(debates, start=1):
            ext_id = debate["DebateSectionExtId"]
            if has_division(ext_id):
                kept.append(debate)
                print(f"  [{i}/{len(debates)}] {ext_id}  HAS a division  ({debate.get('Title')})")
            else:
                print(f"  [{i}/{len(debates)}] {ext_id}  no division")

        total_checked += len(debates)
        total_kept += len(kept)

        if kept:
            # Append immediately — a crash/disconnect on a later month
            # doesn't lose the months already collected.
            file_exists = os.path.exists(args.output)
            pd.DataFrame(kept).to_csv(args.output, mode="a", header=not file_exists, index=False)
            print(f"Appended {len(kept)} division-bearing debate(s) to {args.output}")

    print(f"\n{'=' * 60}")
    print(f"Done. {total_kept} of {total_checked} debate(s) across the full range had a division.")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
