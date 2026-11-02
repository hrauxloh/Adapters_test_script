"""
WHAT THIS SCRIPT DOES
----------------------
Collects every Commons/Lords debate that looks like a Bill's Second
Reading, from 1945-01-01 to today, by querying hansard-api.parliament.uk's
debate search endpoint.

IMPORTANT CAVEAT, read before trusting the output: hansard-api has no
confirmed structured field for a debate's legislative stage (First
Reading / Second Reading / Committee / Report / Third Reading) — the
earlier exploration in this project (hansard/README.md, "Status: Step 1
only") only confirmed the fields DebateSection, SittingDate, House,
Title, Rank, DebateSectionExtId on a debate search result, and NONE of
those is a stage field. So "Second Reading" here is a TEXT filter on the
debate's Title, not a guaranteed, API-native category:
  - Server-side: the search endpoint's searchTerm parameter is passed
    "Second Reading" to narrow what's fetched (reduces how much has to be
    paged through across 80 years). searchTerm's real matching behaviour
    (title-only? full debate text?) has never been confirmed against a
    live response.
  - Client-side (the authoritative filter actually applied before a row
    is saved): the debate's own Title must contain "second reading"
    case-insensitively. This is what "filtering for the inclusion of
    second reading" means in this script's output.
  - Because searchTerm's real behaviour is unconfirmed, this script also
    prints, per month, how many server-returned debates did NOT pass the
    client-side Title check -- a big share of those would mean searchTerm
    is matching on something other than the title (e.g. body text
    mentioning "second reading" in passing), which is worth knowing before
    trusting the totals.
  - SPOT-CHECK A SAMPLE of the output's Title column against the real
    Hansard record before using this for anything load-bearing -- the
    same discipline this project has applied to every other unverified
    API assumption (TotalResultCount, withDivision, etc., all turned out
    to be unreliable on first real use).

Same defensive month-chunking as collect_divisions.py, for the same
confirmed reason: hansard-api's search endpoint was confirmed to silently
cap how far a big multi-year query's pagination reaches, so even an
80-year pull is split into one call-sequence per calendar month rather
than one huge range.

RESUMABLE BY DESIGN: each month's result is its own
second_readings_YYYY-MM.csv in --output-dir, written the moment that
month is done. A month whose file already exists is skipped entirely on
a re-run (including a month that genuinely found zero matches, saved as
an empty-but-headered file, so "checked, found nothing" stays
distinguishable from "not checked yet").

Menu of what happens when you run this file, in order:
  1. Split [--start-date, --end-date] into calendar-month chunks.
  2. For each month, in order: skip if already done.
  3. Otherwise, page through /search/debates.json for that month with
     searchTerm="Second Reading" (and --house, if given), stopping on the
     first empty page -- never trusting TotalResultCount.
  4. Keep only the results whose Title contains "second reading"
     case-insensitively; print how many were dropped by that check.
  5. Save that month's kept rows to its own file before moving on.
  6. Print a final summary and a snippet for combining all months.

Usage:
    python collect_second_readings.py --output-dir second_readings
    python collect_second_readings.py --start-date 1945-01-01 --end-date 2026-11-02 --house Commons --output-dir /content/drive/MyDrive/second_readings

    (re-running the same command later resumes automatically -- already-done months are skipped)
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
SEARCH_TERM = "Second Reading"

# Confirmed fields on a debate search result (hansard/README.md).
DEBATE_FIELDS = ["DebateSection", "SittingDate", "House", "Title", "Rank", "DebateSectionExtId"]


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
    # An 80-year pull makes many thousands of calls -- a short retry with
    # backoff means one transient network hiccup doesn't sink the run.
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


def fetch_month_debates(start_date, end_date, house, take=50, sleep_seconds=0.2):
    """Page through /search/debates.json for one month, searchTerm="Second
    Reading". Stops on the first empty page -- TotalResultCount is
    confirmed unreliable, so it's never used to decide when to stop.
    """
    all_results = []
    skip = 0
    page = 1

    while True:
        params = {
            "queryParameters.searchTerm": SEARCH_TERM,
            "queryParameters.startDate": start_date,
            "queryParameters.endDate": end_date,
            "queryParameters.skip": skip,
            "queryParameters.take": take,
        }
        if house:
            params["queryParameters.house"] = house

        try:
            data = fetch_with_retry(f"{BASE_URL}{SEARCH_PATH}", params)
        except requests.RequestException as exc:
            print(f"  Giving up on page {page} after retries: {exc}")
            print(f"  Returning the {len(all_results)} result(s) collected so far this month.")
            break

        results = (data or {}).get("Results", []) if isinstance(data, dict) else []
        print(f"  page {page} (skip={skip}): {len(results)} result(s)")
        if not results:
            break

        all_results.extend(results)
        skip += take
        page += 1
        time.sleep(sleep_seconds)  # be polite to a public API, don't hammer it

    return all_results


def is_second_reading_title(title):
    return "second reading" in (title or "").lower()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="1945-01-01", help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=date.today().isoformat(), help="YYYY-MM-DD")
    parser.add_argument(
        "--house",
        default=None,
        choices=["Commons", "Lords"],
        help="Restrict to one House. Omit to pass no house filter at all -- whether that "
        "returns both Houses or defaults to one has never been confirmed against a live "
        "response, so check the House column in the output the first time you run this.",
    )
    parser.add_argument("--take", type=int, default=50, help="Results per page.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder to save one second_readings_YYYY-MM.csv per month into. Point this at a "
        "mounted Google Drive folder for a long run, so results survive even if the Colab "
        "runtime itself is lost.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    chunks = list(month_chunks(args.start_date, args.end_date))
    print(f"Splitting {args.start_date} to {args.end_date} into {len(chunks)} month-chunk(s).")
    print(f'Server-side filter: searchTerm="{SEARCH_TERM}"'
          + (f", house={args.house}" if args.house else ", no house filter") + ".")
    print('Client-side (authoritative) filter: Title must contain "second reading" (case-insensitive).\n')

    total_kept = 0
    total_dropped = 0
    skipped_months = 0

    for chunk_i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        month_label = chunk_start[:7]  # "YYYY-MM"
        out_file = os.path.join(args.output_dir, f"second_readings_{month_label}.csv")

        if os.path.exists(out_file):
            skipped_months += 1
            continue

        print(f"\n{'=' * 60}\nMonth {chunk_i}/{len(chunks)}: {chunk_start} to {chunk_end}\n{'=' * 60}")

        results = fetch_month_debates(chunk_start, chunk_end, args.house, args.take)

        kept = [r for r in results if is_second_reading_title(r.get("Title"))]
        dropped = len(results) - len(kept)
        total_kept += len(kept)
        total_dropped += dropped
        if dropped:
            print(f"  {dropped} server-returned result(s) this month did NOT have "
                  '"second reading" in their Title -- dropped.')

        rows = [{field: r.get(field) for field in DEBATE_FIELDS} for r in kept]
        # Written even when empty -- marks this month "checked, found
        # nothing" rather than "not done yet" for the resume check above.
        # An explicit columns= keeps a real header on empty months, so
        # pd.read_csv can still read them back later.
        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=DEBATE_FIELDS)
        df.to_csv(out_file, index=False)
        print(f"Saved {len(rows)} Second Reading debate(s) to {out_file}")

    print(f"\n{'=' * 60}")
    print(f"Done this run: {total_kept} Second Reading debate(s) kept, {total_dropped} dropped by the "
          f"Title check ({skipped_months} already-done month(s) skipped).")
    print(f"Per-month files are in {args.output_dir}")
    print(f"To combine them later:\n"
          f"  import glob, pandas as pd\n"
          f"  second_readings = pd.concat([pd.read_csv(f) for f in glob.glob('{args.output_dir}/second_readings_*.csv')])")


if __name__ == "__main__":
    main()
