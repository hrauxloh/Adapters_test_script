"""
WHAT THIS SCRIPT DOES
----------------------
Collects every recorded House of Commons division (an actual vote) within
a date range, WITH the full vote breakdown — for each division, saves:
  - its own summary (date, title, Aye/No counts, EVEL flags, etc.)
  - every individual member's vote (Aye / No / teller / no-vote-recorded),
    with their party and other details

No turnout/attendance filtering happens here — this collects everything
the API gives, so filtering (e.g. "only divisions with >= 90% turnout")
can be done afterwards from the saved data, without needing to re-fetch
anything.

Queries commonsvotes-api.parliament.uk DIRECTLY — a separate, dedicated
API from hansard-api.parliament.uk that this script used to (wrongly)
chain through. Confirmed by a real run against its live Swagger spec:
  - /data/divisions.json/search takes real startDate/endDate/skip/take
    parameters that actually filter (unlike hansard-api's search, whose
    withDivision filter and TotalResultCount both turned out to be
    unreliable) — no "house" parameter exists because this API only ever
    covers the Commons.
  - Search results already include AyeCount/NoCount/tellers directly, but
    the full member-by-member Ayes/Noes/NoVoteRecorded lists come back
    empty — one follow-up call per division, to
    /data/division/{DivisionId}.json, fills those in.

This is a big simplification over the old approach: no more fetching
every debate from hansard-api, filtering by DebateSection, or guessing
which field holds a division's ID — DivisionId is a confirmed, reliable
field straight from the search results.

Still splits the requested range into calendar-month chunks as a
precaution — hansard-api's search endpoint was confirmed to silently cap
how far a big multi-year query's pagination reaches, and since both APIs
share the same query-parameter conventions (likely the same underlying
platform), the same caution is kept here even though it hasn't been
specifically confirmed to affect this API too.

RESUMABLE BY DESIGN, for long unattended runs (e.g. multiple years, run
from a laptop that might lose wifi, or a Colab session that might
disconnect): each month's results are saved as their own pair of files
(divisions_YYYY-MM.csv, votes_YYYY-MM.csv) in --output-dir, written the
moment that month is done. Before doing any work for a month, the script
checks whether its divisions_YYYY-MM.csv already exists and skips the
month entirely if so (including months that genuinely found zero
divisions, saved as an empty file, so "checked, found nothing" is
distinguishable from "not checked yet"). Point --output-dir at a mounted
Google Drive folder, not local Colab storage, so results survive even if
the runtime itself is lost.

Menu of what happens when you run this file, in order:
  1. Split the requested date range into calendar-month chunks.
  2. For each month, in order: if already done (its divisions file
     exists), skip it.
  3. Otherwise: page through /data/divisions.json/search for that month,
     stopping when a page comes back empty.
  4. For each division found, call /data/division/{DivisionId}.json for
     the full detail (member-level Ayes/Noes/tellers/no-vote-recorded).
  5. Save that month's division summaries and every individual member
     vote to their own two files, before moving to the next month.
  6. Print a short summary at the end.

Usage:
    python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-31 --output-dir /content/drive/MyDrive/hansard_divisions
    python collect_divisions.py --start-date 2010-01-01 --end-date 2023-12-31 --output-dir /content/drive/MyDrive/hansard_divisions

    (re-running the same command later resumes automatically — already-done months are skipped)
"""

import argparse
import calendar
import os
import time
from datetime import date

import pandas as pd
import requests

BASE_URL = "https://commonsvotes-api.parliament.uk"
SEARCH_PATH = "/data/divisions.json/search"

# Scalar fields to keep from a division's full detail record (the nested
# member lists — Ayes, Noes, AyeTellers, NoTellers, NoVoteRecorded — are
# handled separately, one row per member, in the votes table).
DIVISION_FIELDS = [
    "DivisionId", "Date", "PublicationUpdated", "Number", "IsDeferred",
    "EVELType", "EVELCountry", "Title", "AyeCount", "NoCount",
    "DoubleMajorityAyeCount", "DoubleMajorityNoCount", "FriendlyDescription",
    "FriendlyTitle", "RemoteVotingStart", "RemoteVotingEnd",
]

# Each of these list-valued fields on a division's detail becomes rows in
# the votes table, tagged with this Role label.
VOTE_LIST_FIELDS = {
    "Ayes": "Aye",
    "Noes": "No",
    "AyeTellers": "AyeTeller",
    "NoTellers": "NoTeller",
    "NoVoteRecorded": "NoVoteRecorded",
}
MEMBER_FIELDS = [
    "MemberId", "Name", "Party", "SubParty", "PartyColour",
    "PartyAbbreviation", "MemberFrom", "ListAs", "ProxyName",
]


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


def fetch_all_divisions(start_date, end_date, take=25, sleep_seconds=0.2):
    """Page through every division in the date range via commonsvotes-api's
    search endpoint. Confirmed real params: startDate, endDate, skip, take.

    Stops on the first empty page.
    """
    all_results = []
    skip = 0
    page = 1

    while True:
        params = {
            "queryParameters.startDate": start_date,
            "queryParameters.endDate": end_date,
            "queryParameters.skip": skip,
            "queryParameters.take": take,
        }
        try:
            data = fetch_with_retry(f"{BASE_URL}{SEARCH_PATH}", params)
        except requests.RequestException as exc:
            print(f"  Giving up on page {page} after retries: {exc}")
            print(f"  Returning the {len(all_results)} division(s) collected so far.")
            break

        results = data if isinstance(data, list) else []
        print(f"  page {page} (skip={skip}): {len(results)} division(s)")
        if not results:
            break

        all_results.extend(results)
        skip += take
        page += 1
        time.sleep(sleep_seconds)  # be polite to a public API, don't hammer it

    return all_results


def fetch_division_detail(division_id, sleep_seconds=0.2):
    url = f"{BASE_URL}/data/division/{division_id}.json"
    try:
        detail = fetch_with_retry(url)
    except requests.RequestException as exc:
        print(f"    could not fetch full detail for division {division_id}: {exc}")
        return None
    finally:
        time.sleep(sleep_seconds)
    return detail


def build_rows(source):
    """Turn one division's data (either a full detail record, or just the
    search-result summary if the detail call failed) into a
    (division_row, vote_rows) pair.
    """
    division_row = {field: source.get(field) for field in DIVISION_FIELDS}

    vote_rows = []
    for list_field, role in VOTE_LIST_FIELDS.items():
        for member in source.get(list_field) or []:
            vote_row = {"DivisionId": source.get("DivisionId"), "Role": role}
            for field in MEMBER_FIELDS:
                vote_row[field] = member.get(field)
            vote_rows.append(vote_row)

    return division_row, vote_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--take", type=int, default=25, help="Results per page.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder to save one divisions_YYYY-MM.csv + votes_YYYY-MM.csv pair per "
        "month into. Point this at a mounted Google Drive folder for a long run, so "
        "results survive even if the Colab runtime itself is lost.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    chunks = list(month_chunks(args.start_date, args.end_date))
    print(f"Splitting {args.start_date} to {args.end_date} into {len(chunks)} month-chunk(s).")

    total_divisions = 0
    skipped = 0

    for chunk_i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        month_label = chunk_start[:7]  # "YYYY-MM"
        divisions_file = os.path.join(args.output_dir, f"divisions_{month_label}.csv")
        votes_file = os.path.join(args.output_dir, f"votes_{month_label}.csv")

        if os.path.exists(divisions_file):
            print(f"Month {chunk_i}/{len(chunks)}: {month_label} — already done, skipping "
                  f"({divisions_file})")
            skipped += 1
            continue

        print(f"\n{'=' * 60}\nMonth {chunk_i}/{len(chunks)}: {chunk_start} to {chunk_end}\n{'=' * 60}")

        summaries = fetch_all_divisions(chunk_start, chunk_end, args.take)
        print(f"Found {len(summaries)} division(s) this month.")

        division_rows = []
        vote_rows = []

        for i, summary in enumerate(summaries, start=1):
            division_id = summary.get("DivisionId")
            detail = fetch_division_detail(division_id) if division_id is not None else None
            source = detail if detail is not None else summary

            division_row, these_vote_rows = build_rows(source)
            division_rows.append(division_row)
            vote_rows.extend(these_vote_rows)

            status = "full detail" if detail is not None else "search summary only (detail call failed)"
            print(f"  [{i}/{len(summaries)}] Division {division_id}: {summary.get('Title')} "
                  f"({summary.get('AyeCount')} Aye / {summary.get('NoCount')} No) — {status}")

        total_divisions += len(division_rows)

        # Written even when empty — that's what marks this month as
        # "checked, found nothing" rather than "not done yet" for the
        # resume check above.
        pd.DataFrame(division_rows).to_csv(divisions_file, index=False)
        pd.DataFrame(vote_rows).to_csv(votes_file, index=False)
        print(f"Saved {len(division_rows)} division(s) to {divisions_file}")
        print(f"Saved {len(vote_rows)} individual vote row(s) to {votes_file}")

    print(f"\n{'=' * 60}")
    print(f"Done this run: {total_divisions} division(s) found "
          f"({skipped} already-done month(s) skipped).")
    print(f"Per-month files are in {args.output_dir}")
    print(f"To combine them later:\n"
          f"  import glob, pandas as pd\n"
          f"  divisions = pd.concat([pd.read_csv(f) for f in glob.glob('{args.output_dir}/divisions_*.csv')])\n"
          f"  votes = pd.concat([pd.read_csv(f) for f in glob.glob('{args.output_dir}/votes_*.csv')])")


if __name__ == "__main__":
    main()
