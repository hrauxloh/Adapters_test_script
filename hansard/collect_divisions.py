"""
WHAT THIS SCRIPT DOES
----------------------
Collects every recorded House of Commons division (an actual vote) within
a date range, WITH the full vote breakdown — not just "did this debate
have a division." For each division found, saves:
  - its own summary (date, title, Aye/No counts, EVEL flags, etc.)
  - every individual member's vote (Aye / No / teller / no-vote-recorded),
    with their party and other details

No turnout/attendance filtering happens here — this collects everything
the API gives, so filtering (e.g. "only divisions with >= 90% turnout")
can be done afterwards from the saved data, without needing to re-fetch
anything.

THREE CONFIRMED API QUIRKS THIS WORKS AROUND (found by running against the
live API, not guessed):
  1. queryParameters.withDivision=true on /search/debates.json does NOT
     filter anything — it returns every debate regardless. So this
     fetches every debate directly, then checks each one directly.
  2. TotalResultCount does not reflect an accurate result count. Never
     trusted — pagination stops only on an empty page.
  3. A big multi-year date range in one query only returns its most
     recent slice (results come back newest-first, with pagination
     silently capped) — so this splits any range into calendar-month
     chunks and queries each one separately.

ONE THING NOT YET CONFIRMED: what field identifies a division within
/debates/divisions/{debateSectionExtId}.json's list (needed to look up
that division's full detail via /debates/division/{divisionId}.json).
This tries a few likely field names (DivisionId, Id, ExternalId) and
prints a clear warning if none match, rather than silently losing data —
correct this once a real run shows what's actually there.

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
  3. Otherwise: fetch every debate in that month, keep only those
     matching --debate-section (default "Commons Chamber").
  4. For each debate, call /debates/divisions/{id}.json to list its
     division(s); for each division found, call
     /debates/division/{divisionId}.json for the full detail.
  5. Save that month's division summaries and every individual member
     vote to their own two files, before moving to the next month.
  6. Print a short summary at the end.

Usage:
    python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-01 --output-dir /content/drive/MyDrive/hansard_divisions
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

BASE_URL = "https://hansard-api.parliament.uk"
SEARCH_PATH = "/search/debates.json"

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

# Tried in this order against each item in /debates/divisions/{id}.json's
# list to find the ID to look up full detail with — not yet confirmed
# against a real populated response, see module docstring.
DIVISION_ID_CANDIDATE_FIELDS = ["DivisionId", "Id", "ExternalId"]


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


def fetch_division_list(debate_section_ext_id, sleep_seconds=0.2):
    """The list of divisions (if any) in one debate section — usually
    empty, since most debates never have a vote at all.
    """
    url = f"{BASE_URL}/debates/divisions/{debate_section_ext_id}.json"
    try:
        divisions = fetch_with_retry(url)
    except requests.RequestException as exc:
        print(f"    could not check {debate_section_ext_id}: {exc} — treating as no division")
        return []
    finally:
        time.sleep(sleep_seconds)
    return divisions if isinstance(divisions, list) else []


def extract_division_id(division_summary):
    for field in DIVISION_ID_CANDIDATE_FIELDS:
        if division_summary.get(field):
            return division_summary[field]
    return None


def fetch_division_detail(division_id, sleep_seconds=0.2):
    url = f"{BASE_URL}/debates/division/{division_id}.json"
    try:
        detail = fetch_with_retry(url)
    except requests.RequestException as exc:
        print(f"      could not fetch full detail for division {division_id}: {exc}")
        return None
    finally:
        time.sleep(sleep_seconds)
    return detail


def build_rows(debate, division_summary, detail):
    """Turn one division's data into (division_row, vote_rows)."""
    debate_context = {
        "DebateSectionExtId": debate.get("DebateSectionExtId"),
        "DebateSection": debate.get("DebateSection"),
        "SittingDate": debate.get("SittingDate"),
        "House": debate.get("House"),
        "DebateTitle": debate.get("Title"),
    }

    if detail is not None:
        division_row = dict(debate_context)
        for field in DIVISION_FIELDS:
            division_row[field] = detail.get(field)

        vote_rows = []
        for list_field, role in VOTE_LIST_FIELDS.items():
            for member in detail.get(list_field) or []:
                vote_row = {"DivisionId": detail.get("DivisionId"), "Role": role}
                for field in MEMBER_FIELDS:
                    vote_row[field] = member.get(field)
                vote_rows.append(vote_row)
        return division_row, vote_rows

    # No full detail available (request failed, or no ID field matched) —
    # keep whatever the list endpoint gave us rather than losing the row.
    division_row = dict(debate_context)
    division_row.update(division_summary)
    return division_row, []


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

    total_debates_checked = 0
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

        debates = fetch_all_debates(chunk_start, chunk_end, args.house, args.take)
        print(f"Found {len(debates)} debate(s) this month.")

        if args.debate_section:
            before = len(debates)
            debates = [d for d in debates if d.get("DebateSection") == args.debate_section]
            print(f"Kept {len(debates)} of {before} matching DebateSection == {args.debate_section!r}.")

        division_rows = []
        vote_rows = []

        for i, debate in enumerate(debates, start=1):
            ext_id = debate["DebateSectionExtId"]
            division_summaries = fetch_division_list(ext_id)

            if not division_summaries:
                print(f"  [{i}/{len(debates)}] {ext_id}  no division")
                continue

            print(f"  [{i}/{len(debates)}] {ext_id}  {len(division_summaries)} division(s)  "
                  f"({debate.get('Title')})")

            for division_summary in division_summaries:
                division_id = extract_division_id(division_summary)
                if division_id is None:
                    print(f"      WARNING: couldn't find an ID field on this division summary "
                          f"(tried {DIVISION_ID_CANDIDATE_FIELDS}) — raw keys: "
                          f"{list(division_summary.keys())}. Saving what the list gave us, "
                          f"without full member-level detail.")
                    detail = None
                else:
                    detail = fetch_division_detail(division_id)

                division_row, these_vote_rows = build_rows(debate, division_summary, detail)
                division_rows.append(division_row)
                vote_rows.extend(these_vote_rows)

        total_debates_checked += len(debates)
        total_divisions += len(division_rows)

        # Written even when empty — that's what marks this month as
        # "checked, found nothing" rather than "not done yet" for the
        # resume check above.
        pd.DataFrame(division_rows).to_csv(divisions_file, index=False)
        pd.DataFrame(vote_rows).to_csv(votes_file, index=False)
        print(f"Saved {len(division_rows)} division(s) to {divisions_file}")
        print(f"Saved {len(vote_rows)} individual vote row(s) to {votes_file}")

    print(f"\n{'=' * 60}")
    print(f"Done this run: {total_divisions} division(s) found across "
          f"{total_debates_checked} debate(s) checked ({skipped} already-done month(s) skipped).")
    print(f"Per-month files are in {args.output_dir}")
    print(f"To combine them later:\n"
          f"  import glob, pandas as pd\n"
          f"  divisions = pd.concat([pd.read_csv(f) for f in glob.glob('{args.output_dir}/divisions_*.csv')])\n"
          f"  votes = pd.concat([pd.read_csv(f) for f in glob.glob('{args.output_dir}/votes_*.csv')])")


if __name__ == "__main__":
    main()
