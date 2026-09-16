"""
WHAT THIS SCRIPT DOES
----------------------
Checks whether a division's Hansard reference (Volume + column range,
e.g. "413 c93-127") can actually be reconstructed, before building a
batch job that assumes it can.

Two things this checks, using ONE real division as a test case:
  1. Can hansard-api's /debates/topleveldebatebytitle.json find the
     parent debate for a division, given its date + title? (Division
     titles sometimes name a specific amendment/clause rather than the
     debate's own section title, e.g. "High Speed Rail Bill Report Stage:
     New Clause 22" — this tries the full title first, then a shortened
     version before the first colon, since an exact match isn't
     guaranteed.)
  2. If a debate is found (with its VolumeNo), can the division's actual
     column position be located within that debate's raw content
     (/debates/debate/{id}.json), by looking for column-number markers
     near wherever the division is mentioned?

Neither of these has been confirmed to work — this prints everything it
finds (or doesn't) so the real answer is visible before committing to a
1,860-row batch pipeline built on an untested assumption.

Usage:
    python explore_hansard_ref.py
    python explore_hansard_ref.py --date 2016-03-23 --title "s5 European Communities (Amendment) Act 1993"
"""

import argparse
import json

import requests

BASE_URL = "https://hansard-api.parliament.uk"


def find_debate(date, title, house="Commons"):
    print("=" * 70)
    print(f"STEP 1: finding the parent debate for {date!r} / {title!r}")
    print("=" * 70)

    candidates = [title]
    if ":" in title:
        candidates.append(title.split(":", 1)[0].strip())

    for candidate_title in candidates:
        url = f"{BASE_URL}/debates/topleveldebatebytitle.json"
        params = {"house": house, "date": date, "sectionTitle": candidate_title}
        print(f"  Trying sectionTitle={candidate_title!r}")
        try:
            response = requests.get(url, params=params, timeout=20)
        except requests.RequestException as exc:
            print(f"    Request failed: {exc}")
            continue
        print(f"    HTTP {response.status_code}")
        if response.status_code != 200:
            print(f"    Body (first 500 chars): {response.text[:500]}")
            continue
        try:
            data = response.json()
        except ValueError:
            print(f"    Not valid JSON: {response.text[:500]}")
            continue
        if not data:
            print("    Empty response — no match with this title variant.")
            continue
        print(f"    Found something! Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
        print(json.dumps(data, indent=2)[:1500])
        return data

    print("  No match found with any title variant tried.")
    return None


def list_debates_on_date(date, house="Commons"):
    print()
    print("=" * 70)
    print(f"STEP 1b: the exact-title lookup found nothing — listing every real")
    print(f"debate title on {date} instead, to see how they actually relate")
    print(f"to the division's title")
    print("=" * 70)
    url = f"{BASE_URL}/search/debates.json"
    params = {
        "queryParameters.house": house,
        "queryParameters.startDate": date,
        "queryParameters.endDate": date,
        "queryParameters.take": 50,
    }
    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return
    print(f"  HTTP {response.status_code}")
    if response.status_code != 200:
        return
    results = response.json().get("Results", [])
    print(f"  {len(results)} debate(s) on this date:")
    for r in results:
        print(f"    {r.get('DebateSection'):>20}  |  {r.get('Title')}  |  ext_id={r.get('DebateSectionExtId')}")


def find_column_for_division(debate_section_ext_id, division_title, division_aye_count):
    print()
    print("=" * 70)
    print(f"STEP 2: looking for the division's position in debate {debate_section_ext_id}")
    print("=" * 70)
    url = f"{BASE_URL}/debates/debate/{debate_section_ext_id}.json"
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return
    print(f"  HTTP {response.status_code}")
    if response.status_code != 200:
        return
    data = response.json()
    items = data.get("Items", [])
    print(f"  Debate has {len(items)} item(s). Scanning for division-related content and nearby column markers...")

    last_column = None
    for item in items:
        if item.get("HRSTag") == "hs_ColumnNumber":
            last_column = item.get("Value") or item.get("ExternalId")
        value = str(item.get("Value") or "")
        if "Division" in value or "Ayes" in value or "Noes" in value or str(division_aye_count) in value:
            print(f"    Possible division mention (last column marker seen: {last_column}):")
            print(f"      HRSTag={item.get('HRSTag')}  Value={value[:200]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2016-03-23")
    parser.add_argument("--title", default="s5 European Communities (Amendment) Act 1993")
    parser.add_argument("--aye-count", type=int, default=241)
    args = parser.parse_args()

    debate = find_debate(args.date, args.title)
    if not debate:
        list_debates_on_date(args.date)
    if debate and isinstance(debate, dict):
        ext_id = debate.get("ExternalId") or debate.get("Id")
        volume = debate.get("VolumeNo") or debate.get("Overview", {}).get("VolumeNo")
        print(f"\nVolumeNo found: {volume}")
        if ext_id:
            find_column_for_division(ext_id, args.title, args.aye_count)
        else:
            print("Couldn't find an ID field on the debate to fetch its full content.")

    print()
    print("=" * 70)
    print("Report back everything this printed — especially whether Step 1 found a")
    print("debate at all, what VolumeNo looked like, and whether Step 2 found anything")
    print("resembling a column marker near the division. That decides whether a full")
    print("Volume+column HansardRef is actually buildable, or whether Volume-only is")
    print("the realistic ceiling.")
    print("=" * 70)


if __name__ == "__main__":
    main()
