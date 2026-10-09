"""
WHAT THIS SCRIPT DOES
----------------------
Checks whether the votes/divisions detail data (Aye/No lists, tellers,
counts) actually lives on a SEPARATE API — commonsvotes-api.parliament.uk
— rather than hansard-api.parliament.uk, which is what collect_divisions.py
currently (possibly wrongly) assumes.

Real, concrete endpoint URLs were found in the source of an existing R
package (houseofcommonslibrary/clvotes) that wraps this API:
  - search:  https://commonsvotes-api.parliament.uk/data/divisions.json/search
  - single:  https://commonsvotes-api.parliament.uk/data/division/{id}.json
That package does its date filtering client-side in R, so it doesn't
confirm whether the search endpoint also accepts a server-side date-range
parameter — that's the main thing this script checks.

Menu of what happens when you run this file, in order:
  1. Try to fetch this API's Swagger spec and print every endpoint under
     "Divisions", with parameters — the real source of truth.
  2. Make a sample call to the search endpoint for a short, recent date
     range, guessing at query parameter names based on hansard-api's own
     convention (queryParameters.startDate/endDate) — print the raw
     response either way, so it's obvious if the guess was wrong.
  3. If step 2 returns anything with an ID field, follow up with the
     single-division endpoint to confirm the rich schema (AyeCount,
     Ayes, Noes, tellers, etc.) really lives here.

Usage:
    python explore_commonsvotes_api.py
    python explore_commonsvotes_api.py --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import json
from datetime import date, timedelta

import requests

BASE_URL = "https://commonsvotes-api.parliament.uk"
SEARCH_PATH = "/data/divisions.json/search"

CANDIDATE_SWAGGER_PATHS = [
    "/swagger/docs/v1",
    "/swagger/v1/swagger.json",
]


def explore_swagger():
    print("=" * 70)
    print("STEP 1: looking for the real endpoint list via the Swagger spec")
    print("=" * 70)
    for path in CANDIDATE_SWAGGER_PATHS:
        url = f"{BASE_URL}{path}"
        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as exc:
            print(f"  {url} -> request failed: {exc}")
            continue
        print(f"  {url} -> HTTP {response.status_code}")
        if response.status_code != 200:
            continue
        try:
            spec = response.json()
        except ValueError:
            print("    (response wasn't valid JSON, skipping)")
            continue

        paths = spec.get("paths", {})
        division_paths = {p: methods for p, methods in paths.items() if "division" in p.lower()}
        print(f"    Found {len(division_paths)} division-related path(s):")
        for p, methods in division_paths.items():
            for method, details in methods.items():
                params = [param.get("name") for param in details.get("parameters", [])]
                print(f"      {method.upper()} {p}")
                if params:
                    print(f"        parameters: {params}")
        return spec

    print("  Could not fetch a Swagger spec from any candidate path.")
    return None


def explore_search(start_date, end_date):
    print()
    print("=" * 70)
    print(f"STEP 2: sample search call ({start_date} to {end_date})")
    print("=" * 70)
    url = f"{BASE_URL}{SEARCH_PATH}"
    # Guessing at hansard-api's own convention — confirm against Step 1's
    # real parameter list above once it prints.
    params = {
        "queryParameters.startDate": start_date,
        "queryParameters.endDate": end_date,
        "queryParameters.house": "Commons",
    }
    print(f"  GET {url}")
    print(f"  params: {params}")
    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return None

    print(f"  HTTP {response.status_code}")
    if response.status_code != 200:
        print(f"  Response body (first 1000 chars): {response.text[:1000]}")
        return None

    try:
        data = response.json()
    except ValueError:
        print(f"  Response wasn't valid JSON. First 1000 chars: {response.text[:1000]}")
        return None

    if isinstance(data, list):
        print(f"  Response is a list of {len(data)} item(s).")
        if data:
            print("  First item's fields:")
            for key, value in data[0].items():
                print(f"    {key}: {str(value)[:80]}")
        return data
    else:
        print(f"  Response is a {type(data).__name__}: {str(data)[:500]}")
        return None


def explore_single_division(division_id):
    print()
    print("=" * 70)
    print(f"STEP 3: fetching full detail for division {division_id}")
    print("=" * 70)
    url = f"{BASE_URL}/data/division/{division_id}.json"
    print(f"  GET {url}")
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return
    print(f"  HTTP {response.status_code}")
    if response.status_code != 200:
        print(f"  Response body (first 1000 chars): {response.text[:1000]}")
        return
    detail = response.json()
    print(f"  Top-level keys: {list(detail.keys()) if isinstance(detail, dict) else type(detail)}")
    print(json.dumps(detail, indent=2)[:2000])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=90)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    args = parser.parse_args()

    explore_swagger()
    results = explore_search(args.start_date, args.end_date)

    if results:
        first = results[0]
        division_id = first.get("DivisionId") or first.get("Id") or first.get("ExternalId")
        if division_id:
            explore_single_division(division_id)
        else:
            print(f"\nCouldn't find an ID field on the first result to look up — raw keys: "
                  f"{list(first.keys())}")

    print()
    print("=" * 70)
    print("Report back what Step 1 printed for the real parameter names, and whether")
    print("Step 2's date-range guess actually filtered anything — that decides whether")
    print("collect_divisions.py should query this API directly instead of hansard-api.")
    print("=" * 70)


if __name__ == "__main__":
    main()
