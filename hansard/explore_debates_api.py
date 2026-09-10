"""
WHAT THIS SCRIPT DOES
----------------------
Step 1 of the free-vote/whipped-vote project: just look at what the UK
Parliament's Hansard API actually returns, before writing any code that
depends on its exact shape.

This does NOT do the free/whipped cross-referencing yet — that's a
separate, later step, once we've actually seen what a debate record looks
like and know which fields (date, house, debate title, section ID, etc.)
are available to match against the Commons Library free-votes briefing.

Menu of what happens when you run this file, in order:
  1. Try to fetch the API's Swagger/OpenAPI spec, and print every endpoint
     path that mentions "debate" — this tells us the real, current
     endpoint names and parameters directly from the API itself, instead
     of guessing.
  2. Make one sample call to the debates-search endpoint for a short,
     recent date range, and save the raw JSON response to a file so you
     can open it and look at exactly what fields come back.
  3. Print a short summary of what it found (how many debates, what
     fields the first record has).

Why this matters: I (Claude) could not reach hansard-api.parliament.uk
from the sandbox this was written in — every attempt came back blocked by
the sandbox's network policy — so nothing below has been verified against
a live response yet. The endpoint path and parameter names below are my
best understanding of this API's public conventions, but you should treat
step 1's printed endpoint list as the source of truth over any guess in
this script, and let me know what it actually shows so this script (and
the real cross-referencing step after it) can be corrected against reality
rather than assumption.

Usage:
    python explore_debates_api.py
    python explore_debates_api.py --start-date 2024-01-01 --end-date 2024-01-31
"""

import argparse
import json
from datetime import date, timedelta

import requests

BASE_URL = "https://hansard-api.parliament.uk"

# Swagger/OpenAPI is commonly served at one of these paths for this kind of
# API (Swashbuckle/.NET) — tried in order, first one that returns valid JSON
# wins. This is exactly the kind of guess step 1 exists to replace with real
# information: whichever one (if any) works, the printed endpoint list below
# is the actual truth, not this list.
CANDIDATE_SWAGGER_PATHS = [
    "/swagger/docs/v1",
    "/swagger/v1/swagger.json",
]

# Best-understanding guess at the debates search endpoint and its query
# parameters, based on this API's public conventions — NOT yet verified
# against a live response from this sandbox. If this 404s or the fields
# look wrong, the swagger output from step 1 is what to go by instead.
DEBATES_SEARCH_PATH = "/search/debates.json"


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
        debate_paths = {p: methods for p, methods in paths.items() if "debate" in p.lower()}
        if not debate_paths:
            print(f"    Got a valid spec from {url}, but no path contained 'debate'.")
            print(f"    Total paths found: {len(paths)}. Here are the first 20, to sanity-check:")
            for p in list(paths.keys())[:20]:
                print(f"      {p}")
            return spec

        print(f"    Found {len(debate_paths)} debate-related path(s):")
        for p, methods in debate_paths.items():
            for method, details in methods.items():
                params = [param.get("name") for param in details.get("parameters", [])]
                print(f"      {method.upper()} {p}")
                if params:
                    print(f"        parameters: {params}")
        return spec

    print("  Could not fetch a Swagger spec from any candidate path.")
    print("  If you know the real Swagger URL, run this script with it, or paste")
    print("  https://hansard-api.parliament.uk/swagger/ui/index 's 'Raw' link contents back.")
    return None


def explore_sample_debates(start_date, end_date, output_path):
    print()
    print("=" * 70)
    print(f"STEP 2: sample debates-search call ({start_date} to {end_date})")
    print("=" * 70)
    url = f"{BASE_URL}{DEBATES_SEARCH_PATH}"
    params = {
        "queryParameters.startDate": start_date,
        "queryParameters.endDate": end_date,
        "queryParameters.take": 5,  # just a handful, this is only for inspection
    }
    print(f"  GET {url}")
    print(f"  params: {params}")
    try:
        response = requests.get(url, params=params, timeout=20)
    except requests.RequestException as exc:
        print(f"  Request failed: {exc}")
        return

    print(f"  HTTP {response.status_code}")
    if response.status_code != 200:
        print("  Response body (first 1000 chars):")
        print(f"    {response.text[:1000]}")
        return

    try:
        data = response.json()
    except ValueError:
        print("  Response wasn't valid JSON. First 1000 chars:")
        print(f"    {response.text[:1000]}")
        return

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved full raw response to {output_path}")

    # Try to describe what came back without assuming its exact shape —
    # this API's real response could be a list, or a dict with a results
    # key, etc.; print enough to see which it actually is.
    if isinstance(data, list):
        print(f"  Response is a list of {len(data)} item(s).")
        first = data[0] if data else None
    elif isinstance(data, dict):
        print(f"  Response is a dict with top-level keys: {list(data.keys())}")
        first = None
        for key, value in data.items():
            if isinstance(value, list) and value:
                print(f"  '{key}' looks like the results list ({len(value)} item(s)).")
                first = value[0]
                break
    else:
        first = None

    if first is not None:
        print("  Fields on the first debate record:")
        for key, value in first.items():
            preview = str(value)[:80]
            print(f"    {key}: {preview}")
    else:
        print("  Couldn't identify an individual debate record to preview — "
              f"open {output_path} directly and look.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=(date.today() - timedelta(days=14)).isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--output", default="sample_debates_response.json")
    args = parser.parse_args()

    explore_swagger()
    explore_sample_debates(args.start_date, args.end_date, args.output)

    print()
    print("=" * 70)
    print("Next step (not done yet): once you've confirmed the real field names")
    print("above, tell me what you saw and we'll build the actual free-vote /")
    print("whipped-vote cross-referencing against the Commons Library briefing.")
    print("=" * 70)


if __name__ == "__main__":
    main()
