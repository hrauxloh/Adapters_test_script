"""
WHAT THIS SCRIPT DOES
----------------------
Checks what Public Whip (publicwhip.org.uk) actually offers for downloading
structured voting/rebellion data, before building anything that depends on
a guessed file format.

Web search turned up real, named resources, but not confirmed full URLs:
  - Vote-matrix .txt files (tab-separated: every MP's vote per division) —
    one confirmed real URL for the Lords (votematrix-lords.txt); the
    Commons equivalent's exact name isn't confirmed.
  - XML feeds: alldivisions.xml, interestingdivisions.xml (>10 rebellions),
    mp-info.xml (per-MP attendance/rebelliousness, live).
  - A reprocessed mirror via mySociety's publicwhip-data package, with
    tables pw_division, pw_vote, pw_mp, etc.

This tries several plausible URLs for each and reports what actually
responds, rather than committing to one guessed path. Every attempt to
fetch these directly from the sandbox this was written in was blocked by
its network policy, so nothing below has been verified yet.

Menu of what happens when you run this file, in order:
  1. Try several candidate URLs for the Commons vote-matrix file.
  2. Try several candidate URLs for each of the three XML feeds.
  3. For anything that responds with HTTP 200, print its size and first
     few hundred characters so the real format is visible.

Usage:
    python explore_publicwhip.py
"""

import requests

VOTE_MATRIX_CANDIDATES = [
    "https://www.publicwhip.org.uk/data/votematrix.txt",
    "https://www.publicwhip.org.uk/data/votematrix-commons.txt",
    "https://www.publicwhip.org.uk/data/votematrix-2024.txt",
    "https://www.publicwhip.org.uk/data/votematrix-lords.txt",  # confirmed real (Lords)
]

XML_FEED_CANDIDATES = {
    "alldivisions.xml": [
        "https://www.publicwhip.org.uk/xml/alldivisions.xml",
        "https://www.publicwhip.org.uk/data/alldivisions.xml",
    ],
    "interestingdivisions.xml": [
        "https://www.publicwhip.org.uk/xml/interestingdivisions.xml",
        "https://www.publicwhip.org.uk/data/interestingdivisions.xml",
    ],
    "mp-info.xml": [
        "https://www.publicwhip.org.uk/xml/mp-info.xml",
        "https://www.publicwhip.org.uk/data/mp-info.xml",
    ],
}


def try_url(url, preview_chars=300):
    print(f"  Trying {url}")
    try:
        response = requests.get(url, timeout=20)
    except requests.RequestException as exc:
        print(f"    Request failed: {exc}")
        return None
    print(f"    HTTP {response.status_code}, {len(response.content):,} bytes")
    if response.status_code == 200:
        print(f"    First {preview_chars} chars: {response.text[:preview_chars]!r}")
        return response.text
    return None


def check_year_coverage(start_year=2015, end_year=2026):
    print()
    print("=" * 70)
    print(f"Checking which years {start_year}-{end_year} have a real votematrix file")
    print("=" * 70)
    for year in range(start_year, end_year + 1):
        url = f"https://www.publicwhip.org.uk/data/votematrix-{year}.txt"
        try:
            response = requests.get(url, timeout=20)
        except requests.RequestException as exc:
            print(f"  {year}: request failed ({exc})")
            continue
        print(f"  {year}: HTTP {response.status_code}, {len(response.content):,} bytes")


def main():
    print("=" * 70)
    print("Vote-matrix candidates")
    print("=" * 70)
    for url in VOTE_MATRIX_CANDIDATES:
        try_url(url)

    # 2024 worked — look at a lot more of it to understand the real layout
    # (header block, column names, and the first few actual data rows).
    print()
    print("=" * 70)
    print("Full structure of votematrix-2024.txt")
    print("=" * 70)
    text = try_url("https://www.publicwhip.org.uk/data/votematrix-2024.txt", preview_chars=3000)
    if text:
        lines = text.splitlines()
        print(f"\n  Total lines: {len(lines)}, total characters: {len(text):,}")
        print("  Line-by-line, first 15:")
        for i, line in enumerate(lines[:15]):
            print(f"    [{i}] {line[:200]}")

        # The MP-name lookup table (650 rows) plus header only accounts for
        # a small fraction of this file's 47KB — the real vote matrix must
        # be further in. Find where the MP list header row actually is,
        # and print everything from there through a good chunk after it.
        mp_header_idx = next((i for i, l in enumerate(lines) if l.startswith("mpid\t")), None)
        print(f"\n  'mpid\\t...' header row found at line index: {mp_header_idx}")
        print(f"\n  Last 20 lines of the file (tail):")
        for i, line in enumerate(lines[-20:], start=len(lines) - 20):
            print(f"    [{i}] {line[:200]}")

        if mp_header_idx is not None:
            after_mp_list = mp_header_idx + 1 + 650  # skip the 650 MP rows
            print(f"\n  20 lines right after where the MP list should end (index {after_mp_list}):")
            for i, line in enumerate(lines[after_mp_list:after_mp_list + 20], start=after_mp_list):
                print(f"    [{i}] {line[:200]}")

    check_year_coverage()

    for name, candidates in XML_FEED_CANDIDATES.items():
        print()
        print("=" * 70)
        print(f"{name} candidates")
        print("=" * 70)
        for url in candidates:
            try_url(url)

    print()
    print("=" * 70)
    print("Report back which URLs (if any) returned HTTP 200 and what their")
    print("content actually looks like — that decides which format to build")
    print("a real parser/collector around.")
    print("=" * 70)


if __name__ == "__main__":
    main()
