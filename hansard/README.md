# Free-vote / whipped-vote debate tagging

Goal: build a dataset of House of Commons debates, each tagged as `free`
or `whipped` (default), by cross-referencing debate dates against the
Commons Library free-votes briefing. Source of debate data:
[hansard-api.parliament.uk](https://hansard-api.parliament.uk/swagger/ui/index).

## Status: Step 1 only (data inspection) — confirmed decision: tag divisions, not all debates

**Decided:** the free/whipped tag applies to *divisions* (actual recorded
votes), not every debate section — most debates never have a vote at all,
so "free vs. whipped" only means something for the ones that do.

Confirmed from a real run against the live API (swagger spec + a real
`/search/debates.json` call):
- Debate search results have fields `DebateSection`, `SittingDate`,
  `House`, `Title`, `Rank`, `DebateSectionExtId`, wrapped in
  `{"Results": [...], "TotalResultCount": ...}`.
- `TotalResultCount` looks unreliable as a filtered count (it returned
  ~1.37 million for a 14-day window) — don't trust it for pagination;
  page with `skip`/`take` and stop when `Results` comes back empty
  instead.
- `queryParameters.withDivision=true` filters search results down to
  debate sections that had at least one division — this is the filter the
  real pipeline should use.
- `/debates/divisions/{debateSectionExtId}.json` lists the division(s) in
  a debate section; a division's own fields haven't been confirmed yet —
  that's what `explore_debates_api.py`'s Step 3 checks.

`explore_debates_api.py` does three things:
1. Fetches the API's Swagger spec and prints every endpoint whose path
   mentions "debate", with its parameters.
2. Makes a sample `/search/debates.json` call and saves the raw response.
3. Finds a debate section with a division (`withDivision=true`) and fetches
   its `/debates/divisions/{id}.json` record, saving the raw response and
   printing the field names on the first division.

```bash
pip install requests
python explore_debates_api.py
```

By default it looks at the last 14 days; pass `--start-date`/`--end-date`
(YYYY-MM-DD) for a different range if no divisions turn up in that window.

`explore_free_votes_briefing.py` looks at the other half of this project —
the Commons Library's ["Free votes in the House of Commons since
1979"](https://commonslibrary.parliament.uk/research-briefings/SN04793/)
briefing, which reportedly includes a downloadable spreadsheet of known
free votes. This hasn't been inspected yet either (see note below). It
fetches the briefing page, finds any spreadsheet/PDF attachment link in
the HTML, downloads it, and if it's a spreadsheet, prints its columns and
first few rows so we can see the real schema:

```bash
pip install requests pandas openpyxl
python explore_free_votes_briefing.py
```

**Note on anything not yet confirmed against a live response:** the
sandbox this was written in couldn't reach either
`hansard-api.parliament.uk` or `commonslibrary.parliament.uk` at all
(blocked by the sandbox's own network policy — confirmed against several
unrelated domains too, including Wikipedia). Everything marked "confirmed"
above came from an actual run of `explore_debates_api.py` reported back;
anything about the division fields or the briefing spreadsheet is still
unverified — run the scripts above somewhere with normal internet access
and report back what they print.

## Scaled-up collection: division-bearing debates in a date range

`collect_divisions.py` collects every Commons debate section with a
recorded division (an actual vote) in a given date range, as one summary
table — one row per debate section (`DebateSection`, `SittingDate`,
`House`, `Title`, `Rank`, `DebateSectionExtId`), no per-item text.

**`queryParameters.withDivision=true` does not actually filter anything —
confirmed by running it.** It was expected to filter `/search/debates.json`
down to only division-bearing debates server-side, but a real run
returned every debate regardless (this lines up with an earlier hint: a
debate it claimed had a division came back with zero divisions when
checked directly). So this script fetches *every* debate in the date
range, then checks each one individually against
`/debates/divisions/{id}.json`, keeping only the ones that genuinely come
back with at least one division — more API calls, but checking ground
truth instead of trusting a parameter that's now failed twice.

Paginates the initial debate fetch with `skip`/`take` and stops on the
first empty page rather than trusting `TotalResultCount` (confirmed
unreliable earlier too). Retries a failed request a couple of times with
backoff before giving up, and returns whatever was already collected
rather than losing a long pull to one dropped connection.

By default it only keeps debates whose `DebateSection` is exactly
`"Commons Chamber"` (excluding Westminster Hall, Public Bill Committees,
etc.) — pass `--debate-section ""` to keep everything, or a different
value to filter on something else. This filter runs *before* the
division-check step, so it also cuts down on API calls for debates that
would be discarded anyway.

```bash
pip install requests pandas
# validate against one day first
python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-01 --output-dir divisions_test

# then scale up
python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-22 --output-dir divisions_may2025
```

Confirmed working against a live run (see the note above about
`withDivision` — this script no longer relies on it).

**A third API quirk, found when trying a big multi-year range:** requesting
2010-01-01 to 2024-12-31 in one call only ever returned the last couple of
months of that range — the search endpoint appears to return results
newest-first and silently caps how far pagination actually reaches for one
query, rather than paging through everything. The script splits any date
range into calendar-month chunks automatically and queries each one
separately.

**Resumable by design**, for a long unattended run from a laptop that
might lose wifi, or a Colab session that might disconnect: `--output-dir`
gets one file per month (`divisions_YYYY-MM.csv`), written the moment
that month finishes — not held in memory until the end. Before doing any
work for a month, it checks whether that file already exists and skips
it if so (months that genuinely found nothing are still saved, as an
empty file, so "checked, found nothing" is distinguishable from "not
checked yet"). That means if the run stops for any reason, re-running the
exact same command later resumes automatically — verified with a test
covering a fresh run, an identical re-run (confirmed zero network calls,
everything skipped), and extending the date range (confirmed only the
new month gets fetched). Point `--output-dir` at a mounted Google Drive
folder, not local Colab storage, so the files themselves survive even if
the runtime is reclaimed entirely.

```bash
python collect_divisions.py --start-date 2010-01-01 --end-date 2023-12-31 \
  --output-dir /content/drive/MyDrive/hansard_divisions
```

Combine the per-month files into one table later with:
```python
import glob, pandas as pd
combined = pd.concat([pd.read_csv(f) for f in glob.glob("/content/drive/MyDrive/hansard_divisions/divisions_*.csv")])
```

## Next steps (not started)

Once we can see a real division record and the briefing spreadsheet's
actual columns:
- Figure out how the briefing's entries map onto individual divisions (by
  date? by division number? by bill/motion title matched against
  `Title`?) — this determines how precisely "free" can be attached to a
  specific division rather than just a date.
- Build the actual cross-referencing script: for each division found via
  `withDivision=true` search (paginated safely, not trusting
  `TotalResultCount`), look up its date/details against the briefing data,
  defaulting to `whipped` unless the briefing says otherwise.
