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

## Scaled-up collection: full vote breakdown for every division in a date range

**Important correction:** `collect_divisions.py` originally chained
through `hansard-api.parliament.uk` (search debates → filter to Commons
Chamber → check each for a division) to eventually try to reach full vote
detail. That chain is gone. A real division-detail schema turned out to
come from a completely **separate, dedicated API**:
[commonsvotes-api.parliament.uk](https://commonsvotes-api.parliament.uk/swagger/ui/index)
— confirmed by finding real endpoint URLs in an existing R package's
source (`houseofcommonslibrary/clvotes`) and then verifying them against
this API's live Swagger spec and a real search + detail call:

- `/data/divisions.json/search` takes real `startDate`/`endDate`/`skip`/`take`
  parameters that **actually filter** (unlike hansard-api's search, whose
  `withDivision` filter and `TotalResultCount` both turned out to be
  unreliable). There's no `house` parameter, because this API only ever
  covers the Commons.
- Search results already include `AyeCount`/`NoCount`/`AyeTellers`/`NoTellers`
  directly, but the full member-by-member `Ayes`/`Noes`/`NoVoteRecorded`
  lists come back empty — one follow-up call per division, to
  `/data/division/{DivisionId}.json`, fills those in.

So the script now queries `commonsvotes-api` directly: page through the
search endpoint for the date range, then fetch full detail for every
division found. No more debate search, no more `DebateSection` filtering,
no more guessing which field holds an ID — `DivisionId` is a confirmed,
reliable field straight from the search results. `explore_commonsvotes_api.py`
is the (now superseded, but kept for reference) script that discovered
this.

**No turnout/attendance filtering happens here.** Every division found
gets collected regardless of `AyeCount + NoCount` — filter the saved data
afterwards for whatever attendance threshold matters, without needing to
re-fetch anything.

For each month, saves two files:
- `divisions_YYYY-MM.csv` — one row per division (date, title, Aye/No
  counts, EVEL flags, etc.)
- `votes_YYYY-MM.csv` — one row per individual member's vote, tagged
  `Aye` / `No` / `AyeTeller` / `NoTeller` / `NoVoteRecorded`, with their
  party and other details

If a division's detail call fails, its row is still saved using just the
search-result summary (still has Aye/No counts and tellers, just not the
full member lists) rather than being dropped — verified with mocked
responses covering both the full-detail and fallback cases.

**Same defensive month-chunking as before, kept as a precaution:**
hansard-api's search endpoint was confirmed to silently cap how far a big
multi-year query's pagination reaches; since both APIs share the same
query-parameter conventions (likely the same underlying platform), this
script still splits any date range into calendar-month chunks rather than
querying a huge range in one call, even though this specific quirk hasn't
been confirmed on `commonsvotes-api` itself.

**Resumable by design**, for a long unattended run from a laptop that
might lose wifi, or a Colab session that might disconnect: `--output-dir`
gets its own pair of files per month, written the moment that month
finishes. Before doing any work for a month, it checks whether its
`divisions_YYYY-MM.csv` already exists and skips the month entirely if so
(months that genuinely found nothing are still saved, as an empty file,
so "checked, found nothing" is distinguishable from "not checked yet").
Verified with a test covering a fresh run, an identical re-run (zero
network calls, everything skipped), and the full-detail/fallback paths
above. Point `--output-dir` at a mounted Google Drive folder, not local
Colab storage, so the files themselves survive even if the runtime is
reclaimed entirely.

```bash
pip install requests pandas
# validate against one month first
python collect_divisions.py --start-date 2024-01-01 --end-date 2024-01-31 --output-dir divisions_test

# then scale up
python collect_divisions.py --start-date 2010-01-01 --end-date 2023-12-31 \
  --output-dir /content/drive/MyDrive/hansard_divisions
```

Combine the per-month files into one table later with:
```python
import glob, pandas as pd
divisions = pd.concat([pd.read_csv(f) for f in glob.glob("/content/drive/MyDrive/hansard_divisions/divisions_*.csv")])
votes = pd.concat([pd.read_csv(f) for f in glob.glob("/content/drive/MyDrive/hansard_divisions/votes_*.csv")])
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
