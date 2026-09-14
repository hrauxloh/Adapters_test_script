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

It only asks the API for debates that already have a division
(`queryParameters.withDivision=true`), so there's no need to fetch every
debate and cross-reference — the filtering happens server-side. Paginates
with `skip`/`take` and stops on the first empty page rather than trusting
`TotalResultCount` (confirmed unreliable earlier). Retries a failed page
a couple of times with backoff before giving up, and returns whatever was
already collected rather than losing a long pull to one dropped
connection.

```bash
pip install requests pandas
# validate against one day first
python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-01

# then scale up
python collect_divisions.py --start-date 2025-05-01 --end-date 2025-05-22 --output may_divisions.csv
```

Not yet verified against a live response (same sandbox network restriction
as everything else here) — run the single-day command first and check the
output looks sensible before trusting the full-range run.

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
