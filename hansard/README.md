# Free-vote / whipped-vote debate tagging

Goal: build a dataset of House of Commons debates, each tagged as `free`
or `whipped` (default), by cross-referencing debate dates against the
Commons Library free-votes briefing. Source of debate data:
[hansard-api.parliament.uk](https://hansard-api.parliament.uk/swagger/ui/index).

## Status: Step 1 only (data inspection)

Nothing here does the actual free/whipped tagging yet. Before writing that,
we need to see what a real debate record from the API actually looks like
— its exact field names, how dates are represented, whether "house"
(Commons vs Lords) is a field, what identifies one debate uniquely, etc.

`explore_debates_api.py` does two things:
1. Fetches the API's Swagger spec and prints every endpoint whose path
   mentions "debate", with its parameters — the real, current source of
   truth for what this API actually offers.
2. Makes one sample debates-search call for a short date range and saves
   the raw JSON response to a file, printing a quick summary of its shape.

```bash
pip install requests
python explore_debates_api.py
```

By default it looks at the last 14 days; pass `--start-date`/`--end-date`
(YYYY-MM-DD) for a different range, and `--output` for where to save the
raw JSON.

**Note on the endpoint/parameter names in this script:** they're a
best-understanding guess at this API's conventions, not verified against a
live response — the sandbox this was written in couldn't reach
`hansard-api.parliament.uk` at all (blocked by the sandbox's own network
policy, confirmed against several unrelated domains too). Run this
somewhere with normal internet access, and treat step 1's printed endpoint
list as ground truth over anything guessed in the script — if it differs,
that's the real schema to build against, not this one.

## Next steps (not started)

Once we can see a real debate record:
- Confirm what field(s) identify its date and House.
- Find/access the Commons Library free-votes briefing data (likely a PDF
  or a research briefing page — not yet located or parsed) and figure out
  how its dates map onto individual debates (a briefing entry may cover a
  whole day, a specific division, or a specific bill's stages — matters
  for how precisely "free" can be attached to a debate record).
- Build the actual cross-referencing script, defaulting every debate to
  `whipped` unless a briefing entry says otherwise.
