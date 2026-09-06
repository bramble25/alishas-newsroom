# Daily brief

A private morning news page. Pulls about 35 free RSS feeds, groups articles
covering the same story, ranks those groups by how many independent outlets
picked them up, and renders one scannable HTML page.

The number beside each story is the outlet count. That is the whole ranking
idea: nobody publishes article-level click data, but forty newsrooms
independently deciding something matters is a strong signal, and it is free.

## Setup

```bash
pip install feedparser requests

python check_feeds.py     # do this first, some feeds will be dead
python brief.py           # writes docs/index.html
open docs/index.html
```

Run it with no API key to start. You get the full ranking and layout, just
without the summary sentences. Add the key once you like the shape of it.

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

To publish: push to GitHub, then Settings, Pages, source `main` branch and
`/docs` folder. Add `ANTHROPIC_API_KEY` under Settings, Secrets and variables,
Actions. The workflow runs at 6am Toronto time and commits the new page.

Cost is one API call per day covering the whole brief, so a few cents a month.

## Tuning

Everything you will want to change lives in the two config files. You should
not need to touch `brief.py`.

**`config/sources.json`** is the outlet list. Each entry has:

- `weight` how much this outlet counts toward consensus. Wires and papers of
  record are 1.0. Aggregators and niche trades are lower.
- `region` one of `world`, `us`, `ca`. Drives the Canada block.
- `access` `free` means the link goes straight to the article. `proquest`
  means the link goes to a ProQuest search instead, because the article is
  paywalled and you have institutional access.

To add an outlet with no obvious feed, use the Google News fallback:

```
https://news.google.com/rss/search?q=when:24h+site:DOMAIN&hl=en-CA&gl=CA&ceid=CA:en
```

**`config/sections.json`** is the layout. Sections are evaluated in `priority`
order and a story lands in the first one it matches, so it never appears twice.

To add a vertical later, copy the `defense` block, change `entities` and
`keywords`, and give it a priority under 40 so it is evaluated before Top
stories. Nothing else needs to change.

```json
{
  "id": "biotech",
  "title": "Biotech",
  "priority": 15,
  "slots": 4,
  "match": "entity",
  "entities": ["moderna", "vertex", "recursion", "insitro", "abcellera"],
  "keywords": ["fda approval", "phase iii", "clinical trial", "drug pricing"]
}
```

You will also want to add the matching trade feeds to `sources.json`, or the
section will have nothing to draw from.

## Knobs in `brief.py`

- `CLUSTER_THRESHOLD` (0.42). Lower it if one story keeps splitting into two
  entries. Raise it if unrelated stories get merged.
- `MAX_AGE_HOURS` (30). How far back a feed item can be and still count.
- `MODEL`. Swap to a cheaper model via the `BRIEF_MODEL` env var.

## The ProQuest link

Your account id and proxy prefix are read from the `PROQUEST_ACCOUNT` and
`PROQUEST_PREFIX` environment variables, falling back to `link_out` in
`sources.json`. Keep them in the environment so a public repo never names your
library account. Set them as Actions secrets alongside `ANTHROPIC_API_KEY`.

With neither set the link goes to a plain proquest.com search, which works on
campus. From off campus you probably need the proxy prefix.

Test one link from your phone on cell data. If the bare link asks you to log
in, find the prefix your library uses and paste it in. UofT runs EZproxy at
`login.library.utoronto.ca` and OpenAthens for some resources, so it is worth
asking the Milt Harris Library which applies to ProQuest rather than guessing.

These links are for you to click. Nothing in this project retrieves paywalled
content automatically, which is deliberate: automated retrieval through
institutional credentials violates the license and can get access suspended
for the whole university.

## First week

Run it, read it, and keep a note of what you wished was in it and what you
skipped. Then adjust weights and slot counts. The ranking gets good by being
tuned against your own reading, not by being clever on day one.
