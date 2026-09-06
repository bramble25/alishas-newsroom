#!/usr/bin/env python3
"""
Daily brief builder.

Pulls RSS from every source in config/sources.json, groups articles covering the
same story, ranks those groups by how many independent outlets picked them up,
routes them into the sections in config/sections.json, writes one summary pass
through an LLM, and renders a static HTML page.

Every run writes a dated page and rebuilds a calendar index. There is no
database: the git repo is the archive. A year of pages is a few megabytes.

Run:  python brief.py
Out:  docs/YYYY-MM-DD.html   today's page
      docs/index.html        a copy of today, so the site root works
      docs/archive.html      calendar of every day you have
      data/index.json        the list of days
      data/history.json      running-story memory
"""

import json
import os
import re
import html
import shutil
import sys
import calendar as cal
from datetime import datetime, timezone, timedelta, date
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

import feedparser
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs")

MODEL = os.environ.get("BRIEF_MODEL", "claude-sonnet-4-6")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

MAX_AGE_HOURS = 30
CLUSTER_THRESHOLD = 0.42
TIMEZONE = timezone(timedelta(hours=-4))

STOPWORDS = set("""
a an the and or but of to in on at for from by with as is are was were be been being
this that these those it its his her their our your my have has had will would could
should may might can do does did not no nor so than then there here what which who whom
how why when where after before over under again more most other some such only own same
too very s t just now says say said new news report reports amid into out up down about
""".split())


def load_config():
    with open(os.path.join(CONFIG_DIR, "sources.json")) as f:
        sources = json.load(f)
    with open(os.path.join(CONFIG_DIR, "sections.json")) as f:
        sections = json.load(f)
    return sources, sections


# ----------------------------------------------------------------------------
# fetch
# ----------------------------------------------------------------------------

def fetch_one(source):
    try:
        resp = requests.get(source["feed"], timeout=20,
                            headers={"User-Agent": "personal-news-brief/1.0"})
        parsed = feedparser.parse(resp.content)
    except Exception as exc:
        print(f"  [skip] {source['name']}: {exc}", file=sys.stderr)
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(hours=MAX_AGE_HOURS)
    articles = []

    for rank, entry in enumerate(parsed.entries[:60]):
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue

        published = None
        for key in ("published_parsed", "updated_parsed"):
            if entry.get(key):
                published = datetime(*entry[key][:6], tzinfo=timezone.utc)
                break
        if published and published < cutoff:
            continue

        blurb = re.sub(r"<[^>]+>", " ", entry.get("summary", ""))
        blurb = html.unescape(re.sub(r"\s+", " ", blurb)).strip()

        articles.append({
            "title": title, "link": link, "blurb": blurb[:400],
            "published": published or datetime.now(timezone.utc),
            "feed_rank": rank,
            "outlet": source["name"], "weight": source.get("weight", 0.5),
            "region": source.get("region", "world"),
            "access": source.get("access", "free"),
            "proquest_id": source.get("proquest_id"),
        })

    print(f"  {source['name']}: {len(articles)}")
    return articles


def fetch_all(sources):
    print("Fetching feeds")
    with ThreadPoolExecutor(max_workers=12) as pool:
        batches = pool.map(fetch_one, sources)
    return [a for batch in batches for a in batch]


# ----------------------------------------------------------------------------
# cluster and score
# ----------------------------------------------------------------------------

def tokenize(text):
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


def similarity(a, b):
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap < 2:
        return 0.0
    return overlap / len(a | b)


def cluster(articles):
    print("Clustering")
    articles.sort(key=lambda a: (-a["weight"], a["published"]))
    clusters = []

    for article in articles:
        tokens = tokenize(article["title"])
        best, best_score = None, 0.0
        for c in clusters:
            s = similarity(tokens, c["tokens"])
            if s > best_score:
                best, best_score = c, s

        if best and best_score >= CLUSTER_THRESHOLD:
            best["articles"].append(article)
            best["tokens"] |= tokens
        else:
            clusters.append({"tokens": tokens, "articles": [article]})

    print(f"  {len(articles)} articles into {len(clusters)} stories")
    return clusters


def score_clusters(clusters):
    now = datetime.now(timezone.utc)
    for c in clusters:
        outlets = {a["outlet"] for a in c["articles"]}
        c["outlet_count"] = len(outlets)
        c["outlets"] = sorted(outlets)
        c["lead"] = sorted(c["articles"],
                           key=lambda a: (-a["weight"], -a["published"].timestamp()))[0]

        breadth = sum(max(a["weight"] for a in c["articles"] if a["outlet"] == o)
                      for o in outlets)
        newest = max(a["published"] for a in c["articles"])
        recency = max(0.3, 1.0 - ((now - newest).total_seconds() / 3600 / 48))
        c["score"] = breadth * recency
        c["ca_share"] = sum(
            1 for o in outlets
            if any(a["region"] == "ca" for a in c["articles"] if a["outlet"] == o)
        ) / len(outlets)

    clusters.sort(key=lambda c: -c["score"])
    return clusters


# ----------------------------------------------------------------------------
# running-story memory
# ----------------------------------------------------------------------------

def load_history():
    path = os.path.join(DATA_DIR, "history.json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        raw = json.load(f)
    cutoff = (datetime.now(timezone.utc) - timedelta(days=21)).isoformat()
    return [h for h in raw if h["last_seen"] >= cutoff]


def apply_history(clusters, history):
    now_iso = datetime.now(timezone.utc).isoformat()
    today = datetime.now(TIMEZONE).date()

    for c in clusters:
        c["days_running"] = 1
        match = next((h for h in history
                      if similarity(c["tokens"], set(h["tokens"])) >= 0.30), None)
        if match:
            first = datetime.fromisoformat(match["first_seen"]).date()
            c["days_running"] = (today - first).days + 1
            match["tokens"] = sorted(set(match["tokens"]) | c["tokens"])[:40]
            match["last_seen"] = now_iso
        else:
            history.append({"tokens": sorted(c["tokens"])[:40],
                            "first_seen": now_iso, "last_seen": now_iso,
                            "headline": c["lead"]["title"]})

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "history.json"), "w") as f:
        json.dump(history, f, indent=1)
    return clusters


# ----------------------------------------------------------------------------
# routing
# ----------------------------------------------------------------------------

def haystack(c):
    return " ".join([c["lead"]["title"], c["lead"]["blurb"]]
                    + [a["title"] for a in c["articles"][:6]]).lower()


def matches(c, section):
    text = haystack(c)
    if any(bad in text for bad in section.get("exclude_keywords", [])):
        return False

    mode = section["match"]
    has_entity = any(e in text for e in section.get("entities", []))
    has_keyword = any(k in text for k in section.get("keywords", []))

    if mode == "entity":
        return has_entity or has_keyword
    if mode == "entity_and_keyword":
        return has_entity and has_keyword
    if mode == "keyword":
        return has_keyword
    if mode == "region":
        return c["ca_share"] >= section.get("min_share", 0.5)
    return False


def route(clusters, sections):
    """Claim order is match_order, which is separate from page order.

    Verticals claim before Top stories on purpose. If Top stories went first it
    would take the six biggest things regardless of topic and the Defense block
    would be left with scraps.
    """
    print("Routing")
    assigned = {s["id"]: [] for s in sections}
    used = set()
    ordered = sorted(sections, key=lambda s: s["match_order"])

    for section in ordered:
        if section["match"] in ("consensus", "by_outlet"):
            continue
        for i, c in enumerate(clusters):
            if i in used or len(assigned[section["id"]]) >= section["slots"]:
                continue
            if matches(c, section):
                assigned[section["id"]].append(c)
                used.add(i)

    for section in ordered:
        if section["match"] != "consensus":
            continue
        for i, c in enumerate(clusters):
            if i in used or len(assigned[section["id"]]) >= section["slots"]:
                continue
            if c["outlet_count"] < 2:
                continue
            assigned[section["id"]].append(c)
            used.add(i)

    for sid, items in assigned.items():
        if items:
            print(f"  {sid}: {len(items)}")
    return assigned


def build_columns(articles, assigned, section):
    """Group leftover articles by outlet for the bottom grid.

    Anything already shown in a story section is filtered out by link and by
    title, so a column is genuinely what else that newsroom is running.
    """
    shown_links, shown_titles = set(), set()
    for items in assigned.values():
        for c in items:
            for a in c["articles"]:
                shown_links.add(a["link"])
                shown_titles.add(a["title"].lower()[:70])

    columns = []
    for spec in section["columns"]:
        wanted = set(spec["outlets"])
        pool = [a for a in articles
                if a["outlet"] in wanted
                and a["link"] not in shown_links
                and a["title"].lower()[:70] not in shown_titles]
        pool.sort(key=lambda a: (a["feed_rank"], -a["published"].timestamp()))

        picked, seen = [], set()
        for a in pool:
            key = a["title"].lower()[:70]
            if key in seen:
                continue
            seen.add(key)
            picked.append(a)
            if len(picked) >= section.get("per_column", 6):
                break

        if picked:
            columns.append({"label": spec["label"], "articles": picked})
    return columns


# ----------------------------------------------------------------------------
# summarize
# ----------------------------------------------------------------------------

SUMMARY_PROMPT = """You are writing a private morning news brief for one reader.
They work in venture capital in Toronto and are building a working pulse on
world news, Canadian news, technology, startups and defense. Assume they may be
coming to any story cold.

Return ONLY valid JSON, no markdown fences, in this shape:

{
  "stories":  { "<story id>": "<one sentence>", ... },
  "sections": { "<section id>": "<two sentences>", ... },
  "columns":  { "<column label>": "<one short clause>", ... }
}

Story sentences: one sentence, under 30 words, plain and declarative. Say what
happened and why it matters. Do not restate the headline. If days_running is
above 1, write what is new today rather than the background.

Section pairs: two sentences naming the through-line across that section. If
there is no real through-line, say what the biggest item is instead.

Column clauses: under 10 words, describing what that outlet is leading with.

Only use information present in what you are given. Never invent a detail, a
number, an attribution or a cause. Do not use em dashes. Do not use exclamation
marks.

INPUT:
"""


def summarize(assigned, columns, sections):
    payload = []
    for section in sorted(sections, key=lambda s: s["order"]):
        if section["match"] == "by_outlet":
            continue
        items = assigned.get(section["id"], [])
        if not items:
            continue
        payload.append(f"\n## section {section['id']} ({section['title']})")
        for c in items:
            payload.append(json.dumps({
                "id": c["uid"], "headline": c["lead"]["title"],
                "outlets": c["outlet_count"], "days_running": c["days_running"],
                "other_headlines": [a["title"] for a in c["articles"][1:5]],
            }))

    if columns:
        payload.append("\n## columns")
        for col in columns:
            payload.append(json.dumps({
                "label": col["label"],
                "headlines": [a["title"] for a in col["articles"][:4]],
            }))

    if not payload:
        return {"stories": {}, "sections": {}, "columns": {}}
    if not API_KEY:
        print("  no ANTHROPIC_API_KEY, skipping summaries")
        return {"stories": {}, "sections": {}, "columns": {}}

    print("Summarizing")
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"content-type": "application/json", "x-api-key": API_KEY,
                     "anthropic-version": "2023-06-01"},
            json={"model": MODEL, "max_tokens": 4000,
                  "messages": [{"role": "user",
                                "content": SUMMARY_PROMPT + "\n".join(payload)}]},
            timeout=120)
        resp.raise_for_status()
        text = "".join(b.get("text", "") for b in resp.json()["content"]
                       if b["type"] == "text")
        text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
        data = json.loads(text)
        data.setdefault("columns", {})
        return data
    except Exception as exc:
        print(f"  summary failed: {exc}", file=sys.stderr)
        return {"stories": {}, "sections": {}, "columns": {}}


# ----------------------------------------------------------------------------
# render
# ----------------------------------------------------------------------------

CSS = """
:root{
  --paper:#EEEFF1; --ink:#17191C; --soft:#5A5F66; --rule:#D5D8DC;
  --signal:#24409E; --quiet:#9AA1AE; --flag:#8A3324;
}
*{box-sizing:border-box}
body{margin:0; background:var(--paper); color:var(--ink);
  font-family:Newsreader,Georgia,serif; font-size:17px; line-height:1.5;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:760px; margin:0 auto; padding:40px 24px 96px}
nav{display:flex; gap:16px; align-items:baseline; font-size:13px;
  color:var(--quiet); margin-bottom:36px}
nav a{color:var(--soft); text-decoration:none; border-bottom:1px solid var(--rule)}
nav a:hover,nav a:focus{color:var(--signal); border-bottom-color:var(--signal)}
nav .sp{flex:1}
header{margin-bottom:52px}
h1{font-size:15px; font-weight:400; color:var(--soft); margin:0 0 6px}
.count{font-size:34px; font-weight:500; line-height:1.15; margin:0; max-width:22ch}
.count b{font-weight:600; color:var(--signal); font-variant-numeric:tabular-nums}

section{margin-bottom:52px}
h2{font-size:13px; font-weight:600; margin:0 0 4px; padding-bottom:8px;
  border-bottom:1px solid var(--rule)}
.gist{font-size:15px; color:var(--soft); margin:12px 0 24px; max-width:64ch}

.story{display:flex; gap:18px; margin-bottom:22px; align-items:baseline}
.n{flex:0 0 42px; text-align:right; font-variant-numeric:tabular-nums;
  font-size:19px; font-weight:500; color:var(--quiet); padding-top:1px}
.n.mid{color:var(--signal); opacity:.62}
.n.hot{color:var(--signal); font-weight:600}
.bd{flex:1; min-width:0}
.hl{font-size:18px; font-weight:500; line-height:1.35; margin:0 0 3px}
.hl a{color:var(--ink); text-decoration:none; border-bottom:1px solid var(--rule)}
.hl a:hover,.hl a:focus{border-bottom-color:var(--signal); color:var(--signal)}
.sum{font-size:15px; color:var(--soft); margin:0 0 5px; max-width:60ch}
.meta{font-size:13px; color:var(--quiet); margin:0}
.meta .run{color:var(--flag)}
.meta .gate{color:var(--soft)}

.cols{display:grid; grid-template-columns:repeat(3,1fr); gap:30px 26px; margin-top:24px}
.col h3{font-size:13px; font-weight:600; margin:0 0 2px;
  padding-bottom:6px; border-bottom:1px solid var(--rule)}
.col .lede{font-size:12px; color:var(--quiet); margin:7px 0 10px; line-height:1.35}
.col ul{list-style:none; margin:0; padding:0}
.col li{margin-bottom:9px; font-size:14px; line-height:1.35}
.col a{color:var(--ink); text-decoration:none; border-bottom:1px solid transparent}
.col a:hover,.col a:focus{border-bottom-color:var(--signal); color:var(--signal)}

footer{margin-top:64px; padding-top:16px; border-top:1px solid var(--rule);
  font-size:13px; color:var(--quiet)}
a:focus-visible{outline:2px solid var(--signal); outline-offset:3px}
@media (max-width:720px){ .cols{grid-template-columns:repeat(2,1fr)} }
@media (max-width:560px){
  .wrap{padding:28px 18px 72px}
  .count{font-size:27px}
  .story{gap:12px} .n{flex-basis:32px; font-size:17px}
  .cols{grid-template-columns:1fr; gap:26px}
}
"""

HEAD = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Newsreader:opsz,"
        "wght@6..72,400;6..72,500;6..72,600&display=swap' rel='stylesheet'>")


def tier(n):
    return "hot" if n >= 12 else ("mid" if n >= 5 else "")


def read_link(c, link_out):
    lead = c["lead"]
    if lead["access"] == "proquest" and lead.get("proquest_id"):
        url = (f"https://www.proquest.com/results/{quote_plus(lead['title'])}"
               f"?accountid={link_out.get('proquest_account', '')}")
        return (link_out.get("proquest_prefix") or "") + url, "via ProQuest"
    return lead["link"], None


def render(assigned, columns, sections, summaries, link_out, total, when,
           prev_date=None, next_date=None):
    stories = summaries.get("stories", {})
    gists = summaries.get("sections", {})
    col_gists = summaries.get("columns", {})
    shown = sum(len(v) for v in assigned.values())

    nav = []
    if prev_date:
        nav.append(f"<a href='{prev_date}.html'>previous</a>")
    if next_date:
        nav.append(f"<a href='{next_date}.html'>next</a>")
    nav.append("<span class='sp'></span>")
    nav.append("<a href='archive.html'>archive</a>")

    out = [HEAD, f"<title>Brief, {when:%-d %B %Y}</title>",
           f"<style>{CSS}</style></head><body><div class='wrap'>",
           "<nav>" + "".join(nav) + "</nav>", "<header>",
           f"<h1>{when:%A, %-d %B %Y}</h1>",
           f"<p class='count'>{total} stories came in. "
           f"<b>{shown}</b> are worth your time.</p>", "</header>"]

    for section in sorted(sections, key=lambda s: s["order"]):
        if section["match"] == "by_outlet":
            if not columns:
                continue
            out.append(f"<section><h2>{html.escape(section['title'])}</h2>")
            out.append("<p class='gist'>What each newsroom is running that did "
                       "not make a block above.</p><div class='cols'>")
            for col in columns:
                out.append(f"<div class='col'><h3>{html.escape(col['label'])}</h3>")
                lede = col_gists.get(col["label"])
                if lede:
                    out.append(f"<p class='lede'>{html.escape(lede)}</p>")
                out.append("<ul>")
                for a in col["articles"]:
                    out.append(f"<li><a href='{html.escape(a['link'])}'>"
                               f"{html.escape(a['title'])}</a></li>")
                out.append("</ul></div>")
            out.append("</div></section>")
            continue

        items = assigned.get(section["id"], [])
        if not items:
            continue

        out.append(f"<section><h2>{html.escape(section['title'])}</h2>")
        if gists.get(section["id"]):
            out.append(f"<p class='gist'>{html.escape(gists[section['id']])}</p>")

        for c in items:
            url, gate = read_link(c, link_out)
            bits = html.escape(", ".join(c["outlets"][:3]))
            if c["outlet_count"] > 3:
                bits += f" and {c['outlet_count'] - 3} more"
            if c["days_running"] > 1:
                bits += f" &middot; <span class='run'>day {c['days_running']}</span>"
            if gate:
                bits += f" &middot; <span class='gate'>{gate}</span>"

            out.append("<article class='story'>")
            out.append(f"<div class='n {tier(c['outlet_count'])}'>{c['outlet_count']}</div>")
            out.append("<div class='bd'>")
            out.append(f"<p class='hl'><a href='{html.escape(url)}'>"
                       f"{html.escape(c['lead']['title'])}</a></p>")
            if stories.get(c["uid"]):
                out.append(f"<p class='sum'>{html.escape(stories[c['uid']])}</p>")
            out.append(f"<p class='meta'>{bits}</p></div></article>")
        out.append("</section>")

    out.append("<footer>The number beside each story is how many independent "
               "outlets are covering it. Nothing is ranked by clicks, because "
               "no one publishes those.</footer></div></body></html>")
    return "\n".join(out)


ARCHIVE_CSS = CSS + """
.months{margin-top:8px}
.mo{margin-bottom:40px}
.mo h3{font-size:13px; font-weight:600; margin:0 0 12px; padding-bottom:8px;
  border-bottom:1px solid var(--rule)}
.grid{display:grid; grid-template-columns:repeat(7,1fr); gap:5px; max-width:400px}
.dow{font-size:11px; color:var(--quiet); text-align:center; padding-bottom:4px}
.day{aspect-ratio:1; display:flex; align-items:center; justify-content:center;
  font-size:14px; font-variant-numeric:tabular-nums; border-radius:3px}
.day.off{color:transparent}
.day.none{color:var(--quiet)}
.day a{display:flex; align-items:center; justify-content:center; width:100%;
  height:100%; color:#fff; background:var(--signal); text-decoration:none;
  border-radius:3px; font-weight:500}
.day a:hover,.day a:focus{background:var(--ink)}
.lead{font-size:14px; color:var(--soft); margin:14px 0 0; max-width:56ch}
"""


def build_archive(index):
    """Calendar of every day on file. Rebuilt from data/index.json each run."""
    by_month = {}
    for row in index:
        d = date.fromisoformat(row["date"])
        by_month.setdefault((d.year, d.month), {})[d.day] = row

    out = [HEAD, "<title>Brief archive</title>",
           f"<style>{ARCHIVE_CSS}</style></head><body><div class='wrap'>",
           "<nav><a href='index.html'>today</a><span class='sp'></span></nav>",
           "<header><h1>Archive</h1>",
           f"<p class='count'><b>{len(index)}</b> briefs on file.</p>"
           "</header><div class='months'>"]

    for (year, month) in sorted(by_month, reverse=True):
        days = by_month[(year, month)]
        out.append(f"<div class='mo'><h3>{cal.month_name[month]} {year}</h3>")
        out.append("<div class='grid'>")
        for label in ("M", "T", "W", "T", "F", "S", "S"):
            out.append(f"<div class='dow'>{label}</div>")
        for week in cal.Calendar(firstweekday=0).monthdayscalendar(year, month):
            for day in week:
                if day == 0:
                    out.append("<div class='day off'>0</div>")
                elif day in days:
                    out.append(f"<div class='day'><a href='{days[day]['date']}.html'>"
                               f"{day}</a></div>")
                else:
                    out.append(f"<div class='day none'>{day}</div>")
        out.append("</div>")

        newest = days[max(days)]
        if newest.get("lead"):
            out.append(f"<p class='lead'>Latest: {html.escape(newest['lead'])}</p>")
        out.append("</div>")

    out.append("</div></div></body></html>")
    return "\n".join(out)


# ----------------------------------------------------------------------------

def main():
    sources_cfg, sections_cfg = load_config()
    sources = sources_cfg["sources"]
    sections = sections_cfg["sections"]

    articles = fetch_all(sources)
    if not articles:
        print("No articles fetched. Run check_feeds.py.")
        return 1

    clusters = score_clusters(cluster(articles))
    for i, c in enumerate(clusters):
        c["uid"] = f"s{i}"
    clusters = apply_history(clusters, load_history())

    assigned = route(clusters, sections)
    col_section = next(s for s in sections if s["match"] == "by_outlet")
    columns = build_columns(articles, assigned, col_section)
    summaries = summarize(assigned, columns, sections)

    when = datetime.now(TIMEZONE)
    today = when.date().isoformat()

    os.makedirs(DATA_DIR, exist_ok=True)
    index_path = os.path.join(DATA_DIR, "index.json")
    index = json.load(open(index_path)) if os.path.exists(index_path) else []
    index = [r for r in index if r["date"] != today]

    dates = sorted([r["date"] for r in index] + [today])
    pos = dates.index(today)
    prev_date = dates[pos - 1] if pos > 0 else None

    page = render(assigned, columns, sections, summaries,
                  sources_cfg.get("link_out", {}), len(clusters), when,
                  prev_date=prev_date)

    os.makedirs(OUT_DIR, exist_ok=True)
    dated = os.path.join(OUT_DIR, f"{today}.html")
    with open(dated, "w") as f:
        f.write(page)
    shutil.copyfile(dated, os.path.join(OUT_DIR, "index.html"))

    lead = None
    for s in sorted(sections, key=lambda x: x["order"]):
        if assigned.get(s["id"]):
            lead = assigned[s["id"]][0]["lead"]["title"]
            break

    index.append({"date": today, "stories": len(clusters), "lead": lead})
    index.sort(key=lambda r: r["date"])
    with open(index_path, "w") as f:
        json.dump(index, f, indent=1)

    # yesterday's page gains a forward link now that today exists
    if prev_date:
        prev_file = os.path.join(OUT_DIR, f"{prev_date}.html")
        if os.path.exists(prev_file):
            src = open(prev_file).read()
            if "'>next</a>" not in src:
                src = src.replace("<span class='sp'>",
                                  f"<a href='{today}.html'>next</a><span class='sp'>", 1)
                open(prev_file, "w").write(src)

    with open(os.path.join(OUT_DIR, "archive.html"), "w") as f:
        f.write(build_archive(index))

    print(f"\nWrote docs/{today}.html, docs/index.html, docs/archive.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
