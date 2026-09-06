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
from urllib.parse import quote_plus, urljoin, urlparse

import feedparser
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(ROOT, "config")
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "docs")

MODEL = os.environ.get("BRIEF_MODEL", "claude-sonnet-4-6")
API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

IMAGE_TIMEOUT = 6          # seconds per og:image lookup
IMAGE_WORKERS = 8         # concurrent lookups
IMAGE_CACHE_DAYS = 60     # forget entries not seen for this long
IMAGE_HEAD_BYTES = 200000  # og:image lives in <head>, so stop reading early

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
            "image": image_from_entry(entry, source["feed"]) or "",
        })

    print(f"  {source['name']}: {len(articles)}")
    return articles


def fetch_all(sources):
    print("Fetching feeds")
    with ThreadPoolExecutor(max_workers=12) as pool:
        batches = pool.map(fetch_one, sources)
    return [a for batch in batches for a in batch]


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------

# Images are hotlinked, never downloaded. A year of downloaded art would add a
# few hundred megabytes to a repo whose pages are read once and then archived.

IMG_ATTR = re.compile(r"<img[^>]+src=[\"\']([^\"\']+)[\"\']", re.I)
OG_TAG = re.compile(
    r"<meta[^>]+(?:property|name)=[\"\']"
    r"(og:image(?::secure_url|:url)?|twitter:image(?::src)?)[\"\']"
    r"[^>]*?content=[\"\']([^\"\']+)[\"\']", re.I)
OG_TAG_REV = re.compile(
    r"<meta[^>]+content=[\"\']([^\"\']+)[\"\'][^>]*?"
    r"(?:property|name)=[\"\'](og:image(?::secure_url|:url)?|twitter:image(?::src)?)[\"\']",
    re.I)


def usable_image(url, base=None):
    """Absolute http(s) URL, or None. Rejects data URIs and tracking pixels."""
    if not url:
        return None
    url = html.unescape(url.strip())
    if not url or url.startswith("data:"):
        return None
    if base:
        url = urljoin(base, url)
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return None
    if re.search(r"\b1x1\b|/pixel\.|spacer\.gif|blank\.gif", url, re.I):
        return None
    return url


def image_from_entry(entry, base=None):
    """Cheapest path: an image the feed already handed us. No extra request."""
    for media in entry.get("media_content") or []:
        if isinstance(media, dict):
            kind = (media.get("medium") or media.get("type") or "")
            if kind and not kind.startswith("image"):
                continue
            found = usable_image(media.get("url"), base)
            if found:
                return found

    for thumb in entry.get("media_thumbnail") or []:
        if isinstance(thumb, dict):
            found = usable_image(thumb.get("url"), base)
            if found:
                return found

    for link in entry.get("links") or []:
        if isinstance(link, dict) and (link.get("type") or "").startswith("image/"):
            found = usable_image(link.get("href"), base)
            if found:
                return found

    blocks = [entry.get("summary") or ""]
    for content in entry.get("content") or []:
        if isinstance(content, dict):
            blocks.append(content.get("value") or "")
    for block in blocks:
        for candidate in IMG_ATTR.findall(block):
            found = usable_image(candidate, base)
            if found:
                return found
    return None


def fetch_og_image(link):
    """Second path: read the article head for og:image. Never raises."""
    try:
        resp = requests.get(
            link, timeout=IMAGE_TIMEOUT, stream=True, allow_redirects=True,
            headers={"User-Agent": "personal-news-brief/1.0",
                     "Accept": "text/html,application/xhtml+xml"})
        if resp.status_code != 200:
            return None
        if "html" not in (resp.headers.get("content-type") or "").lower():
            return None

        head = b""
        for chunk in resp.iter_content(16384):
            head += chunk
            if b"</head>" in head.lower() or len(head) >= IMAGE_HEAD_BYTES:
                break
        resp.close()

        text = head.decode("utf-8", "ignore")
        for pattern in (OG_TAG, OG_TAG_REV):
            for groups in pattern.findall(text):
                candidate = groups[1] if pattern is OG_TAG else groups[0]
                found = usable_image(candidate, resp.url)
                if found:
                    return found
    except Exception:
        return None
    return None


def load_image_cache():
    path = os.path.join(DATA_DIR, "images.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return {}
    cutoff = (datetime.now(timezone.utc).date()
              - timedelta(days=IMAGE_CACHE_DAYS)).isoformat()
    return {k: v for k, v in raw.items()
            if isinstance(v, dict) and v.get("seen", "") >= cutoff}


def save_image_cache(cache):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "images.json"), "w") as f:
        json.dump(cache, f, indent=1, sort_keys=True)


def resolve_images(articles):
    """Fill in .image for the articles that will actually be shown.

    Anything the feed already gave us is free and already set. Only the
    remainder costs a request, which is why this runs after routing: it is
    roughly 25 lookups a day, not one per article fetched.
    """
    cache = load_image_cache()
    today = datetime.now(timezone.utc).date().isoformat()

    for a in articles:
        if a.get("image"):
            cache[a["link"]] = {"src": a["image"], "seen": today}

    pending = []
    for a in articles:
        if a.get("image"):
            continue
        hit = cache.get(a["link"])
        if hit is not None:
            # a null src is a remembered failure, so we do not retry it daily
            a["image"] = hit.get("src") or ""
            hit["seen"] = today
        else:
            pending.append(a)

    if pending:
        print(f"Resolving images for {len(pending)} articles")
        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as pool:
            found = list(pool.map(lambda a: fetch_og_image(a["link"]), pending))
        for a, src in zip(pending, found):
            a["image"] = src or ""
            cache[a["link"]] = {"src": src, "seen": today}
        print(f"  {sum(1 for s in found if s)} of {len(pending)} resolved")

    save_image_cache(cache)
    return articles


def displayed_leads(assigned, sections):
    """The lead article of every card that will be rendered with an image."""
    by_id = {s["id"]: s for s in sections}
    leads = []
    for sid, items in assigned.items():
        if by_id.get(sid, {}).get("match") == "by_outlet":
            continue
        for c in items:
            leads.append(c["lead"])
    return leads


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
        if section["match"] in ("consensus", "by_outlet", "periodic"):
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
# periodic sections
# ----------------------------------------------------------------------------

# A periodic section carries research publishers that post every few weeks
# rather than every day: Dealroom, CVCA, RBCx, PitchBook-NVCA, Carta. It shows
# only what it has not shown before, and when there is nothing new it does not
# render at all. That absence is the design, not an empty state.
#
# These items are never clustered and never ranked by outlet count. A single
# research post has no consensus signal to measure, so the loud number that
# carries the daily sections would be meaningless here.

SEEN_PERIODIC_DAYS = 730


def periodic_sources(sections):
    """Every source name claimed by any periodic section, as {name: section id}."""
    owners = {}
    for section in sections:
        if section.get("match") != "periodic":
            continue
        for name in section.get("sources", []):
            owners[name] = section["id"]
    return owners


def split_periodic(articles, sections):
    """Hold periodic articles out of the cluster pool.

    They carry no consensus signal, so letting them cluster would either add a
    phantom outlet to somebody else's count or let a research post get claimed
    by a keyword section and appear twice. With no periodic sources configured
    this is a no-op and the article list is returned unchanged.
    """
    owners = periodic_sources(sections)
    if not owners:
        return [], articles
    held = [a for a in articles if a["outlet"] in owners]
    rest = [a for a in articles if a["outlet"] not in owners]
    if held:
        print(f"  holding {len(held)} periodic items out of clustering")
    return held, rest


def load_seen_periodic():
    path = os.path.join(DATA_DIR, "seen_periodic.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        return {}
    cutoff = (datetime.now(timezone.utc).date()
              - timedelta(days=SEEN_PERIODIC_DAYS)).isoformat()
    return {k: v for k, v in raw.items() if isinstance(v, str) and v >= cutoff}


def save_seen_periodic(seen):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "seen_periodic.json"), "w") as f:
        json.dump(seen, f, indent=1, sort_keys=True)


def collect_periodic(held, sections, seen):
    """New items only, wrapped so they travel the same path as a story card."""
    owners = periodic_sources(sections)
    by_section = {}

    for article in sorted(held, key=lambda a: -a["published"].timestamp()):
        sid = owners.get(article["outlet"])
        if sid is None or article["link"] in seen:
            continue
        by_section.setdefault(sid, []).append(article)

    out = {}
    for section in sections:
        if section.get("match") != "periodic":
            continue
        items = by_section.get(section["id"], [])[:section.get("slots", 6)]
        if not items:
            continue
        out[section["id"]] = [{
            "uid": f"p{section['id']}{i}", "lead": a, "articles": [a],
            "outlet_count": 1, "outlets": [a["outlet"]], "days_running": 1,
            "ca_share": 0.0, "score": 0.0, "tokens": set(),
        } for i, a in enumerate(items)]
        print(f"  {section['id']}: {len(items)} new")
    return out


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
  --paper:#F2F3F4; --ink:#101112; --ink-mid:#5B5F63; --ink-quiet:#8D9296;
  --rule:#D6D8DA; --placeholder:#E2E4E6; --accent:#C8102E;
  --col:3;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0; background:var(--paper); color:var(--ink);
  font-family:Archivo,'Helvetica Neue',Helvetica,Arial,sans-serif;
  font-size:14px; line-height:1.45; -webkit-font-smoothing:antialiased}
img{max-width:100%}
a{color:inherit; text-decoration:none}
.wrap{max-width:1280px; margin:0 auto; padding:36px 28px 96px}

/* masthead ---------------------------------------------------------------- */
.mast{display:flex; align-items:flex-start; justify-content:space-between;
  gap:24px; margin-bottom:44px}
.mark{font-size:44px; font-weight:800; letter-spacing:-.03em; line-height:.95;
  margin:0; color:var(--ink)}
.mast-right{text-align:right; padding-top:6px}
.mast-date{font-size:11px; font-weight:500; letter-spacing:.06em;
  color:var(--ink-quiet); margin:0 0 2px}
.mast-tally{font-size:11px; font-weight:400; color:var(--ink-quiet); margin:0 0 8px}
nav{display:flex; justify-content:flex-end; align-items:baseline; gap:14px;
  font-size:11px; font-weight:500; letter-spacing:.06em}
nav a{color:var(--ink-mid); border-bottom:1px solid var(--rule); padding-bottom:1px}
nav a:hover{color:var(--ink); border-bottom-color:var(--ink)}
nav .sp{width:2px}

/* section header ---------------------------------------------------------- */
.sec{margin-bottom:52px}
.sec-head{display:flex; align-items:center; gap:12px; margin:0 0 14px}
.sec-badge{flex:0 0 auto; width:28px; height:28px; border-radius:50%;
  background:var(--ink); color:var(--paper); display:flex; align-items:center;
  justify-content:center; font-size:12px; font-weight:700; text-transform:uppercase}
.sec-title{font-size:34px; font-weight:700; letter-spacing:-.02em; line-height:1;
  margin:0; text-transform:uppercase}
.sec-rule{height:1px; background:var(--rule); margin:0 0 0}
.sec-gist{font-size:12px; color:var(--ink-mid); margin:12px 0 0; max-width:62ch}

/* card grid --------------------------------------------------------------- */
.grid{display:grid; grid-template-columns:repeat(var(--col),1fr); margin-top:0}
.card{display:flex; flex-direction:column; padding:22px 20px 24px;
  border-right:1px solid var(--rule); border-bottom:1px solid var(--rule);
  min-width:0}
.card:nth-child(3n){border-right:none}
.card-hl{font-size:14px; font-weight:600; line-height:1.25; letter-spacing:-.005em;
  margin:0 0 8px; display:-webkit-box; -webkit-line-clamp:3; -webkit-box-orient:vertical;
  overflow:hidden; border-bottom:1px solid transparent}
.card-hl:hover{border-bottom-color:var(--ink)}
.card-sum{font-size:12px; font-weight:400; line-height:1.4; color:var(--ink-mid);
  margin:0 0 18px; display:-webkit-box; -webkit-line-clamp:2;
  -webkit-box-orient:vertical; overflow:hidden}
.card-meta{display:flex; align-items:flex-end; justify-content:space-between;
  gap:10px; margin-top:auto; margin-bottom:12px}
.tags{display:flex; align-items:center; gap:6px; flex-wrap:wrap; min-width:0;
  font-size:10px; font-weight:500; letter-spacing:.05em; color:var(--ink-quiet);
  padding-bottom:5px}
.dot{flex:0 0 auto; width:14px; height:14px; border-radius:50%; background:var(--ink);
  color:var(--paper); display:flex; align-items:center; justify-content:center;
  font-size:8px; font-weight:700; text-transform:uppercase}
.tags .run{color:var(--ink-mid)}
.tags .gate{color:var(--ink-mid)}
.card-n{flex:0 0 auto; font-size:32px; font-weight:500; line-height:1;
  letter-spacing:-.02em; font-variant-numeric:tabular-nums; color:var(--ink-quiet)}
.card-n.mid{color:var(--ink)}
.card-n.hot{color:var(--accent)}
.card-media{position:relative; width:100%; aspect-ratio:4/3;
  background:var(--placeholder); overflow:hidden}
.card-media img{position:absolute; inset:0; width:100%; height:100%;
  object-fit:cover; display:block}

/* by outlet --------------------------------------------------------------- */
.cols{display:grid; grid-template-columns:repeat(var(--col),1fr)}
.col{padding:22px 20px 24px; border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule); min-width:0}
.col:nth-child(3n){border-right:none}
.col h3{font-size:12px; font-weight:700; letter-spacing:.06em; margin:0 0 4px;
  text-transform:uppercase}
.col .lede{font-size:11px; color:var(--ink-quiet); margin:0 0 12px; line-height:1.35}
.col ul{list-style:none; margin:0; padding:0}
.col li{margin-bottom:10px; font-size:13px; font-weight:500; line-height:1.3}
.col li a{border-bottom:1px solid transparent}
.col li a:hover{border-bottom-color:var(--ink)}

/* occasional section ------------------------------------------------------ */
/* Inverted on purpose. It appears only when a research publisher posts, so it
   should be impossible to miss and impossible to confuse with a daily block. */
.sec.occ{background:var(--ink); color:var(--paper); padding:26px 24px 10px;
  margin-left:-24px; margin-right:-24px}
.sec.occ .sec-badge{background:var(--paper); color:var(--ink)}
.sec.occ .sec-title{color:var(--paper)}
.sec.occ .sec-rule{background:#3A3D40}
.sec.occ .sec-gist{color:#A8ACB0}
.occ-list{margin-top:18px}
.occ-item{display:flex; gap:18px; padding:18px 0; border-bottom:1px solid #3A3D40}
.occ-item:last-child{border-bottom:none}
.occ-media{flex:0 0 172px; aspect-ratio:3/2; background:#2A2D30;
  position:relative; overflow:hidden}
.occ-media img{position:absolute; inset:0; width:100%; height:100%;
  object-fit:cover; display:block}
.occ-body{flex:1; min-width:0}
.occ-hl{display:block; font-size:17px; font-weight:600; line-height:1.25;
  letter-spacing:-.01em; margin:0 0 6px; color:var(--paper);
  border-bottom:1px solid transparent}
.occ-hl:hover{border-bottom-color:var(--paper)}
.occ-sum{font-size:13px; line-height:1.45; color:#A8ACB0; margin:0 0 8px;
  max-width:64ch}
.occ-meta{font-size:10px; font-weight:500; letter-spacing:.05em; color:#8A8E92;
  margin:0; display:flex; align-items:center; flex-wrap:wrap}
.occ-meta .sep{display:inline-block; width:3px; height:3px; border-radius:50%;
  background:#5A5E62; margin:0 8px}
.sec.occ a:focus-visible{outline-color:var(--paper)}

footer{margin-top:56px; padding-top:16px; border-top:1px solid var(--rule);
  font-size:11px; color:var(--ink-quiet); max-width:72ch; line-height:1.5}

a:focus-visible,.card-hl:focus-visible{outline:2px solid var(--accent);
  outline-offset:2px; border-bottom-color:transparent}

@media (max-width:1023px){
  :root{--col:2}
  .card:nth-child(3n),.col:nth-child(3n){border-right:1px solid var(--rule)}
  .card:nth-child(2n),.col:nth-child(2n){border-right:none}
  .mark{font-size:36px}
  .sec-title{font-size:28px}
}
@media (max-width:767px){
  :root{--col:1}
  .wrap{padding:24px 16px 72px}
  .mast{flex-direction:column; gap:14px}
  .mast-right{text-align:left; padding-top:0}
  nav{justify-content:flex-start}
  .mark{font-size:34px}
  .sec-title{font-size:24px}
  .sec-badge{width:24px; height:24px; font-size:11px}
  .sec.occ{margin-left:-16px; margin-right:-16px; padding:22px 16px 8px}
  .occ-item{flex-direction:column; gap:12px}
  .occ-media{flex:0 0 auto; width:100%; aspect-ratio:16/9}
  .occ-hl{font-size:16px}
  .card,.col{padding:20px 0 22px; border-right:none}
  .card:nth-child(3n),.card:nth-child(2n),
  .col:nth-child(3n),.col:nth-child(2n){border-right:none}
  .card-n{font-size:28px}
}
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{transition:none!important; animation:none!important;
    scroll-behavior:auto!important}
}
"""

HEAD = ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<link rel='preconnect' href='https://fonts.googleapis.com'>"
        "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>"
        "<link href='https://fonts.googleapis.com/css2?family=Archivo:wght@"
        "400;500;600;700;800&display=swap' rel='stylesheet'>")

WORDMARK = "Alicia's newsroom"

# An image that fails at view time collapses so the tinted block behind it
# shows through. A never-resolved image and a dead one look identical.
IMG_ONERROR = "this.style.display='none'"


def tier(n):
    return "hot" if n >= 12 else ("mid" if n >= 5 else "")


def read_link(c, link_out):
    lead = c["lead"]
    if lead["access"] == "proquest" and lead.get("proquest_id"):
        url = (f"https://www.proquest.com/results/{quote_plus(lead['title'])}"
               f"?accountid={link_out.get('proquest_account', '')}")
        return (link_out.get("proquest_prefix") or "") + url, "via ProQuest"
    return lead["link"], None


def card_media(c, cls="card-media"):
    """The image slot. Always present, so a missing image is a tonal block
    rather than a hole in the grid."""
    url = (c.get("lead") or {}).get("image") or ""
    if not url:
        return f"<div class='{cls}'></div>"
    alt = html.escape(c["lead"]["title"], quote=True)
    return (f"<div class='{cls}'>"
            f"<img src='{html.escape(url, quote=True)}' alt='{alt}' "
            f"loading='lazy' referrerpolicy='no-referrer' "
            f"onerror=\"{IMG_ONERROR}\"></div>")


def section_head(title, gist=None):
    initial = html.escape(title.strip()[:1])
    out = ["<div class='sec-head'>",
           f"<span class='sec-badge' aria-hidden='true'>{initial}</span>",
           f"<h2 class='sec-title'>{html.escape(title)}</h2>",
           "</div>", "<div class='sec-rule'></div>"]
    if gist:
        out.append(f"<p class='sec-gist'>{html.escape(gist)}</p>")
    return "".join(out)


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
           "<header class='mast'>",
           f"<h1 class='mark'>{html.escape(WORDMARK)}</h1>",
           "<div class='mast-right'>",
           f"<p class='mast-date'>{when:%A, %-d %B %Y}</p>",
           f"<p class='mast-tally'>{total} stories in, {shown} shown</p>",
           "<nav>" + "".join(nav) + "</nav>",
           "</div></header>"]

    for section in sorted(sections, key=lambda s: s["order"]):
        if section["match"] == "by_outlet":
            if not columns:
                continue
            out.append("<section class='sec'>")
            out.append(section_head(section["title"],
                                    "What each newsroom is running that did not "
                                    "make a block above."))
            out.append("<div class='cols'>")
            for col in columns:
                out.append(f"<div class='col'><h3>{html.escape(col['label'])}</h3>")
                lede = col_gists.get(col["label"])
                if lede:
                    out.append(f"<p class='lede'>{html.escape(lede)}</p>")
                out.append("<ul>")
                for a in col["articles"]:
                    out.append(f"<li><a href='{html.escape(a['link'], quote=True)}'>"
                               f"{html.escape(a['title'])}</a></li>")
                out.append("</ul></div>")
            out.append("</div></section>")
            continue

        items = assigned.get(section["id"], [])
        if not items:
            # a periodic section with nothing new leaves no trace at all
            continue

        if section["match"] == "periodic":
            out.append("<section class='sec occ'>")
            out.append(section_head(section["title"], gists.get(section["id"])))
            out.append("<div class='occ-list'>")
            for c in items:
                lead = c["lead"]
                url, gate = read_link(c, link_out)
                out.append("<article class='occ-item'>")
                out.append(card_media(c, cls="occ-media"))
                out.append("<div class='occ-body'>")
                out.append(f"<a class='occ-hl' href='{html.escape(url, quote=True)}'>"
                           f"{html.escape(lead['title'])}</a>")
                if stories.get(c["uid"]):
                    out.append(f"<p class='occ-sum'>{html.escape(stories[c['uid']])}</p>")
                bits = [html.escape(lead["outlet"]),
                        f"{lead['published'].astimezone(TIMEZONE):%-d %B %Y}"]
                if gate:
                    bits.append(html.escape(gate))
                out.append("<p class='occ-meta'>" +
                           "<span class='sep'></span>".join(bits) + "</p>")
                out.append("</div></article>")
            out.append("</div></section>")
            continue

        out.append("<section class='sec'>")
        out.append(section_head(section["title"], gists.get(section["id"])))
        out.append("<div class='grid'>")

        for c in items:
            url, gate = read_link(c, link_out)
            lead_outlet = c["lead"]["outlet"]
            full = ", ".join(c["outlets"])

            tags = [f"<span class='dot' aria-hidden='true'>"
                    f"{html.escape(lead_outlet[:1])}</span>",
                    f"<span>{html.escape(lead_outlet)}</span>"]
            if c["days_running"] > 1:
                tags.append(f"<span class='run'>day {c['days_running']}</span>")
            if gate:
                tags.append(f"<span class='gate'>{html.escape(gate)}</span>")

            n = c["outlet_count"]
            label = f"{n} outlets covering this: {html.escape(full, quote=True)}"

            out.append("<article class='card'>")
            out.append(f"<a class='card-hl' href='{html.escape(url, quote=True)}'>"
                       f"{html.escape(c['lead']['title'])}</a>")
            if stories.get(c["uid"]):
                out.append(f"<p class='card-sum'>{html.escape(stories[c['uid']])}</p>")
            out.append("<div class='card-meta'>")
            out.append("<span class='tags'>" + "".join(tags) + "</span>")
            out.append(f"<span class='card-n {tier(n)}' title='{label}'>{n}</span>")
            out.append("</div>")
            out.append(card_media(c))
            out.append("</article>")

        out.append("</div></section>")

    out.append("<footer>The number on each card is how many independent outlets "
               "are covering that story. Nothing is ranked by clicks, because no "
               "one publishes those.</footer></div></body></html>")
    return "\n".join(out)


ARCHIVE_CSS = CSS + """
.months{display:grid; grid-template-columns:repeat(var(--col),1fr)}
.mo{padding:22px 20px 26px; border-right:1px solid var(--rule);
  border-bottom:1px solid var(--rule); min-width:0}
.mo:nth-child(3n){border-right:none}
.mo h3{font-size:12px; font-weight:700; letter-spacing:.06em; margin:0 0 14px;
  text-transform:uppercase}
.cal{display:grid; grid-template-columns:repeat(7,1fr); gap:3px; max-width:280px}
.dow{font-size:9px; font-weight:500; letter-spacing:.04em; color:var(--ink-quiet);
  text-align:center; padding-bottom:4px}
.day{aspect-ratio:1; display:flex; align-items:center; justify-content:center;
  font-size:12px; font-weight:500; font-variant-numeric:tabular-nums}
.day.off{color:transparent}
.day.none{color:var(--ink-quiet)}
.day a{display:flex; align-items:center; justify-content:center; width:100%;
  height:100%; color:var(--paper); background:var(--ink); font-weight:600}
.day a:hover{background:var(--accent)}
.lead{font-size:11px; color:var(--ink-mid); margin:14px 0 0; line-height:1.4}
@media (max-width:1023px){ .mo:nth-child(3n){border-right:1px solid var(--rule)}
  .mo:nth-child(2n){border-right:none} }
@media (max-width:767px){ .mo{padding:20px 0 22px; border-right:none}
  .mo:nth-child(3n),.mo:nth-child(2n){border-right:none} }
"""


def build_archive(index):
    """Calendar of every day on file. Rebuilt from data/index.json each run."""
    by_month = {}
    for row in index:
        d = date.fromisoformat(row["date"])
        by_month.setdefault((d.year, d.month), {})[d.day] = row

    out = [HEAD, "<title>Brief archive</title>",
           f"<style>{ARCHIVE_CSS}</style></head><body><div class='wrap'>",
           "<header class='mast'>",
           f"<h1 class='mark'>{html.escape(WORDMARK)}</h1>",
           "<div class='mast-right'>",
           "<p class='mast-date'>Archive</p>",
           f"<p class='mast-tally'>{len(index)} briefs on file</p>",
           "<nav><a href='index.html'>today</a>"
           "<span class='sp'></span></nav>",
           "</div></header>",
           "<section class='sec'>",
           section_head("Archive"),
           "<div class='months'>"]

    for (year, month) in sorted(by_month, reverse=True):
        days = by_month[(year, month)]
        out.append(f"<div class='mo'><h3>{cal.month_name[month]} {year}</h3>")
        out.append("<div class='cal'>")
        for i, label in enumerate(("M", "T", "W", "T", "F", "S", "S")):
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

    out.append("</div></section></div></body></html>")
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

    held, articles = split_periodic(articles, sections)

    clusters = score_clusters(cluster(articles))
    for i, c in enumerate(clusters):
        c["uid"] = f"s{i}"
    clusters = apply_history(clusters, load_history())

    assigned = route(clusters, sections)

    seen_periodic = load_seen_periodic()
    for sid, items in collect_periodic(held, sections, seen_periodic).items():
        assigned[sid] = items

    col_section = next(s for s in sections if s["match"] == "by_outlet")
    columns = build_columns(articles, assigned, col_section)

    # after routing on purpose: only the cards that will be shown cost a request
    resolve_images(displayed_leads(assigned, sections))

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

    # only after the page exists, so a crashed run does not burn an item
    for section in sections:
        if section.get("match") != "periodic":
            continue
        for c in assigned.get(section["id"], []):
            seen_periodic[c["lead"]["link"]] = today
    save_seen_periodic(seen_periodic)

    print(f"\nWrote docs/{today}.html, docs/index.html, docs/archive.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
