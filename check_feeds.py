#!/usr/bin/env python3
"""
Check every feed in config/sources.json and report which ones are broken.

Run this first, and any time you add a source. Feed URLs rot constantly:
outlets move them, drop them, or put them behind Cloudflare. A source that
returns nothing will quietly skew your consensus ranking, so it is worth
knowing rather than guessing.

Run:  python check_feeds.py
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import feedparser
import requests

ROOT = os.path.dirname(os.path.abspath(__file__))


def check(source):
    try:
        resp = requests.get(
            source["feed"], timeout=20,
            headers={"User-Agent": "personal-news-brief/1.0"},
        )
        if resp.status_code != 200:
            return source, "fail", f"HTTP {resp.status_code}"
        parsed = feedparser.parse(resp.content)
        n = len(parsed.entries)
        if n == 0:
            return source, "fail", "parsed but zero entries"
        if n < 5:
            return source, "thin", f"only {n} entries"
        return source, "ok", f"{n} entries"
    except Exception as exc:
        return source, "fail", str(exc)[:70]


def main():
    with open(os.path.join(ROOT, "config", "sources.json")) as f:
        sources = json.load(f)["sources"]

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(pool.map(check, sources))

    order = {"fail": 0, "thin": 1, "ok": 2}
    results.sort(key=lambda r: (order[r[1]], r[0]["name"]))

    broken = 0
    for source, status, detail in results:
        mark = {"ok": "  ok  ", "thin": " thin ", "fail": " FAIL "}[status]
        print(f"{mark} {source['name']:<22} {detail}")
        if status == "fail":
            broken += 1

    print(f"\n{len(results) - broken} of {len(results)} working.")
    if broken:
        print("Find replacements, or delete the dead entries from sources.json.")
        print("Fallback for any outlet without a feed:")
        print("  https://news.google.com/rss/search?q=when:24h+site:DOMAIN"
              "&hl=en-CA&gl=CA&ceid=CA:en")
    return 0


if __name__ == "__main__":
    sys.exit(main())
