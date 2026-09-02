#!/usr/bin/env python3
"""Scan arXiv for 2025-2026 papers on (a) GFlowNet theory/OT/non-acyclic, (b) OT on graphs / discrete SB / neural OT.
Writes data/scan_gfn.json and data/scan_ot.json with id, title, authors, published, updated, comment, journal_ref, abstract, categories.
Polite: 3s between requests. Usage: python3 scripts/scan_trends.py
"""
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NS = {"a": "http://www.w3.org/2005/Atom", "ar": "http://arxiv.org/schemas/atom"}
DATE = "submittedDate:[202501010000 TO 202609302359]"

QUERIES = {
    "gfn": [
        'all:GFlowNet',
        'all:GFlowNets',
        'all:"generative flow network"',
        'all:"generative flow networks"',
        'all:GFlowNet AND all:"optimal transport"',
        'all:GFlowNet AND (all:"shortest path" OR all:"minimum flow" OR all:"non-acyclic" OR all:cyclic)',
        'all:GFlowNet AND all:"Schrodinger bridge"',
        'all:"flow network" AND all:sampler AND all:reward',
    ],
    "ot": [
        'all:"optimal transport" AND all:graph AND (all:neural OR all:learning) AND all:plan',
        'all:"optimal transport" AND all:graphs AND all:"neural"',
        'all:"Schrodinger bridge" AND (all:discrete OR all:graph OR all:graphs)',
        'all:"minimum cost flow" AND (all:learning OR all:neural)',
        'all:"unbalanced optimal transport" AND (all:graph OR all:neural)',
        'all:"amortized" AND all:"optimal transport"',
        'all:"optimal transport" AND all:"reinforcement learning" AND (all:graph OR all:discrete)',
        'all:"entropic optimal transport" AND all:discrete',
        'all:Beckmann AND all:"optimal transport"',
        'all:"optimal transport" AND all:"error bound" AND (all:dual OR all:certificate)',
    ],
}


def fetch(query, start=0, n=100):
    q = f"({query}) AND {DATE}"
    url = ("http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q)
           + f"&start={start}&max_results={n}&sortBy=submittedDate&sortOrder=descending")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return ET.fromstring(r.read())
        except Exception as exc:
            print(f"  retry {attempt}: {exc}")
            time.sleep(8)
    return None


def parse(tree):
    out = []
    for e in tree.findall("a:entry", NS):
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        g = lambda tag: (e.find(tag, NS).text or "").strip() if e.find(tag, NS) is not None else ""
        out.append({
            "arxiv": aid,
            "title": re.sub(r"\s+", " ", g("a:title")),
            "authors": [a.find("a:name", NS).text for a in e.findall("a:author", NS)],
            "published": g("a:published")[:10],
            "updated": g("a:updated")[:10],
            "comment": re.sub(r"\s+", " ", g("ar:comment")),
            "journal_ref": g("ar:journal_ref"),
            "primary": (e.find("ar:primary_category", NS).attrib.get("term", "") if e.find("ar:primary_category", NS) is not None else ""),
            "abstract": re.sub(r"\s+", " ", g("a:summary")),
        })
    return out


for side, qs in QUERIES.items():
    seen = {}
    for q in qs:
        tree = fetch(q)
        time.sleep(3)
        if tree is None:
            continue
        rows = parse(tree)
        new = 0
        for r in rows:
            if r["arxiv"] not in seen:
                r["hit_queries"] = [q]
                seen[r["arxiv"]] = r
                new += 1
            else:
                seen[r["arxiv"]]["hit_queries"].append(q)
        print(f"[{side}] {len(rows):3d} hits, {new:3d} new  <- {q[:70]}")
    out = ROOT / "data" / f"scan_{side}.json"
    out.write_text(json.dumps(sorted(seen.values(), key=lambda r: r["published"], reverse=True), ensure_ascii=False, indent=1))
    print(f"[{side}] total unique: {len(seen)} -> {out}")
