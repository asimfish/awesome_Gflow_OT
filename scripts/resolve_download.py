#!/usr/bin/env python3
"""Resolve missing arXiv IDs by title search, then download all paper PDFs.

Usage: python3 scripts/resolve_download.py [--download-only]
Writes back resolved IDs into data/papers_resolved.yaml and downloads to papers/.
"""
import sys
import time
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NS = {"a": "http://www.w3.org/2005/Atom"}


def arxiv_search(title: str):
    q = urllib.parse.quote(f'ti:"{title}"')
    url = f"http://export.arxiv.org/api/query?search_query={q}&max_results=5"
    with urllib.request.urlopen(url, timeout=30) as r:
        tree = ET.fromstring(r.read())
    results = []
    for e in tree.findall("a:entry", NS):
        t = re.sub(r"\s+", " ", e.find("a:title", NS).text.strip())
        aid = e.find("a:id", NS).text.rsplit("/", 1)[-1]
        results.append((aid, t))
    return results


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def main():
    data = yaml.safe_load((ROOT / "data/papers.yaml").read_text())
    papers = data["papers"]
    for p in papers:
        if p.get("arxiv"):
            continue
        title = p["title"]
        try:
            results = arxiv_search(title)
        except Exception as exc:
            print(f"[WARN] {p['id']} search failed: {exc}")
            continue
        hit = None
        for aid, t in results:
            if norm(t) == norm(title) or norm(title) in norm(t) or norm(t) in norm(title):
                hit = (aid, t)
                break
        if hit:
            p["arxiv"] = re.sub(r"v\d+$", "", hit[0])
            p["resolved_title"] = hit[1]
            print(f"[OK]   {p['id']} -> {p['arxiv']}  ({hit[1][:70]})")
        else:
            print(f"[MISS] {p['id']} '{title[:60]}' candidates:")
            for aid, t in results[:3]:
                print(f"         {aid}  {t[:80]}")
        time.sleep(3)  # arXiv API ????

    out = ROOT / "data/papers_resolved.yaml"
    out.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False))
    print(f"\nwrote {out}")

    # download
    pdir = ROOT / "papers"
    pdir.mkdir(exist_ok=True)
    for p in papers:
        aid = p.get("arxiv")
        if not aid:
            print(f"[SKIP] {p['id']} no arxiv id")
            continue
        dest = pdir / f"{aid}.pdf"
        if dest.exists() and dest.stat().st_size > 10000:
            print(f"[HAVE] {p['id']} {dest.name}")
            continue
        url = f"https://arxiv.org/pdf/{aid}"
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"[DL]   {p['id']} {dest.name} ({dest.stat().st_size // 1024} KB)")
            time.sleep(3)
        except Exception as exc:
            print(f"[FAIL] {p['id']} {url}: {exc}")


if __name__ == "__main__":
    main()
