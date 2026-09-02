#!/usr/bin/env python3
"""Assemble the consolidated report from reports/report_src/*.md (ordered by filename).

Directives (each on its own line):
  @@INCLUDE <path> [SECTIONS=all|1,2,5] [DEMOTE=n] [SKIP_META]  -- pull a deep-dive report in,
        keeping only numbered top-level sections listed (matching '## N.' headings), demoting
        headings by n levels, optionally dropping the leading one-liner/metadata table.
  @@PAPER_TABLE            -- appendix table of all papers with QA counts
  @@REFERENCES             -- reference list generated from data/meta/*.json
Usage: python3 scripts/build_report.py zh   -> reports/AWESOME_GFLOW_OT_REPORT_zh.md
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data/meta").glob("*.json"))}
ORDER = ["O08", "O07", "T19", "T36", "T00", "T02", "T03", "T05", "T10", "O01", "O02", "O03", "O04", "O05", "O06", "C01", "C02", "C03"]


def demote(text, n):
    if n <= 0:
        return text
    out = []
    in_code = False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,6} ", line):
            line = "#" * n + line
        out.append(line)
    return "\n".join(out)


def include(path, sections="all", demote_n=2, skip_meta=False):
    text = (ROOT / path).read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    lines = text.split("\n")
    # split into preamble (before first '## ') and sections keyed by leading number
    pre, secs, cur, key = [], {}, None, None
    for line in lines:
        m = re.match(r"^## (\d+)\.", line)
        m2 = re.match(r"^## ", line)
        if m:
            key = m.group(1)
            cur = secs.setdefault(key, [])
            cur.append(line)
        elif m2:  # unnumbered ## (e.g. editor's note)
            key = line.strip("# ").strip()
            cur = secs.setdefault(key, [])
            cur.append(line)
        elif cur is None:
            pre.append(line)
        else:
            cur.append(line)
    if skip_meta:
        pre = [l for l in pre if l.startswith("# ")]
    keep = list(secs.keys()) if sections == "all" else [s.strip() for s in sections.split(",")]
    body = pre + sum((secs[k] for k in keep if k in secs), [])
    # drop the top-level '# ' title of the report (we add our own heading)
    body = [l for l in body if not l.startswith("# ")]
    return demote("\n".join(body).strip(), demote_n)


def paper_table():
    hdr = "| ID | Paper | Venue | Type | Report | zh-PDF QA |" if LANG == "en" else "| ?? | ?? | ?? | ?? | ???? | ?? QA |"
    rows = [hdr, "|---|---|---|---|---|---|"]
    for pid in ORDER:
        m = META.get(pid)
        if not m:
            continue
        arxiv = m.get("arxiv", "")
        rep = next(iter((ROOT / "reports").glob(f"{pid}_*.md")), None)
        qj = ROOT / "papers_zh" / f"{arxiv}.inspect.json"
        if qj.exists():
            qa = f"{json.loads(qj.read_text())['issue_count']} issues"
        elif (ROOT / "papers_zh" / f"{arxiv}.zh.pdf").exists():
            qa = "translated, not inspected" if LANG == "en" else "?????"
        else:
            qa = "not translated" if LANG == "en" else "??"
        rows.append(f"| {pid} | {m['title']} | {m.get('venue','')} | {m.get('venue_type','')} | `{rep.name if rep else '-'}` | {qa} |")
    return "\n".join(rows)


def references():
    out = []
    for pid in ORDER:
        m = META.get(pid)
        if not m:
            continue
        authors = ", ".join(m.get("authors", []))
        arxiv = m.get("arxiv")
        link = f" arXiv:[{arxiv}](https://arxiv.org/abs/{arxiv})." if arxiv else ""
        out.append(f"- **[{pid}]** {authors}. *{m['title']}*. {m.get('venue','')}, {m.get('year','')}.{link}")
    return "\n".join(out)


LANG = "zh"


def main(lang):
    global LANG
    LANG = lang
    src_dir = ROOT / "reports" / "report_src" / lang
    parts = []
    for p in sorted(src_dir.glob("*.md")):
        for line in p.read_text(encoding="utf-8").split("\n"):
            if line.startswith("@@INCLUDE "):
                toks = line.split()
                path = toks[1]
                kw = dict(t.split("=", 1) for t in toks[2:] if "=" in t)
                parts.append(include(path, kw.get("SECTIONS", "all"), int(kw.get("DEMOTE", "2")), "SKIP_META" in toks))
            elif line.strip() == "@@PAPER_TABLE":
                parts.append(paper_table())
            elif line.strip() == "@@REFERENCES":
                parts.append(references())
            else:
                parts.append(line)
    out = ROOT / "reports" / f"AWESOME_GFLOW_OT_REPORT_{lang}.md"
    out.write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {out} ({sum(1 for _ in out.open(encoding='utf-8'))} lines)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "zh")
