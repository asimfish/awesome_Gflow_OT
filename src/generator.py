#!/usr/bin/env python3
"""Generate README.md (EN) and README_zh.md (ZH) from data/meta/*.json + data/candidates_*.csv.

Follows the awesome-ml4co convention: data files are the source of truth, README is generated.
Run from repo root:  python3 src/generator.py
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "meta"
REPORTS = ROOT / "reports"
PAPERS = ROOT / "papers"
PAPERS_ZH = ROOT / "papers_zh"

# section -> (title_en, title_zh, ordered paper ids)
SECTIONS = [
    ("core", "GFlowNet x Optimal Transport (core)", "GFlowNet x 最优传输（核心）", ["O08", "O07"]),
    ("nonacyclic", "Non-Acyclic GFlowNet Theory (the bridge to OT)", "非无环 GFlowNet 理论（通向 OT 的桥）", ["T19", "T36"]),
    ("foundations", "GFlowNet Foundations and Training Objectives", "GFlowNet 基础与训练目标", ["T00", "T02", "T03", "T05", "T10"]),
    ("ot-prereq", "Optimal Transport Prerequisites and Graph OT", "最优传输先修与图上 OT", ["O01", "O02"]),
    ("neural-ot", "Neural / Amortized OT and Schrodinger Bridges", "神经·摊销 OT 与 Schrödinger 桥", ["O03", "O04", "O05", "O06"]),
    ("competitors", "Competing and Adjacent Works: OT and SB on Graphs", "竞品与相邻工作：图上 OT 与 SB", ["C01", "C02", "C03"]),
]

VENUE_BADGE = {"main": "", "journal": "", "workshop": " (Workshop)", "preprint": " (preprint)", "lecture-notes": " (lecture notes)"}


def load_meta():
    metas = {}
    for p in sorted(META.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        metas[d["id"]] = d
    return metas


def load_candidates():
    rows = []
    for name in ("candidates_gfn.csv", "candidates_ot.csv"):
        p = ROOT / "data" / name
        if not p.exists():
            continue
        with p.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    r["_rel"] = int(str(r.get("relevance(1-5)", r.get("relevance", "0"))).strip() or 0)
                except ValueError:
                    r["_rel"] = 0
                rows.append(r)
    rows.sort(key=lambda r: (-r["_rel"], -int(r.get("year") or 0)))
    return rows


def report_link(pid, arxiv):
    cands = list(REPORTS.glob(f"{pid}_*.md"))
    return f"reports/{cands[0].name}" if cands else None


def entry(i, m, lang):
    venue = m.get("venue", "")
    vt = m.get("venue_type", "")
    badge = VENUE_BADGE.get(vt, "")
    if vt in ("workshop", "preprint", "lecture-notes"):
        venue_str = f"{venue}"
    else:
        venue_str = venue
    arxiv = m.get("arxiv")
    links = []
    if arxiv:
        links.append(f"[paper](https://arxiv.org/abs/{arxiv})")
    if m.get("code_url"):
        links.append(f"[code]({m['code_url']})")
    rl = report_link(m["id"], arxiv)
    if rl:
        links.append(f"[{'report' if lang == 'en' else '解读'}]({rl})")
    if arxiv and (PAPERS_ZH / f"{arxiv}.zh.pdf").exists():
        links.append(f"[{'zh-PDF' if lang == 'en' else '中译PDF'}](papers_zh/{arxiv}.zh.pdf)")
    if arxiv and (PAPERS / f"{arxiv}.pdf").exists():
        links.append(f"[{'PDF' if lang == 'en' else '原文PDF'}](papers/{arxiv}.pdf)")
    one = m.get("one_line_en" if lang == "en" else "one_line_zh", "")
    authors = ", ".join(m.get("authors", []))
    prio = m.get("priority", "")
    lines = [f"{i}. **{m['title']}.** {venue_str}, {m.get('year', '')}. {' '.join(links)}", ""]
    lines.append(f"    *{authors}*" + (f" · `{prio}`" if prio else ""))
    if one:
        lines += ["", f"    > {one}"]
    lines.append("")
    return "\n".join(lines)


def build(lang):
    metas = load_meta()
    header = (ROOT / "data" / ("header.md" if lang == "en" else "header_zh.md")).read_text(encoding="utf-8")
    out = [header.rstrip(), ""]

    # TOC
    out.append("## Content" if lang == "en" else "## 目录")
    out.append("")
    toc_titles = []
    for k, te, tz, ids in SECTIONS:
        t = te if lang == "en" else tz
        toc_titles.append((k, t))
    extra = [("trends-2026", "2026 Additions and Trends" if lang == "en" else "2026 增补与趋势"),
             ("deep-dive-reports", "Deep-Dive Reports and Translated PDFs" if lang == "en" else "深度解读与中译 PDF"),
             ("insights", "Insights and Open Problems" if lang == "en" else "Insight 与开放问题"),
             ("deliverables", "Reports and Slides" if lang == "en" else "汇总报告与幻灯"),
             ("contributing", "Contributing and Citation" if lang == "en" else "贡献与引用")]
    for n, (k, t) in enumerate(toc_titles + extra, 1):
        out.append(f"{n}. [{t}](#{k})")
    out.append("")

    for k, te, tz, ids in SECTIONS:
        out.append(f'<a id="{k}"></a>')
        out.append(f"### {te if lang == 'en' else tz}")
        out.append("")
        n = 0
        for pid in ids:
            if pid in metas:
                n += 1
                out.append(entry(n, metas[pid], lang))
        out.append("")

    # trends
    out.append('<a id="trends-2026"></a>')
    out.append("### " + ("2026 Additions and Trends" if lang == "en" else "2026 增补与趋势"))
    out.append("")
    cands = load_candidates()
    if cands:
        intro = ("Papers found by the 2026 trend scan (relevance >= 4). Full analysis: "
                 "[reports/TRENDS_GFN_2026.md](reports/TRENDS_GFN_2026.md), [reports/TRENDS_OT_2026.md](reports/TRENDS_OT_2026.md)."
                 if lang == "en" else
                 "2026 趋势扫描收录（relevance >= 4）。完整分析见 "
                 "[reports/TRENDS_GFN_2026.md](reports/TRENDS_GFN_2026.md)、[reports/TRENDS_OT_2026.md](reports/TRENDS_OT_2026.md)。")
        out += [intro, ""]
        n = 0
        for r in cands:
            if r["_rel"] < 4:
                continue
            n += 1
            arxiv = (r.get("arxiv") or "").strip()
            links = [f"[paper](https://arxiv.org/abs/{arxiv})"] if arxiv else []
            if (r.get("code_url") or "").strip():
                links.append(f"[code]({r['code_url'].strip()})")
            if arxiv and (PAPERS / f"{arxiv}.pdf").exists():
                links.append(f"[PDF](papers/{arxiv}.pdf)")
            venue = (r.get("venue") or "").strip()
            out.append(f"{n}. **{r['title'].strip().rstrip('.')}.** {venue}, {r.get('year', '').strip()}. {' '.join(links)}")
            out.append("")
            out.append(f"    *{(r.get('authors') or '').strip()}*")
            one = (r.get("one_line_en") or "").strip()
            if one:
                out += ["", f"    > {one}"]
            out.append("")
    else:
        out.append("_(trend scan pending)_" if lang == "en" else "_（趋势扫描进行中）_")
    out.append("")

    # reports table
    out.append('<a id="deep-dive-reports"></a>')
    out.append("### " + ("Deep-Dive Reports and Translated PDFs" if lang == "en" else "深度解读与中译 PDF"))
    out.append("")
    out.append("| ID | Paper | Report | Original PDF | Chinese PDF (SuperTranslate) | QA |" if lang == "en"
               else "| 编号 | 论文 | 解读报告 | 原文 PDF | 中译 PDF（SuperTranslate） | QA |")
    out.append("|---|---|---|---|---|---|")
    for k, te, tz, ids in SECTIONS:
        for pid in ids:
            m = metas.get(pid)
            if not m:
                continue
            arxiv = m.get("arxiv", "")
            rl = report_link(pid, arxiv)
            rep = f"[{Path(rl).name}]({rl})" if rl else "-"
            pdf = f"[{arxiv}.pdf](papers/{arxiv}.pdf)" if (PAPERS / f"{arxiv}.pdf").exists() else "-"
            zh = f"[{arxiv}.zh.pdf](papers_zh/{arxiv}.zh.pdf)" if (PAPERS_ZH / f"{arxiv}.zh.pdf").exists() else "-"
            qa = "-"
            qj = PAPERS_ZH / f"{arxiv}.inspect.json"
            if qj.exists():
                try:
                    qa = f"{json.loads(qj.read_text())['issue_count']} issues"
                except Exception:
                    qa = "?"
            out.append(f"| {pid} | {m['title'][:70]} | {rep} | {pdf} | {zh} | {qa} |")
    out.append("")

    # tail sections from static files
    tail = ROOT / "data" / ("tail.md" if lang == "en" else "tail_zh.md")
    if tail.exists():
        out.append(tail.read_text(encoding="utf-8").rstrip())
        out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    (ROOT / "README.md").write_text(build("en"), encoding="utf-8")
    (ROOT / "README_zh.md").write_text(build("zh"), encoding="utf-8")
    print("wrote README.md, README_zh.md")
