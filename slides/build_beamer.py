#!/usr/bin/env python3
"""Render slides/deck.json to a Beamer .tex (metropolis, 16:9, 10pt, no overlays) and compile with xelatex.

Same content source as the HTML deck so both stay in sync.
Usage: python3 slides/build_beamer.py [--no-compile]
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))

_PREAMBLE_T = r"""\documentclass[aspectratio=169,10pt]{beamer}
\usetheme{metropolis}
\metroset{numbering=fraction,progressbar=frametitle,block=fill}
\setbeamertemplate{navigation symbols}{}
\usepackage{amsmath,amssymb,mathtools,booktabs,array,tabularx,ragged2e}
\usepackage{xeCJK}
\setCJKmainfont[BoldFont=Hiragino Sans GB W6]{Hiragino Sans GB W3}
\setCJKsansfont[BoldFont=Hiragino Sans GB W6]{Hiragino Sans GB W3}
\setmainfont{Helvetica Neue}
\setsansfont{Helvetica Neue}
\usepackage{newunicodechar}
\newfontfamily\symfont{STIX Two Math}
%%SYMBOLS%%
\definecolor{positive}{HTML}{0173B2}
\definecolor{negative}{HTML}{DE8F05}
\definecolor{emphasis}{HTML}{029E73}
\definecolor{neutral}{gray}{0.55}
\newcommand{\pos}[1]{\textcolor{positive}{#1}}
\newcommand{\con}[1]{\textcolor{negative}{#1}}
\newcommand{\HL}[1]{\textcolor{emphasis}{#1}}
\setbeamerfont{frametitle}{size=\large}
\setbeamerfont{title}{size=\LARGE}
\setlength{\leftmargini}{1.2em}
\newcolumntype{Y}{>{\RaggedRight\arraybackslash}X}
"""
_SYMS = "\u2192\u21d2\u21d4\u2194\u2261\u2264\u2265\u2260\u2208\u221e\u00d7\u00b7\u2460\u2461\u2462\u2463"
_SYMLINES = "".join("\\newunicodechar{%s}{{\\symfont %s}}" % (c, c) for c in _SYMS)
PREAMBLE = _PREAMBLE_T.replace("%%SYMBOLS%%", _SYMLINES)


def tex(s):
    """Inline markup -> LaTeX. Math \\( \\) is preserved (converted to $ $). Escapes the rest."""
    parts = re.split(r"(\\\(.*?\\\))", s)
    out = []
    for p in parts:
        if p.startswith("\\(") and p.endswith("\\)"):
            out.append("$" + p[2:-2] + "$")
            continue
        p = p.replace("\\", "\\textbackslash{}")
        for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"), ("_", "\\_"), ("{", "\\{"), ("}", "\\}"), ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
            p = p.replace(a, b)
        p = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", p)
        p = re.sub(r"`(.+?)`", r"\\texttt{\1}", p)
        p = p.replace("[hl]", "\\HL{").replace("[/hl]", "}").replace("[ok]", "\\pos{").replace("[/ok]", "}").replace("[bad]", "\\con{").replace("[/bad]", "}")
        out.append(p)
    return "".join(out)


def items(lst, small=False):
    o = ["\\begin{itemize}" + ("\\small" if small else "")]
    for it in lst:
        if isinstance(it, list):
            o.append(items(it, small))
        else:
            o.append("\\item " + tex(it))
    o.append("\\end{itemize}")
    return "\n".join(o)


def note(s):
    return ("\n\\vfill\\hrule\\vspace{2pt}{\\footnotesize\\textcolor{neutral}{" + tex(s["note"]) + "}}") if s.get("note") else ""


def frame(s):
    t = s["type"]
    if t == "title":
        return ""  # handled by \maketitle
    if t == "section":
        return f"\\section{{{tex(s['title'])}}}\n"
    title = tex(s["title"])
    if t == "bullets":
        body = items(s["items"], small=len(s["items"]) > 4)
    elif t == "two-col":
        dense = sum(len(i) for c in s["cols"] for i in c["items"])
        sz = "\\scriptsize" if dense > 700 else "\\footnotesize"
        cols = []
        for c in s["cols"]:
            cols.append(f"\\begin{{column}}{{0.48\\textwidth}}\\textbf{{\\pos{{{tex(c['head'])}}}}}\\vspace{{2pt}}\n{{{sz}{items(c['items'])}}}\\end{{column}}")
        body = "\\begin{columns}[T]\n" + "\n\\hfill\n".join(cols) + "\n\\end{columns}"
    elif t == "cards":
        n = len(s["cards"])
        w = {2: 0.48, 3: 0.31, 4: 0.235}.get(n, 0.31)
        cols = [f"\\begin{{column}}{{{w}\\textwidth}}\\begin{{block}}{{{tex(c['head'])}}}\\footnotesize {tex(c['text'])}\\end{{block}}\\end{{column}}" for c in s["cards"]]
        body = "\\begin{columns}[T]\n" + "\n".join(cols) + "\n\\end{columns}"
    elif t == "table":
        ncol = len(s["header"])
        # column widths proportional to average cell length, clamped; last column absorbs the rest
        lens = [max(1, sum(len(r[j]) for r in s["rows"]) / len(s["rows"])) for j in range(ncol)]
        tot = sum(lens)
        fracs = [min(0.55, max(0.09, l / tot)) for l in lens]
        fracs = [f / sum(fracs) for f in fracs]
        spec = "".join(f">{{\\RaggedRight\\arraybackslash}}p{{{f*0.97:.3f}\\textwidth}}" for f in fracs)
        head = " & ".join(f"\\textbf{{{tex(h)}}}" for h in s["header"])
        rows = " \\\\\\addlinespace[2pt]\n".join(" & ".join(tex(c) for c in r) for r in s["rows"])
        dense = sum(len(c) for r in s["rows"] for c in r)
        size = "\\tiny" if dense > 1400 else ("\\scriptsize" if dense > 700 or len(s["rows"]) > 6 else "\\footnotesize")
        body = f"{{{size}\\setlength{{\\tabcolsep}}{{3pt}}\\begin{{tabular}}{{@{{}}{spec}@{{}}}}\\toprule\n{head} \\\\\\midrule\n{rows} \\\\\\bottomrule\\end{{tabular}}}}"
    elif t == "quote":
        body = f"\\begin{{quote}}\\large {tex(s['text'])}\\end{{quote}}\\vspace{{6pt}}{{\\small\\textcolor{{neutral}}{{{tex(s.get('source',''))}}}}}"
    elif t == "formula":
        latex = s["latex"].strip()
        if latex.startswith("\\["):
            latex = latex[2:-2]
        body = f"\\begin{{equation*}}{latex}\\end{{equation*}}\n" + items(s.get("items", []), small=True)
    elif t == "timeline":
        n = len(s["steps"])
        w = round(0.96 / n - 0.01, 3)
        cols = [f"\\begin{{column}}{{{w}\\textwidth}}\\begin{{block}}{{{tex(x['id'])}}}\\footnotesize {tex(x['what'])}\\\\[3pt]\\textcolor{{negative}}{{{tex(x.get('gap',''))}}}\\end{{block}}\\end{{column}}" for x in s["steps"]]
        body = "\\begin{columns}[T]\n" + "\n".join(cols) + "\n\\end{columns}"
    else:
        raise ValueError(t)
    kicker = f"\\framesubtitle{{{tex(s['kicker'])}}}" if s.get("kicker") else ""
    return f"\\begin{{frame}}{{{title}}}{kicker}\n{body}{note(s)}\n\\end{{frame}}\n"


REFS = (HERE / "beamer_refs.tex").read_text(encoding="utf-8")
BACKUP = (HERE / "beamer_backup.tex").read_text(encoding="utf-8")


def build(compile_pdf=True):
    title = SPEC["slides"][0]
    frames = "\n".join(frame(s) for s in SPEC["slides"][1:])
    doc = (PREAMBLE + f"\\title{{{tex(title['title'])}}}\n\\subtitle{{{tex(title.get('subtitle',''))}}}\n"
           f"\\author{{awesome\\_Gflow\\_OT ???}}\n\\institute{{github.com/asimfish/awesome\\_Gflow\\_OT}}\n\\date{{2026-09}}\n"
           "\\begin{document}\n\\maketitle\n" + frames + REFS + "\\begin{frame}[standout]?? · ?????\\end{frame}\n" + BACKUP + "\\end{document}\n")
    out = HERE / "awesome_gflow_ot_slides.tex"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out}")
    if compile_pdf:
        for _ in range(2):
            r = subprocess.run(["xelatex", "-interaction=nonstopmode", out.name], cwd=HERE, capture_output=True, text=True)
        log = (HERE / "awesome_gflow_ot_slides.log").read_text(encoding="utf-8", errors="ignore")
        errs = [l for l in log.split("\n") if l.startswith("!")]
        pages = re.findall(r"Output written on .*\((\d+) pages", log)
        print(f"pages={pages[-1] if pages else '?'} errors={len(errs)}")
        for e in errs[:8]:
            print("  ", e)
        over = len(re.findall(r"Overfull \\vbox", log))
        print(f"overfull vboxes: {over}")


if __name__ == "__main__":
    build(compile_pdf="--no-compile" not in sys.argv)
