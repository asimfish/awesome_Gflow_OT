#!/usr/bin/env python3
"""Build a single-file, zero-dependency HTML slide deck from slides/deck.json.

deck.json schema (UTF-8):
{
  "title": str, "subtitle": str, "footer": str,
  "slides": [
    {"type": "title"|"section"|"bullets"|"two-col"|"table"|"quote"|"formula"|"cards"|"timeline",
     "title": str, "kicker": str (optional), "note": str (optional),
     ... type-specific fields ...}
  ]
}
Usage: python3 slides/build_slides.py  -> slides/awesome_gflow_ot_slides.html
"""
import html
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "deck.json").read_text(encoding="utf-8"))

CSS = r"""
:root{--bg:#0f1720;--panel:#16212d;--ink:#e8eef5;--mute:#9fb0c3;--brand:#4cc2ff;--accent:#ffb454;--ok:#5fd68a;--bad:#ff6b6b;--line:#27364a}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%;background:var(--bg);color:var(--ink);font-family:-apple-system,"Helvetica Neue","PingFang SC","Hiragino Sans GB","Microsoft YaHei",sans-serif;overflow:hidden}
.deck{position:relative;width:100vw;height:100vh}
.slide{position:absolute;inset:0;display:none;padding:6vh 7vw;flex-direction:column}
.slide.active{display:flex;animation:fade .35s ease}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.kicker{color:var(--brand);font-size:1.1vw;letter-spacing:.12em;text-transform:uppercase;margin-bottom:1vh}
h1{font-size:4.2vw;line-height:1.15;font-weight:700}
h2{font-size:2.6vw;line-height:1.2;font-weight:700;margin-bottom:3vh}
h3{font-size:1.5vw;color:var(--accent);margin-bottom:1vh}
.sub{font-size:1.6vw;color:var(--mute);margin-top:2vh;max-width:70vw}
.body{flex:1;display:flex;flex-direction:column;justify-content:center;font-size:1.45vw;line-height:1.55}
ul{list-style:none}li{padding-left:1.6em;position:relative;margin:.55em 0}
li::before{content:"";position:absolute;left:.3em;top:.62em;width:.5em;height:.5em;border-radius:50%;background:var(--brand)}
li li::before{background:var(--mute);width:.35em;height:.35em;top:.7em}
b,strong{color:#fff}.hl{color:var(--accent);font-weight:600}.ok{color:var(--ok)}.bad{color:var(--bad)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:3vw;flex:1;align-content:center}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:2vh 1.6vw}
.cards{display:grid;gap:1.6vw;flex:1;align-content:center}
.cards.c3{grid-template-columns:repeat(3,1fr)}.cards.c2{grid-template-columns:repeat(2,1fr)}.cards.c4{grid-template-columns:repeat(4,1fr)}
.card h3{font-size:1.35vw}.card p{font-size:1.2vw;color:var(--ink);line-height:1.5}
table{border-collapse:collapse;width:100%;font-size:1.15vw}
th{color:var(--brand);text-align:left;border-bottom:2px solid var(--line);padding:.5em .6em;font-weight:600}
td{border-bottom:1px solid var(--line);padding:.5em .6em;vertical-align:top}
tr:nth-child(even) td{background:rgba(255,255,255,.02)}
.quote{font-size:2vw;line-height:1.5;border-left:6px solid var(--accent);padding-left:2vw;max-width:76vw}
.quote small{display:block;color:var(--mute);font-size:1.2vw;margin-top:2vh}
.formula{font-size:2.4vw;text-align:center;padding:3vh 0;color:#fff;font-family:"Latin Modern Math","STIX Two Math","Cambria Math",serif}
.note{color:var(--mute);font-size:1.1vw;margin-top:2vh;border-top:1px solid var(--line);padding-top:1.2vh}
.section{justify-content:center;background:radial-gradient(1200px 600px at 20% 30%,rgba(76,194,255,.15),transparent 60%),var(--bg)}
.section h1{font-size:3.6vw}.section .num{color:var(--brand);font-size:6vw;font-weight:800;opacity:.35;line-height:1}
.timeline{display:flex;gap:1.4vw;align-items:center;flex:1}
.tl{flex:1;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:2.4vh 1.3vw;position:relative;min-height:34vh}
.tl .id{color:var(--brand);font-weight:700;font-size:1.5vw}.tl .what{font-size:1.25vw;margin:1.2vh 0;line-height:1.5}.tl .gap{font-size:1.1vw;color:var(--accent);margin-top:1.5vh;border-top:1px dashed var(--line);padding-top:1vh}
.tl:not(:last-child)::after{content:"\2192";position:absolute;right:-.9vw;top:45%;color:var(--brand);font-size:1.6vw}
.foot{position:absolute;left:7vw;right:7vw;bottom:2.5vh;display:flex;justify-content:space-between;color:var(--mute);font-size:1vw}
.prog{position:absolute;left:0;bottom:0;height:4px;background:var(--brand);transition:width .3s}
.help{position:absolute;right:2vw;top:2vh;color:var(--mute);font-size:.9vw;opacity:.7}
@media print{html,body{overflow:visible;background:#fff;color:#111}.slide{display:flex!important;position:relative;height:100vh;page-break-after:always;padding:5vh 6vw}
 :root{--bg:#fff;--panel:#f3f6fa;--ink:#111;--mute:#555;--brand:#0b5fa5;--accent:#b35c00;--line:#d0d7de}.foot,.help,.prog{display:none}}
"""

JS = r"""
const S=[...document.querySelectorAll('.slide')];let i=Math.max(0,Math.min(S.length-1,parseInt(location.hash.slice(1)||'0')));
const cnt=document.getElementById('cnt'),prog=document.getElementById('prog');
function show(n){S[i].classList.remove('active');i=(n+S.length)%S.length;S[i].classList.add('active');cnt.textContent=(i+1)+' / '+S.length;prog.style.width=((i+1)/S.length*100)+'%';location.hash=i;}
show(i);
addEventListener('keydown',e=>{if(['ArrowRight','ArrowDown','PageDown',' ','Enter'].includes(e.key)){e.preventDefault();show(i+1)}
 else if(['ArrowLeft','ArrowUp','PageUp','Backspace'].includes(e.key)){e.preventDefault();show(i-1)}
 else if(e.key==='Home')show(0);else if(e.key==='End')show(S.length-1);else if(e.key==='f'){document.documentElement.requestFullscreen&&document.documentElement.requestFullscreen()}});
let wl=0;addEventListener('wheel',e=>{const t=Date.now();if(t-wl<450)return;wl=t;show(i+(e.deltaY>0?1:-1))},{passive:true});
let tx=null;addEventListener('touchstart',e=>tx=e.touches[0].clientX,{passive:true});
addEventListener('touchend',e=>{if(tx===null)return;const dx=e.changedTouches[0].clientX-tx;if(Math.abs(dx)>40)show(i+(dx<0?1:-1));tx=null});
addEventListener('click',e=>{if(e.target.closest('a'))return;show(e.clientX>innerWidth*0.7?i+1:(e.clientX<innerWidth*0.3?i-1:i))});
"""


def md(s):
    """Tiny inline markdown: **bold**, `code`, [hl]...[/hl], [ok]...[/ok], [bad]...[/bad]. Escapes HTML first."""
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    for tag in ("hl", "ok", "bad"):
        s = s.replace(f"[{tag}]", f'<span class="{tag}">').replace(f"[/{tag}]", "</span>")
    return s


def ul(items):
    out = ["<ul>"]
    for it in items:
        if isinstance(it, list):
            out.append(ul(it))
        else:
            out.append(f"<li>{md(it)}</li>")
    out.append("</ul>")
    return "".join(out)


def render(s, idx):
    t = s["type"]
    kicker = f'<div class="kicker">{md(s["kicker"])}</div>' if s.get("kicker") else ""
    note = f'<div class="note">{md(s["note"])}</div>' if s.get("note") else ""
    if t == "title":
        return (f'<section class="slide section">{kicker}<h1>{md(s["title"])}</h1>'
                f'<p class="sub">{md(s.get("subtitle",""))}</p>{note}</section>')
    if t == "section":
        return (f'<section class="slide section"><div class="num">{s.get("num","")}</div>'
                f'<h1>{md(s["title"])}</h1><p class="sub">{md(s.get("subtitle",""))}</p></section>')
    head = f'{kicker}<h2>{md(s["title"])}</h2>'
    if t == "bullets":
        body = ul(s["items"])
    elif t == "two-col":
        cols = "".join(f'<div class="card"><h3>{md(c["head"])}</h3>{ul(c["items"])}</div>' for c in s["cols"])
        body = f'<div class="cols">{cols}</div>'
    elif t == "cards":
        n = len(s["cards"])
        cls = "c4" if n >= 4 else ("c3" if n == 3 else "c2")
        cards = "".join(f'<div class="card"><h3>{md(c["head"])}</h3><p>{md(c["text"])}</p></div>' for c in s["cards"])
        body = f'<div class="cards {cls}">{cards}</div>'
    elif t == "table":
        th = "".join(f"<th>{md(h)}</th>" for h in s["header"])
        rows = "".join("<tr>" + "".join(f"<td>{md(c)}</td>" for c in r) + "</tr>" for r in s["rows"])
        body = f"<table><thead><tr>{th}</tr></thead><tbody>{rows}</tbody></table>"
    elif t == "quote":
        body = f'<div class="quote">{md(s["text"])}<small>{md(s.get("source",""))}</small></div>'
    elif t == "formula":
        body = f'<div class="formula">{s["latex"]}</div>{ul(s.get("items", []))}'
    elif t == "timeline":
        steps = "".join(f'<div class="tl"><div class="id">{md(x["id"])}</div><div class="what">{md(x["what"])}</div>'
                        f'<div class="gap">{md(x.get("gap",""))}</div></div>' for x in s["steps"])
        body = f'<div class="timeline">{steps}</div>'
    else:
        raise ValueError(t)
    return f'<section class="slide">{head}<div class="body">{body}</div>{note}</section>'


def build():
    slides = "\n".join(render(s, i) for i, s in enumerate(SPEC["slides"]))
    mathjax = ('<script>window.MathJax={tex:{inlineMath:[["\\\\(","\\\\)"]],displayMath:[["\\\\[","\\\\]"]]},'
               'options:{skipHtmlTags:["script","noscript","style","textarea","pre","code"]}};</script>'
               '<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>')
    doc = f"""<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(SPEC["title"])}</title>
{mathjax}<style>{CSS}</style></head><body><div class="deck">
{slides}
<div class="help">&larr; &rarr; / wheel / tap &middot; f = fullscreen</div>
<div class="foot"><span>{md(SPEC["footer"])}</span><span id="cnt"></span></div><div class="prog" id="prog"></div>
</div><script>{JS}</script></body></html>"""
    out = HERE / "awesome_gflow_ot_slides.html"
    out.write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({len(SPEC['slides'])} slides)")


if __name__ == "__main__":
    build()
