#!/usr/bin/env python3
"""Translate a Chinese Markdown report into English via an OpenAI-compatible chat API (OpenRouter).

Chunks by headings (keeping each chunk under ~3500 chars), translates concurrently, preserves
Markdown structure, LaTeX math, links, code, tables and citation markers verbatim.
Usage: python3 scripts/translate_md.py IN.md OUT.md [--model MODEL] [--workers N] [--limit N]
Key: OPENROUTER_API_KEY env var, or refs/OPENROUTER_API_KEY in ~/.dsh/.credentials.yaml
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

SYSTEM = (
    "You are a meticulous academic translator (Chinese -> English) for a machine-learning survey on "
    "GFlowNets and optimal transport. Translate the user's Markdown fragment into precise, natural academic "
    "English. HARD RULES: (1) Preserve ALL Markdown structure exactly: heading levels, lists, tables (same "
    "number of columns), blockquotes, code fences, links, HTML. (2) Copy LaTeX math verbatim, including "
    "\\( ... \\) and \\[ ... \\] delimiters; never translate or alter anything inside math. (3) Keep paper IDs "
    "(O08, T36, C02, N-GFN-01), arXiv numbers, theorem/equation/table references (Thm. 3.2, Eq. (11), Table 1), "
    "author names, venue names, file paths and URLs unchanged. (4) Keep English terms already in the text. "
    "(5) Do not add, drop, summarize or comment; output only the translated Markdown, no preamble. "
    "(6) Translate the phrase ????? as 'Our assessment:' and ??? as 'Editorial note'."
)


def load_key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if k:
        return k
    import yaml
    return yaml.safe_load(open(Path.home() / ".dsh/.credentials.yaml"))["refs"]["OPENROUTER_API_KEY"]


def chunk(text, limit=3500):
    """Split on headings, then pack consecutive blocks up to limit chars. Never split inside code fences."""
    blocks, cur, in_code = [], [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            in_code = not in_code
        if not in_code and re.match(r"^#{1,6} ", line) and cur:
            blocks.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    packed, buf = [], ""
    for b in blocks:
        if len(b) > limit:  # split long block on blank lines
            paras, pb = [], ""
            for para in b.split("\n\n"):
                if len(pb) + len(para) > limit and pb:
                    paras.append(pb)
                    pb = ""
                pb = (pb + "\n\n" + para) if pb else para
            if pb:
                paras.append(pb)
            sub = paras
        else:
            sub = [b]
        for s in sub:
            if len(buf) + len(s) + 2 > limit and buf:
                packed.append(buf)
                buf = ""
            buf = (buf + "\n\n" + s) if buf else s
    if buf:
        packed.append(buf)
    return packed


def has_cjk(s):
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def call(key, model, text, retries=4):
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
        "temperature": 0.1,
    }).encode()
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "HTTP-Referer": "https://github.com/asimfish/awesome_Gflow_OT", "X-Title": "awesome_Gflow_OT"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            out = d["choices"][0]["message"]["content"].strip()
            out = re.sub(r"^```(?:markdown|md)?\n(.*)\n```$", r"\1", out, flags=re.S)
            return out
        except Exception as exc:
            wait = 5 * (i + 1)
            print(f"  retry {i+1}: {exc} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("translation failed after retries")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--model", default="google/gemini-2.5-flash")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="translate only first N chunks (test)")
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    key = load_key()
    text = Path(a.src).read_text(encoding="utf-8")
    # keep YAML front matter aside
    fm = ""
    m = re.match(r"^---\n.*?\n---\n", text, flags=re.S)
    if m:
        fm, text = m.group(0), text[m.end():]
    chunks = chunk(text)
    if a.limit:
        chunks = chunks[: a.limit]
    cache_path = Path(a.cache or (a.dst + ".cache.json"))
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    print(f"{len(chunks)} chunks, {sum(len(c) for c in chunks)} chars, cached {len(cache)}")

    def work(i):
        c = chunks[i]
        if c in cache:
            return i, cache[c]
        if not has_cjk(c):
            return i, c
        return i, call(key, a.model, c)

    results = [None] * len(chunks)
    with cf.ThreadPoolExecutor(a.workers) as ex:
        for n, (i, out) in enumerate(ex.map(work, range(len(chunks))), 1):
            results[i] = out
            cache[chunks[i]] = out
            if n % 5 == 0 or n == len(chunks):
                cache_path.write_text(json.dumps(cache, ensure_ascii=False))
                print(f"  {n}/{len(chunks)} done", flush=True)
    if fm:
        fm = ('---\ntitle: "GFlowNet x Optimal Transport: From Flow Conservation to Kantorovich Plans"\n'
              'subtitle: "awesome_Gflow_OT Consolidated Report"\nauthor: "awesome_Gflow_OT project"\n'
              'date: "2026-09"\nlang: en\n---\n')
    out = fm + "\n\n".join(results) + "\n"
    Path(a.dst).write_text(out, encoding="utf-8")
    residual = sum(1 for r in results if has_cjk(r))
    print(f"wrote {a.dst}; chunks with residual CJK: {residual}")


if __name__ == "__main__":
    main()
