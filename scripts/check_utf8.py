#!/usr/bin/env python3
"""Fail if any given text file is not valid UTF-8 or looks CJK-corrupted (runs of '?' / stray latin-1 bytes).
Usage: python3 scripts/check_utf8.py FILE [FILE...]   (exit 1 on any failure)
"""
import re
import sys
from pathlib import Path

bad = 0
for arg in sys.argv[1:]:
    p = Path(arg)
    raw = p.read_bytes()
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        print(f"[FAIL] {p}: not utf-8 ({e})"); bad += 1; continue
    runs = len(re.findall(r"\?{3,}", s))
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    flag = "FAIL" if runs > 2 else "ok"
    if flag == "FAIL": bad += 1
    print(f"[{flag}] {p.name}: {len(s)} chars, cjk={cjk}, '???'-runs={runs}")
sys.exit(1 if bad else 0)
