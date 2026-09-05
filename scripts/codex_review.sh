#!/usr/bin/env bash
# Ask local Codex CLI (gpt-6-astra, reasoning=max) to review one Markdown file and write
# a structured JSON review to logs/codex_reviews/<name>.json. Non-interactive, read-only sandbox.
# Usage: bash scripts/codex_review.sh <file.md> [focus-hint]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$1"; HINT="${2:-}"
NAME="$(basename "${SRC%.md}")"
OUT_DIR="$ROOT/logs/codex_reviews"; mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/$NAME.json"; RAW="$OUT_DIR/$NAME.raw.txt"
PROMPT_FILE="$OUT_DIR/$NAME.prompt.txt"
{
cat <<'PEOF'
You are a senior reviewer for a research reading-list repository on GFlowNets x Optimal Transport
(awesome_Gflow_OT). Below is one Chinese Markdown document from the repo. Review it as a demanding
ICML-area-chair-level reader who knows GFlowNet theory (flow matching, DB/TB, non-acyclic flows,
minimum-flow / expected-trajectory-length identity) and optimal transport (Kantorovich duality,
min-cost flow / Beckmann on graphs, entropic OT, Schrodinger bridges).

Your job is NOT to rewrite the document. Produce a JSON object (and nothing else, no code fences) with:
{
 "verdict": "one paragraph: what the document gets right, and the single biggest weakness",
 "factual_risks": [ {"quote": "<exact short quote from the doc>", "issue": "<why it may be wrong or unsupported>", "fix": "<concrete correction or what evidence is needed>"} ],
 "logic_gaps": [ {"where": "<section/quote>", "gap": "<missing step or unjustified leap>", "fix": "<how to close it>"} ],
 "clarity_edits": [ {"quote": "<exact short quote>", "replacement": "<improved Chinese wording, same meaning, no added claims>"} ],
 "missing_content": [ "<concrete thing a strong reader would expect that is absent>" ],
 "keep_as_is": [ "<things that are unusually good and must not be changed>" ]
}
Rules: quotes must be verbatim substrings of the document so they can be located programmatically;
keep clarity_edits to at most 12 high-impact items; do not invent theorem numbers or papers; if a claim
cites a theorem number you cannot verify from the document itself, list it under factual_risks with
fix = "verify against source"; write all free text in Simplified Chinese; preserve LaTeX as-is.
PEOF
[ -n "$HINT" ] && printf '\nReviewer focus for this document: %s\n' "$HINT"
printf '\n===== DOCUMENT (%s) =====\n' "$NAME"
cat "$SRC"
} > "$PROMPT_FILE"
echo "[$(date +%H:%M:%S)] codex review start: $NAME ($(wc -c < "$SRC") bytes)"
codex exec -m gpt-6-astra -c model_reasoning_effort="max" --skip-git-repo-check -s read-only -C "$ROOT" \
  -o "$RAW" - < "$PROMPT_FILE" > "$OUT_DIR/$NAME.stderr.txt" 2>&1
rc=$?
# extract the JSON object from raw output (codex may wrap it)
python3 - "$RAW" "$OUT" <<'PY'
import json,re,sys
raw=open(sys.argv[1],encoding='utf-8',errors='replace').read()
m=re.search(r'\{.*\}',raw,flags=re.S)
try:
    obj=json.loads(m.group(0)) if m else None
except Exception as e:
    obj=None
if obj is None:
    print("JSON parse failed"); sys.exit(1)
json.dump(obj,open(sys.argv[2],'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print(f"ok: {len(obj.get('factual_risks',[]))} risks, {len(obj.get('logic_gaps',[]))} gaps, {len(obj.get('clarity_edits',[]))} edits, {len(obj.get('missing_content',[]))} missing")
PY
echo "[$(date +%H:%M:%S)] codex review done rc=$rc -> $OUT"
