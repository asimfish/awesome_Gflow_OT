#!/usr/bin/env bash
# Translate all paper PDFs to Chinese with SuperTranslate (layout-preserving), priority order.
# Usage: bash scripts/translate_batch.sh            # run all pending
#        bash scripts/translate_batch.sh 2606.06272 # run one
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ST_HOME="${SUPER_TRANSLATE_HOME:-$HOME/Code/awesome_Gflow_OT_build/tools/super_translate}"
PY="$ST_HOME/.venv/bin/python"
OUT="$ROOT/papers_zh"; LOG="$ROOT/logs"; mkdir -p "$OUT" "$LOG"

# key never echoed; read from local credential store into env only
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  export DEEPSEEK_API_KEY="$(python3 -c "import yaml,sys;print(yaml.safe_load(open('$HOME/.dsh/.credentials.yaml'))['refs']['DEEPSEEK_API_KEY'])")"
fi
[ -n "$DEEPSEEK_API_KEY" ] || { echo "no DEEPSEEK_API_KEY"; exit 2; }

ORDER=(2606.06272 2603.01786 2502.07735 2312.15246 2106.04399 2201.13259 2111.09266 \
       2506.12025 2602.04675 2305.07170 2209.12782 1704.08200 2212.00133 2410.01500 \
       2209.14440 2305.11857 2409.09347 2505.06589)
[ $# -gt 0 ] && ORDER=("$@")

cd "$ST_HOME"
for id in "${ORDER[@]}"; do
  src="$ROOT/papers/$id.pdf"; dst="$OUT/$id.zh.pdf"
  [ -f "$src" ] || { echo "[$(date +%H:%M:%S)] MISSING $src"; continue; }
  if [ -f "$dst" ] && [ -f "$OUT/$id.inspect.json" ]; then echo "[$(date +%H:%M:%S)] DONE-ALREADY $id"; continue; fi
  echo "[$(date +%H:%M:%S)] START $id"
  t0=$(date +%s)
  "$PY" -m pdf_zh_translator translate "$src" "$dst" \
      --api-mode deepseek --api-key-env DEEPSEEK_API_KEY \
      --preserve-graphics-text --skip-overflow --quiet \
      --cache-file "$OUT/$id.translation-cache.jsonl" > "$LOG/$id.translate.log" 2>&1
  rc=$?
  echo "[$(date +%H:%M:%S)] translate rc=$rc ($(( $(date +%s)-t0 ))s) $id"
  if [ $rc -eq 0 ] && [ -f "$dst" ]; then
    "$PY" -m pdf_zh_translator inspect "$src" "$dst" --json-out "$OUT/$id.inspect.json" > "$LOG/$id.inspect.log" 2>&1
    echo "[$(date +%H:%M:%S)] inspect rc=$? issues=$(python3 -c "import json;print(json.load(open('$OUT/$id.inspect.json'))['issue_count'])" 2>/dev/null) $id"
  fi
done
echo "[$(date +%H:%M:%S)] BATCH FINISHED"
