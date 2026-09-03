#!/usr/bin/env bash
# One-off: translate O01 (480-page lecture notes) via OpenRouter / Gemini 2.5 Flash, then inspect.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ST_HOME="${SUPER_TRANSLATE_HOME:-$HOME/Code/awesome_Gflow_OT_build/tools/super_translate}"
PY="$ST_HOME/.venv/bin/python"; id=2505.06589
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$(python3 -c "import yaml;print(yaml.safe_load(open('$HOME/.dsh/.credentials.yaml'))['refs']['OPENROUTER_API_KEY'])")}"
cd "$ST_HOME"
echo "[$(date +%H:%M:%S)] START $id"; t0=$(date +%s)
"$PY" -m pdf_zh_translator translate "$ROOT/papers/$id.pdf" "$ROOT/papers_zh/$id.zh.pdf" \
  --api-mode openai-compatible --api-url https://openrouter.ai/api/v1 --api-key-env OPENROUTER_API_KEY \
  --model google/gemini-2.5-flash --batch-size 16 --max-batch-chars 6000 --retries 4 --timeout 180 \
  --preserve-graphics-text --skip-overflow --quiet --cache-file "$ROOT/papers_zh/$id.translation-cache.jsonl"
rc=$?; echo "[$(date +%H:%M:%S)] translate rc=$rc ($(( $(date +%s)-t0 ))s)"
if [ $rc -eq 0 ]; then
  "$PY" -m pdf_zh_translator inspect "$ROOT/papers/$id.pdf" "$ROOT/papers_zh/$id.zh.pdf" --json-out "$ROOT/papers_zh/$id.inspect.json" > "$ROOT/logs/$id.inspect.log" 2>&1
  echo "[$(date +%H:%M:%S)] inspect rc=$? issues=$(python3 -c "import json;print(json.load(open('$ROOT/papers_zh/$id.inspect.json'))['issue_count'])" 2>/dev/null)"
fi
