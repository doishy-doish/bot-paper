#!/usr/bin/env bash
# incremental.sh — ramp VUS level by level (like N people browsing), stop at first fail.
# Same user model as the official bench (upstream bench/load.js: feed -> think ->
# post -> think -> 15% like, 2% post). Same pass bars: p95 < 500ms, p99 < 1000ms,
# failed requests < 1%. Short holds: relative signal only, NEVER the official score
# (that needs the dedicated 1vCPU box + 16-core rig + 5-min holds).
#
#   bash twelve/sim/incremental.sh <base_url> <tokens.json> <load.js> <outdir> [levels]
#   env: DURATION (default 30s), RAMP_UP (default 10s)
#
# Writes: <outdir>/level-<n>.json (k6 summary), <outdir>/result.json + TABLE.md
set -uo pipefail
BASE_URL="${1:?usage: incremental.sh <base_url> <tokens.json> <load.js> <outdir> [levels]}"
TOKENS="$2"; LOADJS="$3"; OUT="$4"
LEVELS="${5:-${LEVELS:-10 25 50 100}}"
DURATION="${DURATION:-30s}"; RAMP_UP="${RAMP_UP:-10s}"
mkdir -p "$OUT"
# warmup: fault in page cache / pools so the ramp measures steady state, not cold start
WARMUP="${WARMUP:-30}"
for _ in $(seq "$WARMUP"); do curl -sf "$BASE_URL/feed" >/dev/null 2>&1 || true; done

pass=0; fail_at=""
echo "| users | req/s | p95 ms | p99 ms | err % | verdict |" | tee "$OUT/TABLE.md"
echo "|---|---|---|---|---|---|" | tee -a "$OUT/TABLE.md"
for n in $LEVELS; do
  BASE_URL="$BASE_URL" VUS="$n" DURATION="$DURATION" RAMP_UP="$RAMP_UP" TOKENS="$TOKENS" \
    k6 run --summary-export "$OUT/level-$n.json" "$LOADJS" > "$OUT/level-$n.log" 2>&1 || true
  row="$(python3 - "$OUT/level-$n.json" "$n" <<'EOF'
import json,sys
p, n = sys.argv[1], sys.argv[2]
def flat(d):
  # k6 >=0.50 summary-export is flat; older nests under .values — accept both
  if isinstance(d, dict) and 'values' in d: return d['values']
  return d if isinstance(d, dict) else {}
try:
  s = json.load(open(p)); m = s.get('metrics', {})
  d = flat(m.get('http_req_duration', {}))
  f = flat(m.get('http_req_failed', {}))
  r = flat(m.get('http_reqs', {}))
  p95, p99 = d.get('p(95)', float('nan')), d.get('p(99)', float('nan'))
  err = 100.0 * f.get('rate', f.get('value', 1.0)) # k6>=0.50 flat rate metrics use `value`, not `rate`
  rps = r.get('rate', float('nan'))
  ok = p95 < 500 and p99 < 1000 and err < 1.0
  print(f"| {n} | {rps:.0f} | {p95:.0f} | {p99:.0f} | {err:.2f} | {'PASS' if ok else 'FAIL'} |")
  print(f"VERDICT={'PASS' if ok else 'FAIL'}", file=sys.stderr)
except Exception as e:
  print(f"| {n} | - | - | - | - | ERROR ({e}) |")
  print("VERDICT=ERROR", file=sys.stderr)
EOF
)"
  verdict="$(echo "$row" | grep '^VERDICT=' | cut -d= -f2)"
  echo "$row" | grep -v '^VERDICT=' | tee -a "$OUT/TABLE.md"
  if [ "$verdict" = "PASS" ]; then pass="$n"; else fail_at="$n"; break; fi
done
python3 - "$OUT" "$pass" "$fail_at" "$LEVELS" <<'EOF'
import json,sys
out, passed, failed, levels = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
json.dump({"levels": levels.split(), "max_pass": passed, "failed_at": failed}, open(out + '/result.json', 'w'))
EOF
echo "max_pass=$pass failed_at=${fail_at:-none}"
