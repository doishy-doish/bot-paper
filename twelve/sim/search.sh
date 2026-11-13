#!/usr/bin/env bash
# search.sh — official-challenge-style max-hold search, short-hold CI edition.
# Same user model (upstream bench/load.js) + same pass bars (p95<500ms, p99<1000ms, err<1%).
# Official: 2-min doubling probes -> bisect ~5% -> 5-min hold, step down 5% until pass.
# CI edition (relative signal only, never the official score): PROBE/RAMP/HOLD durations,
# START/CAP bounds. One fresh DB per server for the whole search (official restarts per
# probe; noted deviation, keeps CI time sane).
#
#   bash twelve/sim/search.sh <base_url> <tokens.json> <load.js> <outdir>
#   env: PROBE (30s), RAMP_UP (10s), HOLD (60s), START (40), CAP (2560), WARMUP (30)
#
# Writes: <outdir>/probe-<n>.json/.log, hold-<n>.json/.log, result.json + TABLE.md
set -uo pipefail
BASE_URL="${1:?usage: search.sh <base_url> <tokens.json> <load.js> <outdir>}"
TOKENS="$2"; LOADJS="$3"; OUT="$4"
PROBE="${PROBE:-30s}"; RAMP_UP="${RAMP_UP:-10s}"; HOLD="${HOLD:-60s}"
START="${START:-40}"; CAP="${CAP:-2560}"
WARMUP="${WARMUP:-30}"
mkdir -p "$OUT"
for _ in $(seq "$WARMUP"); do curl -sf "$BASE_URL/feed" >/dev/null 2>&1 || true; done

run_k6() { # <vus> <duration> <ramp> <tag> -> "VERDICT=.. RPS=.. P95=.. P99=.. ERR=.."
  local n="$1" dur="$2" ramp="$3" tag="$4"
  BASE_URL="$BASE_URL" VUS="$n" DURATION="$dur" RAMP_UP="$ramp" TOKENS="$TOKENS" \
    k6 run --summary-export "$OUT/$tag.json" "$LOADJS" > "$OUT/$tag.log" 2>&1 || true
  python3 - "$OUT/$tag.json" <<'EOF'
import json,sys
def flat(d):
  if isinstance(d, dict) and 'values' in d: return d['values']
  return d if isinstance(d, dict) else {}
try:
  s=json.load(open(sys.argv[1])); m=s.get('metrics',{})
  d=flat(m.get('http_req_duration',{})); f=flat(m.get('http_req_failed',{})); r=flat(m.get('http_reqs',{}))
  p95=d.get('p(95)',float('nan')); p99=d.get('p(99)',float('nan'))
  err=100.0*f.get('rate',f.get('value',1.0)); rps=r.get('rate',float('nan'))
  ok=(p95<500 and p99<1000 and err<1.0)
  print(f"VERDICT={'PASS' if ok else 'FAIL'} RPS={rps:.1f} P95={p95:.0f} P99={p99:.0f} ERR={err:.2f}")
except Exception as e:
  print(f"VERDICT=ERROR RPS=nan P95=nan P99=nan ERR=nan ({e})")
EOF
}

echo "| users | req/s | p95 ms | p99 ms | err % | verdict |" | tee "$OUT/TABLE.md"
echo "|---|---|---|---|---|---|" | tee -a "$OUT/TABLE.md"
probed=""
doprobe() { # <vus> <dur> <ramp> <tag> <label> -> sets PROBE_V; appends table row
  local out line v rps p95 p99 err
  out="$(run_k6 "$1" "$2" "$3" "$4")"
  v="$(echo "$out" | grep -o 'VERDICT=[A-Z]*' | cut -d= -f2)"
  rps="$(echo "$out" | grep -o 'RPS=[^ ]*' | cut -d= -f2)"
  p95="$(echo "$out" | grep -o 'P95=[^ ]*' | cut -d= -f2)"
  p99="$(echo "$out" | grep -o 'P99=[^ ]*' | cut -d= -f2)"
  err="$(echo "$out" | grep -o 'ERR=[^ ]*' | cut -d= -f2)"
  echo "| $5 | $rps | $p95 | $p99 | $err | $v |" | tee -a "$OUT/TABLE.md"
  probed="$probed $1"
  PROBE_V="$v"
}

# doubling probes
lo=0; hi=""
n="$START"
while :; do
  doprobe "$n" "$PROBE" "$RAMP_UP" "probe-$n" "probe $n"
  if [ "$PROBE_V" = "PASS" ]; then
    lo="$n"
    if [ "$n" -ge "$CAP" ]; then hi=""; break; fi
    n=$(( n * 2 ))
  else
    hi="$n"; break
  fi
done

# bisect to ~5% (max 6 steps)
if [ -n "$hi" ] && [ "$lo" -gt 0 ]; then
  tol=$(( lo / 20 )); if [ "$tol" -lt 5 ]; then tol=5; fi
  i=0
  while [ $(( hi - lo )) -gt "$tol" ] && [ "$i" -lt 6 ]; do
    mid=$(( (lo + hi) / 2 ))
    doprobe "$mid" "$PROBE" "$RAMP_UP" "probe-$mid" "bisect $mid"
    if [ "$PROBE_V" = "PASS" ]; then lo="$mid"; else hi="$mid"; fi
    i=$(( i + 1 ))
  done
fi

# the hold: HOLD duration at best probe pass (official: 5 min, step down 5% until pass)
hold="none"
if [ "$lo" -gt 0 ]; then
  doprobe "$lo" "$HOLD" "$RAMP_UP" "hold-$lo" "HOLD $lo"
  hold="$PROBE_V"
fi

if [ "$hold" = "PASS" ]; then max_hold="$lo"; else max_hold="0"; fi
echo "max_hold=$max_hold failed_at=${hi:-none} hold=$hold"
python3 - "$OUT" "$max_hold" "${hi:-none}" "$probed" "$hold" <<'EOF'
import json,sys
out, passed, failed, probed, hold = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
json.dump({"levels": probed.split(), "max_pass": passed, "failed_at": failed, "hold": hold}, open(out + '/result.json', 'w'))
EOF
