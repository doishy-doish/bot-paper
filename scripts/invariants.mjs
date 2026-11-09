// invariants.mjs — L2 public runner (K-free, scrambled outputs).
// Frozen L1 dimensionless harvest, identical code every leg, no per-market tuning.
// Usage: node scripts/invariants.mjs --symbol SPY --out /tmp/inv_SPY.json [--p1 1420070400]
// Yahoo chart API (query1) — works on ubuntu-latest runners. Fails LOUDLY (exit 1) on bad data.
import { writeFileSync } from "fs";

const args = Object.fromEntries(
  process.argv.slice(2).flatMap((a, i, arr) =>
    a.startsWith("--") ? [[a.slice(2), arr[i + 1] && !arr[i + 1].startsWith("--") ? arr[i + 1] : "1"]] : []
  )
);
const SYMBOL = args.symbol || process.env.SYMBOL;
const OUT = args.out || process.env.OUT_FILE;
const P1 = parseInt(args.p1 || "1420070400", 10); // 2015-01-01
const P2 = Math.floor(Date.now() / 1000);
// Frozen params (L1): K=3 W=20 ZT=-2.0 VT=1.5 TM=1.0 SM=3.0 HOLD=10 FR=0.001
const K = 3, W = 20, ZT = -2.0, VT = 1.5, TM = 1.0, SM = 3.0, HOLD = 10, FR = 0.001;

if (!SYMBOL || !OUT) {
  console.error("usage: node scripts/invariants.mjs --symbol SYM --out FILE [--p1 epoch]");
  process.exit(1);
}

function median(a) {
  if (!a.length) return NaN;
  const s = a.slice().sort((x, y) => x - y);
  const m = Math.floor(s.length / 2);
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}
function stdev(a) {
  if (a.length < 2) return NaN;
  const m = a.reduce((s, x) => s + x, 0) / a.length;
  let v = 0;
  for (const x of a) v += (x - m) * (x - m);
  return Math.sqrt(v / (a.length - 1));
}

function parseChart(j) {
  const r = j?.chart?.result?.[0];
  if (!r) throw new Error("no chart result for " + SYMBOL);
  if (r.meta?.symbol && r.meta.symbol !== SYMBOL && !SYMBOL.startsWith("^") && r.meta.symbol !== SYMBOL)
    console.log(`meta symbol ${r.meta.symbol} (asked ${SYMBOL}) — continuing`);
  const ts = r.timestamp || [];
  const q = r.indicators?.quote?.[0] || {};
  const adj = (r.indicators?.adjclose?.[0]?.adjclose) || [];
  const O = q.open || [], H = q.high || [], L = q.low || [], C = q.close || [], V = q.volume || [];
  const bars = [];
  for (let i = 0; i < ts.length; i++) {
    const c = C[i];
    if (c == null) continue;
    const a = adj[i] != null ? adj[i] : c;
    const f = c !== 0 ? a / c : 1;
    const o = O[i], h = H[i], l = L[i];
    if (o == null || h == null || l == null) continue;
    bars.push({ t: ts[i] * 1000, o: o * f, h: h * f, l: l * f, c: a, v: V[i] == null ? 0 : V[i] });
  }
  bars.sort((a, b) => a.t - b.t);
  if (bars.length < 300) throw new Error(`too few bars for ${SYMBOL}: ${bars.length}`);
  return bars;
}

function simulate(bars, requireVolume) {
  const rets = [];
  for (let i = 1; i < bars.length; i++) rets.push(bars[i].c / bars[i - 1].c - 1);
  const trades = [];
  let signals = 0;
  for (let i = Math.max(W, K); i < bars.length - 1; i++) {
    if (i - 19 < 1) continue;
    const win = rets.slice(i - 19, i + 1);
    if (win.some((x) => !isFinite(x))) continue;
    const sd = stdev(win);
    if (!isFinite(sd) || sd <= 0) continue;
    const rk = bars[i].c / bars[i - K].c - 1;
    const z = rk / (sd * Math.sqrt(K));
    const vwin = bars.slice(i - W, i).map((b) => b.v).filter((v) => v != null && v > 0);
    let vrel = NaN;
    if (vwin.length >= W) vrel = bars[i].v / median(vwin);
    if (!isFinite(z)) continue;
    let pass = z <= ZT;
    if (requireVolume) {
      if (!isFinite(vrel) || vrel < VT) pass = false;
    }
    if (!pass) continue;
    signals++;
    const ei = i + 1;
    if (ei >= bars.length) continue;
    const entry = bars[ei].o;
    if (!isFinite(entry) || entry <= 0) continue;
    const tgt = entry * (1 + TM * sd), stp = entry * (1 - SM * sd);
    let exit = null, how = null, xi = null;
    const last = Math.min(ei + HOLD - 1, bars.length - 1);
    for (let j = ei; j <= last; j++) {
      const lo = bars[j].l, hi = bars[j].h;
      const hS = isFinite(lo) && lo <= stp, hT = isFinite(hi) && hi >= tgt;
      if (hS && hT) { exit = stp; how = "stop"; xi = j; break; } // conservative, matches L1/P1
      if (hS) { exit = stp; how = "stop"; xi = j; break; }
      if (hT) { exit = tgt; how = "target"; xi = j; break; }
    }
    if (exit == null) { xi = last; exit = bars[last].c; how = "time"; }
    trades.push({ net: exit / entry - 1 - FR, how, z, vrel: isFinite(vrel) ? vrel : null });
  }
  const n = trades.length;
  const nets = trades.map((t) => t.net);
  const mean = n ? nets.reduce((s, x) => s + x, 0) / n : NaN;
  const sd = n > 1 ? stdev(nets) : NaN;
  const t = n > 1 && sd > 0 ? mean / (sd / Math.sqrt(n)) : NaN;
  const winPct = n ? (trades.filter((t) => t.net > 0).length / n) * 100 : NaN;
  const up = trades.filter((t) => t.how === "target").length;
  const dn = trades.filter((t) => t.how === "stop").length;
  const tm = trades.filter((t) => t.how === "time").length;
  return { N: n, signals, meanPct: isFinite(mean) ? +(mean * 100).toFixed(3) : null, t: isFinite(t) ? +t.toFixed(2) : null, winPct: isFinite(winPct) ? +winPct.toFixed(1) : null, up, dn, tm };
}

const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(SYMBOL)}?interval=1d&period1=${P1}&period2=${P2}&events=div%7Csplit`;
const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0" } });
if (!res.ok) throw new Error(`chart fetch ${res.status} for ${SYMBOL}`);
const bars = parseChart(await res.json());
const gated = simulate(bars, true);
const priceOnly = simulate(bars, false);
const out = {
  symbol: SYMBOL,
  params: { K, W, ZT, VT, TM, SM, HOLD, FR },
  window: { p1: P1, firstDay: new Date(bars[0].t).toISOString().slice(0, 10), lastDay: new Date(bars[bars.length - 1].t).toISOString().slice(0, 10), bars: bars.length },
  gated: { ...gated, volumeGate: true },
  priceOnly: { ...priceOnly, volumeGate: false },
  kSpent: 0,
};
if (gated.N === 0) out.reason = "no trades";
writeFileSync(OUT, JSON.stringify(out));
console.log(`inv ok: ${SYMBOL} bars=${bars.length} gated_N=${gated.N} mean=${gated.meanPct} win=${gated.winPct}`);
