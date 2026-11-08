// Paper tick for GitHub Actions. No secrets: uses the runner's GITHUB_TOKEN
// (git push) and public Yahoo Finance data. Mirrors src/bot/backtest.py rules:
// SMA50/200 long-only, 0.3% friction per round trip.
import {readFileSync, writeFileSync, existsSync, mkdirSync} from "fs";

const SYMBOLS = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","SBIN.NS","LT.NS","TITAN.NS","SUNPHARMA.NS"];
const SLOT = 10000, START_CASH = 100000, FRICTION = 0.003;
const UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"};

async function closes(sym) {
  for (const h of ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]) {
    try {
      const r = await fetch(`${h}/v8/finance/chart/${sym}?interval=1d&range=1y`, {headers: UA});
      if (!r.ok) continue;
      const j = await r.json();
      const res = j.chart?.result?.[0];
      const q = res?.indicators?.quote?.[0];
      if (!q?.close) continue;
      const out = [];
      for (let i = 0; i < res.timestamp.length; i++) {
        if (q.close[i] == null) continue;
        out.push({date: new Date(res.timestamp[i] * 1000).toISOString().slice(0, 10), close: q.close[i]});
      }
      if (out.length > 220) return out;
    } catch (e) { /* next host */ }
  }
  return null;
}

function signal(hist) {
  const c = hist.map(h => h.close);
  if (c.length < 201) return 0;
  const sma = n => c.slice(-n).reduce((a, b) => a + b, 0) / n;
  return sma(50) > sma(200) ? 1 : 0;
}

let state = {cash: START_CASH, positions: {}, trades: [], equity: []};
if (existsSync("data/paper.json"))
  state = JSON.parse(readFileSync("data/paper.json", "utf8"));

const today = new Date().toISOString().slice(0, 10);
if (state.equity.length && state.equity[state.equity.length - 1].date === today) {
  console.log("already ticked today"); process.exit(0);
}
for (const sym of SYMBOLS) {
  const hist = await closes(sym);
  if (!hist || hist.length < 202) { console.log(sym + ": no data"); continue; }
  const bar = hist[hist.length - 1];          // fill bar (fresh close)
  const px = bar.close, held = !!state.positions[sym];
  const sig = signal(hist.slice(0, -1));      // signal on CONFIRMED bars only (t+1)
  if (sig === 1 && !held && state.cash >= SLOT) {
    state.positions[sym] = {entryPx: px, entryDate: hist[hist.length - 1].date, qty: SLOT / px};
    state.cash -= SLOT; console.log(sym + ": ENTER @" + px.toFixed(1));
  } else if (sig === 0 && held) {
    const p = state.positions[sym], proceeds = p.qty * px * (1 - FRICTION);
    state.cash += proceeds;
    state.trades.push({symbol: sym, entryDate: p.entryDate, exitDate: hist[hist.length - 1].date,
      exitPx: Math.round(px * 100) / 100, net: Math.round((proceeds / SLOT - 1) * 1e4) / 1e4});
    delete state.positions[sym]; console.log(sym + ": EXIT @" + px.toFixed(1));
  }
}
let mtm = 0;
for (const sym of Object.keys(state.positions)) {
  const hist = await closes(sym);
  if (hist) mtm += state.positions[sym].qty * hist[hist.length - 1].close;
}
state.equity.push({date: today, equity: Math.round((state.cash + mtm) * 100) / 100});
mkdirSync("data", {recursive: true});
writeFileSync("data/paper.json", JSON.stringify(state, null, 1));
console.log("equity: " + state.equity[state.equity.length - 1].equity);
