// Paper tick for GitHub Actions. No secrets: uses the runner's GITHUB_TOKEN
// (git push) and public Yahoo Finance data. Mirrors src/bot/backtest.py rules:
// SMA50/200 long-only, 0.3% friction per round trip.
// NIFTY_FILTER=1 optionally gates entries on Nifty > SMA200 (Step 1 regime test).
import {readFileSync, writeFileSync, existsSync, mkdirSync} from "fs";
import {createCipheriv, createDecipheriv, createHash, randomBytes} from "crypto";

const SYMBOLS = ["RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS","ITC.NS","SBIN.NS","LT.NS","TITAN.NS","SUNPHARMA.NS"];
const SLOT = 10000, START_CASH = 100000, FRICTION = 0.003;
const NIFTY_FILTER = process.env.NIFTY_FILTER === "1";
const UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"};
// Scramble mode: when SCRAMBLE_KEY (64 hex) is set, the paper file on disk is an
// AES-256-GCM wrapper, and all logs go quiet (counts + hashes only — never
// symbols, prices, or equity, because Actions logs on a public repo are public).
const SCRAMBLE_KEY = (process.env.SCRAMBLE_KEY || "").trim();
const SCRAMBLED = /^[0-9a-fA-F]{64}$/.test(SCRAMBLE_KEY);

function scrambleEncrypt(buf) {
  const key = Buffer.from(SCRAMBLE_KEY, "hex"), nonce = randomBytes(12);
  const c = createCipheriv("aes-256-gcm", key, nonce);
  const ct = Buffer.concat([c.update(buf), c.final()]);
  return JSON.stringify({enc: "AES-256-GCM-v1", nonce: nonce.toString("base64"),
    data: ct.toString("base64"), tag: c.getAuthTag().toString("base64"),
    sha256: createHash("sha256").update(buf).digest("hex"), bytes: buf.length,
    date: new Date().toISOString().slice(0, 10)}, null, 1);
}
function scrambleDecryptIfNeeded(raw) {
  let obj;
  try { obj = JSON.parse(raw.toString("utf8")); } catch { return JSON.parse(raw.toString("utf8")); }
  if (!obj || obj.enc !== "AES-256-GCM-v1") return obj;
  if (!SCRAMBLED) { console.error("paper.json is scrambled but SCRAMBLE_KEY is missing"); process.exit(1); }
  const d = createDecipheriv("aes-256-gcm", Buffer.from(SCRAMBLE_KEY, "hex"), Buffer.from(obj.nonce, "base64"));
  d.setAuthTag(Buffer.from(obj.tag, "base64"));
  const plain = Buffer.concat([d.update(Buffer.from(obj.data, "base64")), d.final()]);
  return JSON.parse(plain.toString("utf8"));
}

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
  state = scrambleDecryptIfNeeded(readFileSync("data/paper.json"));

const today = new Date().toISOString().slice(0, 10);
if (state.equity.length && state.equity[state.equity.length - 1].date === today) {
  console.log("already ticked today"); process.exit(0);
}
let entered = 0, exited = 0, skipped = 0;
let bull = true;
if (NIFTY_FILTER) {
  const nh = await closes("^NSEI");
  if (!nh || nh.length < 201) { console.log("nifty gate shut: no data"); bull = false; }
  else {
    const c = nh.slice(0, -1).map(h => h.close); // confirmed bars only (t+1)
    const sma200 = c.slice(-200).reduce((a, b) => a + b, 0) / 200;
    bull = c[c.length - 1] > sma200;
    if (!SCRAMBLED) console.log("nifty gate: " + (bull ? "BULL" : "BEAR"));
  }
}
for (const sym of SYMBOLS) {
  const hist = await closes(sym);
  if (!hist || hist.length < 202) { skipped++; if (!SCRAMBLED) console.log(sym + ": no data"); continue; }
  const bar = hist[hist.length - 1];          // fill bar (fresh close)
  const px = bar.close, held = !!state.positions[sym];
  let sig = signal(hist.slice(0, -1));      // signal on CONFIRMED bars only (t+1)
  if (NIFTY_FILTER && !bull) sig = 0;       // regime shut: flat, exits fill next open
  if (sig === 1 && !held && state.cash >= SLOT) {
    state.positions[sym] = {entryPx: px, entryDate: hist[hist.length - 1].date, qty: SLOT / px};
    state.cash -= SLOT; entered++; if (!SCRAMBLED) console.log(sym + ": ENTER @" + px.toFixed(1));
  } else if (sig === 0 && held) {
    const p = state.positions[sym], proceeds = p.qty * px * (1 - FRICTION);
    state.cash += proceeds;
    state.trades.push({symbol: sym, entryDate: p.entryDate, exitDate: hist[hist.length - 1].date,
      exitPx: Math.round(px * 100) / 100, net: Math.round((proceeds / SLOT - 1) * 1e4) / 1e4});
    delete state.positions[sym]; exited++; if (!SCRAMBLED) console.log(sym + ": EXIT @" + px.toFixed(1));
  }
}
let mtm = 0;
for (const sym of Object.keys(state.positions)) {
  const hist = await closes(sym);
  if (hist) mtm += state.positions[sym].qty * hist[hist.length - 1].close;
}
state.equity.push({date: today, equity: Math.round((state.cash + mtm) * 100) / 100});
mkdirSync("data", {recursive: true});
if (SCRAMBLED) {
  writeFileSync("data/paper.json", scrambleEncrypt(Buffer.from(JSON.stringify(state, null, 1))) + "\n");
  const h = createHash("sha256").update(JSON.stringify(state.equity)).digest("hex").slice(0, 16);
  console.log("tick ok date=" + today + " positions=" + Object.keys(state.positions).length +
    " trades=" + state.trades.length + " entered=" + entered + " exited=" + exited +
    " skipped=" + skipped + " curve=" + h + "…");
} else {
  writeFileSync("data/paper.json", JSON.stringify(state, null, 1));
  console.log("equity: " + state.equity[state.equity.length - 1].equity);
}
