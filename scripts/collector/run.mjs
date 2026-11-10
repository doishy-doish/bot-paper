// run.mjs — fetch -> build -> store. Counts only on stdout (public-log safe).
// Usage (Actions or Debian):
//   UPSTOX_KEY=... node run.mjs --source upstox --symbols RELIANCE-EQ,INFY-EQ \
//     --from 2023-01-01 --to 2023-03-31 --interval 15 --out data/intraday
//   DHAN_TOKEN=... DHAN_CLIENT_ID=... node run.mjs --source dhan --symbols RELIANCE --segment NSE_EQ ...
// Writes: <out>/<SYM>_<interval>m.json (bars) + manifest.json (counts/ranges).
import { mkdirSync, writeFileSync } from "node:fs";
import { buildBars, manifest } from "./bars.mjs";
import { upstoxCandles, dhanCandles } from "./feed.mjs";

const args = Object.fromEntries(process.argv.slice(2).map((a) => {
  const m = a.match(/^--([^=]+)=(.*)$/); return m ? [m[1], m[2]] : [a.replace(/^--/, ""), true];
}));
const need = (k) => { if (!args[k]) { console.error("missing --" + k); process.exit(2); } return args[k]; };
const source = need("source"), symbols = need("symbols").split(","), from = need("from"), to = need("to");
const nMin = parseInt(need("interval"), 10), outDir = args.out || "data/intraday";
mkdirSync(outDir, { recursive: true });
const man = { source, interval: nMin, from, to, symbols: {} };
for (const sym of symbols) {
  let mins;
  if (source === "upstox") {
    const key = process.env.UPSTOX_KEY; if (!key) { console.error("UPSTOX_KEY unset"); process.exit(2); }
    mins = await upstoxCandles({ key, instrument: sym, interval: nMin === 1 ? "1minute" : "15minute", from, to });
  } else if (source === "dhan") {
    const token = process.env.DHAN_TOKEN, clientId = process.env.DHAN_CLIENT_ID;
    if (!token || !clientId) { console.error("DHAN_TOKEN/DHAN_CLIENT_ID unset"); process.exit(2); }
    mins = await dhanCandles({ clientId, token, symbol: sym, segment: args.segment || "NSE_EQ", interval: String(nMin), from, to });
  } else { console.error("unknown --source (upstox|dhan)"); process.exit(2); }
  const bars = nMin === 1 ? mins : buildBars(mins, nMin);
  const safe = sym.replace(/[^A-Za-z0-9_.-]/g, "_");
  writeFileSync(`${outDir}/${safe}_${nMin}m.json`, JSON.stringify({ symbol: sym, interval: nMin + "m", bars }));
  man.symbols[sym] = manifest(bars);
  console.log(`PROBE_JSON ${sym}: ${JSON.stringify(man.symbols[sym])}`);
}
writeFileSync(`${outDir}/manifest.json`, JSON.stringify(man, null, 1));
console.log("collector done: " + symbols.length + " symbols");
