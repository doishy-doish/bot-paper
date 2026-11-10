// feed.mjs — vendor adapters (Upstox V3 + Dhan v2). Keys ONLY from env.
// No keys here, no defaults, no logging of key material. Shapes follow the
// public docs; first live run with a real key re-verifies field names
// (probe prints row counts + date range, never prices, on mismatch).
const U = (p) => "https://api.upstox.com" + p;

export async function upstoxCandles({ key, instrument, interval, from, to }) {
  // interval: '1minute'|'15minute'; from/to: 'YYYY-MM-DD'
  const url = `${U("/v3/historical-candle/" + encodeURIComponent(instrument) + "/" + interval + "/" + to + "/" + from)}`;
  const r = await fetch(url, { headers: { Authorization: "Bearer " + key, Accept: "application/json" } });
  if (!r.ok) throw new Error("upstox HTTP " + r.status);
  const j = await r.json();
  const rows = j.data && j.data.candles ? j.data.candles : [];
  // Upstox candle: [timestamp, open, high, low, close, volume, oi]
  return rows.map((c) => ({ t: Date.parse(c[0]), o: +c[1], h: +c[2], l: +c[3], c: +c[4], v: +(c[5] || 0) }))
    .filter((b) => Number.isFinite(b.t)).sort((a, b) => a.t - b.t);
}

export async function dhanCandles({ clientId, token, symbol, segment, interval, from, to }) {
  // Dhan v2 charts/historical (verify field names on first keyed run).
  const r = await fetch("https://api.dhan.co/v2/charts/historical", {
    method: "POST",
    headers: { "Content-Type": "application/json", "access-token": token, "client-id": clientId },
    body: JSON.stringify({ symbol, exchangeSegment: segment, instrument: "EQUITY", expiryCode: 0, from, to, interval }),
  });
  if (!r.ok) throw new Error("dhan HTTP " + r.status);
  const j = await r.json();
  const n = j.open ? j.open.length : 0;
  const out = [];
  for (let i = 0; i < n; i++) out.push({ t: (j.start_Time ? Date.parse(j.start_Time[i]) : NaN), o: +j.open[i], h: +j.high[i], l: +j.low[i], c: +j.close[i], v: +(j.volume ? j.volume[i] : 0) });
  return out.filter((b) => Number.isFinite(b.t)).sort((a, b) => a.t - b.t);
}
