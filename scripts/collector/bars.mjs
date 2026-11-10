// bars.mjs — pure 1m -> N-minute bar builder (no I/O, no keys, fully tested).
// Input: ascending [{t: epochMs, o,h,l,c, v}]. Output: ascending N-min bars.
// Session filter keeps 09:15–15:30 IST (NSE cash) unless {session:false}.
export function buildBars(minutes, nMin, { session = true } = {}) {
  const MS = 60 * 1000;
  const out = [];
  let cur = null;
  const flush = () => { if (cur) { out.push(cur); cur = null; } };
  for (const m of minutes) {
    if (session && !inSession(m.t)) continue;
    const bucket = Math.floor(m.t / (nMin * MS)) * (nMin * MS);
    if (!cur || cur.t !== bucket) { flush(); cur = { t: bucket, o: m.o, h: m.h, l: m.l, c: m.c, v: m.v || 0 }; }
    else { if (m.h > cur.h) cur.h = m.h; if (m.l < cur.l) cur.l = m.l; cur.c = m.c; cur.v += m.v || 0; }
  }
  flush();
  return out;
}
// IST wall-clock check via +05:30 offset (no TZ db needed in Actions).
export function inSession(tMs) {
  const ist = new Date(tMs + 5.5 * 3600 * 1000);
  const mins = ist.getUTCHours() * 60 + ist.getUTCMinutes();
  return mins >= 9 * 60 + 15 && mins <= 15 * 60 + 30;
}
export function manifest(bars) {
  if (!bars.length) return { n: 0 };
  return { n: bars.length, first: new Date(bars[0].t).toISOString(), last: new Date(bars[bars.length - 1].t).toISOString() };
}
