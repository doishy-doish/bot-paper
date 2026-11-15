// qwen-alone.mjs: donor-alone acc/ppl on bench queries via llama-server n_probs.
// Diagnostic-grade (competence ordering gate for the big-donor continuity rule, not a verdict):
// first-char aggregation of server token probs, coverage logged. Query sampling replays the
// bench exactly (LCG seed stream, 90/10 cut, W) so report rows pair 1:1 with bench repHit strings.
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
const A = {};
{ const a = process.argv.slice(2); for (let i = 0; i < a.length; i += 2) A[a[i].replace(/^--/, '')] = a[i + 1]; }
function fail(m){ console.error('FAIL ' + m); process.exit(1); }
for (const k of ['server', 'data', 'out']) if (!A[k]) fail('need --' + k);
const W = parseInt(A.W || '16'), TR = parseInt(A.trials || '200'), SEED = parseInt(A.seed || '2001');
function rng(seed){ let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
function gpt2ByteMap(){ // copied from lm-shakespeare.mjs readGraftRowsHF (keep in sync)
  const bs = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) if (bs.indexOf(b) < 0){ bs.push(b); cs.push(256 + n); n++; }
  const m = {};
  for (let i = 0; i < bs.length; i++) m[bs[i]] = String.fromCodePoint(cs[i]);
  return m;
}
const bmap = gpt2ByteMap(), invmap = {};
for (const b of Object.keys(bmap)) invmap[bmap[b]] = parseInt(b);
function pieceFirstChar(piece, toI){
  // llama.cpp pieces use GPT-2 text form (leading U+0120 = space). Take the first char,
  // inverse-map high chars back to bytes, return the corpus char or null.
  let s = piece;
  if (s.charCodeAt(0) === 0x0120) s = ' ' + s.slice(1);
  const chars = [...s];
  if (!chars.length) return null;
  let c0 = chars[0];
  if (toI[c0] !== undefined) return c0;
  if (invmap[c0] !== undefined){ const b = String.fromCharCode(invmap[c0]); if (toI[b] !== undefined) return b; }
  return null;
}
const raw = readFileSync(A.data, 'utf8');
const cut = Math.floor(raw.length * 0.9), train = raw.slice(0, cut), val = raw.slice(cut); // same 90/10 cut as bench
const chset = [...new Set(train)].sort();
const toI = {}; chset.forEach((c, i) => { toI[c] = i; });
async function waitServer(){
  for (let i = 0; i < 120; i++){
    try { const r = await fetch(A.server + '/health'); if (r.ok) return; } catch (e) { /* not up yet */ }
    await new Promise(r => setTimeout(r, 5000));
  }
  fail('server never healthy ' + A.server);
}
async function complete(ctx){
  const r = await fetch(A.server + '/completion', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: ctx, n_predict: 1, n_probs: 100, temperature: 0, cache_prompt: true, post_sampling_probs: true }) });
  if (!r.ok) fail('server completion status ' + r.status);
  const raw = await r.text();
  let j; try { j = JSON.parse(raw); } catch (e) { fail('server: bad JSON ' + raw.slice(0, 400)); }
  const cp0 = j.completion_probabilities && j.completion_probabilities[0];
  const probs = (cp0 && (cp0.probs || cp0.top_probs)) || null;
  if (!probs || !probs.length) fail('server: no probs (keys: ' + Object.keys(cp0 || {}).join(',') + ') body: ' + raw.slice(0, 400));
  return probs;
}
await waitServer();
const rr = rng(SEED);
let acc = 0, ce = 0, covSum = 0, skipped = 0;
const queries = [];
for (let t = 0; t < TR; t++){
  const pos = Math.floor(rr() * (val.length - W - 1));
  const ctx = val.slice(pos, pos + W), tn = val[pos + W];
  if (toI[tn] === undefined){ skipped++; continue; }
  const probs = await complete(ctx);
  const mass = {}; let tot = 0;
  for (const p of probs){
    if (p.token === undefined || typeof p.prob !== 'number') fail('server: prob shape changed (keys: ' + Object.keys(p || {}).join(',') + ')');
    const c = pieceFirstChar(p.token, toI);
    if (c === null) continue;
    mass[c] = (mass[c] || 0) + p.prob; tot += p.prob;
  }
  covSum += tot;
  let best = null, bp = -1;
  for (const c of Object.keys(mass)){ const q = mass[c] / Math.max(tot, 1e-12); if (q > bp){ bp = q; best = c; } }
  const ptn = (mass[tn] || 0) / Math.max(tot, 1e-12);
  if (best === tn) acc++;
  ce += -Math.log(Math.max(ptn, 1e-12));
  queries.push({ pos, tn, pred: best, ptn: +ptn.toFixed(4), cov: +tot.toFixed(3) });
}
const n = queries.length;
const res = { donor: A.donor || null, corpus: A.corpus || null, W, seed: SEED, trials: TR, n, skipped,
  acc: acc / Math.max(n, 1), ppl: Math.exp(ce / Math.max(n, 1)), coverage: covSum / Math.max(n, 1), queries };
const dir = A.out.split('/').slice(0, -1).join('/');
if (dir) mkdirSync(dir, { recursive: true });
writeFileSync(A.out, JSON.stringify(res));
console.log('qwen-alone done donor=' + res.donor + ' corpus=' + res.corpus + ' acc=' + res.acc.toFixed(3) + ' ppl=' + res.ppl.toFixed(2) + ' coverage=' + res.coverage.toFixed(3) + ' n=' + n + ' skipped=' + skipped);
