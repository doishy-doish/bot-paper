// donor-topk.mjs: V4.1 donor-side harvest (D1-alone, no D2).
// Per-trial donor top-10 via OAI /v1/completions (logprobs 10, temp 0, single token),
// same-trial as banked bench REPORT stream (hopBiDists(TR, 2001), LCG regen, no store).
// Pre-reg: scratch/papers/router-v4-prereg-2026-11-16.md + router-v4.1-addendum-2026-11-16.md.
// Fires ONLY on explicit user go. Output is plaintext trial logprobs (no weights, no SCRAMBLE).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const A = {};
{ const a = process.argv.slice(2); for (let i = 0; i < a.length; i += 2) A[a[i].replace(/^--/, '')] = a[i + 1]; }
function fail(m){ console.error('FAIL ' + m); process.exit(1); }
for (const k of ['data', 'tag', 'out', 'server']) if (!A[k]) fail('need --' + k);
const W = parseInt(A.W || '16'), TR = parseInt(A.trials || '200'), SEED_STREAM = 2001;
const DONOR = A.donor || 'qwen2.5-0.5b-instruct-q4_k_m', PIN = A.pin || 'a68c6136';
function rng(seed){ let s = seed >>> 0; return () => { s = (Math.imul(s, 1664525) + 1013904223) >>> 0; return s / 4294967296; }; }
function buildVocab(text){ const chars = [...new Set(text)].sort(); const toI = {}; chars.forEach((c, i) => { toI[c] = i; }); return { chars, toI, V: chars.length }; }
const raw = readFileSync(A.data, 'utf8');
const cut = Math.floor(raw.length * 0.9), train = raw.slice(0, cut), val = raw.slice(cut);
const { toI, V } = buildVocab(train);
const rr = rng(SEED_STREAM);
const trials = [];
for (let t = 0; t < TR; t++){
  const pos = Math.floor(rr() * (val.length - W - 1));
  const ctx = val.slice(pos, pos + W), tnChar = val[pos + W];
  if (ctx.length !== W) fail('short ctx t=' + t);
  trials.push({ t, pos, ctx, tnChar, tnId: toI[tnChar] !== undefined ? toI[tnChar] : -1 });
}
async function waitServer(){
  for (let i = 0; i < 120; i++){
    try { const r = await fetch(A.server + '/health'); if (r.ok) return; } catch (e) {}
    await new Promise(r => setTimeout(r, 5000));
  }
  fail('server never healthy ' + A.server);
}
async function oaiTopk(ctx){
  let last = null;
  for (let a = 0; a < 3; a++){
    const r = await fetch(A.server + '/v1/completions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt: ctx, max_tokens: 1, temperature: 0, logprobs: 10, echo: false }) });
    if (!r.ok){ last = 'http ' + r.status; await new Promise(r2 => setTimeout(r2, 2000)); continue; }
    const j = await r.json();
    const ch = j.choices && j.choices[0], lp = ch && ch.logprobs;
    const c0 = lp && lp.content && lp.content[0];
    const tl = c0 && (c0.top_logprobs || c0.top_logprobs_compat || []);
    if (Array.isArray(tl) && tl.length > 0) return { top: tl.map(x => [x.token !== undefined ? x.token : x.tok_str, x.logprob]), text: ch.text };
    last = 'short topk lpKeys=' + (lp ? Object.keys(lp).join(',') : 'null');
    await new Promise(r2 => setTimeout(r2, 2000));
  }
  return { top: [], err: last, text: null };
}
async function plainText(ctx){
  const r = await fetch(A.server + '/completion', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ prompt: ctx, n_predict: 1, temperature: 0, cache_prompt: true }) });
  if (!r.ok) return { text: null, err: 'http ' + r.status };
  const j = await r.json();
  return { text: j.content || '' };
}
await waitServer();
let depthOk = 0;
for (const tr of trials){
  const o = await oaiTopk(tr.ctx);
  tr.top = o.top; tr.depth = o.top.length; tr.genText = o.text;
  if (o.err) tr.err = o.err;
  if (tr.depth >= 10) depthOk++;
  if (tr.t % 10 === 0){ const s = await plainText(tr.ctx); tr.spot = s; } // 20 spot trials
  if (tr.t % 50 === 0) console.log(`trial ${tr.t}/200 depth>=10 so far ${depthOk}`);
}
const res = { meta: { tag: A.tag, donor: DONOR, pin: PIN, W, trials: TR, stream: 'report-2001-regen', V, corpusChars: raw.length },
  summary: { n: trials.length, depthFrac: depthOk / trials.length, spotN: trials.filter(t => t.spot).length }, trials };
const dir = A.out.split('/').slice(0, -1).join('/');
if (dir) mkdirSync(dir, { recursive: true });
writeFileSync(A.out, JSON.stringify(res));
console.log(`donor-topk done tag=${A.tag} n=${trials.length} depthFrac=${res.summary.depthFrac.toFixed(3)} spotN=${res.summary.spotN}`);
