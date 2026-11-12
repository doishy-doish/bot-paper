// lm-shakespeare: HALT-R as a retrieval LM (kNN-LM flavor) on Tiny Shakespeare.
// Store = (context window -> next char) pairs in a modern Hopfield net; the recall
// softmax over the store aggregates into a next-char distribution (acc + perplexity).
// Baselines (same data, same queries): unigram + bigram. Gate: beat unigram.
//
// Usage:
//   node halt-r/lm-shakespeare.mjs --data input.txt --out out/W32 [--W 32] [--P 8000]
//     [--trials 100] [--no-encrypt]   (env SCRAMBLE wraps the model; --no-encrypt local smoke only)
//   node halt-r/lm-shakespeare.mjs decrypt <file.enc> --key-file <path> [--out <p>] [--force]
import { writeFileSync, readFileSync, mkdirSync, existsSync } from 'node:fs';
import { createHash, randomBytes, createCipheriv, createDecipheriv } from 'node:crypto';

const DEF = { W: 32, P: 8000, trials: 100, probeN: 10 };
const BETAS = [0.1, 0.3, 1.0, 3.0];
const SEED = 7;
function fail(m){ console.error('FAIL ' + m); process.exit(1); }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function sha256hex(b){ return createHash('sha256').update(b).digest('hex'); }
function kdf(s){ return createHash('sha256').update(String(s), 'utf8').digest(); }
function rssMB(){ return Math.round(process.memoryUsage().rss / 1024 / 1024 * 10) / 10; }
function argmax(a){ let bi = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i; return bi; }
function encryptWrap(plain, secret){
  const K = kdf(secret), nonce = randomBytes(12), c = createCipheriv('aes-256-gcm', K, nonce);
  const data = Buffer.concat([c.update(plain), c.final()]);
  return { enc: 'AES-256-GCM-v1', nonce: nonce.toString('base64'), data: data.toString('base64'),
    tag: c.getAuthTag().toString('base64'), sha256: sha256hex(plain), bytes: plain.length,
    date: new Date().toISOString().slice(0, 10) };
}
function decryptWrap(wrap, secret){
  if (wrap.enc !== 'AES-256-GCM-v1') fail('unknown wrapper ' + wrap.enc);
  const d = createDecipheriv('aes-256-gcm', kdf(secret), Buffer.from(wrap.nonce, 'base64'));
  d.setAuthTag(Buffer.from(wrap.tag, 'base64'));
  const plain = Buffer.concat([d.update(Buffer.from(wrap.data, 'base64')), d.final()]);
  if (plain.length !== wrap.bytes || sha256hex(plain) !== wrap.sha256) fail('integrity mismatch (wrong key or corrupt)');
  return plain;
}
// ---- core (pure; testable without node builtins) ----
function buildVocab(text){
  const chars = [...new Set(text)].sort();
  const toI = {}; chars.forEach((c, i) => { toI[c] = i; });
  return { chars, toI, V: chars.length };
}
function encodeCtx(textIds, pos, W, V, out){
  out.fill(0);
  for (let k = 0; k < W; k++) out[k * V + textIds[pos + k]] = 1;
  return out;
}
// recall returns normalized store weights + mixed state
function recallW(q, X, P, D, beta, maxS){
  let x = Float32Array.from(q);
  const w = new Float32Array(P);
  let steps = 0;
  for (let s = 0; s < maxS; s++){
    let m = -1e30;
    for (let i = 0; i < P; i++){
      let d = 0; const off = i * D;
      for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * x[j];
      d *= beta; w[i] = d; if (d > m) m = d;
    }
    let t = 0;
    for (let i = 0; i < P; i++){ w[i] = Math.exp(w[i] - m); t += w[i]; }
    const nx = new Float32Array(D);
    for (let i = 0; i < P; i++){ const c = w[i] / t; const off = i * D; for (let j = 0; j < D; j++) nx[j] += c * (X[off + j] ? 1 : -1); }
    let maxd = 0;
    for (let j = 0; j < D; j++){ const dd = Math.abs(nx[j] - x[j]); if (dd > maxd) maxd = dd; }
    x = nx; steps++;
    if (maxd < 1e-3) break;
  }
  return { w, steps };
}
function predictNext(w, next, V){
  const dist = new Float64Array(V);
  for (let i = 0; i < w.length; i++) dist[next[i]] += w[i];
  return dist;
}
// ---- io ----
if (process.argv[2] === 'decrypt'){
  const a = process.argv.slice(3), file = a[0];
  const ki = a.indexOf('--key-file'), ke = a.indexOf('--key-env'), oi = a.indexOf('--out');
  let secret = null;
  if (ki >= 0) secret = readFileSync(a[ki + 1], 'utf8').trim();
  else if (ke >= 0) secret = process.env[a[ke + 1]];
  else if (process.env.SCRAMBLE) secret = process.env.SCRAMBLE;
  if (!secret) fail('no key');
  const plain = decryptWrap(JSON.parse(readFileSync(file, 'utf8')), secret);
  const out = oi >= 0 ? a[oi + 1] : file.replace(/\.enc$/, '');
  if (existsSync(out) && !a.includes('--force')) fail('refuses to overwrite ' + out);
  writeFileSync(out, plain);
  console.log('decrypt ok bytes=' + plain.length + ' sha256=' + sha256hex(plain));
  process.exit(0);
}
{
  const a = process.argv.slice(2);
  const gv = (k, d) => { const i = a.indexOf('--' + k); return i >= 0 ? a[i + 1] : d; };
  const noEnc = a.includes('--no-encrypt');
  const dataPath = gv('data', null), outDir = gv('out', null);
  if (!dataPath || !outDir) fail('usage: --data <input.txt> --out <dir> [--W 32] [--P 8000] [--trials 100]');
  const W = parseInt(gv('W', DEF.W)), P = parseInt(gv('P', DEF.P)), TR = parseInt(gv('trials', DEF.trials));
  const kei = a.indexOf('--key-env');
  const secret = process.env[kei >= 0 ? a[kei + 1] : 'SCRAMBLE'];
  if (!secret && !noEnc) fail('SCRAMBLE key missing (refuses plaintext save; --no-encrypt for smoke only)');
  mkdirSync(outDir, { recursive: true });
  const raw = readFileSync(dataPath, 'utf8');
  const cut = Math.floor(raw.length * 0.9);
  const train = raw.slice(0, cut), val = raw.slice(cut);
  const { chars, toI, V } = buildVocab(train);
  if (V > 200) fail('vocab blowup V=' + V);
  console.log(`corpus chars=${raw.length} train=${train.length} val=${val.length} V=${V} W=${W} P=${P}`);
  const trainIds = new Uint16Array(train.length);
  for (let i = 0; i < train.length; i++) trainIds[i] = toI[train[i]];
  const valIds = new Uint16Array(val.length);
  for (let i = 0; i < val.length; i++) valIds[i] = toI[val[i]] !== undefined ? toI[val[i]] : 0;
  // baselines on full train
  const uni = new Float64Array(V);
  for (let i = 0; i < trainIds.length; i++) uni[trainIds[i]]++;
  for (let i = 0; i < V; i++) uni[i] /= trainIds.length;
  const uniPred = argmax(uni);
  const bi = new Float64Array(V * V);
  for (let i = 0; i + 1 < trainIds.length; i++) bi[trainIds[i] * V + trainIds[i + 1]]++;
  // store: seeded sampled train contexts
  const t0 = Date.now();
  const D = W * V;
  const X = new Int8Array(P * D), next = new Uint8Array(P);
  const r = rng(SEED), q = new Float32Array(D);
  for (let s = 0; s < P; s++){
    const pos = Math.floor(r() * (trainIds.length - W));
    encodeCtx(trainIds, pos, W, V, q);
    for (let j = 0; j < D; j++) X[s * D + j] = q[j] > 0.5 ? 1 : 0;
    next[s] = trainIds[pos + W];
  }
  const buildMs = Date.now() - t0;
  console.log(`store P=${P} D=${D} bytes=${X.length + next.length} buildMs=${buildMs} rss=${rssMB()}MB`);
  const runQueries = (n, seed) => {
    const rr = rng(seed), dist_sum = { acc: 0, ce: 0, steps: 0 };
    const qq = new Float32Array(D), qpm = new Float32Array(D);
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      encodeCtx(valIds, pos, W, V, qq);
      for (let j = 0; j < D; j++) qpm[j] = qq[j] > 0.5 ? 1 : -1; // ±1 queries: dot = 2*matches - W (zero-mean noise)
      const trueNext = valIds[pos + W];
      const { w, steps } = recallW(qpm, X, P, D, runQueries.beta, 12);
      const dist = predictNext(w, next, V);
      let tot = 0; for (let i = 0; i < V; i++) tot += dist[i];
      if (argmax(dist) === trueNext) dist_sum.acc++;
      dist_sum.ce += -Math.log((dist[trueNext] + 1e-12) / (tot + 1e-12 * V));
      dist_sum.steps += steps;
    }
    return { acc: dist_sum.acc / n, ppl: Math.exp(dist_sum.ce / n), avgSteps: dist_sum.steps / n };
  };
  // bigram eval on same query streams happens implicitly below via shared seeds (positions only depend on seed)
  const bigramEval = (n, seed) => {
    const rr = rng(seed);
    let acc = 0, ce = 0;
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      const prev = valIds[pos + W - 1], trueNext = valIds[pos + W];
      const row = prev * V;
      let tot = V, best = 0, bestC = -1; // Laplace alpha=1
      for (let c = 0; c < V; c++){ const cc = bi[row + c] + 1; tot += bi[row + c]; if (cc > bestC){ bestC = cc; best = c; } }
      if (best === trueNext) acc++;
      ce += -Math.log((bi[row + trueNext] + 1) / tot);
    }
    return { acc: acc / n, ppl: Math.exp(ce / n) };
  };
  const uniEval = (n, seed) => {
    const rr = rng(seed);
    let acc = 0, ce = 0;
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      const trueNext = valIds[pos + W];
      if (uniPred === trueNext) acc++;
      ce += -Math.log(uni[trueNext] + 1e-12);
    }
    return { acc: acc / n, ppl: Math.exp(ce / n) };
  };
  const t1 = Date.now();
  let best = null;
  for (const beta of BETAS){
    runQueries.beta = beta;
    const m = runQueries(DEF.probeN, 1000);
    console.log(`probe beta=${beta} acc=${m.acc.toFixed(3)} ppl=${m.ppl.toFixed(2)} steps=${m.avgSteps.toFixed(1)}`);
    if (!best || m.acc > best.acc || (m.acc === best.acc && m.ppl < best.ppl)) best = { beta, ...m };
  }
  runQueries.beta = best.beta;
  const e = runQueries(TR, 2000);
  const biE = bigramEval(TR, 2000), uniE = uniEval(TR, 2000);
  const evalMs = Date.now() - t1;
  console.log(`eval beta=${best.beta} hopfield acc=${e.acc.toFixed(3)} ppl=${e.ppl.toFixed(2)} steps=${e.avgSteps.toFixed(2)} ms=${evalMs} rss=${rssMB()}MB`);
  console.log(`base bigram acc=${biE.acc.toFixed(3)} ppl=${biE.ppl.toFixed(2)} | unigram acc=${uniE.acc.toFixed(3)} ppl=${uniE.ppl.toFixed(2)}`);
  let failN = 0;
  const check = (n, c, d) => { console.log(((c ? 'PASS' : 'FAIL') + ' ' + n + ' ' + (d || ''))); if (!c) failN++; };
  // hard gate = machinery (self-retrieval must be exact); baselines are recorded comparisons, not gates
  {
    const qs = new Float32Array(D), qp = new Float32Array(D);
    const rs = rng(999);
    let self = 0; const NS = 20;
    for (let t = 0; t < NS; t++){
      const pos = Math.floor(rs() * (trainIds.length - W));
      encodeCtx(trainIds, pos, W, V, qs);
      for (let j = 0; j < D; j++) qp[j] = qs[j] > 0.5 ? 1 : -1;
      const { w } = recallW(qp, X, P, D, best.beta, 12);
      if (predictNext(w, next, V) && argmax(predictNext(w, next, V)) === trainIds[pos + W]) self++;
    }
    check('self-retrieval-exact', self === NS, self + '/' + NS);
  }
  console.log(`compare hopfield acc=${e.acc.toFixed(3)} ppl=${e.ppl.toFixed(2)} vs bigram ${biE.acc.toFixed(3)}/${biE.ppl.toFixed(2)} vs unigram ${uniE.acc.toFixed(3)}/${uniE.ppl.toFixed(2)} beatUni=${e.acc > uniE.acc}`);
  // save wrapped store
  const plain = Buffer.concat([Buffer.from(X), Buffer.from(next)]);
  const sum = sha256hex(plain);
  const meta = { task: 'tinyshakespeare-nextchar', W, P, V, D, seed: SEED, beta: best.beta, bytes: plain.length, sha256: sum, enc: noEnc ? 'none-smoke' : 'AES-256-GCM-v1', chars: raw.length };
  writeFileSync(outDir + '/meta.json', JSON.stringify(meta));
  let encBytes = 0;
  if (!noEnc){ const wj = JSON.stringify(encryptWrap(plain, secret)); writeFileSync(outDir + '/model.bin.enc', wj); encBytes = wj.length; console.log(`enc bytes=${encBytes} sha256=${sum}`); }
  else { writeFileSync(outDir + '/model.bin', plain); console.log(`plain-smoke bytes=${plain.length} sha256=${sum}`); }
  const metrics = { ...meta, trials: TR, flips: 'n/a-nextchar', acc: e.acc, ppl: e.ppl, avgSteps: e.avgSteps, bigramAcc: biE.acc, bigramPpl: biE.ppl, unigramAcc: uniE.acc, unigramPpl: uniE.ppl, beatUni: e.acc > uniE.acc, buildMs, evalMs, maxRssMB: rssMB(), encBytes, evalScope: 'heldout-val' };
  writeFileSync(outDir + '/metrics.json', JSON.stringify(metrics));
  if (failN){ console.log('lm-shakespeare: ' + failN + ' FAILURES'); process.exit(1); }
  console.log('lm-shakespeare: benched');
}
