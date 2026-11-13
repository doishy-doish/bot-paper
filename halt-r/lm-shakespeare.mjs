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
const BETAS = [0.1, 0.3, 1.0, 3.0, 10.0];
const LAMBDAS = [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]; // kNN-LM (Khandelwal et al 2019) interpolation grid
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
  const noGen = a.includes('--no-gen'); // P-sweep: skip generation (costly, unneeded per-P)
  const dataPath = gv('data', null), outDir = gv('out', null);
  if (!dataPath || !outDir) fail('usage: --data <input.txt> --out <dir> [--W 32] [--P 8000] [--trials 100] [--proj raw|sim|sup] [--no-gen]');
  const W = parseInt(gv('W', DEF.W)), TR = parseInt(gv('trials', DEF.trials));
  let P = parseInt(gv('P', DEF.P));
  const P0 = P; // requested store size; P becomes Peff after B2 prune
  const GATE = parseFloat(gv('gate', 0)), KEEP = parseFloat(gv('pruneKeep', 1)); // B1 confidence gate, B2 store prune
  const tag = gv('tag', 'shakes');
  const PROJ = gv('proj', 'raw'), H = parseInt(gv('H', 256));
  const BSEL = gv('betaSel', 'acc');
  if (BSEL !== 'acc' && BSEL !== 'ppl') fail('usage: --betaSel acc|ppl');
  const ADA = gv('ada', '0'); // H4/H5 adaptive-lambda: 0=off(fixed lam), 1=L=lam*top1frac(g), 2=L=lam*negentropy(1-h); default 0
  const kei = a.indexOf('--key-env');
  const secret = process.env[kei >= 0 ? a[kei + 1] : 'SCRAMBLE'];
  if (!secret && !noEnc) fail('SCRAMBLE key missing (refuses plaintext save; --no-encrypt for smoke only)');
  mkdirSync(outDir, { recursive: true });
  const raw = readFileSync(dataPath, 'utf8');
  const cut = Math.floor(raw.length * 0.9);
  const train = raw.slice(0, cut), val = raw.slice(cut);
  const { chars, toI, V } = buildVocab(train);
  if (V > 200) fail('vocab blowup V=' + V);
  console.log(`corpus chars=${raw.length} train=${train.length} val=${val.length} V=${V} W=${W} P=${P} proj=${PROJ}${PROJ === 'raw' ? '' : ' H=' + H}`);
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
  // store: seeded sampled train contexts (raw one-hot or SimHash-projected bits)
  const t0 = Date.now();
  const Din = W * V;
  const D = PROJ === 'raw' ? Din : H;
  function gauss1(rr){ let u = 0, v = 0; while (!u) u = rr(); while (!v) v = rr(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  let M = null; // projection matrix (sim: random Gaussian; sup: supervised from store labels)
  if (PROJ === 'sim'){ // deterministic random projection (own rng stream); sign bits = SimHash
    M = new Float32Array(Din * H);
    const rr2 = rng(SEED + 1), inv = Math.sqrt(1 / Din);
    for (let i = 0; i < M.length; i++) M[i] = gauss1(rr2) * inv;
  }
  function featPM(ids, pos, out){ // ±1 features, length D
    if (PROJ === 'sim' || PROJ === 'sup'){
      for (let h = 0; h < H; h++) out[h] = 0;
      for (let k = 0; k < W; k++){
        const base = (k * V + ids[pos + k]) * H;
        for (let h = 0; h < H; h++) out[h] += M[base + h];
      }
      for (let h = 0; h < H; h++) out[h] = out[h] >= 0 ? 1 : -1;
    } else {
      out.fill(0);
      for (let k = 0; k < W; k++) out[k * V + ids[pos + k]] = 1;
      for (let j = 0; j < out.length; j++) out[j] = out[j] > 0.5 ? 1 : -1;
    }
    return out;
  }
  let X = new Int8Array(P * D), next = new Uint8Array(P), spos = new Int32Array(P);
  let flopSteps = 0; // recall iterations actually executed (all phases) -> evalMflops
  const r = rng(SEED), q = new Float32Array(D);
  if (PROJ === 'sup'){
    // H1 learned encoder: positions that predict the label get bigger projection
    // magnitude, so dots are dominated by predictive (usually suffix) positions and
    // partial matches concentrate instead of diffusing. Same random-direction code
    // path as sim (featPM identical) — the ONLY difference is learned per-position
    // scales a_k from single-position accuracy on the store sample (W params, seconds).
    for (let s = 0; s < P; s++){
      const pos = Math.floor(r() * (trainIds.length - W));
      spos[s] = pos;
      next[s] = trainIds[pos + W];
    }
    const aK = new Float64Array(W); // learned position scales: single-position accuracy
    for (let k = 0; k < W; k++){
      const groups = new Map(); // char -> counts per next-char
      for (let s = 0; s < P; s++){
        const c = trainIds[spos[s] + k], n = next[s];
        let g = groups.get(c); if (!g){ g = new Uint32Array(V); groups.set(c, g); }
        g[n]++;
      }
      let correct = 0;
      for (let s = 0; s < P; s++){
        const g = groups.get(trainIds[spos[s] + k]);
        let b = 0; for (let c = 1; c < V; c++) if (g[c] > g[b]) b = c;
        if (b === next[s]) correct++;
      }
      aK[k] = Math.max(0, correct / P - 1 / V);
    }
    let aMean = 0; for (let k = 0; k < W; k++) aMean += aK[k]; aMean /= W;
    if (aMean > 1e-9){ for (let k = 0; k < W; k++) aK[k] /= aMean; }
    else { for (let k = 0; k < W; k++) aK[k] = 1; } // no signal: fall back to sim
    const rr3 = rng(SEED + 3), inv = Math.sqrt(1 / Din);
    M = new Float32Array(Din * H);
    for (let k = 0; k < W; k++) for (let c = 0; c < V; c++){
      const sc = aK[k], base = (k * V + c) * H;
      for (let h = 0; h < H; h++) M[base + h] = gauss1(rr3) * inv * sc;
    }
    console.log(`sup position scales a=[${[...aK].map(a => a.toFixed(2)).join(' ')}]`);
    console.log(`sup position scales a=[${[...aK].map(a => a.toFixed(2)).join(' ')}]`);
    for (let s = 0; s < P; s++){
      featPM(trainIds, spos[s], q);
      for (let j = 0; j < D; j++) X[s * D + j] = q[j] > 0 ? 1 : 0;
    }
  } else {
  for (let s = 0; s < P; s++){
    const pos = Math.floor(r() * (trainIds.length - W));
    spos[s] = pos;
    featPM(trainIds, pos, q);
    for (let j = 0; j < D; j++) X[s * D + j] = q[j] > 0 ? 1 : 0;
    next[s] = trainIds[pos + W];
  }
  }
  if (KEEP < 1){ // B2: seeded nested random prune (keep=0.25 subset of 0.5 subset of full)
    const pr = rng(SEED + 2), keep = [];
    for (let s = 0; s < P; s++) if (pr() < KEEP) keep.push(s);
    const nX = new Int8Array(keep.length * D), nN = new Uint8Array(keep.length), nS = new Int32Array(keep.length);
    keep.forEach((s, i) => { nX.set(X.subarray(s * D, s * D + D), i * D); nN[i] = next[s]; nS[i] = spos[s]; });
    X = nX; next = nN; spos = nS; P = keep.length;
    console.log(`prune keep=${KEEP} Peff=${P}`);
  }
  const buildMs = Date.now() - t0;
  console.log(`store P=${P} D=${D} bytes=${X.length + next.length} buildMs=${buildMs} rss=${rssMB()}MB`);
  const runQueries = (n, seed) => {
    const rr = rng(seed), dist_sum = { acc: 0, ce: 0, steps: 0 };
    const qq = new Float32Array(D);
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      featPM(valIds, pos, qq);
      const trueNext = valIds[pos + W];
      const { w, steps } = recallW(qq, X, P, D, runQueries.beta, 12);
      const dist = predictNext(w, next, V);
      let tot = 0; for (let i = 0; i < V; i++) tot += dist[i];
      if (argmax(dist) === trueNext) dist_sum.acc++;
      dist_sum.ce += -Math.log((dist[trueNext] + 1e-12) / (tot + 1e-12 * V));
      dist_sum.steps += steps;
    }
    flopSteps += dist_sum.steps;
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
  let best = null, bestPpl = null; // best = acc-first (legacy); bestPpl = ppl-first (H2)
  for (const beta of BETAS){
    runQueries.beta = beta;
    const m = runQueries(DEF.probeN, 1000);
    console.log(`probe beta=${beta} acc=${m.acc.toFixed(3)} ppl=${m.ppl.toFixed(2)} steps=${m.avgSteps.toFixed(1)}`);
    if (!best || m.acc > best.acc || (m.acc === best.acc && m.ppl < best.ppl)) best = { beta, ...m };
    if (!bestPpl || m.ppl < bestPpl.ppl || (m.ppl === bestPpl.ppl && m.acc > bestPpl.acc)) bestPpl = { beta, ...m };
  }
  console.log(`probe best acc beta=${best.beta} acc=${best.acc.toFixed(3)} ppl=${best.ppl.toFixed(2)} | best ppl beta=${bestPpl.beta} acc=${bestPpl.acc.toFixed(3)} ppl=${bestPpl.ppl.toFixed(2)}`);
  runQueries.beta = BSEL === 'ppl' ? bestPpl.beta : best.beta;
  const e = runQueries(TR, 2000);
  const biE = bigramEval(TR, 2000), uniE = uniEval(TR, 2000);
  const evalMs = Date.now() - t1;
  console.log(`eval beta=${best.beta} hopfield acc=${e.acc.toFixed(3)} ppl=${e.ppl.toFixed(2)} steps=${e.avgSteps.toFixed(2)} ms=${evalMs} rss=${rssMB()}MB`);
  console.log(`base bigram acc=${biE.acc.toFixed(3)} ppl=${biE.ppl.toFixed(2)} | unigram acc=${uniE.acc.toFixed(3)} ppl=${uniE.ppl.toFixed(2)}`);
  // generation quality: argmax continuations from fixed held-out prompts, both systems
  const GEN_N = noGen ? 0 : 3, GEN_LEN = 100;
  const wordSet = new Set((train.toLowerCase().match(/[a-z]+/g) || []));
  const gSeed = rng(3000), gPrompts = [];
  for (let g = 0; g < GEN_N; g++) gPrompts.push(Math.floor(gSeed() * (valIds.length - W - 1)));
  function distOf(win){
    const qq = new Float32Array(D);
    // win is an array of ids; materialize via a scratch ids buffer
    const ids = distOf.ids || (distOf.ids = new Uint16Array(4096));
    for (let k = 0; k < W; k++) ids[k] = win[k];
    featPM(ids, 0, qq);
    const { w, steps: stG } = recallW(qq, X, P, D, best.beta, 12);
    flopSteps += stG;
    const dist = predictNext(w, next, V);
    let tot = 0; for (let i = 0; i < V; i++) tot += dist[i];
    for (let i = 0; i < V; i++) dist[i] /= (tot + 1e-12 * V);
    return dist;
  }
  function genText(kind, pos){
    const win = new Array(W);
    for (let k = 0; k < W; k++) win[k] = valIds[pos + k];
    let text = '';
    for (let k = 0; k < W; k++) text += chars[win[k]];
    for (let t = 0; t < GEN_LEN; t++){
      let ni;
      if (kind === 'hop') ni = argmax(distOf(win));
      else { const row = win[W - 1] * V; let b = -1; ni = 0; for (let c = 0; c < V; c++){ if (bi[row + c] > b){ b = bi[row + c]; ni = c; } } }
      text += chars[ni];
      for (let k = 0; k < W - 1; k++) win[k] = win[k + 1];
      win[W - 1] = ni;
    }
    return text;
  }
  function qualityOf(text){
    const body = text.slice(W);
    const grams = new Set();
    let total = 0;
    for (let i = 0; i + 4 <= body.length; i++){ grams.add(body.slice(i, i + 4)); total++; }
    const words = (body.toLowerCase().match(/[a-z]+/g) || []);
    let ok = 0; for (const wd of words) if (wordSet.has(wd)) ok++;
    return { rep4: total ? 1 - grams.size / total : 0, wordValid: words.length ? ok / words.length : 0 };
  }
  const samples = [];
  let qRep = 0, qWord = 0;
  for (const pos of gPrompts){
    const h = genText('hop', pos), b = genText('bi', pos);
    const qh = qualityOf(h);
    qRep += qh.rep4; qWord += qh.wordValid;
    samples.push({ prompt: h.slice(0, W), hop: h.slice(W), bi: b.slice(W) });
  }
  const quality = { n: GEN_N, genLen: GEN_LEN, rep4: GEN_N ? qRep / GEN_N : 0, wordValid: GEN_N ? qWord / GEN_N : 0 };
  console.log(`gen rep4=${quality.rep4.toFixed(3)} wordValid=${quality.wordValid.toFixed(3)}`);
  let failN = 0;
  const check = (n, c, d) => { console.log(((c ? 'PASS' : 'FAIL') + ' ' + n + ' ' + (d || ''))); if (!c) failN++; };
  // kNN-LM prescription (Khandelwal et al 2019, "Generalization through Memorization"):
  // the paper's SOTA (-2.9 ppl, no extra training) comes from p = lam*p_kNN + (1-lam)*p_LM,
  // NOT from head-to-head comparison. We never mixed -- this rung adds the lam sweep.
  // Select lam on seed-2000 queries (same stream as the main eval above, so comparable),
  // lock it, report on fresh seed-2001 queries (honest generalization).
  function hopBiDists(n, seed){
    const rr = rng(seed), qq = new Float32Array(D);
    const rows = [];
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      featPM(valIds, pos, qq);
      const trueNext = valIds[pos + W];
      const { w, steps: stI } = recallW(qq, X, P, D, best.beta, 12);
      flopSteps += stI;
      let wmax = 0, wsum = 0; // B1: top-1 weight fraction = match confidence
      for (let i = 0; i < w.length; i++){ wsum += w[i]; if (w[i] > wmax) wmax = w[i]; }
      let ent = 1; // H5: normalized entropy of store weights (1=uniform/diffuse, 0=one-hot); same loop, no extra pass
      if (wsum > 0){ ent = 0; for (let i = 0; i < w.length; i++){ const p = w[i] / wsum; if (p > 0) ent -= p * Math.log(p); } ent /= Math.log(w.length); }
      const hd = predictNext(w, next, V);
      let tot = 0; for (let i = 0; i < V; i++) tot += hd[i];
      for (let i = 0; i < V; i++) hd[i] /= (tot + 1e-12 * V);
      const off = valIds[pos + W - 1] * V;
      let rtot = V; for (let c = 0; c < V; c++) rtot += bi[off + c];
      const bd = new Float64Array(V);
      for (let c = 0; c < V; c++) bd[c] = (bi[off + c] + 1) / rtot;
      rows.push({ hd, bd, trueNext, g: wmax / (wsum || 1), h: ent });
    }
    return rows;
  }
  function lamStats(rows){
    return LAMBDAS.map(lam => {
      let acc = 0, ce = 0;
      for (const qr of rows){
        const L = (GATE > 0 && qr.g < GATE) ? 0 : lam; // B1: no-match -> bigram only
        let bp = -1, bci = 0;
        for (let c = 0; c < V; c++){ const p = L * qr.hd[c] + (1 - L) * qr.bd[c]; if (p > bp){ bp = p; bci = c; } }
        if (bci === qr.trueNext) acc++;
        ce += -Math.log(Math.max(L * qr.hd[qr.trueNext] + (1 - L) * qr.bd[qr.trueNext], 1e-12));
      }
      return { lam, acc: acc / rows.length, ppl: Math.exp(ce / rows.length) };
    });
  }
  const selRows = hopBiDists(TR, 2000), selStats = lamStats(selRows);
  // ppl-guarded: minimize select-set ppl first (kNN-LM tunes for ppl), tie-break higher acc.
  // acc-first selection picked lam=1 overconfident stores whose ppl exploded on fresh queries.
  let bestLam = selStats[0];
  for (const s of selStats) if (s.ppl < bestLam.ppl || (s.ppl === bestLam.ppl && s.acc > bestLam.acc)) bestLam = s;
  const repRows = hopBiDists(TR, 2001), repStats = lamStats(repRows);
  const repLocked = repStats[LAMBDAS.indexOf(bestLam.lam)];
  // H4 adaptive-lambda: per-query L_i = lam * retrieval confidence (top-1 weight fraction g).
  // Known kNN-LM pathology (TO-VERIFY cites): fixed lam lets noisy retrieval hurt the tokens
  // it cannot help; adaptive lam collapses toward bigram where retrieval is diffuse.
  // Cheap: reuses hopBiDists rows, no extra recall passes.
  function adaStats(rows){
    return LAMBDAS.map(lam => {
      let acc = 0, ce = 0;
      for (const qr of rows){
        const conf = ADA === '2' ? (1 - qr.h) : qr.g; // H5 entropy mode: sharp store->lam, diffuse store->bigram
        const L = (GATE > 0 && qr.g < GATE) ? 0 : lam * conf;
        let bp = -1, bci = 0;
        for (let c = 0; c < V; c++){ const p = L * qr.hd[c] + (1 - L) * qr.bd[c]; if (p > bp){ bp = p; bci = c; } }
        if (bci === qr.trueNext) acc++;
        ce += -Math.log(Math.max(L * qr.hd[qr.trueNext] + (1 - L) * qr.bd[qr.trueNext], 1e-12));
      }
      return { lam, acc: acc / rows.length, ppl: Math.exp(ce / rows.length) };
    });
  }
  let adaLamSel = -1, adaAccSel = 0, adaPplSel = 0, adaAcc = 0, adaPpl = 0;
  if (ADA !== '0'){
    const selAda = adaStats(selRows);
    let bestAda = selAda[0];
    for (const s of selAda) if (s.ppl < bestAda.ppl || (s.ppl === bestAda.ppl && s.acc > bestAda.acc)) bestAda = s;
    const repAda = adaStats(repRows);
    const ra = repAda[LAMBDAS.indexOf(bestAda.lam)];
    adaLamSel = bestAda.lam; adaAccSel = bestAda.acc; adaPplSel = bestAda.ppl; adaAcc = ra.acc; adaPpl = ra.ppl;
    console.log(`adaptive interp select lam=${bestAda.lam} acc=${bestAda.acc.toFixed(3)} ppl=${bestAda.ppl.toFixed(2)} report acc=${ra.acc.toFixed(3)} ppl=${ra.ppl.toFixed(2)}`);
  }
  const gatedFracRep = repRows.length ? repRows.filter(qr => qr.g < GATE).length / repRows.length : 0;
  console.log(`gate tau=${GATE} pruneKeep=${KEEP} Peff=${P} gatedFrac=${gatedFracRep.toFixed(3)}`);
  console.log('interp select: ' + selStats.map(s => `lam=${s.lam} acc=${s.acc.toFixed(3)} ppl=${s.ppl.toFixed(2)}`).join(' | '));
  console.log(`interp locked lam=${bestLam.lam} report acc=${repLocked.acc.toFixed(3)} ppl=${repLocked.ppl.toFixed(2)}`);
  // pairing self-check: lam=0 must reproduce the bigram eval exactly (same queries, same Laplace)
  check('interp-lam0==bigram-select', Math.abs(selStats[0].acc - biE.acc) < 1e-9 && Math.abs(selStats[0].ppl - biE.ppl) / biE.ppl < 1e-9,
    `lam0 ${selStats[0].acc.toFixed(3)}/${selStats[0].ppl.toFixed(2)} vs bigram ${biE.acc.toFixed(3)}/${biE.ppl.toFixed(2)}`);
  // hard gate = machinery (the exact stored KEY must win, whatever its label);
  // labels are ambiguous under duplicate contexts, so patterns — not next-chars — are the check
  {
    const qp = new Float32Array(D);
    const rs = rng(999);
    let self = 0; const NS = 20;
    for (let t = 0; t < NS; t++){
      const s = Math.floor(rs() * P), pos = spos[s]; // query an ACTUAL stored key
      featPM(trainIds, pos, qp);
      const { w } = recallW(qp, X, P, D, best.beta, 1); // single step: argmax must be the key itself
      let bi = 0; for (let i = 1; i < P; i++) if (w[i] > w[bi]) bi = i;
      let same = true;
      for (let j = 0; j < D; j++){ const bit = qp[j] > 0 ? 1 : 0; if (X[bi * D + j] !== bit){ same = false; break; } }
      if (same) self++;
    }
    check('self-key-retrieval', self === NS, self + '/' + NS);
  }
  console.log(`compare hopfield acc=${e.acc.toFixed(3)} ppl=${e.ppl.toFixed(2)} vs bigram ${biE.acc.toFixed(3)}/${biE.ppl.toFixed(2)} vs unigram ${uniE.acc.toFixed(3)}/${uniE.ppl.toFixed(2)} beatUni=${e.acc > uniE.acc}`);
  const evalMflops = Math.round(flopSteps * P * D * 2 / 1e6 * 10) / 10; // 2 FLOP per MAC (dot+mix), measured steps
  const buildMflops = PROJ === 'raw' ? 0 : Math.round(P * W * H * 2 / 1e6 * 10) / 10; // feature-projection estimate (W*H adds per pattern)
  console.log(`flops steps=${flopSteps} evalMflops=${evalMflops} buildMflops=${buildMflops}`);
  // save wrapped store
  const plain = Buffer.concat([Buffer.from(X), Buffer.from(next)]);
  const sum = sha256hex(plain);
  const meta = { task: 'nextchar-heldout', corpus: tag, proj: PROJ, W, P: P0, P0, Peff: P, V, D, seed: SEED, beta: BSEL === 'ppl' ? bestPpl.beta : best.beta, betaSel: BSEL, bytes: plain.length, sha256: sum, enc: noEnc ? 'none-smoke' : 'AES-256-GCM-v1', chars: raw.length };
  writeFileSync(outDir + '/meta.json', JSON.stringify(meta));
  let encBytes = 0;
  if (!noEnc){ const wj = JSON.stringify(encryptWrap(plain, secret)); writeFileSync(outDir + '/model.bin.enc', wj); encBytes = wj.length; console.log(`enc bytes=${encBytes} sha256=${sum}`); }
  else { writeFileSync(outDir + '/model.bin', plain); console.log(`plain-smoke bytes=${plain.length} sha256=${sum}`); }
  const metrics = { ...meta, trials: TR, flips: 'n/a-nextchar', acc: e.acc, ppl: e.ppl, avgSteps: e.avgSteps, bigramAcc: biE.acc, bigramPpl: biE.ppl, unigramAcc: uniE.acc, unigramPpl: uniE.ppl, beatUni: e.acc > uniE.acc, quality, buildMs, evalMs, maxRssMB: rssMB(), encBytes, evalScope: 'heldout-val', evalMflops, buildMflops, flopSteps, betaSel: BSEL, betaAcc: best.beta, betaPpl: bestPpl.beta, ada: parseInt(ADA) || 0, adaLamSel, adaAccSel, adaPplSel, adaAcc, adaPpl };
  // kNN-LM interpolation results (locked-lam report on fresh seed-2001 queries)
  metrics.interpLambda = bestLam.lam;
  metrics.interpAccSel = bestLam.acc; metrics.interpPplSel = bestLam.ppl;
  metrics.interpAcc = repLocked.acc; metrics.interpPpl = repLocked.ppl;
  metrics.hopAccRep = repStats[LAMBDAS.length - 1].acc; metrics.hopPplRep = repStats[LAMBDAS.length - 1].ppl;
  metrics.biAccRep = repStats[0].acc; metrics.biPplRep = repStats[0].ppl;
  metrics.gateTau = GATE; metrics.pruneKeep = KEEP; metrics.Peff = P; metrics.gatedFracRep = gatedFracRep;
  writeFileSync(outDir + '/metrics.json', JSON.stringify(metrics));
  let stxt = '';
  for (const s of samples) stxt += 'PROMPT: ' + JSON.stringify(s.prompt) + '\nHOP: ' + JSON.stringify(s.hop) + '\nBI: ' + JSON.stringify(s.bi) + '\n---\n';
  writeFileSync(outDir + '/samples-' + tag + '-W' + W + '-' + PROJ + '.txt', stxt);
  if (failN){ console.log('lm-shakespeare: ' + failN + ' FAILURES'); process.exit(1); }
  console.log('lm-shakespeare: benched');
}
