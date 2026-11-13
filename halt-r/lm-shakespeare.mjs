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
let rawFastChecked = false;
function recallW(q, X, P, D, beta, maxS, sparse, rawFast, ones, Wd){
  let x = Float32Array.from(q);
  const w = new Float32Array(P);
  let steps = 0;
  for (let s = 0; s < maxS; s++){
    let m = -1e30;
    if (rawFast === 1){
      let qsum = 0; for (let j = 0; j < D; j++) qsum += x[j];
      for (let i = 0; i < P; i++){
        let part = 0; const ob = i * Wd;
        for (let k = 0; k < Wd; k++) part += x[ones[ob + k]];
        const d = (2 * part - qsum) * beta; w[i] = d; if (d > m) m = d;
      }
      if (!rawFastChecked){
        rawFastChecked = true;
        let maxDiff = 0;
        for (let i = 0; i < P; i++){
          let d = 0; const off = i * D;
          for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * x[j];
          d *= beta;
          const df = Math.abs(d - w[i]); if (df > maxDiff) maxDiff = df;
        }
        console.log('rawfast self-check maxAbsDiff=' + maxDiff);
        if (!(maxDiff < 1e-3)) fail('rawfast diverged diff=' + maxDiff);
      }
    } else {
      for (let i = 0; i < P; i++){
        let d = 0; const off = i * D;
        for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * x[j];
        d *= beta; w[i] = d; if (d > m) m = d;
      }
    }
    let t = 0;
    if (sparse === '1'){
      // entmax-1.5 (Correia et al. 2019): p_i = [(s_i - tau)]_+^2 / Z, tau via bisection.
      // Same argmax as softmax (monotone), compact support: dilutes nothing, sharpens votes.
      let mx = -1e30;
      for (let i = 0; i < P; i++) if (w[i] > mx) mx = w[i];
      let lo = mx - 4, hi = mx;
      for (let it = 0; it < 40; it++){
        const mid = (lo + hi) / 2;
        let s = 0;
        for (let i = 0; i < P; i++){ const v = w[i] - mid; if (v > 0) s += v * v; }
        if (s > 1) lo = mid; else hi = mid;
      }
      const tau = (lo + hi) / 2;
      for (let i = 0; i < P; i++){ const v = w[i] - tau; w[i] = v > 0 ? v * v : 0; t += w[i]; }
      if (!(t > 0)){ t = P; for (let i = 0; i < P; i++) w[i] = 1; } // unreachable safety (tau <= max always)
    } else {
      for (let i = 0; i < P; i++){ w[i] = Math.exp(w[i] - m); t += w[i]; }
    }
    const nx = new Float32Array(D);
    if (rawFast === 1){
      let csum = 0;
      for (let i = 0; i < P; i++) csum += w[i] / t;
      for (let i = 0; i < P; i++){ const c = (w[i] / t) * 2, ob = i * Wd; for (let k = 0; k < Wd; k++) nx[ones[ob + k]] += c; }
      for (let j = 0; j < D; j++) nx[j] -= csum;
    } else {
      for (let i = 0; i < P; i++){ const c = w[i] / t; const off = i * D; for (let j = 0; j < D; j++) nx[j] += c * (X[off + j] ? 1 : -1); }
    }
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
function normDist(d, V){ let t = 0; for (let i = 0; i < V; i++) t += d[i]; for (let i = 0; i < V; i++) d[i] /= (t + 1e-12 * V); return d; }
// transplant: read frozen donor token embeddings from a tinygpt GGUF (pure JS, zero deps, zero training)
function readGraftEmb(gpath, V){
  const buf = readFileSync(gpath);
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const dec = new TextDecoder();
  let o = 0;
  const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(o, true)); o += 8; return v; };
  const rstr = () => { const n = u64(); const s = dec.decode(buf.subarray(o, o + n)); o += n; return s; };
  const magic = u32(), ver = u32(), nt = u64(), nk = u64();
  if (magic !== 0x46554747 || ver !== 3) fail('graft: not GGUFv3 ' + gpath);
  const kvr = () => {
    const t = u32();
    if (t === 4) return u32();
    if (t === 8) return rstr();
    if (t === 9){ const et = u32(), n = u64(), a = []; for (let i = 0; i < n; i++){ if (et === 8) a.push(rstr()); else fail('graft: unsupported kv array et=' + et); } return a; }
    fail('graft: unsupported kv type ' + t);
  };
  const kv = {};
  for (let i = 0; i < nk; i++){ const k = rstr(); kv[k] = kvr(); }
  if (kv['general.architecture'] !== 'halt-tinygpt') fail('graft: arch ' + kv['general.architecture']);
  const GE = kv['halt.n_embd'];
  if (kv['halt.vocab_size'] !== V) fail(`graft: donor V ${kv['halt.vocab_size']} vs corpus V ${V}`);
  let tokOff = -1, tokDims = null;
  for (let i = 0; i < nt; i++){
    const name = rstr(), nd = u32(), dims = [];
    for (let j = 0; j < nd; j++) dims.push(u64());
    const tt = u32(), off = u64();
    if (name === 'tok.weight'){ tokOff = off; tokDims = dims; if (tt !== 0) fail('graft: tok.weight not F32'); }
  }
  if (tokOff < 0) fail('graft: tok.weight missing in ' + gpath);
  if (!(tokDims.length === 2 && tokDims[0] === GE && tokDims[1] === V)) fail('graft: tok.weight dims ' + tokDims); // stored reversed; true shape [V, E]
  const dataStart = o + ((32 - (o % 32)) % 32);
  const emb = new Float32Array(V * GE);
  for (let c = 0; c < V; c++) for (let e = 0; e < GE; e++) emb[c * GE + e] = dv.getFloat32(dataStart + tokOff + (c * GE + e) * 4, true);
  return { emb, GE };
}
function zcaWhiten(flatEmb, NV, NE){ // full-covariance whitening (zero training): out = U diag(1/sqrt(lam)) U^T (x - mu), Jacobi eigen, deterministic
  const mu = new Float64Array(NE);
  for (let c = 0; c < NV; c++) for (let e = 0; e < NE; e++) mu[e] += flatEmb[c * NE + e];
  for (let e = 0; e < NE; e++) mu[e] /= NV;
  const C = new Float64Array(NE * NE);
  for (let c = 0; c < NV; c++) for (let i = 0; i < NE; i++){ const xi = flatEmb[c * NE + i] - mu[i]; for (let j = 0; j <= i; j++) C[i * NE + j] += xi * (flatEmb[c * NE + j] - mu[j]); }
  for (let i = 0; i < NE; i++) for (let j = 0; j <= i; j++){ const v = C[i * NE + j] / NV + (i === j ? 1e-6 : 0); C[i * NE + j] = v; C[j * NE + i] = v; }
  const U = new Float64Array(NE * NE);
  for (let i = 0; i < NE; i++) U[i * NE + i] = 1;
  for (let sweep = 0; sweep < 50; sweep++){
    let off = 0;
    for (let i = 0; i < NE; i++) for (let j = i + 1; j < NE; j++) off += C[i * NE + j] * C[i * NE + j];
    if (!(off > 1e-20)) break;
    for (let p = 0; p < NE - 1; p++) for (let q = p + 1; q < NE; q++){
      const apq = C[p * NE + q];
      if (Math.abs(apq) < 1e-30) continue;
      const app = C[p * NE + p], aqq = C[q * NE + q];
      const theta = (aqq - app) / (2 * apq);
      const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < NE; k++){ const akp = C[k * NE + p], akq = C[k * NE + q]; C[k * NE + p] = c * akp - s * akq; C[k * NE + q] = s * akp + c * akq; }
      for (let k = 0; k < NE; k++){ const apk = C[p * NE + k], aqk = C[q * NE + k]; C[p * NE + k] = c * apk - s * aqk; C[q * NE + k] = s * apk + c * aqk; }
      for (let k = 0; k < NE; k++){ const ukp = U[k * NE + p], ukq = U[k * NE + q]; U[k * NE + p] = c * ukp - s * ukq; U[k * NE + q] = s * ukp + c * ukq; }
    }
  }
  const inv = new Float64Array(NE);
  for (let i = 0; i < NE; i++) inv[i] = 1 / Math.sqrt(Math.max(C[i * NE + i], 1e-12));
  const out = new Float32Array(NV * NE), y = new Float64Array(NE);
  for (let c = 0; c < NV; c++){
    for (let i = 0; i < NE; i++){ let d = 0; for (let e = 0; e < NE; e++) d += U[e * NE + i] * (flatEmb[c * NE + e] - mu[e]); y[i] = d * inv[i]; }
    for (let e = 0; e < NE; e++){ let d = 0; for (let i = 0; i < NE; i++) d += U[e * NE + i] * y[i]; out[c * NE + e] = d; }
  }
  return out;
}
function orthSquare(view, N){ // deterministic orthogonal binding basis: Gram-Schmidt on seeded Gaussian square (columns orthonormal)
  const Q = new Float32Array(view);
  for (let j = 0; j < N; j++){
    for (let i = 0; i < j; i++){ let d = 0; for (let r = 0; r < N; r++) d += Q[r * N + j] * Q[r * N + i]; for (let r = 0; r < N; r++) Q[r * N + j] -= d * Q[r * N + i]; }
    let s = 0; for (let r = 0; r < N; r++) s += Q[r * N + j] * Q[r * N + j]; s = Math.sqrt(s) || 1;
    for (let r = 0; r < N; r++) Q[r * N + j] /= s;
  }
  return Q;
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
  if (!dataPath || !outDir) fail('usage: --data <input.txt> --out <dir> [--W 32] [--P 8000] [--trials 100] [--proj raw|...|graft] [--graftgguf <tinygpt.gguf>] [--no-gen]');
  const W = parseInt(gv('W', DEF.W)), TR = parseInt(gv('trials', DEF.trials));
  let P = parseInt(gv('P', DEF.P));
  const P0 = P; // requested store size; P becomes Peff after B2 prune
  const GATE = parseFloat(gv('gate', 0)), KEEP = parseFloat(gv('pruneKeep', 1)); // B1 confidence gate, B2 store prune
  const tag = gv('tag', 'shakes');
  const PROJ = gv('proj', 'raw'), H = parseInt(gv('H', 256));
  const BSEL = gv('betaSel', 'acc');
  if (BSEL !== 'acc' && BSEL !== 'ppl') fail('usage: --betaSel acc|ppl');
  const ADA = gv('ada', '0'); // H4/H5 adaptive-lambda: 0=off(fixed lam), 1=L=lam*top1frac(g), 2=L=lam*negentropy(1-h); default 0
  const SCORE = gv('score', 'whole'); // H7: whole=window dot (default, bit-identical), maxsim=late-interaction sum_j max_k match
  if (SCORE !== 'whole' && SCORE !== 'maxsim') fail('usage: --score whole|maxsim');
  const CAPAB = gv('capab', 'none'); // C-rungs: none (default, bit-identical), canary=C1 edit + C2 exact-forget (three phases, one job)
  if (CAPAB !== 'none' && CAPAB !== 'canary') fail('usage: --capab none|canary');
  const H6 = gv('h6', '0'); // reframed consolidate-wins logistic gate (0=off bit-identical, 1=on)
  if (H6 !== '0' && H6 !== '1') fail('usage: --h6 0|1');
  const PEN = parseFloat(gv('pen', '0')); // H10 degeneration penalty (0=off bit-identical)
  if (![0, 0.5, 1, 2].includes(PEN)) fail('usage: --pen 0|0.5|1|2');
  const SPLIT = gv('split', '0'); // C5 memorized-vs-novel query split (0=off bit-identical, 1=report split)
  if (SPLIT !== '0' && SPLIT !== '1') fail('usage: --split 0|1');
  const PROV = gv('prov', '0'); // C3 provenance cites (0=off bit-identical, 1=attach top-3 store neighbors per sample)
  if (PROV !== '0' && PROV !== '1') fail('usage: --prov 0|1');
  const TOK = gv('tok', 'char'); // H12 subword pilot: char (default, bit-identical), bpe (256 learned merges on train head)
  if (TOK !== 'char' && TOK !== 'bpe') fail('usage: --tok char|bpe');
  const GRAFTGGUF = gv('graftgguf', null); // transplant: donor GGUF path (keys = frozen donor embeddings, zero training)
  if ((PROJ === 'graft' || PROJ === 'graft2' || PROJ === 'graft3' || PROJ === 'graft4') && TOK !== 'char') fail('usage: --proj graft needs --tok char (donor V is char-level)');
  if ((PROJ === 'graft' || PROJ === 'graft2' || PROJ === 'graft3' || PROJ === 'graft4') && !GRAFTGGUF) fail('usage: --proj graft needs --graftgguf <tinygpt.gguf>');
  const MIX = gv('mix', 'none'); // gmix: none (default, bit-identical), gpt (mix hop + tiny-GPT dists from q100-gpt file, joined on pos)
  if (MIX !== 'none' && MIX !== 'gpt') fail('usage: --mix none|gpt');
  const GPTQ = gv('gptq', null);
  const SPARSE = gv('sparse', '0'); // S-rung: 0=softmax (default, bit-identical), 1=entmax-1.5 sparse assignment (same argmax, 100x sparser support; prototype: identical convergence/acc, support 253->2)
  if (SPARSE !== '0' && SPARSE !== '1') fail('usage: --sparse 0|1');
  const RAWFAST = gv('rawfast', '0'); // raw sparse-dot fast path: 0=off (default, bit-identical), 1=O(P·W) dots for raw one-hot rows (fp-equivalent, self-checked)
  if (RAWFAST !== '0' && RAWFAST !== '1') fail('usage: --rawfast 0|1');
  const HEDGE = gv('hedge', '0'); // Hedge/experts adaptive lambda: 0=off (default, bit-identical), 1=online Hedge over the LAMBDAS grid (frozen on select stream, evaluated on report stream; regret-bounded mixing)
  if (HEDGE !== '0' && HEDGE !== '1') fail('usage: --hedge 0|1');
  const HYB = gv('hyb', '0'); // H8 rare-exact hybrid: 0=off (default, bit-identical), 1=mix dense dots + rare-char exact-match (alpha grid 0.25/0.5, select/lock discipline)
  if (HYB !== '0' && HYB !== '1') fail('usage: --hyb 0|1');
  const PK = gv('pk', '0'); // H9 leaders-index: 0=dense scan (default, bit-identical), 1=route via 256 seeded leaders + top-2 member rerank (sublinear eval MACs)
  if (PK !== '0' && PK !== '1') fail('usage: --pk 0|1');
  const TMP = gv('tmp', '0'); // H11 post-hoc temperature on the MIXED dist: 0=off (default, bit-identical), 1=fit T on select, locked report
  if (TMP !== '0' && TMP !== '1') fail('usage: --tmp 0|1');
  const CONF = gv('conf', '0'); // conformal abstention gate (alpha=0.1): 0=off (default, bit-identical), 1=calibrate non-answer rate on select, abstain->bigram on report
  if (CONF !== '0' && CONF !== '1') fail('usage: --conf 0|1');
  const ORF = gv('orf', '0'); // ORF variance cut: 0=iid Gaussian M (default, bit-identical), 1=Gram-Schmidt orthogonalized feature directions (norms preserved)
  if (ORF !== '0' && ORF !== '1') fail('usage: --orf 0|1');
  const C2D = gv('c2d', '0'); // C2 per-implant detail: 0=off (default, bit-identical), 1=emit per-implant remember/edit bit strings
  if (C2D !== '0' && C2D !== '1') fail('usage: --c2d 0|1');
  const DTW = gv('dtw', '0'); // DTW rerank: 0=off (default, bit-identical), 1=top-50 weights reranked by DTW dist (0/1 frame cost, tau=W fixed; screen 2026-11-13: shift 0.74, rev 0.00)
  if (DTW !== '0' && DTW !== '1') fail('usage: --dtw 0|1');
  const SMP = parseFloat(gv('smp', '0')); // E1 energy sampling: 0=off (default, bit-identical argmax/pen path), else sampling temperature on the hop dist
  if (![0, 0.7, 1.0, 1.3].includes(SMP)) fail('usage: --smp 0|0.7|1|1.3');
  const kei = a.indexOf('--key-env');
  const secret = process.env[kei >= 0 ? a[kei + 1] : 'SCRAMBLE'];
  if (!secret && !noEnc) fail('SCRAMBLE key missing (refuses plaintext save; --no-encrypt for smoke only)');
  mkdirSync(outDir, { recursive: true });
  const raw = readFileSync(dataPath, 'utf8');
  const cut = Math.floor(raw.length * 0.9);
  const train = raw.slice(0, cut), val = raw.slice(cut);
  let { chars, toI, V } = buildVocab(train);
  // H12 BPE pilot (JS-only, zero new deps, deterministic): learn 256 merges on the
  // train head (500K chars, representative + 6x cheaper), encode full train/val.
  // Primary verdict is internal (W16 lam* + interp vs base under the SAME tokenizer),
  // so id-level ppl needs no cross-tokenizer BPC conversion in this pilot.
  function learnBPE(sample, nMerges) {
    const ch = [...new Set(sample)].sort();
    const vocab = [...ch], pairRank = new Map();
    let toks = [...sample].map(c => ch.indexOf(c));
    const merges = [];
    for (let m = 0; m < nMerges; m++) {
      const counts = new Map(), firstSeen = new Map();
      let order = 0;
      for (let i = 0; i + 1 < toks.length; i++) {
        const k = toks[i] * 1000003 + toks[i + 1];
        if (!counts.has(k)) { counts.set(k, 0); firstSeen.set(k, order++); }
        counts.set(k, counts.get(k) + 1);
      }
      let bk = -1, bc = 1;
      for (const [k, c] of counts) {
        if (c > bc || (c === bc && firstSeen.get(k) < firstSeen.get(bk))) { bc = c; bk = k; }
      }
      if (bk < 0) break;
      const a = Math.floor(bk / 1000003), b = bk % 1000003;
      const id = vocab.length;
      vocab.push(vocab[a] + vocab[b]);
      merges.push([a, b, id]);
      const nt = [];
      for (let i = 0; i < toks.length; i++) {
        if (i + 1 < toks.length && toks[i] === a && toks[i + 1] === b) { nt.push(id); i++; }
        else nt.push(toks[i]);
      }
      toks = nt;
    }
    return { vocab, merges };
  }
  function bpeEncode(text, ch, merges) {
    const ci = new Map();
    ch.forEach((c, i) => { ci.set(c, i); });
    let toks = [...text].map(c => (ci.has(c) ? ci.get(c) : 0));
    for (const [a, b, id] of merges) {
      const nt = [];
      for (let i = 0; i < toks.length; i++) {
        if (i + 1 < toks.length && toks[i] === a && toks[i + 1] === b) { nt.push(id); i++; }
        else nt.push(toks[i]);
      }
      toks = nt;
    }
    return toks;
  }
  let bpeMerges = 0, bpeMeanLen = 0;
  let trainIds = null, valIds = null;
  if (TOK === 'bpe') {
    const bt0 = Date.now();
    const { vocab, merges } = learnBPE(train.slice(0, 500000), 256);
    bpeMerges = merges.length;
    const ch0 = [...new Set(train)].sort();
    const tIds = bpeEncode(train, ch0, merges), vIds = bpeEncode(val, ch0, merges);
    let tl = 0;
    for (const id of tIds) tl += vocab[id].length;
    bpeMeanLen = tl / tIds.length;
    console.log(`bpe merges=${bpeMerges} V=${vocab.length} meanTokLen=${bpeMeanLen.toFixed(2)} learnMs=${Date.now() - bt0}`);
    chars = vocab; V = vocab.length;
    trainIds = Uint32Array.from(tIds); valIds = Uint32Array.from(vIds);
  } else {
    trainIds = new Uint16Array(train.length);
    for (let i = 0; i < train.length; i++) trainIds[i] = toI[train[i]];
    valIds = new Uint16Array(val.length);
    for (let i = 0; i < val.length; i++) valIds[i] = toI[val[i]] !== undefined ? toI[val[i]] : 0;
  }
  if (V > 2000) fail('vocab blowup V=' + V);
  console.log(`corpus chars=${raw.length} train=${train.length} val=${val.length} V=${V} W=${W} P=${P} proj=${PROJ}${PROJ === 'raw' ? '' : ' H=' + H} score=${SCORE} tok=${TOK}${TOK === 'bpe' ? ' merges=' + bpeMerges : ''}`);
  // baselines on full train
  const uni = new Float64Array(V);
  const tFreq = new Uint32Array(V); // H8: rare-char set = bottom-quartile train frequency (exact-match rescue signal)
  for (let i = 0; i < trainIds.length; i++){ uni[trainIds[i]]++; tFreq[trainIds[i]]++; }
  const rareC = new Uint8Array(V);
  { const ord = [...Array(V).keys()].sort((a, b) => tFreq[a] - tFreq[b]); for (let i = 0; i < Math.max(1, Math.floor(V / 4)); i++) rareC[ord[i]] = 1; }
  for (let i = 0; i < V; i++) uni[i] /= trainIds.length;
  const uniPred = argmax(uni);
  const bi = new Float64Array(V * V);
  for (let i = 0; i + 1 < trainIds.length; i++) bi[trainIds[i] * V + trainIds[i + 1]]++;
  let gptMap = null; // gmix: pos -> {2000: probs, 2001: probs} from the tiny-GPT q100 streams
  if (MIX === 'gpt') {
    if (!GPTQ) fail('usage: --mix gpt needs --gptq <q100-gpt-tag.json>');
    const gj = JSON.parse(readFileSync(GPTQ, 'utf8'));
    if (gj.V !== V) fail('gptq V mismatch ' + gj.V + ' vs ' + V + ' (same corpus required)');
    gptMap = new Map();
    for (const s of ['2000', '2001']) {
      const st = gj.streams[s];
      if (!st) fail('gptq missing stream ' + s);
      for (const q of st.queries) {
        if (!gptMap.has(q.pos)) gptMap.set(q.pos, {});
        gptMap.get(q.pos)[s] = q.probs;
      }
    }
    console.log(`gmix gptq streams=2000,2001 entries=${gptMap.size}`);
  }
  // store: seeded sampled train contexts (raw one-hot or SimHash-projected bits)
  const t0 = Date.now();
  const Din = W * V;
  let graftEmb = null, GE = 0, graftWhite = null, graftBind = null; // transplant: frozen donor token rows (zero training)
  if (PROJ === 'graft' || PROJ === 'graft2' || PROJ === 'graft3' || PROJ === 'graft4'){
    const g = readGraftEmb(GRAFTGGUF, V);
    graftEmb = g.emb; GE = g.GE;
    console.log(`graft donor=${GRAFTGGUF} V=${V} E=${GE} D=${W * GE}`);
  }
  const D = PROJ === 'raw' ? Din : (PROJ === 'graft3' ? V : (((PROJ === 'graft' || PROJ === 'graft2') || PROJ === 'graft4') ? GE : H));
  const FW = PROJ === 'graft3' ? V : ((PROJ === 'graft' || PROJ === 'graft2' || PROJ === 'graft4') ? GE : H); // feature width: graft keys live in donor space (graft3: anchor space)
  function gauss1(rr){ let u = 0, v = 0; while (!u) u = rr(); while (!v) v = rr(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  let M = null; // projection matrix (sim: random Gaussian; sup: supervised from store labels)
  function orthCols(Mm, rows, cols){ // ORF: Gram-Schmidt on column (feature) directions, per-column norms preserved (scales survive rotation)
    const n0 = new Float64Array(cols);
    for (let h = 0; h < cols; h++){ let s = 0; for (let j = 0; j < rows; j++) s += Mm[j * cols + h] * Mm[j * cols + h]; n0[h] = Math.sqrt(s) || 1; }
    for (let h = 0; h < cols; h++){
      for (let g = 0; g < h; g++){ let d = 0; for (let j = 0; j < rows; j++) d += Mm[j * cols + h] * Mm[j * cols + g]; for (let j = 0; j < rows; j++) Mm[j * cols + h] -= d * Mm[j * cols + g]; }
      let s = 0; for (let j = 0; j < rows; j++) s += Mm[j * cols + h] * Mm[j * cols + h]; s = Math.sqrt(s) || 1;
      for (let j = 0; j < rows; j++) Mm[j * cols + h] *= n0[h] / s;
    }
  }
  if (PROJ === 'sim'){ // deterministic random projection (own rng stream); sign bits = SimHash
    M = new Float32Array(Din * H);
    const rr2 = rng(SEED + 1), inv = Math.sqrt(1 / Din);
    for (let i = 0; i < M.length; i++) M[i] = gauss1(rr2) * inv;
    if (ORF === '1') orthCols(M, Din, H);
  }
  if (PROJ === 'graft'){ // transplant keys: frozen donor rows, shared across positions (order enters via the sum)
    M = new Float32Array(Din * GE);
    for (let k = 0; k < W; k++) for (let c = 0; c < V; c++){
      const base = (k * V + c) * GE, src = c * GE;
      for (let e = 0; e < GE; e++) M[base + e] = graftEmb[src + e];
    }
  }
  if (PROJ === 'graft2'){ // aligned transplant: whiten donor space + bind positions (zero training, deterministic)
    const mu = new Float64Array(GE), sd = new Float64Array(GE);
    for (let c = 0; c < V; c++) for (let e = 0; e < GE; e++) mu[e] += graftEmb[c * GE + e];
    for (let e = 0; e < GE; e++) mu[e] /= V;
    for (let c = 0; c < V; c++) for (let e = 0; e < GE; e++){ const d = graftEmb[c * GE + e] - mu[e]; sd[e] += d * d; }
    for (let e = 0; e < GE; e++) sd[e] = Math.sqrt(sd[e] / V + 1e-12) || 1;
    graftWhite = new Float32Array(V * GE);
    for (let c = 0; c < V; c++) for (let e = 0; e < GE; e++) graftWhite[c * GE + e] = (graftEmb[c * GE + e] - mu[e]) / sd[e];
    const br = rng(SEED + 9);
    graftBind = new Float32Array(W * GE);
    for (let k = 0; k < W; k++) for (let e = 0; e < GE; e++) graftBind[k * GE + e] = br() < 0.5 ? 1 : -1;
    M = new Float32Array(Din * GE);
    for (let k = 0; k < W; k++) for (let c = 0; c < V; c++){
      const base = (k * V + c) * GE, src = c * GE;
      for (let e = 0; e < GE; e++) M[base + e] = graftWhite[src + e] * graftBind[k * GE + e];
    }
    console.log('graft2 whiten+bind ready V=' + V + ' E=' + GE + ' W=' + W);
  }
  if (PROJ === 'graft3'){ // relative transplant: cosine-to-anchors rows, centered, bound per position (zero training, translation-free)
    const nn = new Float64Array(V);
    for (let c = 0; c < V; c++){ let s = 0; for (let e = 0; e < GE; e++) s += graftEmb[c * GE + e] * graftEmb[c * GE + e]; nn[c] = Math.sqrt(s) + 1e-12; }
    const rel = new Float32Array(V * V);
    for (let c = 0; c < V; c++) for (let a = 0; a < V; a++){ let d = 0; for (let e = 0; e < GE; e++) d += graftEmb[c * GE + e] * graftEmb[a * GE + e]; rel[c * V + a] = d / (nn[c] * nn[a]); }
    const rm = new Float64Array(V);
    for (let c = 0; c < V; c++) for (let a = 0; a < V; a++) rm[a] += rel[c * V + a];
    for (let a = 0; a < V; a++) rm[a] /= V;
    const br3 = rng(SEED + 11);
    const bind3 = new Float32Array(W * V);
    for (let k = 0; k < W; k++) for (let a = 0; a < V; a++) bind3[k * V + a] = br3() < 0.5 ? 1 : -1;
    M = new Float32Array(Din * V);
    for (let k = 0; k < W; k++) for (let c = 0; c < V; c++){
      const base = (k * V + c) * V;
      for (let a = 0; a < V; a++) M[base + a] = (rel[c * V + a] - rm[a]) * bind3[k * V + a];
    }
    console.log('graft3 relative ready V=' + V + ' E=' + GE + ' W=' + W);
  }
  if (PROJ === 'graft4'){ // procrustes-side transplant: full ZCA + orthogonal position binding (zero training, deterministic)
    const zw = zcaWhiten(graftEmb, V, GE);
    const rr4 = rng(SEED + 12);
    const G = new Float32Array(W * GE * GE);
    for (let i = 0; i < G.length; i++){ let u = 0, v = 0; while (!u) u = rr4(); while (!v) v = rr4(); G[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
    M = new Float32Array(Din * GE);
    const tmp = new Float32Array(GE);
    for (let k = 0; k < W; k++){
      const Rk = orthSquare(G.subarray(k * GE * GE, (k + 1) * GE * GE), GE);
      for (let c = 0; c < V; c++){
        const base = (k * V + c) * GE, src = c * GE;
        for (let e = 0; e < GE; e++){ let d = 0; for (let j = 0; j < GE; j++) d += Rk[e * GE + j] * zw[src + j]; tmp[e] = d; }
        for (let e = 0; e < GE; e++) M[base + e] = tmp[e];
      }
    }
    console.log('graft4 zca+orth ready V=' + V + ' E=' + GE + ' W=' + W);
  }
  function featPM(ids, pos, out){ // ±1 features, length D (every projected PROJ incl. graft shares the path)
    if (PROJ !== 'raw'){
      for (let h = 0; h < FW; h++) out[h] = 0;
      for (let k = 0; k < W; k++){
        const base = (k * V + ids[pos + k]) * FW;
        for (let h = 0; h < FW; h++) out[h] += M[base + h];
      }
      for (let h = 0; h < FW; h++) out[h] = out[h] >= 0 ? 1 : -1;
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
  let encLossFirst = 0, encLossLast = 0; // trunk contrastive metrics (0 unless proj=enc)
  if (PROJ === 'sup' || PROJ === 'sup2'){ // sup2: per-(position,char) scales (strict generalization of sup)
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
    let A2 = null; // sup2: per-(position,char) predictive scales (W*V params, seconds, same code path)
    if (PROJ === 'sup2'){
      A2 = new Float64Array(W * V);
      for (let k = 0; k < W; k++){
        const groups = new Map();
        for (let s = 0; s < P; s++){ const c = trainIds[spos[s] + k], n = next[s]; let g = groups.get(c); if (!g){ g = new Uint32Array(V); groups.set(c, g); } g[n]++; }
        for (let c = 0; c < V; c++){
          const g = groups.get(c);
          if (!g){ A2[k * V + c] = 0; continue; }
          let b = 0; for (let cc = 1; cc < V; cc++) if (g[cc] > g[b]) b = cc;
          let correct = 0, tot = 0;
          for (let s = 0; s < P; s++){ if (trainIds[spos[s] + k] === c){ tot++; if (b === next[s]) correct++; } }
          A2[k * V + c] = tot ? Math.max(0, correct / tot - 1 / V) : 0;
        }
      }
      let m2 = 0; for (let i = 0; i < A2.length; i++) m2 += A2[i]; m2 /= A2.length;
      if (m2 > 1e-9){ for (let i = 0; i < A2.length; i++) A2[i] /= m2; } else { for (let i = 0; i < A2.length; i++) A2[i] = 1; }
      console.log('sup2 scales computed mean-normed');
    }
    const rr3 = rng(SEED + 3), inv = Math.sqrt(1 / Din);
    M = new Float32Array(Din * H);
    for (let k = 0; k < W; k++) for (let c = 0; c < V; c++){
      const sc = PROJ === 'sup2' ? A2[k * V + c] : aK[k], base = (k * V + c) * H;
      for (let h = 0; h < H; h++) M[base + h] = gauss1(rr3) * inv * sc;
    }
    if (ORF === '1') orthCols(M, Din, H);
    console.log(`sup position scales a=[${[...aK].map(a => a.toFixed(2)).join(' ')}]`);
    console.log(`sup position scales a=[${[...aK].map(a => a.toFixed(2)).join(' ')}]`);
    for (let s = 0; s < P; s++){
      featPM(trainIds, spos[s], q);
      for (let j = 0; j < D; j++) X[s * D + j] = q[j] > 0 ? 1 : 0;
    }
  } else if (PROJ === 'enc'){ // TRUNK: contrastive-learned encoder (JS, CPU-cheap, zero python)
    // Same store positions as sup (SEED stream) for comparability; M starts from the
    // sim distribution (SEED+3 stream) and moves by InfoNCE SGD (SEED+4 batches):
    // same-next-char contexts attract, batch peers repel (temp 0.2). Sign bits after
    // training keep the EXACT sim code path (featPM identical).
    for (let s = 0; s < P; s++){
      const pos = Math.floor(r() * (trainIds.length - W));
      spos[s] = pos;
      next[s] = trainIds[pos + W];
    }
    const rrE = rng(SEED + 3), invE = Math.sqrt(1 / Din);
    M = new Float32Array(Din * H);
    for (let i = 0; i < M.length; i++) M[i] = gauss1(rrE) * invE;
    const EB = 128, ESTEPS = 100, ELR = 0.05, ET = 0.2;
    const br = rng(SEED + 4);
    const Z = new Float32Array(EB * H), G = new Float32Array(EB * H), Zn = new Float32Array(EB);
    const eidx = new Uint32Array(EB);
    for (let st = 0; st < ESTEPS; st++){
      for (let i = 0; i < EB; i++) eidx[i] = Math.floor(br() * P);
      for (let i = 0; i < EB; i++){
        for (let h = 0; h < H; h++) Z[i * H + h] = 0;
        for (let k = 0; k < W; k++){
          const base = (k * V + trainIds[spos[eidx[i]] + k]) * H;
          for (let h = 0; h < H; h++) Z[i * H + h] += M[base + h];
        }
        let s2 = 0;
        for (let h = 0; h < H; h++) s2 += Z[i * H + h] * Z[i * H + h];
        Zn[i] = Math.sqrt(s2) + 1e-9;
      }
      G.fill(0);
      let loss = 0, pairs = 0;
      for (let i = 0; i < EB; i++){
        let m = -1e30;
        const sims = new Float32Array(EB);
        for (let j = 0; j < EB; j++){
          if (j === i) continue;
          let d = 0;
          for (let h = 0; h < H; h++) d += Z[i * H + h] * Z[j * H + h];
          sims[j] = d / (Zn[i] * Zn[j] * ET);
          if (sims[j] > m) m = sims[j];
        }
        let t = 0;
        const p = new Float32Array(EB);
        for (let j = 0; j < EB; j++){ if (j === i) continue; p[j] = Math.exp(sims[j] - m); t += p[j]; }
        if (!(t > 0)) continue;
        for (let j = 0; j < EB; j++){ if (j === i) continue; p[j] /= t; }
        let npos = 0;
        for (let j = 0; j < EB; j++) if (j !== i && next[eidx[j]] === next[eidx[i]]) npos++;
        if (!npos) continue;
        pairs++;
        for (let j = 0; j < EB; j++){
          if (j === i) continue;
          const y = (next[eidx[j]] === next[eidx[i]]) ? 1 / npos : 0;
          loss += -y * Math.log(p[j] + 1e-12);
          const dd = (p[j] - y) / (Zn[i] * Zn[j] * ET);
          for (let h = 0; h < H; h++){ G[i * H + h] += dd * Z[j * H + h]; G[j * H + h] += dd * Z[i * H + h]; }
        }
      }
      if (st === 0) encLossFirst = loss / Math.max(1, pairs);
      if (st === ESTEPS - 1) encLossLast = loss / Math.max(1, pairs);
      for (let i = 0; i < EB; i++) for (let k = 0; k < W; k++){
        const base = (k * V + trainIds[spos[eidx[i]] + k]) * H;
        for (let h = 0; h < H; h++) M[base + h] -= ELR * G[i * H + h];
      }
    }
    if (ORF === '1') orthCols(M, Din, H); // ORF applies to the trained encoder too (geometry change, same code path)
    console.log(`enc contrastive steps=${ESTEPS} loss ${encLossFirst.toFixed(3)}->${encLossLast.toFixed(3)}`);
    for (let s = 0; s < P; s++){
      featPM(trainIds, spos[s], q);
      for (let j = 0; j < D; j++) X[s * D + j] = q[j] > 0 ? 1 : 0;
    }
  } else if (PROJ === 'enc2'){ // TRUNK v2 (decision 2026-11-13): supervised linear-probe training.
    // Same store positions as sup/enc (SEED stream) for comparability; M starts from the
    // sim distribution (SEED+3). Probe U (V x H, SEED+6): z=sum_k M[k,c_k], logits=U z,
    // CE SGD batch-128, ~300 steps, CPU-cheap, JS-only. Discards U; keeps M; sign bits
    // via the EXACT sim code path (featPM identical). Rationale: contrastive same-next
    // positives are noisy; the probe optimizes the ppl-relevant signal directly, and lets
    // token rows share statistical strength across repeating units (BPE-compose: run enc2
    // under --tok bpe too). Default-off bit-identical (old PROJ paths untouched).
    for (let s = 0; s < P; s++){
      const pos = Math.floor(r() * (trainIds.length - W));
      spos[s] = pos;
      next[s] = trainIds[pos + W];
    }
    const rrE2 = rng(SEED + 3), invE2 = Math.sqrt(1 / Din);
    M = new Float32Array(Din * H);
    for (let i = 0; i < M.length; i++) M[i] = gauss1(rrE2) * invE2;
    const rrU = rng(SEED + 6), invU = Math.sqrt(1 / H);
    const U = new Float32Array(V * H);
    for (let i = 0; i < U.length; i++) U[i] = gauss1(rrU) * invU;
    const EB2 = 128, ESTEPS2 = 300, ELR2 = 0.05;
    const br2 = rng(SEED + 4);
    const eidx2 = new Uint32Array(EB2);
    const z2 = new Float32Array(EB2 * H), lg2 = new Float32Array(EB2 * V);
    for (let st = 0; st < ESTEPS2; st++){
      for (let i = 0; i < EB2; i++) eidx2[i] = Math.floor(br2() * P);
      for (let i = 0; i < EB2; i++){
        for (let h = 0; h < H; h++) z2[i * H + h] = 0;
        for (let k = 0; k < W; k++){
          const base = (k * V + trainIds[spos[eidx2[i]] + k]) * H;
          for (let h = 0; h < H; h++) z2[i * H + h] += M[base + h];
        }
      }
      let loss = 0;
      const gz = new Float32Array(EB2 * H);
      for (let i = 0; i < EB2; i++){
        let m = -1e30;
        for (let c = 0; c < V; c++){ let d = 0; for (let h = 0; h < H; h++) d += U[c * H + h] * z2[i * H + h]; lg2[i * V + c] = d; if (d > m) m = d; }
        let t = 0;
        for (let c = 0; c < V; c++){ const e = Math.exp(lg2[i * V + c] - m); lg2[i * V + c] = e; t += e; }
        if (!(t > 0)) continue;
        const y = next[eidx2[i]];
        loss += -Math.log(Math.max(lg2[i * V + y] / t, 1e-12));
        for (let c = 0; c < V; c++){
          const dl = (lg2[i * V + c] / t) - (c === y ? 1 : 0);
          for (let h = 0; h < H; h++){ gz[i * H + h] += U[c * H + h] * dl; U[c * H + h] -= ELR2 * dl * z2[i * H + h] / EB2; }
        }
      }
      loss /= EB2;
      if (st === 0) encLossFirst = loss;
      if (st === ESTEPS2 - 1) encLossLast = loss;
      for (let i = 0; i < EB2; i++) for (let k = 0; k < W; k++){
        const base = (k * V + trainIds[spos[eidx2[i]] + k]) * H;
        for (let h = 0; h < H; h++) M[base + h] -= ELR2 * gz[i * H + h] / EB2;
      }
    }
    if (ORF === '1') orthCols(M, Din, H);
    console.log(`enc2 supervised steps=${ESTEPS2} ce ${encLossFirst.toFixed(3)}->${encLossLast.toFixed(3)}`);
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
  let onesList = null; // rawfast: per-row one-hot positions (W per row), built post-prune
  if (RAWFAST === '1' && PROJ === 'raw'){
    onesList = new Int32Array(P * W);
    for (let s = 0; s < P; s++){ let k = 0; for (let j = 0; j < D && k < W; j++) if (X[s * D + j]) onesList[s * W + k++] = j; if (k !== W) fail('rawfast: row ' + s + ' has ' + k + ' ones'); }
    console.log('rawfast ones built P=' + P + ' W=' + W);
  }
  const buildMs = Date.now() - t0;
  console.log(`store P=${P} D=${D} bytes=${X.length + next.length} buildMs=${buildMs} rss=${rssMB()}MB`);
  // H9 leaders-index (sublinear eval): 256 seeded leaders, one assignment pass at build.
  // Eval = leaders (L*D) + top-2 leader members (~2P/L full dots). Build cost is one-time; eval is the series.
  const PK_L = 256;
  let pkLead = null, pkMembers = null, pkMacs = 0;
  if (PK === '1'){
    const tPk = Date.now(), lr5 = rng(SEED + 5);
    pkLead = new Int32Array(PK_L);
    for (let l = 0; l < PK_L; l++) pkLead[l] = Math.floor(lr5() * P);
    const asg = new Uint16Array(P), mem = Array.from({length: PK_L}, () => []);
    const lv = new Float32Array(D);
    for (let s = 0; s < P; s++){
      let bl = 0, bld = -1e18;
      for (let l = 0; l < PK_L; l++){
        const li = pkLead[l]; let d = 0; const off = li * D;
        for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * (X[s * D + j] ? 1 : -1);
        if (d > bld){ bld = d; bl = l; }
      }
      asg[s] = bl; mem[bl].push(s);
    }
    pkMembers = mem;
    console.log(`pk leaders=${PK_L} buildMs=${Date.now() - tPk} rss=${rssMB()}MB`);
  }
  let msOps = 0; // H7 char-compare ops under maxsim scoring (cost axis, cf flopSteps)
  function maxsimW(qIds, qpos, beta){
    const sc = new Float32Array(P);
    let m = -1e30;
    for (let s = 0; s < P; s++){
      let sum = 0; const base = spos[s];
      for (let j = 0; j < W; j++){
        const qc = qIds[qpos + j];
        for (let k = 0; k < W; k++){ if (trainIds[base + k] === qc){ sum++; break; } }
      }
      sc[s] = sum; if (sum > m) m = sum;
    }
    const w = new Float32Array(P);
    for (let s = 0; s < P; s++) w[s] = Math.exp(beta * (sc[s] - m));
    msOps += P * W * W;
    return { w, steps: 1 };
  }
  let hybAlpha = 0.25; // H8: locked alpha for the hybrid arm (grid-selected on select stream)
  function hybRecall(qq, qIds, qpos, alpha, beta, maxS){
    // H8: dense dots + static rare-exact term E (fraction of query rare positions matched, scaled by D to sit in dot units)
    let nR = 0;
    for (let k = 0; k < W; k++) if (rareC[qIds[qpos + k]]) nR++;
    const E = new Float32Array(P);
    if (nR > 0) for (let s = 0; s < P; s++){ let m = 0; const base = spos[s]; for (let k = 0; k < W; k++) if (rareC[qIds[qpos + k]] && trainIds[base + k] === qIds[qpos + k]) m++; E[s] = m / nR; msOps += W; }
    let x = Float32Array.from(qq);
    const w = new Float32Array(P);
    let steps = 0;
    for (let s = 0; s < maxS; s++){
      let m = -1e30;
      for (let i = 0; i < P; i++){
        let d = 0; const off = i * D;
        for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * x[j];
        d = beta * (d + alpha * E[i] * D); w[i] = d; if (d > m) m = d;
      }
      let t = 0;
      if (SPARSE === '1'){
        let mx = -1e30;
        for (let i = 0; i < P; i++) if (w[i] > mx) mx = w[i];
        let lo = mx - 4, hi = mx;
        for (let it = 0; it < 40; it++){ const mid = (lo + hi) / 2; let sm = 0; for (let i = 0; i < P; i++){ const v = w[i] - mid; if (v > 0) sm += v * v; } if (sm > 1) lo = mid; else hi = mid; }
        const tau = (lo + hi) / 2;
        for (let i = 0; i < P; i++){ const v = w[i] - tau; w[i] = v > 0 ? v * v : 0; t += w[i]; }
        if (!(t > 0)){ t = P; for (let i = 0; i < P; i++) w[i] = 1; }
      } else {
        for (let i = 0; i < P; i++){ w[i] = Math.exp(w[i] - m); t += w[i]; }
      }
      const nx = new Float32Array(D);
      for (let i = 0; i < P; i++){ const c = w[i] / t; const off = i * D; for (let j = 0; j < D; j++) nx[j] += c * (X[off + j] ? 1 : -1); }
      let maxd = 0;
      for (let j = 0; j < D; j++){ const dd = Math.abs(nx[j] - x[j]); if (dd > maxd) maxd = dd; }
      x = nx; steps++;
      if (maxd < 1e-3) break;
    }
    return { w, steps };
  }
  function pkRecall(qq, beta, maxS){
    // H9: route via leaders, iterate only over top-2 leaders' members (sublinear eval MACs, counted in pkMacs)
    const ls = new Array(PK_L);
    for (let l = 0; l < PK_L; l++){ const li = pkLead[l]; let d = 0; const off = li * D; for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * qq[j]; ls[l] = [d, l]; }
    pkMacs += PK_L * D;
    ls.sort((a, b) => b[0] - a[0]);
    const cand = [];
    for (let k = 0; k < 2; k++) for (const i of pkMembers[ls[k][1]]) cand.push(i);
    let x = Float32Array.from(qq);
    const w = new Float32Array(P);
    let steps = 0;
    for (let s = 0; s < maxS; s++){
      let m = -1e30;
      for (const i of cand){ let d = 0; const off = i * D; for (let j = 0; j < D; j++) d += (X[off + j] ? 1 : -1) * x[j]; d *= beta; w[i] = d; if (d > m) m = d; }
      pkMacs += cand.length * D;
      let t = 0;
      if (SPARSE === '1'){
        let mx = -1e30;
        for (const i of cand) if (w[i] > mx) mx = w[i];
        let lo = mx - 4, hi = mx;
        for (let it = 0; it < 40; it++){ const mid = (lo + hi) / 2; let sm = 0; for (const i of cand){ const v = w[i] - mid; if (v > 0) sm += v * v; } if (sm > 1) lo = mid; else hi = mid; }
        const tau = (lo + hi) / 2;
        for (const i of cand){ const v = w[i] - tau; w[i] = v > 0 ? v * v : 0; t += w[i]; }
        if (!(t > 0)){ t = cand.length; for (const i of cand) w[i] = 1; }
      } else {
        for (const i of cand){ w[i] = Math.exp(w[i] - m); t += w[i]; }
      }
      const nx = new Float32Array(D);
      for (const i of cand){ const c = w[i] / t; const off = i * D; for (let j = 0; j < D; j++) nx[j] += c * (X[off + j] ? 1 : -1); }
      let maxd = 0;
      for (let j = 0; j < D; j++){ const dd = Math.abs(nx[j] - x[j]); if (dd > maxd) maxd = dd; }
      x = nx; steps++;
      if (maxd < 1e-3) break;
    }
    return { w, steps };
  }
  function scoreW(qq, qIds, qpos, beta, maxS){
    if (SCORE === 'maxsim') return maxsimW(qIds, qpos, beta);
    if (PK === '1') return pkRecall(qq, beta, maxS);
    if (HYB === '1') return hybRecall(qq, qIds, qpos, hybAlpha, beta, maxS);
    return recallW(qq, X, P, D, beta, maxS, SPARSE, (RAWFAST === '1' && PROJ === 'raw') ? 1 : 0, onesList, W);
  }
  let trainWinSet = null; // C5: memorized-vs-novel split (descriptive, no lam mix)
  function wkey(ids, p){
    let h1 = 0, h2 = 0;
    for (let k = 0; k < W; k++){ h1 = (h1 * 131 + ids[p + k]) % 1000000007; h2 = (h2 * 137 + ids[p + k]) % 1000000009; }
    return h1 + ':' + h2;
  }
  function splitEval(n, seed){
    if (!trainWinSet){
      trainWinSet = new Set();
      for (let p = 0; p + W <= trainIds.length; p++) trainWinSet.add(wkey(trainIds, p));
    }
    const rr = rng(seed), qq = new Float32Array(D);
    let mA = 0, mB = 0, mN = 0, nA = 0, nB = 0, nN = 0;
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      const mem = trainWinSet.has(wkey(valIds, pos));
      featPM(valIds, pos, qq);
      const { w, steps: sS } = scoreW(qq, valIds, pos, runQueries.beta, 12);
      flopSteps += sS;
      const dist = predictNext(w, next, V);
      const trueNext = valIds[pos + W];
      const ha = argmax(dist) === trueNext ? 1 : 0;
      const off = valIds[pos + W - 1] * V;
      let best = 0, bestC = -1;
      for (let c = 0; c < V; c++){ const cc = bi[off + c] + 1; if (cc > bestC){ bestC = cc; best = c; } }
      const ba = best === trueNext ? 1 : 0;
      if (mem){ mA += ha; mB += ba; mN++; } else { nA += ha; nB += ba; nN++; }
    }
    return { memAcc: mN ? mA / mN : -1, memBi: mN ? mB / mN : -1, memN: mN, novAcc: nN ? nA / nN : -1, novBi: nN ? nB / nN : -1, novN: nN };
  }
  const runQueries = (n, seed) => {
    const rr = rng(seed), dist_sum = { acc: 0, ce: 0, steps: 0 };
    const qq = new Float32Array(D);
    for (let t = 0; t < n; t++){
      const pos = Math.floor(rr() * (valIds.length - W - 1));
      featPM(valIds, pos, qq);
      const trueNext = valIds[pos + W];
      const { w, steps } = scoreW(qq, valIds, pos, runQueries.beta, 12);
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
  const split = SPLIT === '1' ? splitEval(TR, 2002) : null;
  // ---- C1+C2 canary: implanted facts, edit flips, exact forget (store surgery, zero gradients) ----
  let canRemember = -1, canEditFlip = -1, canForgetLeak = -1, collatBase = -1, collatAfter = -1;
  let canRemDetail = '', canEditDetail = '';
  if (CAPAB === 'canary'){
    const CAN_N = 20, crr = rng(SEED + 99);
    const cfreq = new Uint32Array(V);
    for (let i = 0; i < trainIds.length; i++) cfreq[trainIds[i]]++;
    let rare = 0, common = 0;
    for (let c = 1; c < V; c++){ if (cfreq[c] < cfreq[rare]) rare = c; if (cfreq[c] > cfreq[common]) common = c; }
    const cidx = [];
    for (let i = 0; i < CAN_N; i++) cidx.push(Math.floor(crr() * P));
    const cpos = cidx.map(s => spos[s]);
    const cqq = new Float32Array(D), cbeta = runQueries.beta;
    const askRare = (positions) => {
      let hit = 0;
      for (const p of positions){
        featPM(trainIds, p, cqq);
        const rw = scoreW(cqq, trainIds, p, cbeta, 12);
        if (argmax(predictNext(rw.w, next, V)) === rare) hit++;
      }
      return hit / positions.length;
    };
    collatBase = e.acc;
    for (const s of cidx) next[s] = rare;
    canRemember = askRare(cpos);
    if (C2D === '1'){ for (const p of cpos){ featPM(trainIds, p, cqq); const rw = scoreW(cqq, trainIds, p, cbeta, 12); flopSteps += rw.steps; canRemDetail += argmax(predictNext(rw.w, next, V)) === rare ? '1' : '0'; } }
    for (let i = 0; i < CAN_N / 2; i++) next[cidx[i]] = common;
    let flip = 0;
    for (let i = 0; i < CAN_N / 2; i++){
      featPM(trainIds, cpos[i], cqq);
      const rw = scoreW(cqq, trainIds, cpos[i], cbeta, 12);
      if (argmax(predictNext(rw.w, next, V)) === common) flip++;
      if (C2D === '1'){ flopSteps += rw.steps; canEditDetail += argmax(predictNext(rw.w, next, V)) === common ? '1' : '0'; }
    }
    canEditFlip = flip / (CAN_N / 2);
    const drop = new Set(cidx);
    const keep = [];
    for (let s = 0; s < P; s++) if (!drop.has(s)) keep.push(s);
    const cX = new Int8Array(keep.length * D), cN = new Uint8Array(keep.length), cS = new Int32Array(keep.length);
    keep.forEach((s, i) => { cX.set(X.subarray(s * D, s * D + D), i * D); cN[i] = next[s]; cS[i] = spos[s]; });
    X = cX; next = cN; spos = cS; P = keep.length;
    canForgetLeak = askRare(cpos);
    collatAfter = runQueries(TR, 2000).acc;
  }
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
    const { w, steps: stG } = scoreW(qq, win, 0, best.beta, 12);
    flopSteps += stG;
    const dist = predictNext(w, next, V);
    let tot = 0; for (let i = 0; i < V; i++) tot += dist[i];
    for (let i = 0; i < V; i++) dist[i] /= (tot + 1e-12 * V);
    return dist;
  }
  const sRng = rng(3002); // E1 sampling stream: reproducible draws, same prompts across jobs
  function smpPick(dist){ // E1: sample with temperature SMP on the hop dist (SMP=0 never calls this)
    const T = SMP;
    const p = new Float64Array(V);
    let tot = 0;
    for (let c = 0; c < V; c++){ p[c] = Math.exp(Math.log(dist[c] + 1e-18) / T); tot += p[c]; }
    let u = sRng() * tot;
    for (let c = 0; c < V; c++){ u -= p[c]; if (u <= 0) return c; }
    return V - 1;
  }
  function penPick(dist, win){ // H10: degeneration penalty over top-8 (PEN=0 -> plain argmax)
    if (PEN === 0) return argmax(dist);
    const K = Math.min(8, V);
    const order = [...Array(V).keys()].sort((a, b) => dist[b] - dist[a]).slice(0, K);
    let bi = order[0], bs = -1e30;
    for (const c of order){
      let rep = 0;
      for (let k = 0; k < W; k++) if (win[k] === c){ rep = 1; break; }
      const s = Math.log(dist[c] + 1e-12) - PEN * rep;
      if (s > bs){ bs = s; bi = c; }
    }
    return bi;
  }
  function genText(kind, pos){
    const win = new Array(W);
    for (let k = 0; k < W; k++) win[k] = valIds[pos + k];
    let text = '';
    for (let k = 0; k < W; k++) text += chars[win[k]];
    for (let t = 0; t < GEN_LEN; t++){
      let ni;
      if (kind === 'hop'){ const dd = distOf(win); ni = SMP > 0 ? smpPick(dd) : penPick(dd, win); }
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
  function provCite(pos){ // C3: top-3 retrieved store windows for the prompt, as text
    const cqq = new Float32Array(D);
    featPM(valIds, pos, cqq);
    const rw = scoreW(cqq, valIds, pos, runQueries.beta, 12);
    const idx = [...Array(P).keys()].sort((a, b) => rw.w[b] - rw.w[a]).slice(0, 3);
    return idx.map(s => {
      const sp = spos[s];
      let str = '';
      for (let k = 0; k < W + 1 && sp + k < trainIds.length; k++) str += chars[trainIds[sp + k]];
      return { store: s, next: chars[next[s]], ctx: str };
    });
  }
  const samples = [];
  let qRep = 0, qWord = 0;
  for (const pos of gPrompts){
    const h = genText('hop', pos), b = genText('bi', pos);
    const qh = qualityOf(h);
    qRep += qh.rep4; qWord += qh.wordValid;
    const cite = (PROV === '1') ? provCite(pos) : [];
    samples.push({ prompt: h.slice(0, W), hop: h.slice(W), bi: b.slice(W), cite });
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
      const { w, steps: stI } = scoreW(qq, valIds, pos, best.beta, 12);
      flopSteps += stI;
      let wmax = 0, wsum = 0; // B1: top-1 weight fraction = match confidence
      for (let i = 0; i < w.length; i++){ wsum += w[i]; if (w[i] > wmax) wmax = w[i]; }
      let ent = 1; // H5: normalized entropy of store weights (1=uniform/diffuse, 0=one-hot); same loop, no extra pass
      if (wsum > 0){ ent = 0; for (let i = 0; i < w.length; i++){ const p = w[i] / wsum; if (p > 0) ent -= p * Math.log(p); } ent /= Math.log(w.length); }
      let w2 = 0, kth = 0, cov = 0; // H6: margin + kth-weight + coverage from same w, zero extra passes
      if (wsum > 0){
        const thr = wsum / w.length, top = [];
        for (let i = 0; i < w.length; i++){ if (w[i] > thr) cov++; top.push(w[i]); }
        top.sort((a, b) => b - a);
        w2 = top.length > 1 ? top[1] : 0; kth = top.length > 9 ? top[9] : top[top.length - 1];
      }
      const hd = predictNext(w, next, V);
      let tot = 0; for (let i = 0; i < V; i++) tot += hd[i];
      for (let i = 0; i < V; i++) hd[i] /= (tot + 1e-12 * V);
      const off = valIds[pos + W - 1] * V;
      let rtot = V; for (let c = 0; c < V; c++) rtot += bi[off + c];
      const bd = new Float64Array(V);
      for (let c = 0; c < V; c++) bd[c] = (bi[off + c] + 1) / rtot;
      let be = 0; for (let c = 0; c < V; c++){ const p = bd[c]; if (p > 0) be -= p * Math.log(p); } be /= Math.log(V);
      rows.push({ pos, hd, bd, trueNext, g: wmax / (wsum || 1), h: ent, m: (wmax - w2) / (wsum || 1), k10: kth / (wsum || 1), cov, be });
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
  // DTW rerank (screen verdict 2026-11-13: shift 0.74, rev 0.00 — rerank-only behind
  // the sup prefilter, K=50, tau=W fixed; costs one extra recall pass per query, disclosed)
  let dtwLamSel = -1, dtwAccSel = 0, dtwPplSel = 0, dtwAcc = 0, dtwPpl = 0, repHitD = '';
  function dtwDist(a, b, n){ // token-id windows length n, 0/1 frame cost, full band
    const C = new Float64Array((n + 1) * (n + 1)).fill(1e18); C[0] = 0;
    for (let i = 1; i <= n; i++) for (let j = 1; j <= n; j++){
      const c = (a[i - 1] === b[j - 1]) ? 0 : 1;
      let m = C[(i - 1) * (n + 1) + j];
      const m2 = C[i * (n + 1) + j - 1]; if (m2 < m) m = m2;
      const m3 = C[(i - 1) * (n + 1) + j - 1]; if (m3 < m) m = m3;
      C[i * (n + 1) + j] = c + m;
    }
    return C[n * (n + 1) + n];
  }
  function dtwMix(qr){
    const K = 50, TAU = W;
    const qq2 = new Float32Array(D);
    featPM(valIds, qr.pos, qq2);
    const r2 = scoreW(qq2, valIds, qr.pos, best.beta, 12);
    flopSteps += r2.steps;
    const w = r2.w;
    const idx = [...Array(w.length).keys()];
    idx.sort((x, y) => w[y] - w[x]);
    const w2 = Float32Array.from(w);
    const qw = new Array(W), sw = new Array(W);
    for (let k = 0; k < W; k++) qw[k] = valIds[qr.pos + k];
    for (let k = 0; k < Math.min(K, idx.length); k++){
      const sp = spos[idx[k]];
      for (let j = 0; j < W; j++) sw[j] = trainIds[sp + j];
      w2[idx[k]] = w[idx[k]] * Math.exp(-dtwDist(qw, sw, W) / TAU);
    }
    return predictNext(w2, next, V);
  }
  function dtwStats(rows){
    return LAMBDAS.map(lam => {
      let acc = 0, ce = 0;
      for (const qr of rows){
        const dd = qr.dd;
        let bp = -1, bci = 0;
        for (let c = 0; c < V; c++){ const p = lam * dd[c] + (1 - lam) * qr.bd[c]; if (p > bp){ bp = p; bci = c; } }
        if (bci === qr.trueNext) acc++;
        ce += -Math.log(Math.max(lam * dd[qr.trueNext] + (1 - lam) * qr.bd[qr.trueNext], 1e-12));
      }
      return { lam, acc: acc / rows.length, ppl: Math.exp(ce / rows.length) };
    });
  }
  if (DTW === '1'){
    for (const qr of selRows) qr.dd = normDist(dtwMix(qr), V);
    const dSelStats = dtwStats(selRows);
    let bestD = dSelStats[0];
    for (const s of dSelStats) if (s.ppl < bestD.ppl || (s.ppl === bestD.ppl && s.acc > bestD.acc)) bestD = s;
    dtwLamSel = bestD.lam; dtwAccSel = bestD.acc; dtwPplSel = bestD.ppl;
    for (const qr of repRows) qr.dd = normDist(dtwMix(qr), V);
    const dRepStats = dtwStats(repRows);
    const dLocked = dRepStats[LAMBDAS.indexOf(dtwLamSel)];
    dtwAcc = dLocked.acc; dtwPpl = dLocked.ppl;
    for (const qr of repRows){
      let dp = -1, dci = 0;
      for (let c = 0; c < V; c++){ const p = dtwLamSel * qr.dd[c] + (1 - dtwLamSel) * qr.bd[c]; if (p > dp){ dp = p; dci = c; } }
      repHitD += dci === qr.trueNext ? '1' : '0';
    }
    console.log(`dtw lam*=${dtwLamSel} acc=${dtwAcc.toFixed(3)} ppl=${dtwPpl.toFixed(2)}`);
  }
  // H8 hybrid arm: alpha grid on select (same beta probe as dense arm; probe ran with hybAlpha=0.25 default), locked report
  let hybA = 0.25, hybLamSel = -1, hybAccSel = 0, hybPplSel = 0, hybAcc = 0, hybPpl = 0, repHitH = '';
  if (HYB === '1'){
    let bestHyb = null;
    for (const a of [0.25, 0.5]){
      hybAlpha = a;
      const hSel = hopBiDists(TR, 2000), hSelStats = lamStats(hSel);
      let b = hSelStats[0];
      for (const s of hSelStats) if (s.ppl < b.ppl || (s.ppl === b.ppl && s.acc > b.acc)) b = s;
      if (!bestHyb || b.ppl < bestHyb.ppl || (b.ppl === bestHyb.ppl && b.acc > bestHyb.acc)) bestHyb = { a, ...b };
    }
    hybAlpha = bestHyb.a; hybA = bestHyb.a; hybLamSel = bestHyb.lam; hybAccSel = bestHyb.acc; hybPplSel = bestHyb.ppl;
    const hRep = hopBiDists(TR, 2001), hRepStats = lamStats(hRep);
    const hr = hRepStats[LAMBDAS.indexOf(bestHyb.lam)]; hybAcc = hr.acc; hybPpl = hr.ppl;
    for (const qr of hRep){ const lam = bestHyb.lam; let ip = -1, ici = 0; for (let c = 0; c < V; c++){ const p = lam * qr.hd[c] + (1 - lam) * qr.bd[c]; if (p > ip){ ip = p; ici = c; } } repHitH += ici === qr.trueNext ? '1' : '0'; }
    console.log(`hyb select alpha=${bestHyb.a} lam=${bestHyb.lam} acc=${bestHyb.acc.toFixed(3)} ppl=${bestHyb.ppl.toFixed(2)} report acc=${hr.acc.toFixed(3)} ppl=${hr.ppl.toFixed(2)}`);
  }
  // H11 post-hoc temperature on the MIXED distribution (lam*): fit T on select, locked report
  const TMPS = [0.5, 0.75, 1, 1.5, 2];
  let tmpT = 1, tmpAccSel = 0, tmpPplSel = 0, tmpAcc = 0, tmpPpl = 0;
  function tmpStats(rows, Tx, lam){
    let acc = 0, ce = 0;
    for (const qr of rows){
      let s = 0; const pw = new Float64Array(V);
      for (let c = 0; c < V; c++){ const p = lam * qr.hd[c] + (1 - lam) * qr.bd[c]; const v = Math.pow(p, 1 / Tx); pw[c] = v; s += v; }
      let bp = -1, bci = 0;
      for (let c = 0; c < V; c++){ const p = pw[c] / s; if (p > bp){ bp = p; bci = c; } }
      if (bci === qr.trueNext) acc++;
      ce += -Math.log(Math.max(pw[qr.trueNext] / s, 1e-12));
    }
    return { acc: acc / rows.length, ppl: Math.exp(ce / rows.length) };
  }
  if (TMP === '1'){
    let bT = null;
    for (const Tx of TMPS){ const st = tmpStats(selRows, Tx, bestLam.lam); if (!bT || st.ppl < bT.ppl || (st.ppl === bT.ppl && st.acc > bT.acc)) bT = { T: Tx, ...st }; }
    tmpT = bT.T; tmpAccSel = bT.acc; tmpPplSel = bT.ppl;
    const rt = tmpStats(repRows, tmpT, bestLam.lam); tmpAcc = rt.acc; tmpPpl = rt.ppl;
    console.log(`tmp select T=${tmpT} acc=${tmpAccSel.toFixed(3)} ppl=${tmpPplSel.toFixed(2)} report acc=${rt.acc.toFixed(3)} ppl=${rt.ppl.toFixed(2)}`);
  }
  // Conformal abstention (alpha=0.1): nonconformity s = 1 - max_c p_mix; qhat = 90th pct on select; abstain -> bigram on report
  let confAbs = 0, confAcc = 0, confPpl = 0;
  if (CONF === '1'){
    const lamC = bestLam.lam;
    const sSel = selRows.map(qr => { let m = 0; for (let c = 0; c < V; c++){ const p = lamC * qr.hd[c] + (1 - lamC) * qr.bd[c]; if (p > m) m = p; } return 1 - m; }).sort((a, b) => a - b);
    const qhat = sSel[Math.min(sSel.length - 1, Math.floor(0.9 * sSel.length))];
    let acc = 0, ce = 0, abs = 0;
    for (const qr of repRows){
      let m = 0, mci = 0;
      for (let c = 0; c < V; c++){ const p = lamC * qr.hd[c] + (1 - lamC) * qr.bd[c]; if (p > m){ m = p; mci = c; } }
      if (1 - m <= qhat){ if (mci === qr.trueNext) acc++; ce += -Math.log(Math.max(m, 1e-12)); }
      else { abs++; let bp = -1, bci = 0; for (let c = 0; c < V; c++) if (qr.bd[c] > bp){ bp = qr.bd[c]; bci = c; } if (bci === qr.trueNext) acc++; ce += -Math.log(Math.max(qr.bd[qr.trueNext], 1e-12)); }
    }
    confAbs = abs / repRows.length; confAcc = acc / repRows.length; confPpl = Math.exp(ce / repRows.length);
    console.log(`conf qhat=${qhat.toFixed(3)} absRate=${confAbs.toFixed(3)} report acc=${confAcc.toFixed(3)} ppl=${confPpl.toFixed(2)}`);
  }
  // gmix: hop + tiny-GPT (the paper's actual prescription — kNN mixed with a real LM,
  // not a bigram). Joined on pos: full coverage iff W == trainer block (32).
  function gmixStats(rows, stream) {
    const cov = rows.filter(qr => gptMap.has(qr.pos) && gptMap.get(qr.pos)[stream]);
    const stats = LAMBDAS.map(lam => {
      let acc = 0, ce = 0;
      for (const qr of cov) {
        const gd = gptMap.get(qr.pos)[stream];
        let bp = -1, bci = 0;
        for (let c = 0; c < V; c++) { const p = lam * qr.hd[c] + (1 - lam) * gd[c]; if (p > bp) { bp = p; bci = c; } }
        if (bci === qr.trueNext) acc++;
        ce += -Math.log(Math.max(lam * qr.hd[qr.trueNext] + (1 - lam) * gd[qr.trueNext], 1e-12));
      }
      return { lam, acc: cov.length ? acc / cov.length : 0, ppl: cov.length ? Math.exp(ce / cov.length) : 0 };
    });
    return { cov: cov.length, stats };
  }
  let gptCover = 0, gptMixLamSel = -1, gptMixAccSel = 0, gptMixPplSel = 0, gptMixAcc = 0, gptMixPpl = 0, bestGlam = 0;
  if (MIX === 'gpt') {
    const gSel = gmixStats(selRows, '2000');
    let bestG = gSel.stats[0];
    for (const s of gSel.stats) if (s.ppl < bestG.ppl || (s.ppl === bestG.ppl && s.acc > bestG.acc)) bestG = s;
    const gRep = gmixStats(repRows, '2001');
    const gr = gRep.stats[LAMBDAS.indexOf(bestG.lam)];
    gptCover = gRep.cov; bestGlam = bestG.lam; gptMixLamSel = bestG.lam; gptMixAccSel = bestG.acc; gptMixPplSel = bestG.ppl; gptMixAcc = gr.acc; gptMixPpl = gr.ppl;
    console.log(`gmix select lam=${bestG.lam} acc=${bestG.acc.toFixed(3)} ppl=${bestG.ppl.toFixed(2)} cover=${gSel.cov}/${selRows.length} report acc=${gr.acc.toFixed(3)} ppl=${gr.ppl.toFixed(2)} cover=${gRep.cov}/${repRows.length}`);
  }
  // H4 adaptive-lambda: per-query L_i = lam * retrieval confidence (top-1 weight fraction g).
  // Known kNN-LM pathology (TO-VERIFY cites): fixed lam lets noisy retrieval hurt the tokens
  // it cannot help; adaptive lam collapses toward bigram where retrieval is diffuse.
  // Cheap: reuses hopBiDists rows, no extra recall passes.
  // Hedge/experts (Freund & Schapire): weights over the LAMBDAS grid fit ONLINE on the
  // select stream (loss = -log p(trueNext), eta=1), FROZEN, then the mixture predicts the
  // report stream — same select/report discipline as lam. Regret-bounded by construction.
  let hedgeAcc = 0, hedgePpl = 0, hedgeNeff = 0;
  if (HEDGE === '1'){
    const NE = LAMBDAS.length;
    const hw = new Array(NE).fill(1 / NE);
    for (const qr of selRows){
      let tot = 0;
      for (let e = 0; e < NE; e++){
        const lam = LAMBDAS[e];
        hw[e] *= Math.max(lam * qr.hd[qr.trueNext] + (1 - lam) * qr.bd[qr.trueNext], 1e-12);
        tot += hw[e];
      }
      for (let e = 0; e < NE; e++) hw[e] /= tot;
    }
    let ent = 0;
    for (let e = 0; e < NE; e++) if (hw[e] > 0) ent -= hw[e] * Math.log(hw[e]);
    hedgeNeff = Math.exp(ent);
    let acc = 0, ce = 0;
    for (const qr of repRows){
      const mix = new Float64Array(V);
      for (let e = 0; e < NE; e++){ const lam = LAMBDAS[e]; for (let c = 0; c < V; c++) mix[c] += hw[e] * (lam * qr.hd[c] + (1 - lam) * qr.bd[c]); }
      let bp = -1, bci = 0;
      for (let c = 0; c < V; c++) if (mix[c] > bp){ bp = mix[c]; bci = c; }
      if (bci === qr.trueNext) acc++;
      ce += -Math.log(Math.max(mix[qr.trueNext], 1e-12));
    }
    hedgeAcc = acc / repRows.length; hedgePpl = Math.exp(ce / repRows.length);
    console.log(`hedge frozen-w acc=${hedgeAcc.toFixed(3)} ppl=${hedgePpl.toFixed(2)} neff=${hedgeNeff.toFixed(2)}`);
  }
  // Paired-testing machinery (faster verdicts without quality loss): per-query hit
  // strings on the REPORT stream. Same seeds => same queries across jobs, so collects
  // can run exact McNemar tests (higher power per query than eyeballing aggregates).
  let repHitI = '', repHitB = '', repHitG = '';
  {
    for (const qr of repRows){
      let bp = -1, bci = 0;
      for (let c = 0; c < V; c++) if (qr.bd[c] > bp){ bp = qr.bd[c]; bci = c; }
      repHitB += bci === qr.trueNext ? '1' : '0';
      const lam = bestLam.lam;
      let ip = -1, ici = 0;
      for (let c = 0; c < V; c++){ const p = lam * qr.hd[c] + (1 - lam) * qr.bd[c]; if (p > ip){ ip = p; ici = c; } }
      repHitI += ici === qr.trueNext ? '1' : '0';
      if (MIX === 'gpt'){
        const g = gptMap.get(qr.pos);
        if (g && g['2001']){
          const gd = g['2001'], gl = bestGlam;
          let gp = -1, gci = 0;
          for (let c = 0; c < V; c++){ const p = gl * qr.hd[c] + (1 - gl) * gd[c]; if (p > gp){ gp = p; gci = c; } }
          repHitG += gci === qr.trueNext ? '1' : '0';
        } else repHitG += 'x';
      }
    }
  }
  function adaStats(rows){    return LAMBDAS.map(lam => {
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
  if (CAPAB === 'canary') console.log(`canary remember=${canRemember.toFixed(2)} editFlip=${canEditFlip.toFixed(2)} forgetLeak=${canForgetLeak.toFixed(3)} collat=${collatBase.toFixed(3)}->${collatAfter.toFixed(3)}`);
  // ---- H6 reframed: consolidate-wins logistic gate (6 features, zero pretraining) ----
  let h6LamSel = -1, h6AccSel = 0, h6PplSel = 0, h6Acc = 0, h6Ppl = 0, h6AUC = 0;
  if (H6 === '1'){
    const F = (qr) => [qr.g, 1 - qr.h, qr.m, qr.k10, qr.cov / P, 1 - qr.be];
    const ME = 6, mu = new Array(ME).fill(0), sd = new Array(ME).fill(0);
    for (const qr of selRows){ const f = F(qr); for (let j = 0; j < ME; j++) mu[j] += f[j]; }
    for (let j = 0; j < ME; j++) mu[j] /= selRows.length;
    for (const qr of selRows){ const f = F(qr); for (let j = 0; j < ME; j++) sd[j] += (f[j] - mu[j]) * (f[j] - mu[j]); }
    for (let j = 0; j < ME; j++) sd[j] = Math.sqrt(sd[j] / selRows.length) || 1;
    const Z = selRows.map(qr => F(qr).map((v, j) => (v - mu[j]) / sd[j]));
    const Y = selRows.map(qr => (-Math.log(Math.max(qr.hd[qr.trueNext], 1e-12)) < -Math.log(Math.max(qr.bd[qr.trueNext], 1e-12))) ? 1 : 0);
    let hw = new Array(ME).fill(0), hb = 0;
    for (let e = 0; e < 500; e++){
      const gw = new Array(ME).fill(0); let gb = 0;
      for (let i = 0; i < Z.length; i++){
        let z = hb; for (let j = 0; j < ME; j++) z += hw[j] * Z[i][j];
        const p = 1 / (1 + Math.exp(-z)), d = p - Y[i]; gb += d;
        for (let j = 0; j < ME; j++) gw[j] += d * Z[i][j];
      }
      for (let j = 0; j < ME; j++) hw[j] -= 0.1 * gw[j] / Z.length; hb -= 0.1 * gb / Z.length;
    }
    const P1 = (qr) => { const f = F(qr); let z = hb; for (let j = 0; j < ME; j++) z += hw[j] * (f[j] - mu[j]) / sd[j]; return 1 / (1 + Math.exp(-z)); };
    const order = selRows.map((qr, i) => i).sort((a, b) => P1(selRows[b]) - P1(selRows[a]));
    let tp = 0, auc = 0; const npos = Y.reduce((a, v) => a + v, 0), nneg = Y.length - npos;
    for (const i of order){ if (Y[i]) tp++; else auc += tp; }
    h6AUC = (npos && nneg) ? auc / (npos * nneg) : 0;
    function h6Stats(rows){
      return LAMBDAS.map(lam => {
        let acc = 0, ce = 0;
        for (const qr of rows){
          const L = lam * P1(qr);
          let bp = -1, bci = 0;
          for (let c = 0; c < V; c++){ const p = L * qr.hd[c] + (1 - L) * qr.bd[c]; if (p > bp){ bp = p; bci = c; } }
          if (bci === qr.trueNext) acc++;
          ce += -Math.log(Math.max(L * qr.hd[qr.trueNext] + (1 - L) * qr.bd[qr.trueNext], 1e-12));
        }
        return { lam, acc: acc / rows.length, ppl: Math.exp(ce / rows.length) };
      });
    }
    const selH6 = h6Stats(selRows);
    let bestH6 = selH6[0];
    for (const s of selH6) if (s.ppl < bestH6.ppl || (s.ppl === bestH6.ppl && s.acc > bestH6.acc)) bestH6 = s;
    const repH6 = h6Stats(repRows);
    const rh = repH6[LAMBDAS.indexOf(bestH6.lam)];
    h6LamSel = bestH6.lam; h6AccSel = bestH6.acc; h6PplSel = bestH6.ppl; h6Acc = rh.acc; h6Ppl = rh.ppl;
    console.log(`h6 gate select lam=${bestH6.lam} acc=${bestH6.acc.toFixed(3)} ppl=${bestH6.ppl.toFixed(2)} auc=${h6AUC.toFixed(3)} report acc=${rh.acc.toFixed(3)} ppl=${rh.ppl.toFixed(2)}`);
  }
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
      const { w } = recallW(qp, X, P, D, best.beta, 1, SPARSE, (RAWFAST === '1' && PROJ === 'raw') ? 1 : 0, onesList, W); // single step: argmax must be the key itself (store machinery: always whole-path, independent of SCORE)
      let bi = 0; for (let i = 1; i < P; i++) if (w[i] > w[bi]) bi = i;
      let same = true;
      for (let j = 0; j < D; j++){ const bit = qp[j] > 0 ? 1 : 0; if (X[bi * D + j] !== bit){ same = false; break; } }
      if (same) self++;
    }
    check('self-key-retrieval', self === NS, self + '/' + NS);
  }
  console.log(`compare hopfield acc=${e.acc.toFixed(3)} ppl=${e.ppl.toFixed(2)} vs bigram ${biE.acc.toFixed(3)}/${biE.ppl.toFixed(2)} vs unigram ${uniE.acc.toFixed(3)}/${uniE.ppl.toFixed(2)} beatUni=${e.acc > uniE.acc}`);
  const evalMflops = Math.round(flopSteps * P * D * 2 / 1e6 * 10) / 10; // 2 FLOP per MAC (dot+mix), measured steps
  const pkMflops = Math.round(pkMacs * 2 / 1e6 * 10) / 10; // H9 leaders-index MACs (routing + candidate dots), 0 unless pk=1
  const msMflops = Math.round(msOps * 2 / 1e6 * 10) / 10; // H7 maxsim char-compare ops (0 under whole)
  const buildMflops = PROJ === 'raw' ? 0 : Math.round(P * W * D * 2 / 1e6 * 10) / 10; // feature-projection estimate (W*D adds per pattern; ==old W*H for every prior arm)
  console.log(`flops steps=${flopSteps} evalMflops=${evalMflops} buildMflops=${buildMflops}`);
  // save wrapped store
  const plain = Buffer.concat([Buffer.from(X), Buffer.from(next)]);
  const sum = sha256hex(plain);
  const meta = { task: 'nextchar-heldout', corpus: tag, proj: PROJ, W, P: P0, P0, Peff: P, V, D, seed: SEED, score: SCORE, beta: BSEL === 'ppl' ? bestPpl.beta : best.beta, betaSel: BSEL, bytes: plain.length, sha256: sum, enc: noEnc ? 'none-smoke' : 'AES-256-GCM-v1', chars: raw.length };
  writeFileSync(outDir + '/meta.json', JSON.stringify(meta));
  let encBytes = 0;
  if (!noEnc){ const wj = JSON.stringify(encryptWrap(plain, secret)); writeFileSync(outDir + '/model.bin.enc', wj); encBytes = wj.length; console.log(`enc bytes=${encBytes} sha256=${sum}`); }
  else { writeFileSync(outDir + '/model.bin', plain); console.log(`plain-smoke bytes=${plain.length} sha256=${sum}`); }
  const metrics = { ...meta, trials: TR, flips: 'n/a-nextchar', acc: e.acc, ppl: e.ppl, avgSteps: e.avgSteps, bigramAcc: biE.acc, bigramPpl: biE.ppl, unigramAcc: uniE.acc, unigramPpl: uniE.ppl, beatUni: e.acc > uniE.acc, quality, buildMs, evalMs, maxRssMB: rssMB(), encBytes, evalScope: 'heldout-val', evalMflops, buildMflops, flopSteps, msOps, msMflops, score: SCORE, betaSel: BSEL, betaAcc: best.beta, betaPpl: bestPpl.beta, ada: parseInt(ADA) || 0, adaLamSel, adaAccSel, adaPplSel, adaAcc, adaPpl };
  // kNN-LM interpolation results (locked-lam report on fresh seed-2001 queries)
  metrics.interpLambda = bestLam.lam;
  metrics.interpAccSel = bestLam.acc; metrics.interpPplSel = bestLam.ppl;
  metrics.interpAcc = repLocked.acc; metrics.interpPpl = repLocked.ppl;
  metrics.mix = MIX; metrics.gptCover = gptCover; metrics.gptMixLamSel = gptMixLamSel;
  metrics.gptMixAccSel = gptMixAccSel; metrics.gptMixPplSel = gptMixPplSel;
  metrics.gptMixAcc = gptMixAcc; metrics.gptMixPpl = gptMixPpl;
  metrics.tok = TOK; metrics.bpeMerges = bpeMerges; metrics.bpeMeanLen = bpeMeanLen;
  metrics.encLossFirst = encLossFirst; metrics.encLossLast = encLossLast;
  // PARADIGM LADDER (user, 2026-11-13): every verdict carries its cost-class. Same
  // equation, different series: exact brute-force is linear-O(P·D) (whole) or
  // linear-O(P·W²) (maxsim); H9 product-key/ANN will register 'sublinear' here.
  metrics.costClass = DTW === '1' ? 'linear-O(P·D+K·W^2)' : (PK === '1' ? 'sublinear-O(sqrt(P)·D)' : (SCORE === 'maxsim' ? 'linear-O(P·W²)' : 'linear-O(P·D)'));
  metrics.sparse = parseInt(SPARSE) || 0; metrics.hedge = parseInt(HEDGE) || 0;
  metrics.hyb = parseInt(HYB) || 0; metrics.hybA = hybA; metrics.hybLamSel = hybLamSel;
  metrics.hybAccSel = hybAccSel; metrics.hybPplSel = hybPplSel; metrics.hybAcc = hybAcc; metrics.hybPpl = hybPpl;
  metrics.pk = parseInt(PK) || 0; metrics.pkMflops = pkMflops;
  metrics.tmp = parseInt(TMP) || 0; metrics.tmpT = tmpT; metrics.tmpAccSel = tmpAccSel; metrics.tmpPplSel = tmpPplSel; metrics.tmpAcc = tmpAcc; metrics.tmpPpl = tmpPpl;
  metrics.conf = parseInt(CONF) || 0; metrics.confAbs = confAbs; metrics.confAcc = confAcc; metrics.confPpl = confPpl;
  metrics.orf = parseInt(ORF) || 0;
  metrics.rawfast = parseInt(RAWFAST) || 0;
  metrics.graft = PROJ === 'graft4' ? 4 : (PROJ === 'graft3' ? 3 : (PROJ === 'graft2' ? 2 : (PROJ === 'graft' ? 1 : 0))); metrics.graftE = PROJ === 'graft3' ? V : GE; metrics.graftFile = (PROJ === 'graft' || PROJ === 'graft2' || PROJ === 'graft3' || PROJ === 'graft4') ? GRAFTGGUF : null;
  metrics.c2d = parseInt(C2D) || 0; metrics.canRemDetail = canRemDetail; metrics.canEditDetail = canEditDetail;
  metrics.repHitH = repHitH;
  metrics.hedgeAcc = hedgeAcc; metrics.hedgePpl = hedgePpl; metrics.hedgeNeff = hedgeNeff;
  metrics.repHitI = repHitI; metrics.repHitB = repHitB; metrics.repHitG = repHitG;
  metrics.hopAccRep = repStats[LAMBDAS.length - 1].acc; metrics.hopPplRep = repStats[LAMBDAS.length - 1].ppl;
  metrics.biAccRep = repStats[0].acc; metrics.biPplRep = repStats[0].ppl;
  metrics.gateTau = GATE; metrics.pruneKeep = KEEP; metrics.Peff = P; metrics.gatedFracRep = gatedFracRep;
  metrics.canRemember = canRemember; metrics.canEditFlip = canEditFlip; metrics.canForgetLeak = canForgetLeak; metrics.collatBase = collatBase; metrics.collatAfter = collatAfter; metrics.canaryN = CAPAB === 'canary' ? 20 : 0;
  metrics.pen = PEN; metrics.prov = parseInt(PROV) || 0;
  metrics.dtw = parseInt(DTW) || 0; metrics.dtwLamSel = dtwLamSel; metrics.dtwAccSel = dtwAccSel; metrics.dtwPplSel = dtwPplSel; metrics.dtwAcc = dtwAcc; metrics.dtwPpl = dtwPpl; metrics.repHitD = repHitD;
  metrics.smp = SMP;
  if (split){ metrics.memAcc = split.memAcc; metrics.memBiAcc = split.memBi; metrics.memN = split.memN; metrics.novAcc = split.novAcc; metrics.novBiAcc = split.novBi; metrics.novN = split.novN; }
  metrics.h6 = parseInt(H6) || 0; metrics.h6LamSel = h6LamSel; metrics.h6AccSel = h6AccSel; metrics.h6PplSel = h6PplSel; metrics.h6Acc = h6Acc; metrics.h6Ppl = h6Ppl; metrics.h6AUC = h6AUC;
  writeFileSync(outDir + '/metrics.json', JSON.stringify(metrics));
  let stxt = '';
  for (const s of samples) stxt += 'PROMPT: ' + JSON.stringify(s.prompt) + '\nHOP: ' + JSON.stringify(s.hop) + '\nBI: ' + JSON.stringify(s.bi) + '\nCITE: ' + JSON.stringify(s.cite || []) + '\n---\n';
  writeFileSync(outDir + '/samples-' + tag + '-W' + W + '-' + PROJ + '.txt', stxt);
  if (failN){ console.log('lm-shakespeare: ' + failN + ' FAILURES'); process.exit(1); }
  console.log('lm-shakespeare: benched');
}
