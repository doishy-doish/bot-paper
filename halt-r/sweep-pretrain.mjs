// sweep-pretrain: scale ladder for the HALT-R memory substrate + scramble-wrapped model save.
// Synthetic seeded data only (scramble-plaintext-safe to generate); saved models are
// AES-256-GCM wrapped ("under wraps": ciphertext + hashes in public, key stays in
// vault / private repo / SCRAMBLE secret). Quiet logs: counts + hashes only.
//
// Usage:
//   node halt-r/sweep-pretrain.mjs --scale S|M|L|XL --out out/S
//     (env SCRAMBLE must hold the wrap key; --key-env NAME to rename;
//      --no-encrypt for local smoke only, never in CI)
//   node halt-r/sweep-pretrain.mjs decrypt <file.enc> --key-file <path> [--out <p>] [--force]
//
// Scales: S=1Kx256 M=10Kx512 L=50Kx1024 XL=200Kx1024(full eval, honest cost)
// Model = Int8 flat P*D (0/1) + meta.json. Eval = 10% flips, beta picked by mini-probe.
import { writeFileSync, readFileSync, mkdirSync, existsSync } from 'node:fs';
import { createHash, randomBytes, createCipheriv, createDecipheriv } from 'node:crypto';

const SCALES = {
  S:  { P: 1000,   D: 256,  trials: 50, probeN: 3 },
  M:  { P: 10000,  D: 512,  trials: 30, probeN: 3 },
  L:  { P: 50000,  D: 1024, trials: 20, probeN: 3 },
  XL: { P: 200000, D: 1024, trials: 8,  probeN: 2 },
};
const SEED = 7, FLIP_FRAC = 0.1;
const BETAS = [0.5, 1, 2, 4, 8, 16, 32, 64]; // multiplied by 1/D (top covers XL's ln(P) threshold)

function fail(msg){ console.error('FAIL ' + msg); process.exit(1); }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function sha256hex(buf){ return createHash('sha256').update(buf).digest('hex'); }
function kdf(secret){ return createHash('sha256').update(String(secret), 'utf8').digest(); } // 32B from any secret string
function rssMB_(){ return Math.round(process.memoryUsage().rss / 1024 / 1024 * 10) / 10; }

function genPatterns(P, D, r){
  const X = new Int8Array(P * D);
  for (let i = 0; i < X.length; i++) X[i] = r() < 0.5 ? 0 : 1;
  return X;
}
function recall(q, X, P, D, beta, maxS){
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
  return { x, steps };
}
function nearest(x, X, P, D){
  let bi = 0, bd = 1e18;
  for (let i = 0; i < P; i++){
    let d = 0; const off = i * D;
    for (let j = 0; j < D; j++){ const bit = x[j] >= 0 ? 1 : 0; if (bit !== X[off + j]) d++; }
    if (d < bd){ bd = d; bi = i; }
  }
  return bi;
}
function trials(X, P, D, beta, n, seed){
  const rr = rng(seed);
  const flips = Math.round(D * FLIP_FRAC);
  let ok = 0, steps = 0;
  for (let t = 0; t < n; t++){
    const k = Math.floor(rr() * P);
    const q = new Float32Array(D);
    for (let j = 0; j < D; j++) q[j] = X[k * D + j] ? 1 : -1;
    for (let f = 0; f < flips; f++){ const j = Math.floor(rr() * D); q[j] = -q[j]; }
    const res = recall(q, X, P, D, beta, 12);
    steps += res.steps;
    if (nearest(res.x, X, P, D) === k) ok++;
  }
  return { acc: ok / n, avgSteps: steps / n };
}
function encryptWrap(plainBuf, secret){
  const K = kdf(secret), nonce = randomBytes(12);
  const c = createCipheriv('aes-256-gcm', K, nonce);
  const data = Buffer.concat([c.update(plainBuf), c.final()]);
  return {
    enc: 'AES-256-GCM-v1', nonce: nonce.toString('base64'), data: data.toString('base64'),
    tag: c.getAuthTag().toString('base64'), sha256: sha256hex(plainBuf),
    bytes: plainBuf.length, date: new Date().toISOString().slice(0, 10),
  };
}
function decryptWrap(wrap, secret){
  if (wrap.enc !== 'AES-256-GCM-v1') fail('unknown wrapper ' + wrap.enc);
  const K = kdf(secret);
  const d = createDecipheriv('aes-256-gcm', K, Buffer.from(wrap.nonce, 'base64'));
  d.setAuthTag(Buffer.from(wrap.tag, 'base64'));
  const plain = Buffer.concat([d.update(Buffer.from(wrap.data, 'base64')), d.final()]);
  if (plain.length !== wrap.bytes) fail('length mismatch');
  if (sha256hex(plain) !== wrap.sha256) fail('sha256 mismatch (wrong key or corrupt)');
  return plain;
}

// ---- decrypt mode ----
if (process.argv[2] === 'decrypt'){
  const a = process.argv.slice(3);
  const file = a[0];
  const ki = a.indexOf('--key-file'), ke = a.indexOf('--key-env'), oi = a.indexOf('--out');
  if (!file) fail('decrypt <file.enc> --key-file <path> [--out <p>]');
  let secret = null;
  if (ki >= 0) secret = readFileSync(a[ki + 1], 'utf8').trim();
  else if (ke >= 0) secret = process.env[a[ke + 1]];
  else if (process.env.SCRAMBLE) secret = process.env.SCRAMBLE;
  if (!secret) fail('no key (pass --key-file or set SCRAMBLE)');
  const wrap = JSON.parse(readFileSync(file, 'utf8'));
  const plain = decryptWrap(wrap, secret);
  const out = oi >= 0 ? a[oi + 1] : file.replace(/\.enc$/, '');
  if (existsSync(out) && !a.includes('--force')) fail('refuses to overwrite ' + out + ' (pass --force)');
  writeFileSync(out, plain);
  console.log('decrypt ok bytes=' + plain.length + ' sha256=' + sha256hex(plain));
  process.exit(0);
}

// ---- sweep mode ----
{
  const a = process.argv.slice(2);
  const si = a.indexOf('--scale'), oii = a.indexOf('--out'), kei = a.indexOf('--key-env');
  const noEnc = a.includes('--no-encrypt');
  if (si < 0 || oii < 0) fail('usage: --scale S|M|L|XL --out <dir> [--no-encrypt]');
  const cfg = SCALES[a[si + 1]];
  if (!cfg) fail('unknown scale');
  const outDir = a[oii + 1];
  const secret = process.env[kei >= 0 ? a[kei + 1] : 'SCRAMBLE'];
  if (!secret && !noEnc) fail('SCRAMBLE key missing (refuses plaintext save; pass --no-encrypt for local smoke only)');
  mkdirSync(outDir, { recursive: true });
  const { P, D } = cfg;
  let t0 = Date.now(), maxRss = rssMB_();
  const X = genPatterns(P, D, rng(SEED));
  const buildMs = Date.now() - t0; maxRss = Math.max(maxRss, rssMB_());
  console.log(`build P=${P} D=${D} ms=${buildMs} rss=${maxRss}MB`);
  // beta mini-probe (same query sequence per beta: fixed seed)
  t0 = Date.now();
  let best = null;
  for (const k of BETAS){
    const beta = k / D;
    const r = trials(X, P, D, beta, cfg.probeN, 1000);
    console.log(`probe betaD=${k} acc=${r.acc.toFixed(2)} steps=${r.avgSteps.toFixed(1)}`);
    if (!best || r.acc > best.acc || (r.acc === best.acc && r.avgSteps < best.avgSteps)) best = { beta, betaD: k, ...r };
  }
  // eval
  const e = trials(X, P, D, best.beta, cfg.trials, 200);
  const evalMs = Date.now() - t0; maxRss = Math.max(maxRss, rssMB_());
  console.log(`eval betaD=${best.betaD} acc=${e.acc.toFixed(3)} steps=${e.avgSteps.toFixed(2)} ms=${evalMs} rss=${maxRss}MB`);
  if (e.acc < 0.5) fail(`acc floor breached (${e.acc}) — code bug, not science`);
  // save model (plaintext bytes in memory only; ciphertext to disk)
  const plain = Buffer.from(X);
  const sum = sha256hex(plain);
  const meta = { scale: a[si + 1], P, D, seed: SEED, beta: best.beta, betaD: best.betaD, bytes: plain.length, sha256: sum, enc: noEnc ? 'none-smoke' : 'AES-256-GCM-v1' };
  writeFileSync(outDir + '/meta.json', JSON.stringify(meta));
  let encBytes = 0;
  if (!noEnc){
    const wrap = encryptWrap(plain, secret);
    writeFileSync(outDir + '/model.bin.enc', JSON.stringify(wrap));
    encBytes = JSON.stringify(wrap).length;
    console.log(`enc bytes=${encBytes} sha256=${sum}`);
  } else {
    writeFileSync(outDir + '/model.bin', plain);
    console.log(`plain-smoke bytes=${plain.length} sha256=${sum}`);
  }
  const metrics = { ...meta, trials: cfg.trials, flips: FLIP_FRAC, acc: e.acc, avgSteps: e.avgSteps, buildMs, evalMs, maxRssMB: maxRss, encBytes, evalScope: 'full' };
  writeFileSync(outDir + '/metrics.json', JSON.stringify(metrics));
  console.log(`sweep-${a[si + 1]}: all measured`);
}
