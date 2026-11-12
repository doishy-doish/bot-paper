// rung: mnist-capacity — real correlated data (binarized MNIST train); temperature is everything.
// Usage: node mnist-capacity.mjs <path-to-mnist-train-images-idx3-ubyte(.gz)>  (CI downloads it first)
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const gzPath = process.argv[2] || process.env.MNIST_GZ;
if (!gzPath){ console.log('SKIP mnist-capacity: no file (pass path or MNIST_GZ)'); process.exit(0); }
let raw;
try {
  const buf = readFileSync(gzPath);
  raw = new Uint8Array(gzPath.endsWith('.gz') ? gunzipSync(buf) : buf);
} catch (e){ console.log('SKIP mnist-capacity: cannot read ' + gzPath + ' (' + e.message + ')'); process.exit(0); }
const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
if (dv.getUint32(0) !== 2051){ console.log('FAIL mnist-capacity: bad IDX magic'); process.exit(1); }
const D = dv.getUint32(8) * dv.getUint32(12);
const P = 200;
const pats = [];
for (let p = 0; p < P; p++){ const v = new Array(D); for (let i = 0; i < D; i++) v[i] = raw[16 + p * D + i] > 128 ? 1 : -1; pats.push(v); }
function recall(q, beta, maxS){
  let x = q.slice(), steps = 0;
  for (let s = 0; s < maxS; s++){
    const w = new Array(P); let m = -1e30;
    for (let i = 0; i < P; i++){ let d = 0; for (let j = 0; j < D; j++) d += pats[i][j] * x[j]; d *= beta; w[i] = d; if (d > m) m = d; }
    let t = 0; for (let i = 0; i < P; i++){ w[i] = Math.exp(w[i] - m); t += w[i]; }
    const nx = new Array(D).fill(0);
    for (let i = 0; i < P; i++) for (let j = 0; j < D; j++) nx[j] += w[i] / t * pats[i][j];
    let maxd = 0; for (let j = 0; j < D; j++) maxd = Math.max(maxd, Math.abs(nx[j] - x[j]));
    x = nx; steps++;
    if (maxd < 1e-3) break;
  }
  return { x, steps };
}
const r = rng(3);
const flips = Math.round(D * 0.1);
function trials(beta, n, seed){
  const rr = rng(seed);
  let ok = 0, steps = 0;
  for (let t = 0; t < n; t++){
    const k = Math.floor(rr() * P);
    const q = pats[k].slice();
    for (let f = 0; f < flips; f++){ const j = Math.floor(rr() * D); q[j] = -q[j]; }
    const res = recall(q, beta, 12);
    steps += res.steps;
    let bi = 0, bd = 1e18;
    for (let i = 0; i < P; i++){ let d = 0; for (let j = 0; j < D; j++) d += Math.abs((res.x[j] >= 0 ? 1 : -1) - pats[i][j]); if (d < bd){ bd = d; bi = i; } }
    if (bi === k) ok++;
  }
  return { acc: ok / n, avgSteps: steps / n };
}
for (const beta of [0.005, 0.02, 0.1]){
  const res = trials(beta, 50, 200);
  console.log('beta=' + beta + ' retrieval=' + res.acc.toFixed(3) + ' avgSteps=' + res.avgSteps.toFixed(2));
  if (beta === 0.005) check('cold-dead', res.acc < 0.1, res.acc.toFixed(3));
  if (beta === 0.1) check('hot-exact', res.acc === 1.0, res.acc.toFixed(3));
}
if (fail){ console.log('mnist-capacity: ' + fail + ' FAILURES'); process.exit(1); }
console.log('mnist-capacity: all green');
