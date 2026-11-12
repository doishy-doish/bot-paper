// halt-r rust-recall CI test: wasm core vs JS reference. node 20+, zero deps.
// Run from this dir after: cargo build --release --target wasm32-unknown-unknown
//   && cp target/wasm32-unknown-unknown/release/halt_r_recall.wasm recall.wasm
import { readFile } from 'node:fs/promises';

function rng(seed) { let s = seed >>> 0; return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296); }
const P = 8, D = 64, BETA = 8, ITERS = 12;
const rand = rng(7);
const pats = [];
for (let i = 0; i < P; i++) { const v = []; for (let j = 0; j < D; j++) v.push(rand() < 0.5 ? -1 : 1); pats.push(v); }
const k = 3;
const q = [...pats[k]];
for (let f = 0; f < 8; f++) { const j = Math.floor(rand() * D); q[j] = -q[j]; }

// JS reference (f64) of the same update
function refStep(x) {
  const w = []; let m = -1e30;
  for (let i = 0; i < P; i++) { let s = 0; for (let j = 0; j < D; j++) s += pats[i][j] * x[j]; s *= BETA; w.push(s); if (s > m) m = s; }
  let t = 0; for (let i = 0; i < P; i++) { w[i] = Math.exp(w[i] - m); t += w[i]; }
  const o = new Array(D).fill(0);
  for (let i = 0; i < P; i++) for (let j = 0; j < D; j++) o[j] += (w[i] / t) * pats[i][j];
  return o;
}
let xr = [...q];
for (let s = 0; s < ITERS; s++) xr = refStep(xr);

// wasm run
const bytes = await readFile(new URL('./recall.wasm', import.meta.url));
const { instance } = await WebAssembly.instantiate(bytes, {});
const mem = () => new Float32Array(instance.exports.memory.buffer);
const F = mem();
const PB = 0, SB = P * D * 4, CB = (P * D + D) * 4;
pats.flat().forEach((v, i) => { F[i] = v; });
q.forEach((v, i) => { F[P * D + i] = v; });
const n = instance.exports.recall(PB, SB, CB, P, D, BETA, ITERS, 0.0);
// determinism: run again on a fresh query copy, must match bit-for-bit
q.forEach((v, i) => { F[P * D + i] = v; });
const n2 = instance.exports.recall(PB, SB, CB, P, D, BETA, ITERS, 0.0);
const after = Array.from(mem().slice(P * D, P * D + D));

let maxd = 0;
for (let j = 0; j < D; j++) maxd = Math.max(maxd, Math.abs(after[j] - xr[j]));
console.log('iters=' + n + ' maxAbsDiff=' + maxd.toExponential(2));

const ham = (a, b) => { let d = 0; for (let j = 0; j < D; j++) d += Math.abs(a[j] - b[j]); return d; };
let bi = 0, bd = 1e18;
for (let i = 0; i < P; i++) { const d = ham(after, pats[i]); if (d < bd) { bd = d; bi = i; } }
console.log('nearest=' + bi + ' target=' + k + ' hamDist=' + bd.toFixed(1));

let fail = 0;
const check = (name, cond, detail) => { console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; };
check('wasm-runs-iters', n === ITERS && n2 === ITERS, 'n=' + n + ',' + n2);
check('wasm-matches-ref', maxd < 1e-3, 'maxd=' + maxd.toExponential(2));
check('wasm-retrieves', bi === k, 'nearest=' + bi);
if (fail) { console.log('WASM-RECALL: ' + fail + ' FAILURES'); process.exit(1); }
console.log('WASM-RECALL: all green');
