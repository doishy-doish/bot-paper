// rung: capacity — classic collapse vs modern scaling. Seeded, CPU-only, zero deps.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function makePatterns(rand, n, d){ const P = []; for (let i = 0; i < n; i++){ const v = []; for (let j = 0; j < d; j++) v.push(rand() < 0.5 ? -1 : 1); P.push(v); } return P; }
function corrupt(v, rand, flips){ const o = v.slice(); for (let k = 0; k < flips; k++){ const j = Math.floor(rand() * v.length); o[j] *= -1; } return o; }
function hebbian(P){ const d = P[0].length, W = Array.from({ length: d }, () => new Array(d).fill(0)); for (const x of P) for (let i = 0; i < d; i++) for (let j = 0; j < d; j++) W[i][j] += x[i] * x[j] / P.length; for (let i = 0; i < d; i++) W[i][i] = 0; return W; }
function recallClassic(x0, W, maxSteps){ let x = x0.slice(); let steps = 0; for (let t = 0; t < maxSteps; t++){ let flipped = false; for (let i = 0; i < x.length; i++){ let h = 0; for (let j = 0; j < x.length; j++) h += W[i][j] * x[j]; const nv = h >= 0 ? 1 : -1; if (nv !== x[i]){ x[i] = nv; flipped = true; } } steps++; if (!flipped) break; } return { x, steps }; }
function recallModern(q0, P, beta, maxSteps, tol){ const d = q0.length; let q = q0.slice(); let steps = 0; for (let t = 0; t < maxSteps; t++){ const s = P.map(p => { let dot = 0; for (let j = 0; j < d; j++) dot += p[j] * q[j]; return beta * dot / P[0].length; }); const m = Math.max(...s), ex = s.map(v => Math.exp(v - m)), Z = ex.reduce((a, b) => a + b, 0); const nq = new Array(d).fill(0); for (let k = 0; k < P.length; k++) for (let j = 0; j < d; j++) nq[j] += P[k][j] * ex[k] / Z; let diff = 0; for (let j = 0; j < d; j++) diff += (nq[j] - q[j]) ** 2; q = nq; steps++; if (Math.sqrt(diff / d) < tol) break; } const qb = q.map(v => v >= 0 ? 1 : -1); let best = 0, bd = 1e9; P.forEach((p, i) => { let dd = 0; for (let j = 0; j < d; j++) dd += (p[j] - qb[j]) ** 2; if (dd < bd){ bd = dd; best = i; } }); return { qb, best, steps }; }
const D = 64, TRIALS = 200, FLIPS = 8;
for (const NPAT of [8, 16, 32]){
  const rand = rng(7);
  const P = makePatterns(rand, NPAT, D);
  const W = hebbian(P);
  let ok = 0;
  for (let i = 0; i < TRIALS; i++){ const k = Math.floor(rand() * NPAT); const r = recallClassic(corrupt(P[k], rand, FLIPS), W, 12); if (r.x.every((v, j) => v === P[k][j])) ok++; }
  const acc = ok / TRIALS;
  console.log('classic NPAT=' + NPAT + ' retrieval=' + acc.toFixed(3));
  if (NPAT === 8) check('classic-8', acc >= 0.90, acc.toFixed(3));
  if (NPAT === 16) check('classic-16-collapse', acc <= 0.30, acc.toFixed(3));
  if (NPAT === 32) check('classic-32-zero', acc <= 0.05, acc.toFixed(3));
}
for (const NPAT of [8, 16, 32]){
  const rand = rng(7);
  const P = makePatterns(rand, NPAT, D);
  let ok = 0, s2 = 0;
  for (let i = 0; i < TRIALS; i++){ const k = Math.floor(rand() * NPAT); const r = recallModern(corrupt(P[k], rand, FLIPS).map(v => v * 1.0), P, 8.0, 12, 1e-3); s2 += r.steps; if (r.best === k) ok++; }
  const acc = ok / TRIALS;
  console.log('modern NPAT=' + NPAT + ' retrieval=' + acc.toFixed(3) + ' avgSteps=' + (s2 / TRIALS).toFixed(2));
  check('modern-' + NPAT, acc >= 0.95, acc.toFixed(3));
}
if (fail){ console.log('capacity: ' + fail + ' FAILURES'); process.exit(1); }
console.log('capacity: all green');
