// rung: causal-scale — 5-node exhaustive BIC recovers exact graph obs-only (29,281 DAGs, CPU ~0.1s).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const D = 5;
const W = { '0->2': 2.0, '0->4': -2.0, '1->3': 2.0, '1->4': 1.0, '2->3': -1.5, '2->4': 2.0 };
function gen(n, r){
  const X = [];
  for (let s = 0; s < n; s++){
    const x = [0, 0, 0, 0, 0];
    for (let j = 0; j < 5; j++){ let v = randn(r); for (let i = 0; i < j; i++) if (W[i + '->' + j]) v += W[i + '->' + j] * x[i]; x[j] = v; }
    X.push(x);
  }
  return X;
}
function olsRSS(X, y, plist){
  const n = X.length, k = plist.length + 1;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++){ const row = [1]; for (const j of plist) row.push(X[i][j]); for (let a = 0; a < k; a++){ Xty[a] += row[a] * X[i][y]; for (let b = 0; b < k; b++) XtX[a][b] += row[a] * row[b]; } }
  const M = XtX.map((row, a) => row.concat([Xty[a]]));
  for (let c = 0; c < k; c++){ let piv = c; for (let q = c + 1; q < k; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < k; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= k; cc++) M[q][cc] -= f * M[c][cc]; } }
  const beta = M.map((row, c) => row[k] / (row[c] || 1e-12));
  let rss = 0;
  for (let i = 0; i < n; i++){ let pr = beta[0]; plist.forEach((j, a) => pr += beta[a + 1] * X[i][j]); const e = X[i][y] - pr; rss += e * e; }
  return { rss, beta };
}
function bic(X){
  const n = X.length, pen = Math.log(n), cache = {};
  const fit = (y, mask) => { const key = y + ':' + mask; if (!cache[key]){ const p = []; for (let j = 0; j < D; j++) if (mask >> j & 1) p.push(j); cache[key] = olsRSS(X, y, p); } return cache[key]; };
  const nodes = [0, 1, 2, 3, 4]; let best = null, evals = 0;
  const perms = (a) => a.length <= 1 ? [a] : a.flatMap((v, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [v, ...p]));
  const bitcount = (m) => { let c = 0; while (m){ c += m & 1; m >>= 1; } return c; };
  for (const ord of perms(nodes)){
    const opts = ord.map((y, idx) => { const allowed = ord.slice(0, idx); const out = []; for (let m = 0; m < (1 << allowed.length); m++){ let mask = 0; for (let b = 0; b < allowed.length; b++) if (m >> b & 1) mask |= (1 << allowed[b]); out.push(mask); } return out; });
    const rec = (i, masks, score) => {
      if (i === D){ evals++; if (!best || score < best.score) best = { score, masks: [...masks] }; return; }
      const y = ord[i];
      for (const mask of opts[i]){ masks.push([y, mask]); rec(i + 1, masks, score + n * Math.log(fit(y, mask).rss / n) + pen * (bitcount(mask) + 1)); masks.pop(); }
    };
    rec(0, [], 0);
  }
  return { masks: best.masks, evals };
}
const truth = new Set(Object.keys(W));
const X = gen(500, rng(11));
const t0 = Date.now();
const { masks, evals } = bic(X);
const ms = Date.now() - t0;
const learned = new Set();
for (const [y, mask] of masks) for (let j = 0; j < D; j++) if (mask >> j & 1) learned.add(j + '->' + y);
let shd = 0;
for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++){ const a = learned.has(i + '->' + j) ? 1 : 0, b = learned.has(j + '->' + i) ? 1 : 0, c = truth.has(i + '->' + j) ? 1 : 0, d = truth.has(j + '->' + i) ? 1 : 0; if (a !== c || b !== d) shd++; }
console.log('obs SHD=' + shd + ' evals=' + evals + ' ms=' + ms + ' ' + [...learned].sort().join(','));
check('scale5-exact', shd === 0, 'SHD=' + shd);
check('scale5-fast', ms < 5000, ms + 'ms');
if (fail){ console.log('scale5: ' + fail + ' FAILURES'); process.exit(1); }
console.log('scale5: all green');
