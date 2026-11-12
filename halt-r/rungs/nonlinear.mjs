// rung: nonlinear — linear-Gaussian BIC blind to quadratic edge; quadratic expansion recovers it.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
function olsRSS(X, y, plist){
  const n = X.length, k = plist.length + 1;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++){ const row = [1]; for (const j of plist) row.push(X[i][j]); for (let a = 0; a < k; a++){ Xty[a] += row[a] * X[i][y]; for (let b = 0; b < k; b++) XtX[a][b] += row[a] * row[b]; } }
  const M = XtX.map((row, a) => row.concat([Xty[a]]));
  let ok = true;
  for (let c = 0; c < k; c++){ let piv = c; for (let q = c + 1; q < k; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; if (Math.abs(M[piv][c]) < 1e-9){ ok = false; break; } const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c]; for (let q = 0; q < k; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= k; cc++) M[q][cc] -= f * M[c][cc]; } }
  if (!ok) return { rss: 1e18, beta: [] };
  const beta = M.map((row, c) => row[k] / (row[c] || 1e-12));
  let rss = 0;
  for (let i = 0; i < n; i++){ let pr = beta[0]; plist.forEach((j, a) => pr += beta[a + 1] * X[i][j]); const e = X[i][y] - pr; rss += e * e; }
  return { rss, beta };
}
function bicNodes(X, D, forbid){
  const n = X.length, pen = Math.log(n), cache = {};
  const fit = (y, mask) => { const key = y + ':' + mask; if (!cache[key]){ const p = []; for (let j = 0; j < D; j++) if (mask >> j & 1) p.push(j); cache[key] = olsRSS(X, y, p); } return cache[key]; };
  const nodes = [...Array(D).keys()]; let best = null;
  const perms = (a) => a.length <= 1 ? [a] : a.flatMap((v, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [v, ...p]));
  const bitcount = (m) => { let c = 0; while (m){ c += m & 1; m >>= 1; } return c; };
  for (const ord of perms(nodes)){
    const opts = ord.map((y, idx) => { const allowed = ord.slice(0, idx); const out = []; for (let m = 0; m < (1 << allowed.length); m++){ let mask = 0; for (let b = 0; b < allowed.length; b++) if (m >> b & 1) mask |= (1 << allowed[b]); out.push(mask); } return out; });
    const rec = (i, masks, score) => {
      if (i === D){ if (!best || score < best.score) best = { score, masks: [...masks] }; return; }
      const y = ord[i];
      for (const mask of opts[i]){ if (forbid && forbid(y, mask)) continue; masks.push([y, mask]); rec(i + 1, masks, score + n * Math.log(Math.max(fit(y, mask).rss, 1e-9) / n) + pen * (bitcount(mask) + 1)); masks.pop(); }
    };
    rec(0, [], 0);
  }
  return best.masks;
}
const r = rng(21), N = 500, X = [];
for (let i = 0; i < N; i++){ const x0 = randn(r), x1 = randn(r); X.push([x0, x1, x0 * x0 + x1 + randn(r)]); }
const lin = bicNodes(X, 3, null);
const linSkel = new Set();
for (const [y, m] of lin) for (let j = 0; j < 3; j++) if (m >> j & 1) linSkel.add(Math.min(j, y) + '-' + Math.max(j, y));
console.log('linear skeleton=[' + [...linSkel].sort().join(' ') + ']');
check('linear-misses-quad', !linSkel.has('0-2'), '');
const XQ = X.map(row => [row[0], row[1], row[0] * row[0], row[2]]);
const quad = bicNodes(XQ, 4, (y, mask) => y === 2 && mask !== 0);
const names = [0, 1, '0sq', 2];
const qEdges = [];
for (const [y, m] of quad) for (let j = 0; j < 4; j++) if (m >> j & 1) qEdges.push(names[j] + '->' + names[y]);
console.log('quad edges=[' + qEdges.sort().join(' ') + ']');
check('quad-recovers', qEdges.includes('0sq->2') && qEdges.includes('1->2'), qEdges.join(' '));
if (fail){ console.log('nonlinear: ' + fail + ' FAILURES'); process.exit(1); }
console.log('nonlinear: all green');
