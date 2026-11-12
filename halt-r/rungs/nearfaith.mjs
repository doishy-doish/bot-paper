// rung: nearfaith — exact cancellation hides 0->2 from marginal AND score-based discovery; do(X1) nails it.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const D = 4, A = 2, B = 2, C = -4.0, E = 1.5;
function gen(n, r, interv){
  const X = [];
  for (let i = 0; i < n; i++){
    let x0, x1, x2, x3;
    if (interv && interv.t === 1){ x0 = randn(r); x1 = interv.v; x2 = B * x1 + C * x0 + randn(r); x3 = E * x2 + randn(r); }
    else { x0 = randn(r); x1 = A * x0 + randn(r); x2 = B * x1 + C * x0 + randn(r); x3 = E * x2 + randn(r); }
    X.push([x0, x1, x2, x3]);
  }
  return X;
}
function corr(X, a, b){ const n = X.length; let ma = 0, mb = 0; for (const row of X){ ma += row[a]; mb += row[b]; } ma /= n; mb /= n; let sab = 0, sa = 0, sb = 0; for (const row of X){ sab += (row[a] - ma) * (row[b] - mb); sa += (row[a] - ma) ** 2; sb += (row[b] - mb) ** 2; } return sab / Math.sqrt(sa * sb); }
function phi(x){ const t = 1 / (1 + 0.2316419 * x); const d = 0.3989423 * Math.exp(-x * x / 2); return 1 - d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))); }
function fisherP(rho, n){ const z = Math.atanh(Math.max(-0.9999999, Math.min(0.9999999, rho))) * Math.sqrt(n - 3); return 2 * (1 - phi(Math.abs(z))); }
function ols(X, y, plist){
  const n = X.length, k = plist.length + 1;
  const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
  const Xty = new Array(k).fill(0);
  for (let i = 0; i < n; i++){ const row = [1]; for (const j of plist) row.push(X[i][j]); for (let a = 0; a < k; a++){ Xty[a] += row[a] * X[i][y]; for (let b = 0; b < k; b++) XtX[a][b] += row[a] * row[b]; } }
  const M = XtX.map((row, a) => row.concat([Xty[a]]));
  for (let c = 0; c < k; c++){ let piv = c; for (let q = c + 1; q < k; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < k; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= k; cc++) M[q][cc] -= f * M[c][cc]; } }
  return M.map((row, c) => row[k] / (row[c] || 1e-12));
}
function olsRSS(X, y, plist){ const beta = ols(X, y, plist); let rss = 0; for (let i = 0; i < X.length; i++){ let pr = beta[0]; plist.forEach((j, a) => pr += beta[a + 1] * X[i][j]); const e = X[i][y] - pr; rss += e * e; } return { rss, beta }; }
function bic(X){
  const n = X.length, pen = Math.log(n), cache = {};
  const fit = (y, mask) => { const key = y + ':' + mask; if (!cache[key]){ const p = []; for (let j = 0; j < D; j++) if (mask >> j & 1) p.push(j); cache[key] = olsRSS(X, y, p); } return cache[key]; };
  const nodes = [0, 1, 2, 3]; let best = null;
  const perms = (a) => a.length <= 1 ? [a] : a.flatMap((v, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [v, ...p]));
  const bitcount = (m) => { let c = 0; while (m){ c += m & 1; m >>= 1; } return c; };
  for (const ord of perms(nodes)){
    const opts = ord.map((y, idx) => { const allowed = ord.slice(0, idx); const out = []; for (let m = 0; m < (1 << allowed.length); m++){ let mask = 0; for (let b = 0; b < allowed.length; b++) if (m >> b & 1) mask |= (1 << allowed[b]); out.push(mask); } return out; });
    const rec = (i, masks, score) => {
      if (i === D){ if (!best || score < best.score) best = { score, masks: [...masks] }; return; }
      const y = ord[i];
      for (const mask of opts[i]){ masks.push([y, mask]); rec(i + 1, masks, score + n * Math.log(fit(y, mask).rss / n) + pen * (bitcount(mask) + 1)); masks.pop(); }
    };
    rec(0, [], 0);
  }
  return best.masks;
}
function skel(masks){ const e = new Set(); for (const [y, mask] of masks) for (let j = 0; j < D; j++) if (mask >> j & 1) e.add(Math.min(j, y) + '-' + Math.max(j, y)); return e; }
const r = rng(7);
const obs = gen(500, r, null);
const rho = corr(obs, 0, 2), p = fisherP(rho, 500);
console.log('marginal corr(X0,X2)=' + rho.toFixed(4) + ' p=' + p.toFixed(4));
check('marginal-blind', p > 0.05, 'p=' + p.toFixed(4));
const truthSkel = new Set(['0-1', '0-2', '1-2', '2-3']);
const sk = skel(bic(obs));
const recall = [...truthSkel].filter(e => sk.has(e)).length;
console.log('BIC skeleton=[' + [...sk].sort().join(' ') + '] recall=' + recall + '/4');
check('bic-also-drops', recall === 3, recall + '/4');
const rI = rng(8);
const int1 = gen(250, rI, { t: 1, v: 2.5 }).concat(gen(250, rI, { t: 1, v: -2.5 }));
const bI = ols(int1, 2, [0, 1]);
console.log('do(X1) OLS w02=' + bI[1].toFixed(3) + ' w12=' + bI[2].toFixed(3) + ' (truth -4.0, 2.0)');
check('intervention-nails-it', Math.abs(bI[1] + 4.0) < 0.5 && Math.abs(bI[2] - 2.0) < 0.5, bI[1].toFixed(3));
if (fail){ console.log('nearfaith: ' + fail + ' FAILURES'); process.exit(1); }
console.log('nearfaith: all green');
