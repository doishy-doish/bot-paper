// rung: causal-diamond — obs equivalence ambiguity resolved by two interventions (N=300, exhaustive 543 DAGs).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const D = 4, W = { '0->1': 1, '0->2': 1, '1->3': 1, '2->3': 1 };
function gen(n, r, interv){
  const X = [];
  for (let s = 0; s < n; s++){
    const x = [0, 0, 0, 0];
    for (let j = 0; j < 4; j++){
      if (interv && interv.t === j){ x[j] = interv.v; continue; }
      let v = randn(r);
      for (let i = 0; i < j; i++) if (W[i + '->' + j]) v += W[i + '->' + j] * x[i];
      x[j] = v;
    }
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
  const nodes = [0, 1, 2, 3]; let best = null;
  const perms = (a) => a.length <= 1 ? [a] : a.flatMap((v, i) => perms([...a.slice(0, i), ...a.slice(i + 1)]).map(p => [v, ...p]));
  for (const ord of perms(nodes)){
    const opts = ord.map((y, idx) => { const allowed = ord.slice(0, idx); const out = []; for (let m = 0; m < (1 << allowed.length); m++){ let mask = 0; for (let b = 0; b < allowed.length; b++) if (m >> b & 1) mask |= (1 << allowed[b]); out.push(mask); } return out; });
    const rec = (i, masks, score) => {
      if (i === D){ if (!best || score < best.score) best = { score, masks: [...masks] }; return; }
      const y = ord[i];
      for (const mask of opts[i]){ masks.push([y, mask]); rec(i + 1, masks, score + n * Math.log(fit(y, mask).rss / n) + pen * (String(mask.toString(2)).split('').filter(c => c === '1').length + 1)); masks.pop(); }
    };
    rec(0, [], 0);
  }
  return best.masks;
}
function edgeSet(masks){ const e = new Set(); for (const [y, mask] of masks) for (let j = 0; j < D; j++) if (mask >> j & 1) e.add(j + '->' + y); return e; }
function shd(L, T){ let s = 0; for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++){ const a = L.has(i + '->' + j) ? 1 : 0, b = L.has(j + '->' + i) ? 1 : 0, c = T.has(i + '->' + j) ? 1 : 0, d = T.has(j + '->' + i) ? 1 : 0; if (a !== c || b !== d) s++; } return s; }
const truth = new Set(['0->1', '0->2', '1->3', '2->3']);
const r = rng(3);
const obs = gen(300, r, null);
const eObs = edgeSet(bic(obs));
const sObs = shd(eObs, truth);
console.log('obs-only SHD=' + sObs + ' ' + [...eObs].sort().join(','));
check('obs-ambiguous', sObs <= 2, 'SHD=' + sObs);
const r2 = rng(4);
const joint = obs.concat(gen(60, r2, { t: 2, v: 2.5 }), gen(60, r2, { t: 2, v: -2.5 }), gen(60, r2, { t: 0, v: 2.5 }), gen(60, r2, { t: 0, v: -2.5 }));
const eJ = edgeSet(bic(joint));
const sJ = shd(eJ, truth);
console.log('joint SHD=' + sJ + ' ' + [...eJ].sort().join(','));
check('joint-exact', sJ === 0, 'SHD=' + sJ);
// total effect 0->3 via path tracing on learned weights (truth = 1*1 + 1*1 = 2)
let w = {};
for (const [y, mask] of bic(joint)){ const p = []; for (let j = 0; j < D; j++) if (mask >> j & 1) p.push(j); const f = olsRSS(joint, y, p); p.forEach((j, a) => w[j + '->' + y] = f.beta[a + 1]); }
const est = (w['0->1'] || 0) * (w['1->3'] || 0) + (w['0->2'] || 0) * (w['2->3'] || 0);
console.log('effect 0->3 est=' + est.toFixed(3) + ' truth=2.000');
check('effect-accurate', Math.abs(est - 2) < 0.35, est.toFixed(3));
if (fail){ console.log('diamond: ' + fail + ' FAILURES'); process.exit(1); }
console.log('diamond: all green');
