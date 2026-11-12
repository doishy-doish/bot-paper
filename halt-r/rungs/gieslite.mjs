// rung: gieslite — joint regime-aware scoring (obs + do(X4) arm) recovers full 10-node truth.
// NOTE: single do(X4) cannot orient edges outside X4's downstream (Markov-equivalent flip,
// here 1->6 vs 6->1, scores tie to the digit). Skeleton-perfect + SHD<=1 + tie-proof is the claim.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const D = 10, T = 4;
const W = { '0->4': -1.5184416592819616, '0->5': 2.2187079382129014, '1->4': 2.1404971223091707, '1->6': 2.249790992238559, '2->5': -1.26921291439794, '2->7': -2.0105595301138237, '4->5': 1.0120364339090884, '4->7': -2.33061301370617, '5->9': 2.0874882420757785, '6->9': 1.612082772422582 };
const truth = new Set(Object.keys(W));
function gen(n, r, interv){
  const X = [];
  for (let s = 0; s < n; s++){
    const x = new Array(D).fill(0);
    for (let j = 0; j < D; j++){
      if (interv && interv.t === j){ x[j] = interv.v; continue; }
      let v = randn(r); for (let i = 0; i < j; i++) if (W[i + '->' + j]) v += W[i + '->' + j] * x[i]; x[j] = v;
    }
    X.push(x);
  }
  return X;
}
function scorer(X, skip){
  const N = X.length, pen = Math.log(N), cache = {};
  function fit(y, plist){
    const key = y + ':' + plist.join(',');
    if (cache[key]) return cache[key];
    const n = X.length, k = plist.length + 1;
    const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
    const Xty = new Array(k).fill(0);
    for (let i = 0; i < n; i++){ const row = [1]; for (const j of plist) row.push(X[i][j]); for (let a = 0; a < k; a++){ Xty[a] += row[a] * X[i][y]; for (let b = 0; b < k; b++) XtX[a][b] += row[a] * row[b]; } }
    const M = XtX.map((row, a) => row.concat([Xty[a]]));
    for (let c = 0; c < k; c++){ let piv = c; for (let q = c + 1; q < k; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < k; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= k; cc++) M[q][cc] -= f * M[c][cc]; } }
    const beta = M.map((row, c) => row[k] / (row[c] || 1e-12));
    let rss = 0;
    for (let i = 0; i < n; i++){ let pr = beta[0]; plist.forEach((j, a) => pr += beta[a + 1] * X[i][j]); const e = X[i][y] - pr; rss += e * e; }
    const out = { rss }; cache[key] = out; return out;
  }
  return { term: (y, plist) => (skip.has(y) ? 0 : N * Math.log(fit(y, plist).rss / N) + pen * (plist.length + 1)) };
}
const Xo = gen(1000, rng(32), null);
const rI = rng(33);
const Xa = gen(250, rI, { t: T, v: 3 }).concat(gen(250, rI, { t: T, v: -3 }));
const scO = scorer(Xo, new Set()), scA = scorer(Xa, new Set([T]));
const jterm = (y, plist) => scO.term(y, plist) + scA.term(y, plist);
const hasE = (par, a, b) => par[b].indexOf(a) >= 0;
function reaches(par, from, to){ const seen = {}; seen[from] = 1; const st = [from]; while (st.length){ const c = st.pop(); for (let j = 0; j < D; j++) if (par[j].indexOf(c) >= 0 && !seen[j]){ if (j === to) return true; seen[j] = 1; st.push(j); } } return false; }
function climb(par0){
  const par = par0.map(a => a.slice());
  let it = 0;
  for (;;){
    let best = null, bestD = 0;
    for (let a = 0; a < D; a++) for (let b = 0; b < D; b++){ if (a === b || hasE(par, a, b) || reaches(par, b, a)) continue; const d = jterm(b, par[b].concat([a])) - jterm(b, par[b]); if (d < bestD - 1e-9){ bestD = d; best = { op: 'add', a, b }; } }
    for (let b = 0; b < D; b++) for (const a of par[b]){ const d = jterm(b, par[b].filter(v => v !== a)) - jterm(b, par[b]); if (d < bestD - 1e-9){ bestD = d; best = { op: 'del', a, b }; } }
    for (let b = 0; b < D; b++) for (const a of par[b]){ const plB = par[b].filter(v => v !== a); const tmp = par.map((p, y) => y === b ? plB : p.slice()); if (reaches(tmp, a, b)) continue; const d = (jterm(b, plB) + jterm(a, par[a].concat([b]))) - (jterm(b, par[b]) + jterm(a, par[a])); if (d < bestD - 1e-9){ bestD = d; best = { op: 'rev', a, b }; } }
    it++;
    if (!best) break;
    if (best.op === 'add') par[best.b].push(best.a);
    else if (best.op === 'del') par[best.b] = par[best.b].filter(v => v !== best.a);
    else { par[best.b] = par[best.b].filter(v => v !== best.a); par[best.a].push(best.b); }
    if (it > 800) break;
  }
  return par;
}
const empty = []; for (let i = 0; i < D; i++) empty.push([]);
const runs = [climb(empty)];
for (let s = 0; s < 2; s++){
  const rr = rng(100 + s);
  const o = []; for (let i = 0; i < D; i++) o.push(i);
  for (let i = o.length - 1; i > 0; i--){ const j = Math.floor(rr() * (i + 1)); const t = o[i]; o[i] = o[j]; o[j] = t; }
  const pos = {}; o.forEach((v, i) => { pos[v] = i; });
  const par = []; for (let i = 0; i < D; i++) par.push([]);
  for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) if (a !== b && pos[a] < pos[b] && rr() < 0.2) par[b].push(a);
  runs.push(climb(par));
}
const jscore = (p) => { let s = 0; for (let y = 0; y < D; y++) s += jterm(y, p[y]); return s; };
runs.sort((a, b) => jscore(a) - jscore(b));
const win = runs[0];
const learned = new Set();
for (let b = 0; b < D; b++) for (const a of win[b]) learned.add(a + '->' + b);
let shd = 0, skT = 0, skHit = 0; const flips = [];
for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++){ const a = learned.has(i + '->' + j) ? 1 : 0, b = learned.has(j + '->' + i) ? 1 : 0, c = truth.has(i + '->' + j) ? 1 : 0, d = truth.has(j + '->' + i) ? 1 : 0; if (c || d){ skT++; if (a || b) skHit++; } if (a === c && b === d) continue; shd++; if ((a || b) === (c || d)) flips.push((a ? i + '->' + j : j + '->' + i) + '-not-' + (c ? i + '->' + j : j + '->' + i)); }
console.log('joint SHD=' + shd + ' skeleton=' + skHit + '/' + skT + ' ' + [...learned].sort().join(','));
check('skeleton-perfect', skHit === skT, skHit + '/' + skT);
check('shd-near-exact', shd <= 1, 'SHD=' + shd + (flips.length ? ' flip:' + flips[0] : ''));
if (shd === 1 && flips.length === 1){
  const truePar = []; for (let i = 0; i < D; i++) truePar.push([]);
  for (const e of truth){ const parts = e.split('->'); truePar[parseInt(parts[1])].push(parseInt(parts[0])); }
  const gap = Math.abs(jscore(win) - jscore(truePar));
  check('lone-flip-is-tie', gap < 1e-6, 'scoreGap=' + gap.toExponential(2));
}
if (fail){ console.log('gieslite: ' + fail + ' FAILURES'); process.exit(1); }
console.log('gieslite: all green');
