// rung: active — truth-free target picking: restart disagreement votes the intervention targets; joint GIES closes.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const D = 20, SEED = 41;
const r0 = rng(SEED);
const truth = new Set(), W = {};
for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++){ if (r0() < 0.15){ truth.add(i + '->' + j); W[i + '->' + j] = ((r0() < 0.5 ? -1 : 1) * (1 + r0() * 1.5)); } }
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
function climber(terms){
  const jterm = (y, plist) => terms.reduce((s, sc) => s + sc.term(y, plist), 0);
  const hasE = (par, a, b) => par[b].indexOf(a) >= 0;
  function reaches(par, from, to){ const seen = {}; seen[from] = 1; const st = [from]; while (st.length){ const c = st.pop(); for (let j = 0; j < D; j++) if (par[j].indexOf(c) >= 0 && !seen[j]){ if (j === to) return true; seen[j] = 1; st.push(j); } } return false; }
  function one(par0){
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
      if (it > 1500) break;
    }
    return par;
  }
  const jscore = (p) => { let s = 0; for (let y = 0; y < D; y++) s += jterm(y, p[y]); return s; };
  return { one, jscore };
}
function randStart(seed, p){
  const rr = rng(seed);
  const o = []; for (let i = 0; i < D; i++) o.push(i);
  for (let i = o.length - 1; i > 0; i--){ const j = Math.floor(rr() * (i + 1)); const t = o[i]; o[i] = o[j]; o[j] = t; }
  const pos = {}; o.forEach((v, i) => { pos[v] = i; });
  const par = []; for (let i = 0; i < D; i++) par.push([]);
  for (let a = 0; a < D; a++) for (let b = 0; b < D; b++) if (a !== b && pos[a] < pos[b] && rr() < p) par[b].push(a);
  return par;
}
function edgeSet(par){ const e = new Set(); for (let b = 0; b < D; b++) for (const a of par[b]) e.add(a + '->' + b); return e; }
function shdVs(L, T){ let s = 0; for (let i = 0; i < D; i++) for (let j = i + 1; j < D; j++){ const a = L.has(i + '->' + j) ? 1 : 0, b = L.has(j + '->' + i) ? 1 : 0, c = T.has(i + '->' + j) ? 1 : 0, d = T.has(j + '->' + i) ? 1 : 0; if (a !== c || b !== d) s++; } return s; }
// Phase 0: obs restarts (empty + randoms, same protocol as every rung) -> instability vote (NO truth used)
const Xo = gen(2000, rng(SEED + 1), null);
const scO = scorer(Xo, new Set());
const { one } = climber([scO]);
const emptyPar = []; for (let i = 0; i < D; i++) emptyPar.push([]);
const restarts = [one(emptyPar)].concat([100, 101, 102, 103, 104, 105].map(s => one(randStart(s, 0.15))));
const votes = {};
for (const par of restarts) for (const e of edgeSet(par)) votes[e] = (votes[e] || 0) + 1;
const inst = {};
for (const [e, v] of Object.entries(votes)){
  const [a, b] = e.split('->').map(Number);
  const rev = votes[b + '->' + a] || 0;
  if (v < 7 || rev > 0){ const wgt = (v < 7 ? 1 : 0) + (rev > 0 ? 2 : 0); inst[a] = (inst[a] || 0) + wgt; inst[b] = (inst[b] || 0) + wgt; }
}
const ranked = Object.entries(inst).sort((a, b) => b[1] - a[1]);
const targets = ranked.slice(0, 2).map(e => parseInt(e[0]));
console.log('truth-free targets=[' + targets.join(',') + '] instabilityTop=' + ranked.slice(0, 3).map(e => e[0] + ':' + e[1]).join(' '));
// Phase 1: intervene truth-free targets, joint climb
const rI = rng(78);
const datas = [{ X: Xo, skip: new Set() }];
for (const t of targets){ const Xa = gen(200, rI, { t, v: 3 }).concat(gen(200, rI, { t, v: -3 })); datas.push({ X: Xa, skip: new Set([t]) }); }
const scorers = datas.map(d => scorer(d.X, d.skip));
const J = climber(scorers);
const jr = [J.one(emptyPar)].concat([100, 101, 102].map(s => J.one(randStart(s, 0.15))));
jr.sort((a, b) => J.jscore(a) - J.jscore(b));
const win = edgeSet(jr[0]);
const shd = shdVs(win, truth);
console.log('truth-free joint SHD=' + shd);
check('active-closes', shd <= 4, 'SHD=' + shd);
check('targets-sane', targets.length === 2, targets.join(','));
if (fail){ console.log('active: ' + fail + ' FAILURES'); process.exit(1); }
console.log('active: all green');
