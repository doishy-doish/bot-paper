// rung: operator-learning — least squares learns linear rules exactly (closed form, CPU ms).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
// 4x4 bipolar grids; rules: mirror L-R, shift-east wrap, fill (bottom half = top half)
function genGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function mirror(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[rr * 4 + c] = g[rr * 4 + (3 - c)]; return o; }
function shiftE(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[rr * 4 + c] = g[rr * 4 + ((c + 3) % 4)]; return o; }
function fill(g){ const o = g.slice(); for (let i = 8; i < 16; i++) o[i] = g[i - 8]; return o; }
function noisy(g, r, p){ return g.map(v => r() < p ? -v : v); }
// least squares: Y = X W (X has bias col). Returns W ((17)x16).
function fitMap(X, Y){
  const K = 17, n = X.length;
  const XtX = Array.from({ length: K }, () => new Array(K).fill(0));
  const XtY = Array.from({ length: K }, () => new Array(16).fill(0));
  for (let i = 0; i < n; i++){ const row = [1].concat(X[i]); for (let a = 0; a < K; a++){ for (let b = 0; b < K; b++) XtX[a][b] += row[a] * row[b]; for (let o = 0; o < 16; o++) XtY[a][o] += row[a] * Y[i][o]; } }
  for (let a = 0; a < K; a++) XtX[a][a] += 1e-9;
  const W = [];
  for (let o = 0; o < 16; o++){
    const M = XtX.map((row, a) => row.concat([XtY[a][o]]));
    for (let c = 0; c < K; c++){ let piv = c; for (let q = c + 1; q < K; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < K; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= K; cc++) M[q][cc] -= f * M[c][cc]; } }
    W.push(M.map((row, c) => row[K] / (row[c] || 1e-12)));
  }
  return W; // W[o][a]
}
function applyMap(W, x){ const row = [1].concat(x); const o = []; for (let i = 0; i < 16; i++){ let s = 0; for (let a = 0; a < 17; a++) s += W[i][a] * row[a]; o.push(s >= 0 ? 1 : -1); } return o; }
function runTask(rule, K, noise, seed){
  const r = rng(seed);
  const X = [], Y = [];
  for (let k = 0; k < K; k++){ const g = genGrid(r); X.push(noise > 0 ? noisy(g, r, noise) : g); Y.push(rule(g)); }
  const W = fitMap(X, Y);
  let bits = 0, tot = 0, exact = 0;
  const tg = rng(9000), tn = rng(9500);
  for (let k = 0; k < 200; k++){ const g = genGrid(tg); const q = noise > 0 ? noisy(g, tn, noise) : g; const p = applyMap(W, q); const y = rule(g); let be = 0; for (let o = 0; o < 16; o++){ bits += (p[o] === y[o] ? 1 : 0); if (p[o] !== y[o]) be++; } tot += 16; if (be === 0) exact++; }
  return { bit: bits / tot, exact: exact / 200 };
}
const K = 200;
for (const [name, rule] of [['mirror', mirror], ['shiftE', shiftE], ['fill', fill]]){
  const res = runTask(rule, K, 0, 100);
  console.log(name + ' clean bit=' + res.bit.toFixed(4) + ' exact=' + res.exact.toFixed(3));
  check(name + '-clean-exact', res.exact === 1.0, res.exact.toFixed(3));
}
const nz = runTask(mirror, 500, 0.1, 100);
console.log('mirror-noisy10 bit=' + nz.bit.toFixed(4) + ' exact=' + nz.exact.toFixed(3));
check('mirror-noisy10-bit', nz.bit > 0.85, nz.bit.toFixed(4));
check('mirror-noisy10-exact-low', nz.exact < 0.7, nz.exact.toFixed(3));
if (fail){ console.log('operator: ' + fail + ' FAILURES'); process.exit(1); }
console.log('operator: all green');
