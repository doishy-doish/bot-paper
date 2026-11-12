// rung: polyauto — BIC over expansion degrees auto-selects deg 2 with exact features; no human-picked degree.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function randn(r){ let u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
const r = rng(21), N = 500;
const X0 = [], X1 = [], X2 = [];
for (let i = 0; i < N; i++){ const a = randn(r), b = randn(r); X0.push(a); X1.push(b); X2.push(a * a + b + randn(r)); }
function colsFor(deg){ const cols = [[X0, 'X0'], [X1, 'X1']]; for (let d = 2; d <= deg; d++){ cols.push([X0.map(v => v ** d), 'X0^' + d]); cols.push([X1.map(v => v ** d), 'X1^' + d]); } return cols; }
function bestSubset(cols){
  const m = cols.length, pen = Math.log(N);
  let best = null;
  for (let mask = 0; mask < (1 << m); mask++){
    const idx = []; for (let b = 0; b < m; b++) if (mask >> b & 1) idx.push(b);
    const k = idx.length + 1;
    const XtX = Array.from({ length: k }, () => new Array(k).fill(0));
    const Xty = new Array(k).fill(0);
    for (let i = 0; i < N; i++){ const row = [1]; for (const b of idx) row.push(cols[b][0][i]); for (let a = 0; a < k; a++){ Xty[a] += row[a] * X2[i]; for (let c = 0; c < k; c++) XtX[a][c] += row[a] * row[c]; } }
    const M = XtX.map((row, a) => row.concat([Xty[a]]));
    let ok = true;
    for (let c = 0; c < k; c++){ let piv = c; for (let q = c + 1; q < k; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; if (Math.abs(M[piv][c]) < 1e-9){ ok = false; break; } const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c]; for (let q = 0; q < k; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= k; cc++) M[q][cc] -= f * M[c][cc]; } }
    if (!ok) continue;
    const beta = M.map((row, c) => row[k] / (row[c] || 1e-12));
    let rss = 0;
    for (let i = 0; i < N; i++){ let pr = beta[0]; idx.forEach((b, a) => pr += beta[a + 1] * cols[b][0][i]); const e = X2[i] - pr; rss += e * e; }
    const score = N * Math.log(rss / N) + pen * k;
    if (!best || score < best.score) best = { score, features: idx.map(b => cols[b][1]) };
  }
  return best;
}
const res = {};
for (const deg of [1, 2, 3]) res[deg] = bestSubset(colsFor(deg));
for (const deg of [1, 2, 3]) console.log('deg' + deg + ' score=' + res[deg].score.toFixed(1) + ' features={' + res[deg].features.join(',') + '}');
const winner = [1, 2, 3].sort((a, b) => res[a].score - res[b].score)[0];
check('winner-is-deg2', winner === 2, 'deg' + winner);
check('deg2-exact-features', res[2].features.includes('X0^2') && res[2].features.includes('X1') && res[2].features.length === 2, res[2].features.join(','));
check('big-margin', res[1].score - res[2].score > 100, (res[1].score - res[2].score).toFixed(1));
check('cubic-rejected', res[3].features.length === 2, res[3].features.join(','));
if (fail){ console.log('polyauto: ' + fail + ' FAILURES'); process.exit(1); }
console.log('polyauto: all green');
