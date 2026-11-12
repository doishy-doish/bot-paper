// rung: hebbian-stack — single map already AT the noise ceiling; stacking adds nothing (negative result).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function genGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function mirror(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[rr * 4 + c] = g[rr * 4 + (3 - c)]; return o; }
function noisy(g, r, p){ return g.map(v => r() < p ? -v : v); }
function fitMap(X, Y){
  const K = 17;
  const XtX = Array.from({ length: K }, () => new Array(K).fill(0));
  const XtY = Array.from({ length: K }, () => new Array(16).fill(0));
  for (let i = 0; i < X.length; i++){ const row = [1].concat(X[i]); for (let a = 0; a < K; a++){ for (let b = 0; b < K; b++) XtX[a][b] += row[a] * row[b]; for (let o = 0; o < 16; o++) XtY[a][o] += row[a] * Y[i][o]; } }
  for (let a = 0; a < K; a++) XtX[a][a] += 1e-9;
  const W = [];
  for (let o = 0; o < 16; o++){
    const M = XtX.map((row, a) => row.concat([XtY[a][o]]));
    for (let c = 0; c < K; c++){ let piv = c; for (let q = c + 1; q < K; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < K; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= K; cc++) M[q][cc] -= f * M[c][cc]; } }
    W.push(M.map((row, c) => row[K] / (row[c] || 1e-12)));
  }
  return W;
}
function applyMap(W, x){ const row = [1].concat(x); const o = []; for (let i = 0; i < 16; i++){ let s = 0; for (let a = 0; a < 17; a++) s += W[i][a] * row[a]; o.push(s >= 0 ? 1 : -1); } return o; }
function ham(a, b){ let d = 0; for (let i = 0; i < a.length; i++) d += Math.abs(a[i] - b[i]); return d; }
function run(noise){
  const r = rng(11);
  const K = 300, X = [], Y = [], clean = [];
  for (let k = 0; k < K; k++){ const g = genGrid(r); clean.push(g); X.push(noisy(g, r, noise)); Y.push(mirror(g)); }
  const direct = fitMap(X, Y);
  const cleanMap = fitMap(clean, clean.map(mirror));
  let dHit = 0, sHit = 0, ceil = 0, tot = 200;
  const rg = rng(4000), rn = rng(4500);
  for (let k = 0; k < tot; k++){
    const g = genGrid(rg);
    const q = noisy(g, rn, noise);
    if (ham(q, g) === 0) ceil++;
    if (applyMap(direct, q).every((v, i) => v === mirror(g)[i])) dHit++;
    // stacked: nearest-clean denoiser then clean map
    let bi = 0, bd = 1e9;
    for (let j = 0; j < K; j++){ const d = ham(q, clean[j]); if (d < bd){ bd = d; bi = j; } }
    if (applyMap(cleanMap, clean[bi]).every((v, i) => v === mirror(g)[i])) sHit++;
  }
  return { direct: dHit / tot, stacked: sHit / tot, ceiling: ceil / tot };
}
for (const nz of [0.02, 0.1]){
  const res = run(nz);
  console.log('noise=' + nz + ' direct=' + res.direct.toFixed(3) + ' stacked=' + res.stacked.toFixed(3) + ' ceiling=' + res.ceiling.toFixed(3));
  check('direct-at-ceiling-n' + nz, Math.abs(res.direct - res.ceiling) < 0.05, res.direct.toFixed(3));
  check('stacking-dead-n' + nz, res.stacked <= res.direct + 0.05, res.stacked.toFixed(3));
}
if (fail){ console.log('hebbian: ' + fail + ' FAILURES'); process.exit(1); }
console.log('hebbian: all green');
