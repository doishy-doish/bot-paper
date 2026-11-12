// rung: systematicity — monolithic map can't compose. Disjoint pattern sets per direction:
// set A shifted east (d1), set B shifted south (d2); held-out = southeast diagonal (d3) on either set.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function rndGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function shiftGrid(g, dr, dc){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[((rr + dr + 4) % 4) * 4 + ((c + dc + 4) % 4)] = g[rr * 4 + c]; return o; }
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
function acc(W, pairs){ let hit = 0; for (const [x, y] of pairs){ const row = [1].concat(x); let be = 0; for (let o = 0; o < 16; o++){ let s = 0; for (let a = 0; a < 17; a++) s += W[o][a] * row[a]; if ((s >= 0 ? 1 : -1) !== y[o]) be++; } if (be === 0) hit++; } return hit / pairs.length; }
const r = rng(21);
const setA = [], setB = [];
for (let k = 0; k < 9; k++) setA.push(rndGrid(r));
for (let k = 0; k < 9; k++) setB.push(rndGrid(r));
const train = [];
const rt = rng(22);
for (let k = 0; k < 150; k++){ const g = setA[Math.floor(rt() * 9)]; train.push([g, shiftGrid(g, 0, 1)]); }
for (let k = 0; k < 150; k++){ const g = setB[Math.floor(rt() * 9)]; train.push([g, shiftGrid(g, 1, 0)]); }
const W = fitMap(train.map(p => p[0]), train.map(p => p[1]));
const seenA = acc(W, setA.map(g => [g, shiftGrid(g, 0, 1)]));
const seenB = acc(W, setB.map(g => [g, shiftGrid(g, 1, 0)]));
const heldA = acc(W, setA.map(g => [g, shiftGrid(g, 1, 1)]));
const heldB = acc(W, setB.map(g => [g, shiftGrid(g, 1, 1)]));
console.log('seenA=' + seenA.toFixed(3) + ' seenB=' + seenB.toFixed(3) + ' heldA=' + heldA.toFixed(3) + ' heldB=' + heldB.toFixed(3));
check('seen-learned', seenA > 0.9 && seenB > 0.9, seenA.toFixed(3) + '/' + seenB.toFixed(3));
check('heldout-fails', heldA < 0.35 && heldB < 0.35, heldA.toFixed(3) + '/' + heldB.toFixed(3));
if (fail){ console.log('systematic: ' + fail + ' FAILURES'); process.exit(1); }
console.log('systematic: all green');
