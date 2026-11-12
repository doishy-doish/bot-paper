// rung: masked 4x4 — random masks ~prior; block masks on block data 100%.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function rndGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function blockGrid(r){ const g = new Array(16); for (let br = 0; br < 2; br++) for (let bc = 0; bc < 2; bc++){ const v = r() < 0.5 ? -1 : 1; for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) g[(br * 2 + i) * 4 + bc * 2 + j] = v; } return g; }
function fitFill(gen, maskFn, K){
  const r = rng(41);
  const D = 17, XtX = Array.from({ length: D }, () => new Array(D).fill(0)), XtY = Array.from({ length: D }, () => new Array(16).fill(0));
  for (let k = 0; k < K; k++){
    const g = gen(r), m = maskFn(r), ms = new Set(m);
    const row = [1]; for (let i = 0; i < 16; i++) row.push(ms.has(i) ? 0 : g[i]);
    for (let a = 0; a < D; a++){ for (let b = 0; b < D; b++) XtX[a][b] += row[a] * row[b]; for (let o = 0; o < 16; o++) XtY[a][o] += row[a] * g[o]; }
  }
  for (let a = 0; a < D; a++) XtX[a][a] += 1e-6;
  const W = [];
  for (let o = 0; o < 16; o++){
    const M = XtX.map((row, a) => row.concat([XtY[a][o]]));
    for (let c = 0; c < D; c++){ let piv = c; for (let q = c + 1; q < D; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < D; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= D; cc++) M[q][cc] -= f * M[c][cc]; } }
    W.push(M.map((row, c) => row[D] / (row[c] || 1e-12)));
  }
  let bits = 0, tot = 0;
  for (let k = 0; k < 200; k++){
    const g = gen(rng(7000 + k)), m = maskFn(rng(7100 + k));
    const row = [1]; const ms = new Set(m); for (let i = 0; i < 16; i++) row.push(ms.has(i) ? 0 : g[i]);
    for (const o of m){ let s = 0; for (let a = 0; a < D; a++) s += W[o][a] * row[a]; bits += ((s >= 0 ? 1 : -1) === g[o] ? 1 : 0); tot++; }
  }
  return bits / tot;
}
function mask3(r){ const idx = [...Array(16).keys()]; for (let i = idx.length - 1; i > 0; i--){ const j = Math.floor(r() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; } return idx.slice(0, 3); }
function blockBits(r){ const br = Math.floor(r() * 2) * 2, bc = Math.floor(r() * 2) * 2; const m = []; const keep = Math.floor(r() * 4); let n = 0; for (let i = 0; i < 2 && n < 3; i++) for (let j = 0; j < 2 && n < 3; j++){ if (i * 2 + j === keep) continue; m.push((br + i) * 4 + bc + j); n++; } return m; }
const rnd = fitFill(rndGrid, mask3, 800);
const blk = fitFill(blockGrid, blockBits, 800);
console.log('random-mask3 bit=' + rnd.toFixed(4) + ' block-mask3 bit=' + blk.toFixed(4));
check('random-near-prior', rnd > 0.5 && rnd < 0.75, rnd.toFixed(4));
check('block-perfect', blk >= 0.99, blk.toFixed(4));
if (fail){ console.log('masked4x4: ' + fail + ' FAILURES'); process.exit(1); }
console.log('masked4x4: all green');
