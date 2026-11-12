// rung: masked-scale — self-supervision works iff the mask spares signal (8x8 independent 2x2 blocks).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const S = 8, B = 2, NB = S / B;
function genGrid(r){ const g = new Array(S * S); for (let br = 0; br < NB; br++) for (let bc = 0; bc < NB; bc++){ const v = r() < 0.5 ? -1 : 1; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) g[(br * B + i) * S + bc * B + j] = v; } return g; }
function maskIdx(r, frac, mode){
  if (mode === 'random'){ const idx = [...Array(S * S).keys()]; for (let i = idx.length - 1; i > 0; i--){ const j = Math.floor(r() * (i + 1)); const t = idx[i]; idx[i] = idx[j]; idx[j] = t; } return idx.slice(0, Math.round(S * S * frac)); }
  const blocks = [...Array(NB * NB).keys()];
  for (let i = blocks.length - 1; i > 0; i--){ const j = Math.floor(r() * (i + 1)); const t = blocks[i]; blocks[i] = blocks[j]; blocks[j] = t; }
  const out = []; const nb = Math.max(1, Math.round(S * S * frac / (B * B)));
  for (let k = 0; k < nb && k < blocks.length; k++){ const br = Math.floor(blocks[k] / NB), bc = blocks[k] % NB; for (let i = 0; i < B; i++) for (let j = 0; j < B; j++) out.push((br * B + i) * S + bc * B + j); }
  return out;
}
function run(frac, mode){
  const r = rng(5);
  const K = 800, D = 65;
  const XtX = Array.from({ length: D }, () => new Array(D).fill(0));
  const Xty = Array.from({ length: D }, () => new Array(64).fill(0));
  for (let k = 0; k < K; k++){
    const g = genGrid(r), m = maskIdx(r, frac, mode), ms = new Set(m);
    const row = [1]; for (let i = 0; i < 64; i++) row.push(ms.has(i) ? 0 : g[i]);
    for (let a = 0; a < D; a++){ for (let b = 0; b < D; b++) XtX[a][b] += row[a] * row[b]; for (let o = 0; o < 64; o++) Xty[a][o] += row[a] * g[o]; }
  }
  for (let a = 0; a < D; a++) XtX[a][a] += 1e-6;
  const W = [];
  for (let o = 0; o < 64; o++){
    const M = XtX.map((row, a) => row.concat([Xty[a][o]]));
    for (let c = 0; c < D; c++){ let piv = c; for (let q = c + 1; q < D; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < D; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= D; cc++) M[q][cc] -= f * M[c][cc]; } }
    W.push(M.map((row, c) => row[D] / (row[c] || 1e-12)));
  }
  let bits = 0, tot = 0;
  const rt = rng(777);
  for (let k = 0; k < 200; k++){
    const g = genGrid(rt), m = maskIdx(rt, frac, mode), ms = new Set(m);
    const row = [1]; for (let i = 0; i < 64; i++) row.push(ms.has(i) ? 0 : g[i]);
    for (const o of m){ let s = 0; for (let a = 0; a < D; a++) s += W[o][a] * row[a]; bits += ((s >= 0 ? 1 : -1) === g[o] ? 1 : 0); tot++; }
  }
  return bits / tot;
}
const r6 = run(0.0625, 'random'), b6 = run(0.0625, 'block');
const r50 = run(0.5, 'random'); let bmax = 0; for (const f of [0.0625, 0.125, 0.25, 0.5]) bmax = Math.max(bmax, run(f, 'block'));
console.log('random6=' + r6.toFixed(4) + ' block6=' + b6.toFixed(4) + ' random50=' + r50.toFixed(4) + ' blockMax=' + bmax.toFixed(4));
check('random-sparse-perfect', r6 >= 0.99, r6.toFixed(4));
check('random-dense-holds', r50 > 0.9, r50.toFixed(4));
check('block-always-chance', bmax < 0.6, bmax.toFixed(4));
if (fail){ console.log('masked-scale: ' + fail + ' FAILURES'); process.exit(1); }
console.log('masked-scale: all green');
