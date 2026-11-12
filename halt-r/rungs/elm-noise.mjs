// rung: elm-noise — size H to the noise, not the task. 8-bit 1D mirror + xor-neighbor.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const D = 8;
function genPair(r, task){ const x = []; for (let i = 0; i < D; i++) x.push(r() < 0.5 ? -1 : 1); let y; if (task === 'mirror') y = x.slice().reverse(); else y = x.map((v, i) => v !== x[(i + 1) % D] ? 1 : -1); return [x, y]; }
function noisy(v, r, p){ return v.map(x => r() < p ? -x : x); }
function trainELM(H, pairs, noise, seed){
  const r = rng(seed);
  const Win = Array.from({ length: H }, () => { const w = []; for (let i = 0; i < D; i++) w.push((r() * 2 - 1) * Math.sqrt(2 / D)); return w; });
  const bh = []; for (let h = 0; h < H; h++) bh.push((r() * 2 - 1) * 0.5);
  const HH = [], YY = [];
  for (const [x, y] of pairs){ const xn = noisy(x, r, noise); const h = Win.map((w, hh) => Math.tanh(w.reduce((s, v, i) => s + v * xn[i], 0) + bh[hh])); HH.push(h); YY.push(y); }
  const D1 = H + 1, lam = 1e-3;
  const A = Array.from({ length: D1 }, () => new Array(D1).fill(0));
  const B = Array.from({ length: D1 }, () => new Array(D).fill(0));
  for (let k = 0; k < HH.length; k++){ const row = [1].concat(HH[k]); for (let a = 0; a < D1; a++){ for (let b = 0; b < D1; b++) A[a][b] += row[a] * row[b]; for (let o = 0; o < D; o++) B[a][o] += row[a] * YY[k][o]; } }
  for (let a = 0; a < D1; a++) A[a][a] += lam;
  const Wout = [];
  for (let o = 0; o < D; o++){
    const M = A.map((row, a) => row.concat([B[a][o]]));
    for (let c = 0; c < D1; c++){ let piv = c; for (let q = c + 1; q < D1; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < D1; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= D1; cc++) M[q][cc] -= f * M[c][cc]; } }
    Wout.push(M.map((row, c) => row[D1] / (row[c] || 1e-12)));
  }
  return { Win, bh, Wout };
}
function evalELM(model, task, noise, seed, n){
  const r = rng(seed); const { Win, bh, Wout } = model;
  let bits = 0, tot = 0, exact = 0;
  for (let k = 0; k < n; k++){
    const [x, y] = genPair(r, task);
    const xn = noisy(x, r, noise);
    const h = Win.map((w, hh) => Math.tanh(w.reduce((s, v, i) => s + v * xn[i], 0) + bh[hh]));
    const row = [1].concat(h);
    let be = 0;
    for (let o = 0; o < D; o++){ let s = 0; for (let a = 0; a <= Win.length; a++) s += Wout[o][a] * row[a]; const p = s >= 0 ? 1 : -1; bits += (p === y[o] ? 1 : 0); if (p !== y[o]) be++; }
    tot += D; if (be === 0) exact++;
  }
  return { bit: bits / tot, exact: exact / n };
}
const K = 500;
function run(task, noise, H){
  const rp = rng(2000); const ps = []; for (let k = 0; k < K; k++) ps.push(genPair(rp, task));
  const model = trainELM(H, ps, noise, 11);
  return evalELM(model, task, noise, 9000, 200);
}
const m64 = run('mirror', 0, 64), m256 = run('mirror', 0, 256);
console.log('mirror clean H64=' + m64.bit.toFixed(4) + '/' + m64.exact.toFixed(3) + ' H256=' + m256.bit.toFixed(4) + '/' + m256.exact.toFixed(3));
check('mirror-clean', m64.bit === 1.0 && m256.bit === 1.0, '');
const m64n = run('mirror', 0.1, 64), m256n = run('mirror', 0.1, 256);
console.log('mirror noise10 H64=' + m64n.bit.toFixed(4) + '/' + m64n.exact.toFixed(3) + ' H256=' + m256n.bit.toFixed(4) + '/' + m256n.exact.toFixed(3));
check('small-beats-big-under-noise', m64n.bit > m256n.bit && m64n.bit > 0.85, m64n.bit.toFixed(4) + ' vs ' + m256n.bit.toFixed(4));
const x256 = run('xorneighbor', 0, 256), x64 = run('xorneighbor', 0, 64);
console.log('xor clean H256=' + x256.bit.toFixed(4) + '/' + x256.exact.toFixed(3) + ' H64=' + x64.bit.toFixed(4));
check('xor-clean', x256.bit > 0.95, x256.bit.toFixed(4));
const x20 = run('xorneighbor', 0.2, 256);
console.log('xor noise20 H256 bit=' + x20.bit.toFixed(4));
check('xor-noise20-near-chance', x20.bit < 0.65, x20.bit.toFixed(4));
if (fail){ console.log('elm-noise: ' + fail + ' FAILURES'); process.exit(1); }
console.log('elm-noise: all green');
