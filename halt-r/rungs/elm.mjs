// rung: elm — hidden layers without backprop (random features + closed-form readout).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function genGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function mirror(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[rr * 4 + c] = g[rr * 4 + (3 - c)]; return o; }
function fill(g){ const o = g.slice(); for (let i = 8; i < 16; i++) o[i] = g[i - 8]; return o; }
function noisy(g, r, p){ return g.map(v => r() < p ? -v : v); }
// ELM: tanh random hidden, ridge readout. X dim inDim, Y dim outDim.
function trainELM(H, inDim, outDim, pairs, noise, seed, lam){
  const r = rng(seed);
  const Win = Array.from({ length: H }, () => { const w = []; for (let i = 0; i < inDim; i++) w.push((r() * 2 - 1) * Math.sqrt(1 / inDim)); return w; });
  const bh = []; for (let h = 0; h < H; h++) bh.push((r() * 2 - 1) * 0.5);
  const HH = [], YY = [];
  for (const [x, y] of pairs){ const xn = noise > 0 ? noisy(x, r, noise) : x; const h = Win.map((w, hh) => Math.tanh(w.reduce((s, v, i) => s + v * xn[i], 0) + bh[hh])); HH.push(h); YY.push(y); }
  const D1 = H + 1;
  const A = Array.from({ length: D1 }, () => new Array(D1).fill(0));
  const B = Array.from({ length: D1 }, () => new Array(outDim).fill(0));
  for (let k = 0; k < HH.length; k++){ const row = [1].concat(HH[k]); for (let a = 0; a < D1; a++){ for (let b = 0; b < D1; b++) A[a][b] += row[a] * row[b]; for (let o = 0; o < outDim; o++) B[a][o] += row[a] * YY[k][o]; } }
  for (let a = 0; a < D1; a++) A[a][a] += lam;
  const Wout = [];
  for (let o = 0; o < outDim; o++){
    const M = A.map((row, a) => row.concat([B[a][o]]));
    for (let c = 0; c < D1; c++){ let piv = c; for (let q = c + 1; q < D1; q++) if (Math.abs(M[q][c]) > Math.abs(M[piv][c])) piv = q; const t = M[c]; M[c] = M[piv]; M[piv] = t; const dd = M[c][c] || 1e-12; for (let q = 0; q < D1; q++){ if (q === c) continue; const f = M[q][c] / dd; for (let cc = c; cc <= D1; cc++) M[q][cc] -= f * M[c][cc]; } }
    Wout.push(M.map((row, c) => row[D1] / (row[c] || 1e-12)));
  }
  return { Win, bh, Wout };
}
function evalELM(model, pairs, noise, seed){
  const r = rng(seed); const { Win, bh, Wout } = model;
  const inDim = Win[0].length, outDim = Wout.length;
  let bits = 0, tot = 0, exact = 0;
  for (const [x, y] of pairs){
    const xn = noise > 0 ? noisy(x, r, noise) : x;
    const h = Win.map((w, hh) => Math.tanh(w.reduce((s, v, i) => s + v * xn[i], 0) + bh[hh]));
    const row = [1].concat(h);
    let be = 0;
    for (let o = 0; o < outDim; o++){ let s = 0; for (let a = 0; a <= Win.length; a++) s += Wout[o][a] * row[a]; const p = s >= 0 ? 1 : -1; bits += (p === y[o] ? 1 : 0); if (p !== y[o]) be++; }
    tot += outDim; if (be === 0) exact++;
  }
  return { bit: bits / tot, exact: exact / pairs.length };
}
function pairs4x4(rule, K, seed){ const r = rng(seed); const ps = []; for (let k = 0; k < K; k++){ const x = genGrid(r); ps.push([x, rule(x)]); } return ps; }
let m = trainELM(256, 16, 16, pairs4x4(fill, 200, 51), 0, 52, 1e-3);
let e = evalELM(m, pairs4x4(fill, 200, 53), 0, 54);
console.log('fill-clean H256 bit=' + e.bit.toFixed(4) + ' exact=' + e.exact.toFixed(3));
check('fill-clean', e.exact === 1.0 && e.bit === 1.0, e.exact.toFixed(3));
m = trainELM(256, 16, 16, pairs4x4(fill, 1000, 51), 0.1, 52, 1e-3);
e = evalELM(m, pairs4x4(fill, 200, 53), 0.1, 54);
console.log('fill-noisy10 H256 bit=' + e.bit.toFixed(4) + ' exact=' + e.exact.toFixed(3));
check('fill-noisy10', e.exact > 0.25 && e.bit > 0.85, e.exact.toFixed(3));
m = trainELM(256, 16, 16, pairs4x4(mirror, 1000, 51), 0.1, 52, 1e-3);
e = evalELM(m, pairs4x4(mirror, 200, 53), 0.1, 54);
console.log('mirror-noisy10 H256 bit=' + e.bit.toFixed(4) + ' exact=' + e.exact.toFixed(3));
check('mirror-noisy10', e.bit > 0.85 && e.exact < 0.3, e.exact.toFixed(3));
// xor-sanity: 2-bit XOR, H=64
m = trainELM(64, 2, 1, [[[-1, -1], [-1]], [[-1, 1], [1]], [[1, -1], [1]], [[1, 1], [-1]]], 0, 55, 1e-3);
e = evalELM(m, [[[ -1, -1], [-1]], [[-1, 1], [1]], [[1, -1], [1]], [[1, 1], [-1]]], 0, 56);
console.log('xor-sanity H64 bit=' + e.bit.toFixed(4));
check('xor-sanity', e.bit === 1.0, e.bit.toFixed(3));
if (fail){ console.log('elm: ' + fail + ' FAILURES'); process.exit(1); }
console.log('elm: all green');
