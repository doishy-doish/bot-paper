// rung: contrastive — online perceptron-form contrastive-Hebbian learns mirror; xor control stays ~chance.
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function genGrid(r){ const g = []; for (let i = 0; i < 16; i++) g.push(r() < 0.5 ? -1 : 1); return g; }
function mirror(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++) o[rr * 4 + c] = g[rr * 4 + (3 - c)]; return o; }
function xorNeigh(g){ const o = new Array(16); for (let rr = 0; rr < 4; rr++) for (let c = 0; c < 4; c++){ const a = g[rr * 4 + c], b = g[rr * 4 + ((c + 1) % 4)]; o[rr * 4 + c] = (a !== b ? 1 : -1); } return o; }
// online contrastive: free = sign(Wx+b); clamped = target; dW = lr*(clamped-free) x^T (perceptron form)
function trainContrastive(rule, K, epochs, lr, seed){
  const r = rng(seed);
  const W = Array.from({ length: 16 }, () => new Array(16).fill(0));
  const b = new Array(16).fill(0);
  const data = []; { const dr = mulberry(seed); for (let k = 0; k < K; k++){ const x = genGrid(dr); data.push({ x, y: rule(x) }); } }
  function mulberry(s){ let t = s >>> 0; return () => { t = (t + 0x6D2B79F5) >>> 0; let z = t; z = Math.imul(z ^ (z >>> 15), z | 1); z ^= z + Math.imul(z ^ (z >>> 7), z | 61); return ((z ^ (z >>> 14)) >>> 0) / 4294967296; }; }
  const free = (x) => { const o = []; for (let j = 0; j < 16; j++){ let s = b[j]; for (let i = 0; i < 16; i++) s += W[j][i] * x[i]; o.push(s >= 0 ? 1 : -1); } return o; };
  for (let ep = 0; ep < epochs; ep++) for (const { x, y } of data){ const yf = free(x); for (let j = 0; j < 16; j++){ const e = y[j] - yf[j]; if (e !== 0){ for (let i = 0; i < 16; i++) W[j][i] += lr * e * x[i]; b[j] += lr * e; } } }
  let bits = 0, tot = 0, exact = 0;
  for (let k = 0; k < 200; k++){ const x = genGrid(rng(5000 + k)); const p = free(x); const y = rule(x); let be = 0; for (let o = 0; o < 16; o++){ bits += (p[o] === y[o] ? 1 : 0); if (p[o] !== y[o]) be++; } tot += 16; if (be === 0) exact++; }
  return { bit: bits / tot, exact: exact / 200 };
}
const mir = trainContrastive(mirror, 500, 60, 0.05, 7);
console.log('mirror K=500 bit=' + mir.bit.toFixed(4) + ' exact=' + mir.exact.toFixed(3));
check('contrastive-mirror', mir.exact >= 0.99 && mir.bit >= 0.999, mir.exact.toFixed(3));
const xo = trainContrastive(xorNeigh, 500, 60, 0.05, 7);
console.log('xor-neighbor control bit=' + xo.bit.toFixed(4) + ' exact=' + xo.exact.toFixed(3));
check('xor-control-chance', xo.bit < 0.65 && xo.exact < 0.05, xo.bit.toFixed(3));
if (fail){ console.log('contrastive: ' + fail + ' FAILURES'); process.exit(1); }
console.log('contrastive: all green');
