// rung: ep-hidden — equilibrium propagation solves 2-bit XOR (credit assignment without backprop works).
let fail = 0;
function check(name, cond, detail){ console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || '')); if (!cond) fail++; }
function rng(seed){ let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
const DATA = [{ x: [-1, -1], y: -1 }, { x: [-1, 1], y: 1 }, { x: [1, -1], y: 1 }, { x: [1, 1], y: -1 }];
function trainEP(H, beta, lr, epochs, steps){
  const I = 2, O = 1, N = I + H + O;
  const r = rng(99);
  const W = Array.from({ length: N }, () => new Array(N).fill(0));
  const bb = new Array(N).fill(0);
  for (let i = 0; i < N; i++) for (let j = i + 1; j < N; j++){ const w = (r() * 2 - 1) * 0.3; W[i][j] = w; W[j][i] = w; }
  function settle(x, t, nudge){
    const u = new Array(N).fill(0);
    for (let s = 0; s < steps; s++){
      const ru = u.map(Math.tanh);
      const nu = new Array(N);
      for (let i = 0; i < N; i++){
        if (i < I){ nu[i] = x[i]; continue; }
        let net = bb[i];
        for (let j = 0; j < N; j++) net += W[i][j] * ru[j];
        if (nudge && i >= I + H) net += beta * (t - ru[i]);
        nu[i] = net;
      }
      for (let i = 0; i < N; i++) u[i] = nu[i];
    }
    return u.map(Math.tanh);
  }
  for (let ep = 0; ep < epochs; ep++) for (const { x, y } of DATA){
    const rf = settle(x, y, false), rn = settle(x, y, true);
    const f = lr / beta;
    for (let i = I; i < N; i++) for (let j = i + 1; j < N; j++){ const d = f * (rn[i] * rn[j] - rf[i] * rf[j]); W[i][j] += d; W[j][i] += d; }
    for (let i = I; i < N; i++) bb[i] += f * (rn[i] - rf[i]);
    for (let i = 0; i < I; i++) for (let j = I; j < N; j++){ const d = f * (rn[j] * x[i] - rf[j] * x[i]); W[i][j] += d; W[j][i] += d; }
  }
  let ok = 0;
  for (const { x, y } of DATA){ const ru = settle(x, y, false); if ((ru[I + H] >= 0 ? 1 : -1) === y) ok++; }
  return ok;
}
const ok = trainEP(8, 0.5, 0.05, 500, 50);
console.log('xor-2bit EP 4-pattern acc=' + ok + '/4');
check('ep-xor', ok === 4, ok + '/4');
if (fail){ console.log('ep-hidden: ' + fail + ' FAILURES'); process.exit(1); }
console.log('ep-hidden: all green');
