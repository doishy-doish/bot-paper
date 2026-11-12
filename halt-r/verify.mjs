// halt-r/verify.mjs — public-runner verification: Hopfield recall + Task-C memory baseline.
// CPU seconds, zero deps, no secrets, no network. Exit non-zero on regression.
import { makePatterns, corrupt, hebbian, recallClassic, recallModern } from './ref-hopfield.mjs';
function rng(seed){ let s=seed>>>0; return ()=>((s=(s*1664525+1013904223)>>>0)/4294967296); }
let fail = 0;
function check(name, cond, detail){
  console.log((cond ? 'PASS' : 'FAIL') + ' ' + name + ' ' + (detail || ''));
  if(!cond) fail++;
}
// 1. Classic Hopfield: seen-pattern retrieval
{
  const rand = rng(7), D = 64, NPAT = 8, TRIALS = 200, FLIPS = 8;
  const P = makePatterns(rand, NPAT, D);
  const W = hebbian(P);
  let ok = 0, steps = 0;
  for(let i=0;i<TRIALS;i++){
    const k = Math.floor(rand()*NPAT);
    const q = corrupt(P[k], rand, FLIPS);
    const r = recallClassic(q, W, 12);
    steps += r.steps;
    if(r.x.every((v,j)=>v===P[k][j])) ok++;
  }
  const acc = ok/TRIALS, avg = steps/TRIALS;
  check('classic-retrieval', acc >= 0.90, 'acc='+acc.toFixed(3));
  check('classic-halt-saves', avg < 6.0, 'avgSteps='+avg.toFixed(2)+' (max 12)');
}
// 2. Modern Hopfield beta=8: seen-pattern retrieval
{
  const rand = rng(7), D = 64, NPAT = 8, TRIALS = 200, FLIPS = 8;
  const P = makePatterns(rand, NPAT, D);
  let ok = 0, steps = 0;
  for(let i=0;i<TRIALS;i++){
    const k = Math.floor(rand()*NPAT);
    const q = corrupt(P[k], rand, FLIPS).map(v=>v*1.0);
    const r = recallModern(q, P, 8.0, 12, 1e-3);
    steps += r.steps;
    if(r.best === k) ok++;
  }
  const acc = ok/TRIALS, avg = steps/TRIALS;
  check('modern-retrieval', acc >= 0.95, 'acc='+acc.toFixed(3));
  check('modern-halt-saves', avg < 6.0, 'avgSteps='+avg.toFixed(2)+' (max 12)');
}
// 3. Task-C: mirror must FAIL on memory alone (guards the baseline honest)
{
  const r = rng(101);
  const ham = (a,b)=>{ let d=0; for(let i=0;i<a.length;i++) d+=Math.abs(a[i]-b[i]); return d; };
  const ex = (a,b)=>a.every((v,i)=>v===b[i]);
  const mir = (g)=>{ const o=new Array(16); for(let rr=0;rr<4;rr++) for(let c=0;c<4;c++) o[rr*4+c]=g[rr*4+(3-c)]; return o; };
  const rnd = ()=>{ const v=[]; for(let i=0;i<16;i++) v.push(r()<0.35?1:0); return v; };
  const K=60, T=150, store=[];
  for(let i=0;i<K;i++){ const x=rnd(); store.push({x, y:mir(x)}); }
  let hit=0;
  for(let i=0;i<T;i++){
    const x=rnd(), y=mir(x);
    let bi=0, bd=1e9;
    for(let k=0;k<K;k++){ const d=ham(x,store[k].x); if(d<bd){bd=d;bi=k;} }
    if(ex(store[bi].y,y)) hit++;
  }
  const acc = hit/T;
  check('mirror-memory-fails', acc < 0.20, 'acc='+acc.toFixed(3)+' (memory must NOT solve rules)');
}
if(fail){ console.log('VERIFY: '+fail+' FAILURES'); process.exit(1); }
console.log('VERIFY: all green');
