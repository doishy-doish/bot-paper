// (c) 2026 Andrale Misquitta. All Rights Reserved. See halt-r/LICENSE.
// ref-hopfield.mjs — Nobel-grounded recursive recall, pure JS, zero deps, CPU-ms.
// Classic Hebbian storage (one-shot, no gradients) + modern Hopfield update
// (softmax attention) + adaptive halt on energy delta. Run: node ref-hopfield.mjs
function rng(seed){ let s=seed>>>0; return ()=>{ s=(s*1664525+1013904223)>>>0; return s/4294967296; }; }

// sign patterns in {-1,+1}
function makePatterns(rand, n, d){
  const P=[];
  for(let i=0;i<n;i++){ const v=[]; for(let j=0;j<d;j++) v.push(rand()<0.5?-1:1); P.push(v); }
  return P;
}
function corrupt(v, rand, flips){
  const o=v.slice();
  for(let k=0;k<flips;k++){ const j=Math.floor(rand()*v.length); o[j]*=-1; }
  return o;
}
// classic Hopfield: W = sum x x^T / n, zero diagonal
function hebbian(P){
  const d=P[0].length, W=Array.from({length:d},()=>new Array(d).fill(0));
  for(const x of P) for(let i=0;i<d;i++) for(let j=0;j<d;j++) W[i][j]+=x[i]*x[j]/P.length;
  for(let i=0;i<d;i++) W[i][i]=0;
  return W;
}
function energy(x,W){
  let e=0; const d=x.length;
  for(let i=0;i<d;i++) for(let j=0;j<d;j++) e-=0.5*W[i][j]*x[i]*x[j];
  return e;
}
// async classic update with adaptive halt (stop when no unit flips)
function recallClassic(x0,W,maxSteps){
  let x=x0.slice(); const trace=[energy(x,W)];
  let steps=0;
  for(let t=0;t<maxSteps;t++){
    let flipped=false;
    for(let i=0;i<x.length;i++){
      let h=0; for(let j=0;j<x.length;j++) h+=W[i][j]*x[j];
      const nv=h>=0?1:-1;
      if(nv!==x[i]){ x[i]=nv; flipped=true; }
    }
    steps++;
    trace.push(energy(x,W));
    if(!flipped) break; // adaptive halt: fixed point reached
  }
  return {x,steps,trace};
}
// modern Hopfield single-head update: xi <- X softmax(beta X^T xi)
function recallModern(q0,P,beta,maxSteps,tol){
  const d=q0.length;
  let q=q0.slice(); let steps=0;
  for(let t=0;t<maxSteps;t++){
    const s=P.map(p=>{ let dot=0; for(let j=0;j<d;j++) dot+=p[j]*q[j]; return beta*dot/P[0].length; });
    const m=Math.max(...s), ex=s.map(v=>Math.exp(v-m)), Z=ex.reduce((a,b)=>a+b,0);
    const nq=new Array(d).fill(0);
    for(let k=0;k<P.length;k++) for(let j=0;j<d;j++) nq[j]+=P[k][j]*ex[k]/Z;
    let diff=0; for(let j=0;j<d;j++) diff+=(nq[j]-q[j])**2;
    q=nq; steps++;
    if(Math.sqrt(diff/d)<tol) break; // adaptive halt
  }
  // binarize + nearest stored
  const qb=q.map(v=>v>=0?1:-1);
  let best=0,bd=1e9;
  P.forEach((p,i)=>{ let dd=0; for(let j=0;j<d;j++) dd+=(p[j]-qb[j])**2; if(dd<bd){bd=dd;best=i;} });
  return {qb,best,steps};
}
function main(){
  const rand=rng(7), D=64, NPAT=8, TRIALS=200, FLIPS=8;
  const P=makePatterns(rand,NPAT,D);
  const W=hebbian(P);
  let ok=0, totSteps=0;
  for(let i=0;i<TRIALS;i++){
    const k=Math.floor(rand()*NPAT);
    const q=corrupt(P[k],rand,FLIPS);
    const {x,steps}=recallClassic(q,W,12);
    totSteps+=steps;
    let same=true; for(let j=0;j<D;j++) if(x[j]!==P[k][j]){same=false;break;}
    if(same) ok++;
  }
  console.log("classic: retrieval="+(ok/TRIALS).toFixed(3)+" avgSteps="+(totSteps/TRIALS).toFixed(2)+" (max 12, halt on fixed point)");
  let ok2=0,s2=0;
  for(let i=0;i<TRIALS;i++){
    const k=Math.floor(rand()*NPAT);
    const q=corrupt(P[k],rand,FLIPS).map(v=>v*1.0);
    const r=recallModern(q,P,8.0,12,1e-3);
    s2+=r.steps;
    if(r.best===k) ok2++;
  }
  console.log("modern:  retrieval="+(ok2/TRIALS).toFixed(3)+" avgSteps="+(s2/TRIALS).toFixed(2)+" (max 12, halt on tol)");
  console.log("note: storage was Hebbian one-shot (zero gradient steps); savings come from halting at fixed point.");
}
if(typeof process!=="undefined" && process.argv && process.argv[1] && String(process.argv[1]).endsWith("ref-hopfield.mjs")) main();
export { makePatterns, corrupt, hebbian, energy, recallClassic, recallModern };
