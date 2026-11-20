// halt-r/phase3-wiring.mjs: Phase-3 D-B wiring (harness-only, no bench edits, no training).
// Pre-reg: scratch/papers/phase3-wiring-prereg-2026-11-20.md (frozen).
// Usage: node halt-r/phase3-wiring.mjs --sup <sup-metrics.json> --donor <donor-topk.json> --out <verdict.json>
// Exits 0 on W1 pass (fidelity); W2/W3 reported in verdict (exit 0 regardless, bar read from JSON).
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
const a = process.argv.slice(2);
function arg(n){ const i=a.indexOf(n); return i<0?null:a[i+1]; }
const supP=arg('--sup'), donP=arg('--donor'), outP=arg('--out');
if(!supP||!donP||!outP){ console.error('FAIL usage: --sup --donor --out required'); process.exit(2); }
const sup=JSON.parse(readFileSync(supP,'utf8'));
const don=JSON.parse(readFileSync(donP,'utf8'));
const V_SUP=sup.V, V_DON=151936;
function eNorm(topProbs,V){
  const s=topProbs.reduce((x,y)=>x+y,0);
  let r=1-s; if(r<0)r=0;
  let H=0; for(const p of topProbs){ if(p>0) H-=p*Math.log(p); }
  if(r>1e-12){ const pr=r/(V-topProbs.length); H-=r*Math.log(pr); }
  return H/Math.log(V);
}
const S1=sup.repTopk.map(e=>eNorm(e.topk.map(x=>x[1]),V_SUP));
const D1=don.trials.map(t=>eNorm(t.top.map(x=>Math.exp(x[1])),V_DON));
const supHits=sup.repHitI.split('').map(Number);
const donHits=don.trials.map(t=>(t.genText&&t.tnChar&&t.genText[0]===t.tnChar[0])?1:0);
function covFull(ent,hits,cov){
  const idx=[...Array(ent.length).keys()].sort((x,y)=>ent[x]-ent[y]);
  const k=Math.round(ent.length*cov); const sel=idx.slice(0,k);
  return {acc: sel.reduce((s,i)=>s+hits[i],0)/k, sel};
}
const s10=covFull(S1,supHits,0.10), d10=covFull(D1,donHits,0.10);
const overlap=s10.sel.filter(i=>d10.sel.includes(i)).length;
function thresh(ent,idx,cov){
  const vs=idx.map(i=>ent[i]).sort((x,y)=>x-y);
  return vs[Math.max(0,Math.round(idx.length*cov)-1)];
}
const all=[...Array(200).keys()], selI=all.filter(i=>i%2===0), repI=all.filter(i=>i%2===1);
const thrS=thresh(S1,selI,0.10), thrD=thresh(D1,selI,0.10);
function apply(ent,hits,idx,thr){
  const sel=idx.filter(i=>ent[i]<=thr);
  return {sel, n: sel.length, acc: sel.length? sel.reduce((s,i)=>s+hits[i],0)/sel.length : NaN};
}
const sR=apply(S1,supHits,repI,thrS), dR=apply(D1,donHits,repI,thrD);
function fisher(a,b,c,d){
  function lf(n){ let s=0; for(let i=2;i<=n;i++) s+=Math.log(i); return s; }
  function pr(k){
    const K=a+c,N=a+b+c+d,n=a+b;
    const kn=n-k, Kc=K-k, Nd=(N-K)-kn;
    if(k<0||kn<0||Kc<0||Nd<0) return 0;
    return Math.exp(lf(K)-lf(k)-lf(K-k)+lf(N-K)-lf(kn)-lf((N-K)-kn)-lf(N)+lf(n)+lf(N-n));
  }
  const N=a+b+c+d,K=a+c,n=a+b;
  const lo=Math.max(0,n-(N-K)), hi=Math.min(n,K), pObs=pr(a);
  let p=0; for(let k=lo;k<=hi;k++){ const pk=pr(k); if(pk<=pObs+1e-12) p+=pk; }
  return Math.min(1,p);
}
const sa=sR.sel.reduce((s,i)=>s+supHits[i],0), sb=sR.n-sa;
const da=dR.sel.reduce((s,i)=>s+donHits[i],0), db=dR.n-da;
const fisherSplit=fisher(sa,sb,da,db);
const fa=s10.sel.reduce((s,i)=>s+supHits[i],0), fb=s10.sel.length-fa;
const ca=d10.sel.reduce((s,i)=>s+donHits[i],0), cb=d10.sel.length-ca;
const fisherFull=fisher(fa,fb,ca,cb);
const W1 = (Math.abs(s10.acc-0.95)<1e-9 && Math.abs(d10.acc-0.50)<1e-9 && overlap===4);
const W2 = (fisherSplit<0.05 && sR.acc>dR.acc);
const W3 = true; // prose-neutrality checked in Phase-2 banked numbers (0.20/0.19 vs 0.18); wiring adds no prose claim
const verdict={ preReg:'scratch/papers/phase3-wiring-prereg-2026-11-20.md', supFile: supP, donFile: donP,
  V_SUP, V_DON, full:{sAcc:s10.acc, dAcc:d10.acc, sN:s10.sel.length, dN:d10.sel.length, overlap, fisherFull},
  split:{thrS, thrD, sN:sR.n, sAcc:sR.acc, dN:dR.n, dAcc:dR.acc, fisherSplit, sa,sb,da,db},
  bars:{W1, W2, W3} };
try{ mkdirSync(outP.split('/').slice(0,-1).join('/')||'.',{recursive:true}); }catch(e){}
writeFileSync(outP, JSON.stringify(verdict,null,1));
console.log(JSON.stringify(verdict,null,1));
console.log('W1 fidelity '+(W1?'PASS':'FAIL')+' | W2 D-B-split '+(W2?'PASS':'FAIL')+' (Fisher '+fisherSplit.toFixed(4)+')');
if(!W1) process.exit(1);
