# gmix: hop + tiny-GPT (the paper's actual prescription)
Same protocol, W32 = trainer block (full pos join). hop+bigram from lm-bench W32 rows; GPT-alone from baseline rev-2 q100. Cost-class is first-class: hop is linear-O(P·D), GPT is parametric (fixed weights, cost in params).

| corpus | hop+bigram acc/ppl | hop+GPT acc/ppl (lam*) | GPT-alone acc/ppl | GPT params | cover |
|---|---|---|---|---|---|
| shakes | 0.380/9.82 | 0.460/5.31 (0) | 0.490/6.12 | 106688 | 100/100 |

shakes W32: McNemar gmix-vs-bigrammix b/c=14/6 p=0.115. VERDICT: leans mix, not significant.
| ts | 0.350/9.70 | 0.650/3.30 (0) | 0.570/4.36 | 108480 | 100/100 |

ts W32: McNemar gmix-vs-bigrammix b/c=34/4 p=0.000 SIGNIFICANT. VERDICT: MIX WINS (hop+GPT beats hop+bigram).
