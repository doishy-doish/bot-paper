# COMBO: grafted keys (K=256) + donor values + router (measured, CI)
Same protocol (P=8000, char, no-gen, T100). interp arm = graft5 keys + bigram (absorption alone). gpt arm = graft5 keys + frozen donor dists (absorption + delegation, the feast). Cross-job paired test (same seeds/queries).

| corpus | W | arm | interp acc/ppl (lam*) | gpt-mix acc/ppl (lam*) | cover |
|---|---|---|---|---|---|
| ts | 32 | interp | 0.330/9.86 (0) | — | — |
| ts | 32 | gpt | — | 0.650/3.30 (0) | 100/100 |

ts W32 combo: interp 9.86 vs gpt-mix 3.30 d=6.56. McNemar gpt-vs-interp b/c=36/4 p=0.000 SIGNIFICANT. VERDICT: BLEND WINS (keys + values + router beat keys alone).
| shakes | 32 | interp | 0.380/9.82 (0) | — | — |
| shakes | 32 | gpt | — | 0.460/5.31 (0) | 100/100 |

shakes W32 combo: interp 9.82 vs gpt-mix 5.31 d=4.52. McNemar gpt-vs-interp b/c=14/6 p=0.115. VERDICT: blend stands (values add nothing here).
