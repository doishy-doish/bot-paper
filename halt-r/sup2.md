# sup2: per-(position,char) scales vs per-position scales
Same protocol (P=8000, H=256, fixed lam, no-gen). sup2 = W*V predictive scales (strict generalization of sup). C5 split rides along (descriptive).

| corpus | W | proj | hop acc/ppl | interp acc/ppl (lam*) | mem acc/n | nov acc/n |
|---|---|---|---|---|---|
| shakes | 16 | sup | 0.260/763853976.31 | 0.270/12.41 (0.1) | 1.00/1 vs 0.29/99 |
| shakes | 16 | sup2 | 0.170/28.71 | 0.260/13.68 (0) | 0.00/1 vs 0.18/99 |
| shakes | 8 | sup | 0.280/3159.66 | 0.270/13.18 (0.25) | 0.32/50 vs 0.20/50 |
| shakes | 8 | sup2 | 0.130/28.81 | 0.300/13.49 (0) | 0.24/50 vs 0.16/50 |
| ts | 16 | sup | 0.290/226.92 | 0.350/11.47 (0.25) | 0.28/39 vs 0.23/61 |
| ts | 16 | sup2 | 0.200/20.49 | 0.270/11.95 (0) | 0.10/39 vs 0.20/61 |
| ts | 8 | sup | 0.500/762790.74 | 0.540/7.56 (0.25) | 0.49/86 vs 0.57/14 |
| ts | 8 | sup2 | 0.140/21.55 | 0.260/11.03 (0) | 0.24/86 vs 0.14/14 |

shakes W8: sup interpPpl=13.18 vs sup2 interpPpl=13.49 d=-0.31. VERDICT: sup HOLDS (finer scales add nothing). Memorized 0.24 (n=50) vs novel 0.16 (n=50).

shakes W16: sup interpPpl=12.41 vs sup2 interpPpl=13.68 d=-1.27. VERDICT: sup HOLDS (finer scales add nothing). Memorized 0.00 (n=1) vs novel 0.18 (n=99).

ts W8: sup interpPpl=7.56 vs sup2 interpPpl=11.03 d=-3.47. VERDICT: sup HOLDS (finer scales add nothing). Memorized 0.24 (n=86) vs novel 0.14 (n=14).

ts W16: sup interpPpl=11.47 vs sup2 interpPpl=11.95 d=-0.47. VERDICT: sup HOLDS (finer scales add nothing). Memorized 0.10 (n=39) vs novel 0.20 (n=61).
