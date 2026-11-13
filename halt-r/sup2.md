# sup2: per-(position,char) scales vs per-position scales
Same protocol (P=8000, H=256, fixed lam, no-gen). sup2 = W*V predictive scales (strict generalization of sup). C5 split rides along (descriptive). Cost-class is first-class: same equation, linear series until H9.

| corpus | W | proj | cost | hop acc/ppl | interp acc/ppl (lam*) | mem acc/n | nov acc/n |
|---|---|---|---|---|---|---|---|
| shakes | 16 | sup | linear-O(P·D) | 0.260/763853976.31 | 0.270/12.41 (0.1) | 1.00/1 vs 0.29/99 |
| shakes | 16 | sup2 | linear-O(P·D) | 0.240/1021008920.72 | 0.260/12.19 (0.1) | 0.00/1 vs 0.25/99 |
| shakes | 8 | sup | linear-O(P·D) | 0.280/3159.66 | 0.270/13.18 (0.25) | 0.32/50 vs 0.20/50 |
| shakes | 8 | sup2 | linear-O(P·D) | 0.260/5646.54 | 0.360/12.88 (0.1) | 0.34/50 vs 0.24/50 |
| ts | 16 | sup | linear-O(P·D) | 0.290/226.92 | 0.350/11.47 (0.25) | 0.28/39 vs 0.23/61 |
| ts | 16 | sup2 | linear-O(P·D) | 0.410/224.85 | 0.360/10.81 (0.25) | 0.31/39 vs 0.31/61 |
| ts | 8 | sup | linear-O(P·D) | 0.500/762790.74 | 0.540/7.56 (0.25) | 0.49/86 vs 0.57/14 |
| ts | 8 | sup2 | linear-O(P·D) | 0.460/527.64 | 0.470/8.24 (0.25) | 0.51/86 vs 0.29/14 |

shakes W8: sup interpPpl=13.18 vs sup2 interpPpl=12.88 d=0.30. VERDICT: SCALES-REFINE WINS. Memorized 0.34 (n=50) vs novel 0.24 (n=50).

shakes W16: sup interpPpl=12.41 vs sup2 interpPpl=12.19 d=0.22. VERDICT: SCALES-REFINE WINS. Memorized 0.00 (n=1) vs novel 0.25 (n=99).

ts W8: sup interpPpl=7.56 vs sup2 interpPpl=8.24 d=-0.68. VERDICT: sup HOLDS (finer scales add nothing). Memorized 0.51 (n=86) vs novel 0.29 (n=14).

ts W16: sup interpPpl=11.47 vs sup2 interpPpl=10.81 d=0.66. VERDICT: SCALES-REFINE WINS. Memorized 0.31 (n=39) vs novel 0.31 (n=61).
