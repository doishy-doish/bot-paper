# HALT-R on text8 (pilot, measured, CI)
Same protocol as lm bench (90/10 split, seed-2000 q100, P=8000). BPC = log2(char-ppl). 2018-era target: Al-Rfou et al. 12L Transformer 44M = 1.18 BPC on text8 (Transformer-XL Table 3).

| W | proj | hop acc/ppl/BPC | bigram acc/ppl/BPC | interp acc/ppl/BPC (locked lam) |
|---|---|---|---|---|
| 2 | raw | 0.270/21.70/4.44 | 0.260/11.54/3.53 | 0.370/9.94/3.31 (lam=0.5) |
| 2 | sim | 0.280/36.29/5.18 | 0.260/11.54/3.53 | 0.370/9.94/3.31 (lam=0.5) |
| 4 | raw | 0.300/13043145.26/23.64 | 0.290/9.89/3.31 | 0.360/9.26/3.21 (lam=0.25) |
| 4 | sim | 0.320/1377.44/10.43 | 0.290/9.89/3.31 | 0.330/9.79/3.29 (lam=0.25) |
| 8 | raw | 0.340/8656.62/13.08 | 0.320/11.95/3.58 | 0.290/11.30/3.50 (lam=0.25) |
| 8 | sim | 0.330/1288.98/10.33 | 0.320/11.95/3.58 | 0.320/11.17/3.48 (lam=0.25) |

Best interp BPC: 3.21 (W=4 raw) vs 2018 target 1.18. Gap: 2.03 bpc.
