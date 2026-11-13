# P-sweep under sup: does the learned encoder keep moving the wall with P?
Same protocol as H1 (90/10 split, seed-2000 select, seed-2001 locked report, H=256, no-gen), proj=sup only.

| corpus | W | P | interp acc/ppl (lam*) | hop acc/ppl | bigram acc/ppl |
|---|---|---|---|---|---|
| shakes | 16 | 128000 | 0.250/13.96 (0.25) | 0.410/9854267.56 | 0.270/10.44 |
| shakes | 16 | 2000 | 0.260/13.68 (0) | 0.220/203.50 | 0.270/10.44 |
| shakes | 16 | 32000 | 0.270/12.64 (0.1) | 0.310/179468271.94 | 0.270/10.44 |
| shakes | 16 | 8000 | 0.270/12.41 (0.1) | 0.260/763853976.31 | 0.270/10.44 |
| shakes | 8 | 128000 | 0.410/10.59 (0.5) | 0.470/1178029.07 | 0.270/12.05 |
| shakes | 8 | 2000 | 0.270/13.69 (0.1) | 0.230/1749887995.51 | 0.270/12.05 |
| shakes | 8 | 32000 | 0.360/11.34 (0.25) | 0.360/273.05 | 0.270/12.05 |
| shakes | 8 | 8000 | 0.270/13.18 (0.25) | 0.280/3159.66 | 0.270/12.05 |
| ts | 16 | 128000 | 0.410/10.32 (0.5) | 0.450/29.46 | 0.350/8.84 |
| ts | 16 | 2000 | 0.270/11.50 (0.1) | 0.240/817.03 | 0.350/8.84 |
| ts | 16 | 32000 | 0.330/11.06 (0.25) | 0.460/2404325.65 | 0.350/8.84 |
| ts | 16 | 8000 | 0.350/11.47 (0.25) | 0.290/226.92 | 0.350/8.84 |
| ts | 8 | 128000 | 0.630/5.25 (0.75) | 0.630/425.27 | 0.370/7.94 |
| ts | 8 | 2000 | 0.410/9.16 (0.25) | 0.430/7015905.02 | 0.370/7.94 |
| ts | 8 | 32000 | 0.590/6.04 (0.5) | 0.580/16843.00 | 0.370/7.94 |
| ts | 8 | 8000 | 0.540/7.56 (0.25) | 0.500/762790.74 | 0.370/7.94 |

shakes W8: P2000 interpPpl=13.69 -> P128000 interpPpl=10.59 (slope -0.516/dbl, max-P lam*=0.5). VERDICT: WALL KEEPS MOVING with P under sup.

shakes W16: P2000 interpPpl=13.68 -> P128000 interpPpl=13.96 (slope 0.046/dbl, max-P lam*=0.25). VERDICT: PLATEAU under sup at this range.

ts W8: P2000 interpPpl=9.16 -> P128000 interpPpl=5.25 (slope -0.652/dbl, max-P lam*=0.75). VERDICT: WALL KEEPS MOVING with P under sup.

ts W16: P2000 interpPpl=11.50 -> P128000 interpPpl=10.32 (slope -0.196/dbl, max-P lam*=0.5). VERDICT: WALL KEEPS MOVING with P under sup.
