# H-enc2: supervised-probe encoder vs sup scales (measured, CI)
Same protocol (P=8000, H=256, no-gen). enc2: 300 SGD steps, batch 128, CE linear probe (JS-only, zero python), probe discarded, M kept, sign bits via identical featPM path. sup rows are same-workflow controls. Bar: 0.2 ppl + McNemar p<0.05. Cost: both linear-O(P·D); sublinear is H9.

| corpus | W | tok | proj | cost | encLoss first->last | interp acc/ppl (lam*) |
|---|---|---|---|---|---|---|
| shakes | 16 | bpe | enc2 | linear-O(P·D) | 5.764->3.896 | 0.260/13.37 (0) |
| shakes | 16 | char | enc2 | linear-O(P·D) | 4.189->3.139 | 0.260/13.68 (0) |
| shakes | 16 | bpe | sup | linear-O(P·D) | 0.000->0.000 | 0.260/13.37 (0) |
| shakes | 16 | char | sup | linear-O(P·D) | 0.000->0.000 | 0.270/12.41 (0.1) |
| shakes | 8 | bpe | enc2 | linear-O(P·D) | 5.762->4.703 | 0.250/14.50 (0.1) |
| shakes | 8 | char | enc2 | linear-O(P·D) | 4.169->3.299 | 0.350/13.12 (0.1) |
| shakes | 8 | bpe | sup | linear-O(P·D) | 0.000->0.000 | 0.290/14.06 (0.1) |
| shakes | 8 | char | sup | linear-O(P·D) | 0.000->0.000 | 0.270/13.18 (0.25) |
| ts | 16 | bpe | enc2 | linear-O(P·D) | 5.807->4.059 | 0.330/11.08 (0) |
| ts | 16 | char | enc2 | linear-O(P·D) | 4.391->3.080 | 0.320/11.85 (0.1) |
| ts | 16 | bpe | sup | linear-O(P·D) | 0.000->0.000 | 0.360/10.23 (0.25) |
| ts | 16 | char | sup | linear-O(P·D) | 0.000->0.000 | 0.350/11.47 (0.25) |
| ts | 8 | bpe | enc2 | linear-O(P·D) | 5.806->4.524 | 0.350/11.70 (0.25) |
| ts | 8 | char | enc2 | linear-O(P·D) | 4.378->3.035 | 0.290/10.01 (0.1) |
| ts | 8 | bpe | sup | linear-O(P·D) | 0.000->0.000 | 0.420/9.92 (0.25) |
| ts | 8 | char | sup | linear-O(P·D) | 0.000->0.000 | 0.540/7.56 (0.25) |

shakes W8 char: sup 13.18 vs enc2 13.12 (lam*=0.1, d=0.06). McNemar enc2-vs-sup b/c=11/3 p=0.057. VERDICT: under bar (<0.2 ppl).

shakes W8 bpe: sup 14.06 vs enc2 14.50 (lam*=0.1, d=-0.44). McNemar enc2-vs-sup b/c=2/6 p=0.289. VERDICT: SUP STANDS.

shakes W16 char: sup 12.41 vs enc2 13.68 (lam*=0, d=-1.27). McNemar enc2-vs-sup b/c=3/4 p=1.000. VERDICT: SUP STANDS.

shakes W16 bpe: sup 13.37 vs enc2 13.37 (lam*=0, d=0.00). McNemar enc2-vs-sup b/c=0/0 p=1.000. VERDICT: SUP STANDS.

ts W8 char: sup 7.56 vs enc2 10.01 (lam*=0.1, d=-2.45). McNemar enc2-vs-sup b/c=2/27 p=0.000 SIGNIFICANT. VERDICT: SUP STANDS.

ts W8 bpe: sup 9.92 vs enc2 11.70 (lam*=0.25, d=-1.78). McNemar enc2-vs-sup b/c=7/14 p=0.189. VERDICT: SUP STANDS.

ts W16 char: sup 11.47 vs enc2 11.85 (lam*=0.1, d=-0.38). McNemar enc2-vs-sup b/c=2/5 p=0.453. VERDICT: SUP STANDS.

ts W16 bpe: sup 10.23 vs enc2 11.08 (lam*=0, d=-0.85). McNemar enc2-vs-sup b/c=8/11 p=0.648. VERDICT: SUP STANDS.
