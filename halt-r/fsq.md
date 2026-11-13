# FSQ: discrete compositional keys vs sup (measured, CI)
Same protocol (P=8000, char, no-gen, H=256). fsq: supervised probe THROUGH FSQ L=8 straight-through (300 steps, same streams as enc2), sign via sim path + quantize-before-sign. sup = same-workflow strongest-fixed-key control.

| corpus | W | proj | cost | interp acc/ppl (lam*) | fsqReuse |
|---|---|---|---|---|---|
| shakes | 16 | fsq | linear-O(P·D) | 0.260/13.68 (0) | 0.933 |
| shakes | 16 | sup | linear-O(P·D) | 0.270/12.41 (0.1) | 0.000 |
| shakes | 8 | fsq | linear-O(P·D) | 0.300/13.49 (0) | 0.990 |
| shakes | 8 | sup | linear-O(P·D) | 0.270/13.18 (0.25) | 0.000 |
| ts | 16 | fsq | linear-O(P·D) | 0.290/12.30 (0.1) | 0.828 |
| ts | 16 | sup | linear-O(P·D) | 0.350/11.47 (0.25) | 0.000 |
| ts | 8 | fsq | linear-O(P·D) | 0.260/11.03 (0) | 0.971 |
| ts | 8 | sup | linear-O(P·D) | 0.540/7.56 (0.25) | 0.000 |

shakes W8 fsq: sup 13.18 (lam*=0.25) vs fsq 13.49 (lam*=0) d=-0.31. McNemar fsq-vs-sup b/c=10/7 p=0.629. VERDICT: WALL STANDS (fsq adds nothing over sup).

shakes W16 fsq: sup 12.41 (lam*=0.1) vs fsq 13.68 (lam*=0) d=-1.27. McNemar fsq-vs-sup b/c=3/4 p=1.000. VERDICT: WALL STANDS (fsq adds nothing over sup).

ts W8 fsq: sup 7.56 (lam*=0.25) vs fsq 11.03 (lam*=0) d=-3.47. McNemar fsq-vs-sup b/c=2/30 p=0.000 SIGNIFICANT. VERDICT: WALL STANDS (fsq adds nothing over sup).

ts W16 fsq: sup 11.47 (lam*=0.25) vs fsq 12.30 (lam*=0.1) d=-0.83. McNemar fsq-vs-sup b/c=1/7 p=0.070. VERDICT: leans fsq, not significant.
