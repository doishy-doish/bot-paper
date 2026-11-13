# ICT: contriever-style span-view keys vs sup (measured, CI)
Same protocol (P=8000, char, no-gen, H=256). ict: two 15%-dropout views of same window attract, batch peers repel (200 steps, zero labels), sign via sim path. sup = same-workflow strongest-fixed-key control.

| corpus | W | proj | cost | interp acc/ppl (lam*) |
|---|---|---|---|---|
| shakes | 16 | ict | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | sup | linear-O(P·D) | 0.270/12.41 (0.1) |
| shakes | 8 | ict | linear-O(P·D) | 0.300/13.49 (0) |
| shakes | 8 | sup | linear-O(P·D) | 0.270/13.18 (0.25) |
| ts | 16 | ict | linear-O(P·D) | 0.300/11.65 (0.1) |
| ts | 16 | sup | linear-O(P·D) | 0.350/11.47 (0.25) |
| ts | 8 | ict | linear-O(P·D) | 0.410/9.05 (0.25) |
| ts | 8 | sup | linear-O(P·D) | 0.540/7.56 (0.25) |

shakes W8 ict: sup 13.18 (lam*=0.25) vs ict 13.49 (lam*=0) d=-0.31. McNemar ict-vs-sup b/c=10/7 p=0.629. VERDICT: WALL STANDS (ict adds nothing over sup).

shakes W16 ict: sup 12.41 (lam*=0.1) vs ict 13.68 (lam*=0) d=-1.27. McNemar ict-vs-sup b/c=3/4 p=1.000. VERDICT: WALL STANDS (ict adds nothing over sup).

ts W8 ict: sup 7.56 (lam*=0.25) vs ict 9.05 (lam*=0.25) d=-1.49. McNemar ict-vs-sup b/c=5/18 p=0.011 SIGNIFICANT. VERDICT: leans ict, not significant.

ts W16 ict: sup 11.47 (lam*=0.25) vs ict 11.65 (lam*=0.1) d=-0.18. McNemar ict-vs-sup b/c=0/5 p=0.063. VERDICT: leans ict, not significant.
