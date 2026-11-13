# H-ORF: orthogonal random features vs iid Gaussian (measured, CI)
Same protocol (P=8000, sim, no-gen). ORF = Gram-Schmidt orthogonalized directions, norms preserved. Verdict on interpPpl (variance cut, not geometry change).

| corpus | W | orf | cost | interp acc/ppl (lam*) | buildMs |
|---|---|---|---|---|---|
| shakes | 16 | 0 | linear-O(P·D) | 0.260/13.68 (0) | 86 |
| shakes | 16 | 1 | linear-O(P·D) | 0.260/13.68 (0) | 272 |
| shakes | 8 | 0 | linear-O(P·D) | 0.350/13.40 (0.1) | 98 |
| shakes | 8 | 1 | linear-O(P·D) | 0.340/13.52 (0.1) | 185 |
| ts | 16 | 0 | linear-O(P·D) | 0.310/11.51 (0.1) | 93 |
| ts | 16 | 1 | linear-O(P·D) | 0.340/11.10 (0.1) | 435 |
| ts | 8 | 0 | linear-O(P·D) | 0.430/8.84 (0.25) | 83 |
| ts | 8 | 1 | linear-O(P·D) | 0.420/8.65 (0.25) | 114 |

shakes W8 iid 13.40 -> orf 13.52 McNemar b/c=0/1 p=1.000. VERDICT: ORF FAILS.

shakes W16 iid 13.68 -> orf 13.68 McNemar b/c=0/0 p=1.000. VERDICT: ORF FAILS.

ts W8 iid 8.84 -> orf 8.65 McNemar b/c=4/5 p=1.000. VERDICT: leans ORF, not significant.

ts W16 iid 11.51 -> orf 11.10 McNemar b/c=3/0 p=0.250. VERDICT: leans ORF, not significant.
