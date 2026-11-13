# S: entmax-1.5 sparse assignment + Hedge mixing (measured, CI)
Same protocol (P=8000, sup, no-gen). Sparse keeps the argmax, sharpens the mixture (support 253->2 in prototype). Hedge mixes the LAMBDAS grid with regret bounds. Paired McNemar on identical queries.

| corpus | W | sparse | cost | interp acc/ppl (lam*) | hedge acc/ppl (neff) |
|---|---|---|---|---|---|
| shakes | 16 | 0 | linear-O(P·D) | 0.270/12.41 (0.1) | 0.260/12.42 (1.3) |
| shakes | 16 | 1 | linear-O(P·D) | 0.270/12.41 (0.1) | 0.260/12.42 (1.3) |
| shakes | 8 | 0 | linear-O(P·D) | 0.270/13.18 (0.25) | 0.280/13.03 (1.9) |
| shakes | 8 | 1 | linear-O(P·D) | 0.340/12.77 (0.1) | 0.330/12.73 (1.7) |
| ts | 16 | 0 | linear-O(P·D) | 0.350/11.47 (0.25) | 0.320/11.42 (2.0) |
| ts | 16 | 1 | linear-O(P·D) | 0.370/10.42 (0.25) | 0.340/10.44 (2.0) |
| ts | 8 | 0 | linear-O(P·D) | 0.540/7.56 (0.25) | 0.530/7.47 (1.4) |
| ts | 8 | 1 | linear-O(P·D) | 0.540/7.56 (0.25) | 0.530/7.40 (1.6) |

shakes W8 sparse-vs-softmax: 13.18 -> 12.77 McNemar b/c=9/2 p=0.065. VERDICT: leans sparse, not significant.
shakes W8 hedge-vs-bestlam: 12.77 -> 12.73 (neff=1.7). VERDICT: ONLINE MIXING WINS.

shakes W16 sparse-vs-softmax: 12.41 -> 12.41 McNemar b/c=0/0 p=1.000. VERDICT: SPARSE FAILS.
shakes W16 hedge-vs-bestlam: 12.41 -> 12.42 (neff=1.3). VERDICT: fixed lam stands.

ts W8 sparse-vs-softmax: 7.56 -> 7.56 McNemar b/c=0/0 p=1.000. VERDICT: leans sparse, not significant.
ts W8 hedge-vs-bestlam: 7.56 -> 7.40 (neff=1.6). VERDICT: ONLINE MIXING WINS.

ts W16 sparse-vs-softmax: 11.47 -> 10.42 McNemar b/c=9/7 p=0.804. VERDICT: leans sparse, not significant.
ts W16 hedge-vs-bestlam: 10.42 -> 10.44 (neff=2.0). VERDICT: fixed lam stands.
