# GRAFT5K: anchor-count sweep on the won config (measured, CI)
ts W8 only, P=8000, char, no-gen, T100. Question: how small can the equation get before the win dies?

| proj | K (graftE) | interp acc/ppl (lam*) |
|---|---|---|
| graft5 | 128 | 0.260/11.03 (0) |
| graft5 | 256 | 0.330/9.37 (0.1) |
| graft5 | 512 | 0.430/8.55 (0.25) |
| graft5 | 64 | 0.260/11.03 (0) |
| raw | 632 | 0.260/11.03 (0) |
| raw | 632 | 0.260/11.03 (0) |
| raw | 632 | 0.260/11.03 (0) |
| raw | 632 | 0.260/11.03 (0) |

K=512: raw 11.03 vs graft5 8.55 d=2.48 b/c=19/2 p=0.000 → HOLDS.

K=256: raw 11.03 vs graft5 9.37 d=1.66 b/c=7/0 p=0.016 → HOLDS.

K=128: raw 11.03 vs graft5 11.03 d=0.00 b/c=0/0 p=1.000 → falls.

K=64: raw 11.03 vs graft5 11.03 d=0.00 b/c=0/0 p=1.000 → falls.

FIND: minimal K keeping the win = 256.
