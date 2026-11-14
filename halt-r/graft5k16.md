# GRAFT5K16: anchor-count sweep on the won config (measured, CI)
ts W16, P=8000, char, no-gen, T100. Question: how small can the equation get before the win dies?

| proj | K (graftE) | interp acc/ppl (lam*) |
|---|---|---|
| graft5 | 128 | 0.310/11.99 (0.1) |
| graft5 | 256 | 0.330/11.43 (0.1) |
| graft5 | 512 | 0.310/11.60 (0.1) |
| graft5 | 64 | 0.300/11.80 (0.1) |
| raw | 1264 | 0.320/11.53 (0.1) |
| raw | 1264 | 0.320/11.53 (0.1) |
| raw | 1264 | 0.320/11.53 (0.1) |
| raw | 1264 | 0.320/11.53 (0.1) |

K=512: raw 11.53 vs graft5 11.60 d=-0.07 b/c=1/2 p=1.000 → falls.

K=256: raw 11.53 vs graft5 11.43 d=0.11 b/c=1/0 p=1.000 → falls.

K=128: raw 11.53 vs graft5 11.99 d=-0.45 b/c=1/2 p=1.000 → falls.

K=64: raw 11.53 vs graft5 11.80 d=-0.26 b/c=1/3 p=0.625 → falls.

FIND: minimal K keeping the win = NONE (win dies below 512 — capacity was load-bearing).
