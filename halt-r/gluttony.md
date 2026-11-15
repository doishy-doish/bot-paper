# GLUTTONY: ownership-pattern transplant at scale (measured, CI)
Arith-revbal W16, P=8000, T200 same seeds as T200, frozen donor (REMOVAL-1/2). Control = sup arm (T200 interp).

| arm | interp acc/ppl (lam*) |
|---|---|---|
| graft5 K=128 | 0.320/8.68 (0.25) |
| graft5 K=256 | 0.370/8.17 (0.25) |
| graft5 K=512 | 0.335/8.59 (0.25) |
| graft5 K=64 | 0.320/8.78 (0.25) |
| sup | 0.295/8.82 (0.25) |

K=512: sup 8.82 vs graft5 8.59 d=0.23 b/c=19/11 p=0.2005 falls.

K=256: sup 8.82 vs graft5 8.17 d=0.66 b/c=23/8 p=0.0107 HOLDS.

K=128: sup 8.82 vs graft5 8.68 d=0.14 b/c=13/8 p=0.3833 falls.

K=64: sup 8.82 vs graft5 8.78 d=0.05 b/c=14/9 p=0.4049 falls.

PRIMARY K=512: NULL — KILL LINE: revbal line closed, transfer unhooked.

FIND: minimal K keeping the win = 256 (constant floor vs 2^N question).
