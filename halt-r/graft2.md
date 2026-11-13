# GRAFT2: aligned transplant (whiten+bind) vs one-hot (measured, CI)
Same protocol (P=8000, char, no-gen). graft2: donor rows mean-centered + per-dim standardized, per-position sign binding (SEED+9), summed+sign, zero training, D=E=64. graft rows = shot-1 reference in same seeds. raw rows = same-workflow controls.

| corpus | W | proj | cost | interp acc/ppl (lam*) |
|---|---|---|---|---|
| shakes | 16 | graft | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | graft2 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | raw | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 8 | graft | linear-O(P·D) | 0.300/13.49 (0) |
| shakes | 8 | graft2 | linear-O(P·D) | 0.300/13.49 (0) |
| shakes | 8 | raw | linear-O(P·D) | 0.300/13.49 (0) |
| ts | 16 | graft | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | graft2 | linear-O(P·D) | 0.320/11.95 (0.1) |
| ts | 16 | raw | linear-O(P·D) | 0.320/11.53 (0.1) |
| ts | 8 | graft | linear-O(P·D) | 0.260/11.03 (0) |
| ts | 8 | graft2 | linear-O(P·D) | 0.310/9.62 (0.1) |
| ts | 8 | raw | linear-O(P·D) | 0.260/11.03 (0) |

shakes W8: raw 13.49 (lam*=0) vs graft2 13.49 (lam*=0) d=0.00. McNemar graft2-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (aligned geometry adds nothing).

shakes W16: raw 13.68 (lam*=0) vs graft2 13.68 (lam*=0) d=0.00. McNemar graft2-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (aligned geometry adds nothing).

ts W8: raw 11.03 (lam*=0) vs graft2 9.62 (lam*=0.1) d=1.41. McNemar graft2-vs-raw b/c=5/0 p=0.063. VERDICT: leans graft, not significant.

ts W16: raw 11.53 (lam*=0.1) vs graft2 11.95 (lam*=0.1) d=-0.42. McNemar graft2-vs-raw b/c=1/1 p=1.000. VERDICT: leans graft, not significant.
