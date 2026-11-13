# GRAFT5: anchor-energy rank keys transplant (measured, CI)
Same protocol (P=8000, char, no-gen). graft5: K=512 shared-probe anchors (SEED+13) in whitened donor space, RANKED cosine token keys (rank-robust, isometry-invariant — no alignment step), per-position sign bind (SEED+14), D=512 via exact sim path. graft2 = campaign-best fixed-key reference in same seeds. raw = same-workflow control.

| corpus | W | proj | cost | interp acc/ppl (lam*) |
|---|---|---|---|---|
| shakes | 16 | graft2 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | graft5 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | raw | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 8 | graft2 | linear-O(P·D) | 0.300/13.49 (0) |
| shakes | 8 | graft5 | linear-O(P·D) | 0.310/13.45 (0.1) |
| shakes | 8 | raw | linear-O(P·D) | 0.300/13.49 (0) |
| ts | 16 | graft2 | linear-O(P·D) | 0.320/11.95 (0.1) |
| ts | 16 | graft5 | linear-O(P·D) | 0.310/11.60 (0.1) |
| ts | 16 | raw | linear-O(P·D) | 0.320/11.53 (0.1) |
| ts | 8 | graft2 | linear-O(P·D) | 0.310/9.62 (0.1) |
| ts | 8 | graft5 | linear-O(P·D) | 0.430/8.55 (0.25) |
| ts | 8 | raw | linear-O(P·D) | 0.260/11.03 (0) |

shakes W8 graft5: raw 13.49 (lam*=0) vs graft5 13.45 (lam*=0.1) d=0.04. McNemar graft5-vs-raw b/c=4/3 p=1.000. VERDICT: leans graft5, not significant.

shakes W8 graft2: raw 13.49 (lam*=0) vs graft2 13.49 (lam*=0) d=0.00. McNemar graft2-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (graft2 adds nothing).

shakes W16 graft5: raw 13.68 (lam*=0) vs graft5 13.68 (lam*=0) d=0.00. McNemar graft5-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (graft5 adds nothing).

shakes W16 graft2: raw 13.68 (lam*=0) vs graft2 13.68 (lam*=0) d=0.00. McNemar graft2-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (graft2 adds nothing).

ts W8 graft5: raw 11.03 (lam*=0) vs graft5 8.55 (lam*=0.25) d=2.48. McNemar graft5-vs-raw b/c=19/2 p=0.000 SIGNIFICANT. VERDICT: TRANSPLANT WINS (graft5 beats one-hot).

ts W8 graft2: raw 11.03 (lam*=0) vs graft2 9.62 (lam*=0.1) d=1.41. McNemar graft2-vs-raw b/c=5/0 p=0.063. VERDICT: leans graft2, not significant.

ts W16 graft5: raw 11.53 (lam*=0.1) vs graft5 11.60 (lam*=0.1) d=-0.07. McNemar graft5-vs-raw b/c=1/2 p=1.000. VERDICT: leans graft5, not significant.

ts W16 graft2: raw 11.53 (lam*=0.1) vs graft2 11.95 (lam*=0.1) d=-0.42. McNemar graft2-vs-raw b/c=1/1 p=1.000. VERDICT: leans graft2, not significant.
