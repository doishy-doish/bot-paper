# GRAFT34: relative (graft3) vs procrustes-side (graft4) transplant, head-to-head (measured, CI)
Same protocol (P=8000, char, no-gen). graft3: cosine-to-anchors rows centered + sign binding (SEED+11), D=V. graft4: full ZCA + orthogonal binding (SEED+12), D=E=64. graft/graft2 = shot-1/2 references in same seeds. raw = same-workflow control.

| corpus | W | proj | cost | interp acc/ppl (lam*) |
|---|---|---|---|---|
| shakes | 16 | graft | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | graft2 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | graft3 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | graft4 | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | raw | linear-O(P·D) | 0.260/13.68 (0) |
| ts | 16 | graft | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | graft2 | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | graft3 | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | graft4 | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | raw | linear-O(P·D) | 0.310/11.50 (0.1) |

shakes W16 graft3: raw 13.68 (lam*=0) vs graft3 13.68 (lam*=0) d=0.00. McNemar graft3-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (graft3 adds nothing).

shakes W16 graft4: raw 13.68 (lam*=0) vs graft4 13.68 (lam*=0) d=0.00. McNemar graft4-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (graft4 adds nothing).

ts W16 graft3: raw 11.50 (lam*=0.1) vs graft3 11.95 (lam*=0) d=-0.45. McNemar graft3-vs-raw b/c=0/4 p=0.125. VERDICT: WALL STANDS (graft3 adds nothing).

ts W16 graft4: raw 11.50 (lam*=0.1) vs graft4 11.95 (lam*=0) d=-0.45. McNemar graft4-vs-raw b/c=0/4 p=0.125. VERDICT: WALL STANDS (graft4 adds nothing).
