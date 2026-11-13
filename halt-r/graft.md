# GRAFT: transplant (donor embeddings as keys) vs one-hot (measured, CI)
Same protocol (P=8000, char, no-gen). graft: frozen donor tok rows summed+sign, zero training, D=E=64. raw rows are same-workflow controls (same store positions, same seeds).

| corpus | W | proj | cost | interp acc/ppl (lam*) |
|---|---|---|---|---|
| shakes | 16 | graft | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 16 | raw | linear-O(P·D) | 0.260/13.68 (0) |
| shakes | 8 | graft | linear-O(P·D) | 0.300/13.49 (0) |
| shakes | 8 | raw | linear-O(P·D) | 0.300/13.49 (0) |
| ts | 16 | graft | linear-O(P·D) | 0.270/11.95 (0) |
| ts | 16 | raw | linear-O(P·D) | 0.320/11.53 (0.1) |
| ts | 8 | graft | linear-O(P·D) | 0.260/11.03 (0) |
| ts | 8 | raw | linear-O(P·D) | 0.260/11.03 (0) |

shakes W8: raw 13.49 (lam*=0) vs graft 13.49 (lam*=0) d=0.00. McNemar graft-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (donor geometry adds nothing).

shakes W16: raw 13.68 (lam*=0) vs graft 13.68 (lam*=0) d=0.00. McNemar graft-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (donor geometry adds nothing).

ts W8: raw 11.03 (lam*=0) vs graft 11.03 (lam*=0) d=0.00. McNemar graft-vs-raw b/c=0/0 p=1.000. VERDICT: WALL STANDS (donor geometry adds nothing).

ts W16: raw 11.53 (lam*=0.1) vs graft 11.95 (lam*=0) d=-0.41. McNemar graft-vs-raw b/c=0/5 p=0.063. VERDICT: WALL STANDS (donor geometry adds nothing).
