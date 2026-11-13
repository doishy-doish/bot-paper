# OWNSHIP: grafted keys alone vs one-hot (measured, CI)
Same protocol (P=8000, char, no-gen, T100). No q100, no donor values. Donor GGUF frozen on disk, never forwarded.

| corpus | W | proj | interp acc/ppl (lam*) |
|---|---|---|---|
| shakes | 16 | graft5 | 0.260/13.68 (0) |
| shakes | 16 | raw | 0.260/13.68 (0) |
| shakes | 8 | graft5 | 0.320/13.77 (0.1) |
| shakes | 8 | raw | 0.300/13.49 (0) |
| ts | 16 | graft5 | 0.330/11.43 (0.1) |
| ts | 16 | raw | 0.320/11.53 (0.1) |
| ts | 8 | graft5 | 0.330/9.37 (0.1) |
| ts | 8 | raw | 0.260/11.03 (0) |

shakes W8: raw 13.49 vs graft5 13.77 d=-0.29 b/c=5/3 p=0.727 VERDICT: stands (rent-only).

shakes W16: raw 13.68 vs graft5 13.68 d=0.00 b/c=0/0 p=1.000 VERDICT: stands (rent-only).

ts W8: raw 11.03 vs graft5 9.37 d=1.66 b/c=7/0 p=0.016 VERDICT: OWNERSHIP WINS (donor removable).

ts W16: raw 11.53 vs graft5 11.43 d=0.11 b/c=1/0 p=1.000 VERDICT: stands (rent-only).
