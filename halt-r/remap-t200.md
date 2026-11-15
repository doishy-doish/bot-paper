# REMAP-T200: powered replicate of the GOLD-1 lean (measured, CI)
Remap corpus (digit-permuted revbal mirror, W16), 200 distinct queries, same SEED streams as GOLD-1 T100. Graft5-K256 vs same-run sup.

| arm | acc/ppl (lam*) | trials |
|---|---|---|
| sup | 0.325/8.28 (0.25) | 200 |
| graft5 | 0.410/7.41 (0.5) | 200 |

graft5-vs-sup d=0.88 McNemar b/c=33/16 p=0.0213 ratio=2.06. VERDICT: WIN REPLICATES at power (remap within-task transfer holds; C1 next on a properly constructed donor-unknown corpus).
