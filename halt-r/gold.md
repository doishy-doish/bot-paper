# GOLD-1: grokkable-but-unknown transfer + reliability (measured, CI)
remap/nat/revbal W16, P=8000, T100 same seeds, graft5-K256. Donor-alone q05 diagnostic (competence gate, unpaired).

| corpus | arm | interp acc/ppl (lam*) |
|---|---|---|
| nat | graft | 0.300/11.65 (0.25) |
| nat | sup | 0.350/10.91 (0.25) |
| remap | graft | 0.430/7.05 (0.5) |
| remap | sup | 0.340/7.90 (0.25) |
| revbal | sup | 0.300/8.62 (0.25) |
| revbal | tiny | 0.380/7.87 (0.25) |

| donor | corpus | alone acc/ppl (coverage) |
|---|---|---|
| q05 | nat | 0.190/5248074602.50 (0.990) |
| q05 | remap | 0.280/436515832.24 (1.000) |
| q05 | revbal | 0.300/251188643.15 (0.990) |

Donor-floor strata (exact, positions banked): {"remap":{"carry0":{"n":32,"acc":0.281},"carry1":{"n":40,"acc":0.25},"carry2":{"n":28,"acc":0.321},"lineMismatches":0,"floor":0.154,"verdict":"GO (competent on carry)"},"nat":{"rareN":0,"rareAcc":0,"commonN":100,"commonAcc":0.19,"q1":40,"floor":0.02,"verdict":"VOID-competence"}}

remap graft: sup 7.90 vs graft 7.05 d=0.85 b/c=18/9 p=0.1221 falls.

nat graft: sup 10.91 vs graft 11.65 d=-0.74 b/c=0/5 p=0.0625 falls.

revbal tiny: sup 8.62 vs graft 7.87 d=0.75 b/c=13/5 p=0.0963 falls.

H-holdout (committed, CI-verified): abstain-alone utility 0.41x1=0.41 vs donor 0.69 — D-B abstain-alone CLOSED under utility (redirect to provenance/draft-verify).
