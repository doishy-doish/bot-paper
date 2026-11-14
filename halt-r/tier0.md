# Tier0: draft-accept harness v0 (measured, CI)
Donor-only, 20 queries, fixed cost 5 gens vs MV8 8 (speedup 1.6).

| metric | value |
|---|---|
| acceptRate | 0.750 |
| draftRate (greedy alone) | 0.900 |
| tier0acc | 0.800 |
| mv8 | 0.850 |

PASS (reopens same-vocab draft/verify pair v1) — pre-reg: kill iff acceptRate*1.6<1 OR tier0acc<mv8-0.05.

