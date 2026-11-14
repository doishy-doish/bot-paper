# Triage: Tier G v2 router signals (measured, CI)
Donor Qwen2.5-0.5B Q4_K_M, 20 queries, K=8 + greedy, harness clean.

| metric | value |
|---|---|
| mv8 | 0.850 |
| directFrac | 0.750 |
| directAcc (MV8 on direct) | 1.000 |

PASS (build invoke-gate router) — pre-reg: directFrac>=0.5 AND directAcc>=mv8-0.05.

