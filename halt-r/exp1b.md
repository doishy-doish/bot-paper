# Exp1b: verifier-vs-majority (measured, CI)
Donor Qwen2.5-0.5B Q4_K_M, 20 queries, K=8 temp 0.7, last-integer grading, harness clean.

| metric | acc |
|---|---|
| single | 0.600 |
| majority-8 | 0.850 |
| oracle | 0.950 |
| verifier (abstain->MV8) | 0.800 |
| headroom (oracle - mv8, descriptive) | 0.100 |
| backfire (mv8 wrong, single right) | 0 |
| abstained | 2 |
| verifier-vs-mv8 b/c | 0/1 |
| McNemar p | 1.0000 |
| margin-2 meanStop / stopmaj-acc | 3.80 / 0.850 |

D1: ROOM (mv8 below oracle). D2: selector stays vote-only — pre-reg keep iff verifier top-1 beats MV8 same bench McNemar p<0.05.

