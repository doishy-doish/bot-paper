# Exp1: verifier-vs-majority headroom (measured, CI)
Donor Qwen2.5-0.5B Q4_K_M, 20 short-answer prompts, K=8 at temp 0.7, last-integer grading.

| metric | acc |
|---|---|
| single | 0.000 |
| majority-4 | 0.000 |
| majority-8 | 0.000 |
| oracle | 0.000 |
| headroom (oracle - mv8) | 0.000 |

GATE: KILL verifier-selection at scout (headroom < 0.10)
