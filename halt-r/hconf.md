# H-conf: conformal abstention gate (measured, CI)
Same protocol (P=8000, sup, no-gen). alpha=0.1: calibrate 90th-pct nonconformity on select, abstain->bigram on report. Bounded claim: selective risk down at measured answer rate.

| corpus | W | conf | interp acc/ppl (lam*) | conf acc/ppl (absRate) |
|---|---|---|---|---|
| shakes | 16 | 0 | 0.270/12.41 (0.1) | 0.000/0.00 (0.00) |
| shakes | 16 | 1 | 0.270/12.41 (0.1) | 0.290/4.13 (0.09) |
| shakes | 8 | 0 | 0.270/13.18 (0.25) | 0.000/0.00 (0.00) |
| shakes | 8 | 1 | 0.270/13.18 (0.25) | 0.270/3.78 (0.10) |
| ts | 16 | 0 | 0.350/11.47 (0.25) | 0.000/0.00 (0.00) |
| ts | 16 | 1 | 0.350/11.47 (0.25) | 0.350/4.46 (0.13) |
| ts | 8 | 0 | 0.540/7.56 (0.25) | 0.000/0.00 (0.00) |
| ts | 8 | 1 | 0.540/7.56 (0.25) | 0.490/4.25 (0.23) |

shakes W8 full 13.18 -> selective 3.78 answerRate=0.90. VERDICT: ABSTENTION PAYS.

shakes W16 full 12.41 -> selective 4.13 answerRate=0.91. VERDICT: ABSTENTION PAYS.

ts W8 full 7.56 -> selective 4.25 answerRate=0.77. VERDICT: ABSTENTION PAYS.

ts W16 full 11.47 -> selective 4.46 answerRate=0.87. VERDICT: ABSTENTION PAYS.
