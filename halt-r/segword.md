# SEGWORD: content-native word windows (measured, CI)
W in words (TOK=word, vocab cap 2000), proj raw, P=8000, no-gen, T100. Paired bar within-tokenizer; cross-tokenizer reads descriptive.

| corpus | W | tok | interp acc/ppl (lam*) | bigram ppl |
|---|---|---|---|---|
| shakes | 8 | char | 0.300/13.49 (0) | 12.05 |
| shakes | 8 | word | 0.330/115.93 (0.5) | 238.26 |
| ts | 16 | char | 0.320/11.53 (0.1) | 8.84 |
| ts | 16 | word | 0.190/125.71 (0.1) | 89.81 |
| ts | 8 | char | 0.260/11.03 (0) | 7.94 |
| ts | 8 | word | 0.280/101.22 (0) | 72.81 |

ts W16 word PRIMARY (walled): bigram 89.81 vs interp 125.71 (lam*=0.1) d=-35.90 McNemar b/c=3/1 p=0.625. VERDICT: NULL (sub-bar).

ts W8 word MUST-RETAIN (won): bigram 72.81 vs interp 101.22 (lam*=0) d=-28.41 McNemar b/c=0/0 p=1.000. VERDICT: WALL STANDS at W8 word (adds nothing).

shakes W8 word DIAGNOSTIC (works): bigram 238.26 vs interp 115.93 (lam*=0.5) d=122.33 McNemar b/c=3/3 p=1.000. VERDICT: NULL (sub-bar).
