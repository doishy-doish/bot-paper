# HALT-R bout: tiny-110K vs 1M-frontier (measured, CI, 20-way)
Same data, same 90/10 split, same 100-query seed-2000 protocol (q100 trailing-block context). 5 train seeds per cell; mean plus stdev over seeds.

| corpus | model | params | iters | q100 acc mean+-sd | q100 ppl mean+-sd |
|---|---|---|---|---|---|
| shakes | tiny | 110464 | 6000 | 0.444+-0.036 | 6.53+-0.57 |
| shakes | onem | 818176 | 6000 | 0.450+-0.009 | 7.25+-0.39 |
| ts | tiny | 112256 | 9000 | 0.588+-0.020 | 4.38+-0.14 |
| ts | onem | 821760 | 9000 | 0.676+-0.017 | 2.96+-0.05 |
