# HALT-R bout: tiny-110K vs 1M-frontier (measured, CI, 20-way)
Same data, same 90/10 split, same 100-query seed-2000 protocol (q100 trailing-block context). 5 train seeds per cell; mean plus stdev over seeds.

| corpus | model | params | iters | q100 acc mean+-sd | q100 ppl mean+-sd |
|---|---|---|---|---|---|
| shakes | tiny | 106688 | 6000 | 0.460+-0.032 | 6.08+-0.25 |
| shakes | onem | 803712 | 6000 | 0.480+-0.021 | 6.73+-0.19 |
| ts | tiny | 108480 | 9000 | 0.596+-0.016 | 4.11+-0.17 |
| ts | onem | 807296 | 9000 | 0.674+-0.005 | 2.76+-0.11 |
