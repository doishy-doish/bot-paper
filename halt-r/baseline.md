# HALT-R baseline: tiny transformer (measured, CI)
Same data, same 90/10 split, same 100-query seed-2000 protocol as the HALT-R bench (q100 uses trailing-32 context). Full-val is the whole held-out 10%.

| corpus | params | iters | trainSec | full-val acc/ppl | q100 acc/ppl |
|---|---|---|---|---|---|
| shakes | 110464 | 6000 | 108 | 0.433/6.83 | 0.440/7.11 |
| ts | 112256 | 9000 | 126 | 0.596/3.73 | 0.590/4.19 |
