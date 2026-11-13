# HALT-R baseline: tiny transformer (measured, CI)
Same data, same 90/10 split, same 100-query seed-2000 protocol as the HALT-R bench (q100 uses trailing-32 context). Full-val is the whole held-out 10%.

| corpus | params | iters | trainSec | full-val acc/ppl | q100 acc/ppl |
|---|---|---|---|---|---|
| shakes | 106688 | 6000 | 135 | 0.434/6.77 | 0.490/6.12 |
| ts | 108480 | 9000 | 192 | 0.600/3.70 | 0.570/4.36 |
