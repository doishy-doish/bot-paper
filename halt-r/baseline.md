# HALT-R baseline: tiny transformer (measured, CI)
Same data, same 90/10 split, same 100-query seed-2000 protocol as the HALT-R bench (q100 uses trailing-32 context). Full-val is the whole held-out 10%.

| corpus | params | iters | trainSec | full-val acc/ppl | q100 acc/ppl |
|---|---|---|---|---|---|
| shakes | 110464 | 2000 | 167 | 0.380/8.19 | 0.380/8.00 |
| ts | 112256 | 3000 | 424 | 0.535/4.65 | 0.510/5.44 |
