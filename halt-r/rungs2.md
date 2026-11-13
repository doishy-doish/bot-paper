# HALT-R rungs2: gate + prune + slope (measured, CI, concurrent)
BPC = log2(char-ppl). text8 W4. Pilot anchor (P=8000, no gate): interp 3.21 BPC vs bigram 3.31.

## B1 confidence gate (tau sweep, P=8000 raw)
| tau | gatedFrac | interp acc/ppl/BPC | vs bigram ppl |
|---|---|---|---|
| 0 | 0.000 | 0.360/9.26/3.21 | at/below bigram |
| 0.1 | 0.150 | 0.340/9.45/3.24 | at/below bigram |
| 0.25 | 0.430 | 0.320/9.71/3.28 | at/below bigram |
| 0.5 | 0.590 | 0.320/9.96/3.32 | ABOVE bigram by 0.06 |

## B2 store prune (nested random subsets, P=8000 raw, no gate)
| keep | Peff | interp acc/ppl/BPC | retention vs keep=1 | evalMflops |
|---|---|---|---|---|
| 1 | 8000 | 0.360/9.26/3.21 | 100.0% | 2676.7 |
| 0.5 | 4064 | 0.330/9.37/3.23 | 91.7% | 1434.4 |
| 0.25 | 2039 | 0.250/10.54/3.40 | 69.4% | 707.3 |

## H3 text8 slope 8K->32K (2 doublings, W4)
| proj | P=8000 BPC (pilot) | P=32000 BPC | slope/dbl | vs -0.11 family |
|---|---|---|---|---|
| raw | 3.21 | 2.87 | -0.168 | in family |
| sim | 3.29 | 2.88 | -0.206 | in family |
