# COMBO816: grafted keys (K=256) + W-matched donor values + router (measured, CI)
Same protocol (P=8000, char, no-gen, T100). interp arm = graft5 keys + bigram (absorption alone). gpt arm = graft5 keys + frozen donor dists at the same W (absorption + delegation). Cross-job paired test (same seeds/queries).

| corpus | W | arm | interp acc/ppl (lam*) | gpt-mix acc/ppl (lam*) | cover |
|---|---|---|---|---|---|
| ts | 8 | interp | 0.330/9.37 (0.1) | — | — |
| ts | 8 | gpt | — | 0.650/3.46 (0) | 100/100 |

ts W8 combo816: interp 9.37 vs gpt-mix 3.46 d=5.90. McNemar gpt-vs-interp b/c=34/2 p=0.000 SIGNIFICANT. VERDICT: BLEND WINS (keys + values + router beat keys alone).
| ts | 16 | interp | 0.330/11.43 (0.1) | — | — |
| ts | 16 | gpt | — | 0.530/4.89 (0) | 100/100 |

ts W16 combo816: interp 11.43 vs gpt-mix 4.89 d=6.54. McNemar gpt-vs-interp b/c=23/3 p=0.000 SIGNIFICANT. VERDICT: BLEND WINS (keys + values + router beat keys alone).
| shakes | 8 | interp | 0.320/13.77 (0.1) | — | — |
| shakes | 8 | gpt | — | 0.470/7.98 (0) | 100/100 |

shakes W8 combo816: interp 13.77 vs gpt-mix 7.98 d=5.79. McNemar gpt-vs-interp b/c=19/4 p=0.003 SIGNIFICANT. VERDICT: BLEND WINS (keys + values + router beat keys alone).
| shakes | 16 | interp | 0.260/13.68 (0) | — | — |
| shakes | 16 | gpt | — | 0.400/7.53 (0) | 100/100 |

shakes W16 combo816: interp 13.68 vs gpt-mix 7.53 d=6.15. McNemar gpt-vs-interp b/c=17/3 p=0.003 SIGNIFICANT. VERDICT: BLEND WINS (keys + values + router beat keys alone).
