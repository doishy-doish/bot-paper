# COMBO: grafted keys (K=256) + donor values + router (measured, CI)
Same protocol (P=8000, char, no-gen, T100). interp arm = graft5 keys + bigram (absorption alone). gpt arm = graft5 keys + frozen donor dists (absorption + delegation, the feast). Cross-job paired test (same seeds/queries).

| corpus | W | arm | interp acc/ppl (lam*) | gpt-mix acc/ppl (lam*) | cover |
|---|---|---|---|---|---|
| ts | 8 | interp | 0.330/9.37 (0.1) | — | — |
| ts | 8 | gpt | — | 0.000/1811.33 (0.5) | 1/100 |

ts W8 combo: interp 9.37 vs gpt-mix 1811.33 d=-1801.96. McNemar gpt-vs-interp b/c=0/0 p=1.000. VERDICT: coverage-starved (1/100 — diagnose pos-join, not a model result).
| ts | 16 | interp | 0.330/11.43 (0.1) | — | — |
| ts | 16 | gpt | — | 0.000/768.76 (0.5) | 1/100 |

ts W16 combo: interp 11.43 vs gpt-mix 768.76 d=-757.33. McNemar gpt-vs-interp b/c=0/0 p=1.000. VERDICT: coverage-starved (1/100 — diagnose pos-join, not a model result).
| shakes | 8 | interp | 0.320/13.77 (0.1) | — | — |
| shakes | 8 | gpt | — | 0.000/38392.15 (1) | 2/100 |

shakes W8 combo: interp 13.77 vs gpt-mix 38392.15 d=-38378.38. McNemar gpt-vs-interp b/c=0/0 p=1.000. VERDICT: coverage-starved (2/100 — diagnose pos-join, not a model result).
| shakes | 16 | interp | 0.260/13.68 (0) | — | — |
| shakes | 16 | gpt | — | 0.000/109.25 (0.5) | 1/100 |

shakes W16 combo: interp 13.68 vs gpt-mix 109.25 d=-95.57. McNemar gpt-vs-interp b/c=0/0 p=1.000. VERDICT: coverage-starved (1/100 — diagnose pos-join, not a model result).
