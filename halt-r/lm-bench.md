# HALT-R LM bench (measured, CI)
Task: next-char on held-out 10%. Hopfield retrieval store vs same-query n-gram baselines. Models AES-wrapped.

| corpus | W | P | store | beta | hop-acc | hop-ppl | bigram acc/ppl | unigram acc/ppl | steps | sha |
|---|---|---|---|---|---|---|---|---|---|---|
| shakes | 2 | 8000 | 1.0 MB | 3 | 0.310 | 72.12 | 0.280/12.26 | 0.140/27.57 | 1.19 | `079295b37a8f` |
| shakes | 4 | 8000 | 2.0 MB | 3 | 0.330 | 27285.92 | 0.280/12.13 | 0.140/27.64 | 2.85 | `0c1b7572ecec` |
| shakes | 8 | 8000 | 4.0 MB | 0.3 | 0.130 | 28.91 | 0.270/12.05 | 0.130/28.72 | 6.75 | `958183c4329f` |
| shakes | 16 | 8000 | 7.9 MB | 0.1 | 0.170 | 28.81 | 0.270/10.44 | 0.170/28.55 | 4.41 | `a01540dbf293` |
| shakes | 32 | 8000 | 15.9 MB | 0.1 | 0.120 | 28.95 | 0.200/12.02 | 0.120/29.04 | 4.68 | `2b4075015b2a` |
