# ARITH gluttony: ingest a skill we lack (measured, CI)
Donor: tiny transformer trained on sums 0-49 only (frozen after train). Store: same 0-49 text. Test: sums 50-99 (held-out range, bench 90/10 cut lands exactly on the boundary by construction, match-free 1.0000).

| pole | acc/ppl (report stream) | lam* |
|---|---|---|
| store+bigram (interp) | 0.360/65.16 | 0.25 |
| donor alone (frozen) | 0.300/73.76 | — |
| blend (ingested) | 0.300/39.92 | 0.1 |
| graft5 keys, no donor at query | 0.380/76.71 | 0.5 |
bigram 0.130/110.50 (chance 1/13=0.077). mem/nov on test queries: memN=0 novN=100 (novFrac 1.000).

 donor-vs-interp McNemar b/c=7/13 p=0.263. blend-vs-interp b/c=7/13 p=0.263 d=25.24.

Difference (post-ingestion vs donor put in): agree(blend,donor)=1.000 agree(blend,interp)=0.800 agree(donor,interp)=0.800.

VERDICTS: store~=chance FAILS. donor strong FAILS. blend~=donor FAILS. graft5 survival STANDS (rent-only for algorithmic skill) d5=-11.54 p=0.815.
