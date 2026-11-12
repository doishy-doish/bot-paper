# HALT-R pretrain sweep (measured, CI)
Date: 2026-11-12. Seeded synthetic (LCG 7), 10% flips, beta via mini-probe. Models AES-wrapped; hashes below.

| scale | P×D | store | beta·D | acc | steps | build | eval | maxRSS | sha256 |
|---|---|---|---|---|---|---|---|---|---|
| S | 1000×256 | 0.2 MB | 32 | 1.000 | 2.00 | 0s | 1s | 50.4 MB | `699241e1a0ee2558…` |
| M | 10000×512 | 4.9 MB | 32 | 1.000 | 2.00 | 0s | 10s | 57.2 MB | `73740ea0a30ea3d8…` |
| L | 50000×1024 | 48.8 MB | 32 | 1.000 | 2.00 | 1s | 76s | 102.8 MB | `18ba6e13fca8eba9…` |
| XL | 200000×1024 | 195.3 MB | 32 | 1.000 | 2.00 | 2s | 155s | 264.7 MB | `5fd346df9057ac48…` |
