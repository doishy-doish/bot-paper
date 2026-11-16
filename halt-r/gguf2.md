# GGUF rung 2: Rust f32 twin parity (measured, CI)
Same committed tinygpt-*.gguf + ref-*.json the baseline exporter verified byte-exact. Rust reader parses GGUFv3 (no deps) and runs the exact rev-2 forward. PASS = all 5 argmax equal AND worst max|diff| < 1e-2 (warn only below).

| corpus | refs | argmax | worst maxdiff | verdict |
|---|---|---|---|---|
| shakes | 5 | 0/5 | 4.50e+0 | PARITY FAILS. |
| ts | 5 | 0/5 | 6.17e+0 | PARITY FAILS. |
