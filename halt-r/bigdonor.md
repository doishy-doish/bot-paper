# BIGDONOR: frozen-Qwen graft matrix (measured, CI)
revbal/shakes/ts W16, P=8000, T200 same seeds, graft5-K256 (floor), donor rows via safetensors ranges (exact tok rows, zero training, zero full download).

| corpus | arm | interp acc/ppl (lam*) | graftSpec |
|---|---|---|---|
| revbal | q05 | 0.405/7.50 (0.5) | hf,Qwen/Qwen2.5-0.5B-Instruct,model.safetensors |
| revbal | q15 | 0.395/7.17 (0.5) | hf,Qwen/Qwen2.5-1.5B-Instruct,model.safetensors |
| revbal | sup | 0.295/8.82 (0.25) | - |
| revbal | tiny | 0.370/8.17 (0.25) | halt-r/tinygpt-arith-revbal.gguf |
| shakes | q05 | 0.305/11.93 (0) | hf,Qwen/Qwen2.5-0.5B-Instruct,model.safetensors |
| shakes | q15 | 0.305/11.93 (0) | hf,Qwen/Qwen2.5-1.5B-Instruct,model.safetensors |
| shakes | sup | 0.350/11.11 (0.1) | - |
| ts | q05 | 0.335/10.43 (0.1) | hf,Qwen/Qwen2.5-0.5B-Instruct,model.safetensors |
| ts | q15 | 0.345/10.32 (0.1) | hf,Qwen/Qwen2.5-1.5B-Instruct,model.safetensors |
| ts | sup | 0.355/10.03 (0.1) | - |

| donor | corpus | alone acc/ppl (coverage) |
|---|---|---|
| q05 | revbal | 0.325/125892541.18 (0.995) |
| q05 | shakes | 0.220/2290867652.77 (0.865) |
| q05 | ts | 0.216/2552908068.24 (0.960) |
| q15 | revbal | 0.310/190546071.80 (1.000) |
| q15 | shakes | 0.270/575439937.34 (0.870) |
| q15 | ts | 0.245/1148153621.50 (0.980) |

revbal tiny: sup 8.82 vs graft 8.17 d=0.66 b/c=23/8 p=0.0107 HOLDS.

revbal q05: sup 8.82 vs graft 7.50 d=1.33 b/c=31/9 p=0.0007 HOLDS.

revbal q15: sup 8.82 vs graft 7.17 d=1.65 b/c=29/9 p=0.0017 HOLDS.

shakes q05: sup 11.11 vs graft 11.93 d=-0.81 b/c=4/13 p=0.0490 falls.

shakes q15: sup 11.11 vs graft 11.93 d=-0.81 b/c=4/13 p=0.0490 falls.

ts q05: sup 10.03 vs graft 10.43 d=-0.41 b/c=1/5 p=0.2188 falls.

ts q15: sup 10.03 vs graft 10.32 d=-0.30 b/c=1/3 p=0.6250 falls.

CONTINUITY: corpus=shakes d05=-0.81 d15=-0.81 → 7B does NOT fire.

TRANSFER (ts, donor-known control): read descriptively against the confound note; C1 arXiv-unknown fallback staged if ts is uninformative either way.
