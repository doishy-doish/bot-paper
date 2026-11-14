# GRAFTBPE: BPE-graft compose (measured, CI)
ts W8/W16, P=8000, BPE-256, no-gen, T100. Internal verdict (same tokenizer; no cross-tokenizer claims).

| W | proj | interp acc/ppl (lam*) |
|---|---|---|
| 8 | graft5 | 0.330/12.17 (0.25) |
| 8 | raw | 0.380/11.34 (0.25) |
| 16 | graft5 | 0.360/10.62 (0.1) |
| 16 | raw | 0.330/11.08 (0) |

W=16: raw-bpe 11.08 vs graft5-bpe 10.62 d=0.46 b/c=3/0 p=0.250 → stands.

W=8: raw-bpe 11.34 vs graft5-bpe 12.17 d=-0.83 b/c=1/6 p=0.125 → stands.

Read: W16 win = scope expands (compositional alignment); W8-retain fail = pooling destroys geometry (compose dead, Track 3 only).
