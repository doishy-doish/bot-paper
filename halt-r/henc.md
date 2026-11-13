# H-enc: contrastive-learned encoder vs sup scales (measured, CI)
Same protocol (P=8000, H=256, no-gen). enc: 100 SGD steps, batch 128, InfoNCE same-next attract (JS-only, zero python). sup rows are same-workflow controls. Cost-class is first-class: both linear-O(P·D) — the encoder changes the constant and the geometry, not the series; sublinear is H9.

| corpus | W | proj | cost | encLoss first->last | interp acc/ppl (lam*) |
|---|---|---|---|---|---|
| shakes | 16 | enc | linear-O(P·D) | 4.874->4.524 | 0.230/14.12 (0.1) |
| shakes | 16 | sup | linear-O(P·D) | 0.000->0.000 | 0.270/12.41 (0.1) |
| shakes | 8 | enc | linear-O(P·D) | 4.961->4.749 | 0.310/12.54 (0.1) |
| shakes | 8 | sup | linear-O(P·D) | 0.000->0.000 | 0.270/13.18 (0.25) |
| ts | 16 | enc | linear-O(P·D) | 4.897->4.563 | 0.350/10.71 (0.25) |
| ts | 16 | sup | linear-O(P·D) | 0.000->0.000 | 0.350/11.47 (0.25) |
| ts | 8 | enc | linear-O(P·D) | 4.971->4.600 | 0.430/8.00 (0.25) |
| ts | 8 | sup | linear-O(P·D) | 0.000->0.000 | 0.540/7.56 (0.25) |

shakes W8: sup 13.18 vs enc 12.54 (lam*=0.1). VERDICT: LEARNING CRACKS IT (encoder beats scales).

shakes W16: sup 12.41 vs enc 14.12 (lam*=0.1). VERDICT: SCALES STAND (learning adds nothing).

ts W8: sup 7.56 vs enc 8.00 (lam*=0.25). VERDICT: SCALES STAND (learning adds nothing).

ts W16: sup 11.47 vs enc 10.71 (lam*=0.25). VERDICT: LEARNING CRACKS IT (encoder beats scales).
