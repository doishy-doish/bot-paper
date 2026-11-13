# DTW rerank vs sup prefilter
Same protocol (P=8000, sup, W8/W16, --no-gen). Top-50 store weights reranked by DTW distance (0/1 frame cost, tau=W). Pre-reg: dtwPpl < interpPpl - 0.2 AND McNemar p<0.05.

| corpus | W | dtw | lam* | acc | ppl | cost |
|---|---|---|---|---|---|---|
| shakes | 16 | 0 | 0.1 | 0.270 | 12.41 | linear-O(P·D) |
| shakes | 16 | 1 | 0.1 | 0.270 | 12.41 | linear-O(P·D+K·W^2) |
| shakes | 8 | 0 | 0.25 | 0.270 | 13.18 | linear-O(P·D) |
| shakes | 8 | 1 | 0.25 | 0.270 | 13.17 | linear-O(P·D+K·W^2) |
| ts | 16 | 0 | 0.25 | 0.350 | 11.47 | linear-O(P·D) |
| ts | 16 | 1 | 0.25 | 0.350 | 11.47 | linear-O(P·D+K·W^2) |
| ts | 8 | 0 | 0.25 | 0.540 | 7.56 | linear-O(P·D) |
| ts | 8 | 1 | 0.25 | 0.540 | 7.56 | linear-O(P·D+K·W^2) |

shakes W8: interp 13.18 vs dtw 13.17 (d=0.01, McNemar b=0 c=0 p=1.0000). VERDICT: DTW HOLDS (rerank does not carry — wall stays semantic).

shakes W16: interp 12.41 vs dtw 12.41 (d=-0.00, McNemar b=0 c=0 p=1.0000). VERDICT: DTW HOLDS (rerank does not carry — wall stays semantic).

ts W8: interp 7.56 vs dtw 7.56 (d=0.00, McNemar b=0 c=0 p=1.0000). VERDICT: DTW HOLDS (rerank does not carry — wall stays semantic).

ts W16: interp 11.47 vs dtw 11.47 (d=0.00, McNemar b=0 c=0 p=1.0000). VERDICT: DTW HOLDS (rerank does not carry — wall stays semantic).
