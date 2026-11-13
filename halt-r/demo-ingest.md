# Ingest demo (measured)

## ts
Donor tinygpt-ts.gguf: 436064 bytes, sha 26402e2d9eac..., 108480 params, GGUFv3 halt-tinygpt rev 2, 17 tensors byte-exact layout. Ref 5/5 argmax-consistent. Voter streams: s2000 acc=0.57 ppl=4.36, s2001 acc=0.65 ppl=3.3, cover 100/100 both. All in pre-reg band.

## shakes
Donor tinygpt-shakes.gguf: 428768 bytes, sha fcd797be68c5..., 106688 params, GGUFv3 halt-tinygpt rev 2, 17 tensors byte-exact layout. Ref 5/5 argmax-consistent. Voter streams: s2000 acc=0.49 ppl=6.12, s2001 acc=0.46 ppl=5.31, cover 100/100 both. All in pre-reg band.

Post-ingestion artifact: demo-ingest-model.json (donor pins + voter replay + router recipe). Boot loads it; nothing rebuilds. Zero training.
