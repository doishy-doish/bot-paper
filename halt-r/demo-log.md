demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/tinygpt-ts.gguf
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/vocab-ts.json
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/ref-ts.json
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/q100-gpt-ts.json
demo: ts gguf OK bytes=436064 params=108480 sha=26402e2d9eac...
demo: ts ref OK refs=5
demo: ts stream2000 acc=0.570 ppl=4.36 cover=100/100 IN BAND
demo: ts stream2001 acc=0.650 ppl=3.30 cover=100/100 IN BAND
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/tinygpt-shakes.gguf
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/vocab-shakes.json
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/ref-shakes.json
demo: pull https://raw.githubusercontent.com/doishy-doish/bot-paper/main/halt-r/q100-gpt-shakes.json
demo: shakes gguf OK bytes=428768 params=106688 sha=fcd797be68c5...
demo: shakes ref OK refs=5
demo: shakes stream2000 acc=0.490 ppl=6.12 cover=100/100 IN BAND
demo: shakes stream2001 acc=0.460 ppl=5.31 cover=100/100 IN BAND
# Ingest demo (measured)

## ts
Donor tinygpt-ts.gguf: 436064 bytes, sha 26402e2d9eac..., 108480 params, GGUFv3 halt-tinygpt rev 2, 17 tensors byte-exact layout. Ref 5/5 argmax-consistent. Voter streams: s2000 acc=0.57 ppl=4.36, s2001 acc=0.65 ppl=3.3, cover 100/100 both. All in pre-reg band.

## shakes
Donor tinygpt-shakes.gguf: 428768 bytes, sha fcd797be68c5..., 106688 params, GGUFv3 halt-tinygpt rev 2, 17 tensors byte-exact layout. Ref 5/5 argmax-consistent. Voter streams: s2000 acc=0.49 ppl=6.12, s2001 acc=0.46 ppl=5.31, cover 100/100 both. All in pre-reg band.

Post-ingestion artifact: demo-ingest-model.json (donor pins + voter replay + router recipe). Boot loads it; nothing rebuilds. Zero training.
