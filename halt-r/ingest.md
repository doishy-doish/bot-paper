# Ingest demo: frozen models in, zero training, self-routing core, self-editing memory
Voters frozen: bigram (counts), HALT-R store (one-shot outer product), tiny-GPT (frozen q100 streams, no finetune). Core: Hedge online routing over the lam grid (frozen on select stream). Surgery: C2 implant/verify/delete/verify transcript. No phase of this demo trains anything.

## Routing (ingest jobs)

| corpus | hedgeAcc | hedgePpl | interpAcc | interpPpl | gptCover | gptMixAcc | gptMixPpl |
|---|---|---|---|---|---|---|---|
| shakes | 0.380 | 9.91 | 0.380 | 9.82 | 100 | 0.460 | 5.31 |
| ts | 0.350 | 9.70 | 0.350 | 9.70 | 100 | 0.650 | 3.30 |

## Surgery (surgery jobs)

| corpus | remember | editFlip | forgetLeak | collatBase | collatAfter | n |
|---|---|---|---|---|---|---|
| shakes | 0.95 | 0.9 | 0 | 0.28 | 0.28 | 20 |
| ts | 0.85 | 0.9 | 0 | 0.5 | 0.5 | 20 |

VERDICT: demo shows the unpublished thesis running — frozen models ingested as voters, a small core routing among them, memory it can edit and delete exactly. Zero gradient steps anywhere in the pipeline.
