# Ingest demo: frozen models in, zero training, self-routing core, self-editing memory
Voters frozen: bigram (counts), HALT-R store (one-shot outer product), tiny-GPT (frozen q100 streams, no finetune). Core: Hedge online routing over the lam grid (frozen on select stream). Surgery: C2 implant/verify/delete/verify transcript. No phase of this demo trains anything.

## Routing (ingest jobs)

| corpus | hedgeAcc | hedgePpl | interpAcc | interpPpl | gptCover | gptMixAcc | gptMixPpl |
|---|---|---|---|---|---|---|---|
| shakes | 0.280 | 13.03 | 0.270 | 13.18 | 2 | 0.000 | 3438.19 |
| ts | 0.530 | 7.47 | 0.540 | 7.56 | 1 | 1.000 | 1.00 |

## Surgery (surgery jobs)

| corpus | remember | editFlip | forgetLeak | collatBase | collatAfter | n |
|---|---|---|---|---|---|---|
| shakes | 0.95 | 0.9 | 0 | 0.28 | 0.28 | 20 |
| ts | 0.85 | 0.9 | 0 | 0.5 | 0.5 | 20 |

VERDICT: demo shows the unpublished thesis running — frozen models ingested as voters, a small core routing among them, memory it can edit and delete exactly. Zero gradient steps anywhere in the pipeline.
