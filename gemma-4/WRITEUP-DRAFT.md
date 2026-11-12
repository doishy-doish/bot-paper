# Calibration Is Real, Verification Is Not Enough

## Subtitle: A pre-registered falsification chain on small-model code repair — frozen 31B agent, 47 failed baseline attempts, 14/32 on a calibrated slice, two controlled nulls, and a flat held-out probe

**Track:** Gemma 4 Developer Agentre. **Status:** draft (2026-11-12, ~12:00 UTC). A temperature probe (Best-of-3, T=0.75, 5 tasks x 3 rollouts) is running on kernel `andralemisquitta/gemma-bestofk-temp-probe`; its readout lands in Section 3.5 before the 23:59 UTC fuse. Everything else below is landed and trace-backed.

## Abstract

We ask a narrow question: what actually converts failures into fixes for a small repair agent, once the harness is frozen? Starting from 0/47 resolved attempts across five pre-registered configurations, a frozen 31-billion-parameter agent resolves 14/32 tasks (43.8%) on a calibrated single-file slice — 5/10 on the first ten, 9/22 on the held-out remainder — then scores a flat 3/12 (25.0%; 3/11 = 27.3% after removing a harness-killed run) on twelve brand-new tasks, exactly 1/4 in each of three groups (small-rich, large-rich, non-rich). Two controlled interventions both score clean nulls: adding verify-then-fix instructions converts 0/5 misses (W2B, 5/10 with the identical resolved set), and injecting the exact target function name converts 0/5 (G2v2), with the same 687-character wrong patch recurring on one task with and without the injection. A 44-trace zero-GPU audit cuts the training pool to 10 source-only surgical greens, rejects a compile-guard hypothesis (all four exit-2 failures are semantic, zero are syntax), and finds only 3/44 traces ever invoked pytest. Claim: calibration is real and large; verification prompting and specification injection are insufficient; no task group predicts success; the remaining levers are sampling/weights (attractor basins) and task selection (locality). Method and failure modes are the contribution; there is no score-chasing in this paper.

## 1. Introduction

Small models are the economic frontier of software repair: if a 31B agent can fix real bugs, every team can run it. The literature reports headline numbers on large models, but the decision a practitioner faces is narrower — given a frozen harness and a fixed budget, which change actually converts a miss into a fix? Prompt tweaks are cheap to propose and expensive to believe, because every run costs GPU hours and every post-hoc story fits the last readout.

We ran this lane the boring way. One agent shape, frozen early and never re-tuned. Every subsequent run pre-registered with its own tasks, its own pass/fail bar, and a hard stop on re-fires: a re-run is a new arm, never a second attempt. Five configurations failed first (0/47 combined), which gave us a calibrated floor instead of a selected story. From there we tested three candidate converters — verification discipline, specification injection, and family/size structure — and one measurement-hygiene program that decides what, if anything, is clean enough to train on.

The headline result is a split, not a slope. Calibration (picking the right tasks and freezing the procedure) moved the rate from 0% to 43.8% on 32 tasks. Interventions moved nothing: 0/5 and 0/5 on controlled probes. A held-out family probe then came back perfectly flat (1/4 x 3), killing both of our structural hypotheses at once. What survives is a residual with two attractors: a handful of tasks where the agent converges to the same wrong patch regardless of information, and a locality pattern where single-file, sub-1200-character fixes succeed and everything else fails. A sampling probe now running tests whether the attractor tasks are escapable basins or locked weights.

## 2. Methods

### 2.1 The frozen agent

All runs use one agent: a single-root loop on the 31B instruction-tuned base (`gemma-4-31b-it-qat-w4a16-ct`), nine tools (run, read, edit, write, four code-navigation tools, submit; three graph tools declared but banned by the prompt), and a fixed envelope — 28 tool calls, 8 minutes, 80 turns, 240 s task timeout, 300 s verification timeout, offline sandbox, served on 4x L4 via vLLM. The system prompt is the "winner-shape" procedure: SPEC first (three lines: trigger, expected, acceptance), locate-then-read, best-guess edit by call 12, verify with one named test file plus `-k`, never touch tests or config, finish with `git diff` then submit. Sampling is temperature 0.2 / top_p 0.95 — verified three ways (the sampling config, the fire-notebook asserts, and the kernel boot log, which additionally records `top_k` as missing). This matters: at T=0.2 the agent is a near-deterministic function of the task, so repeated runs measure the basin, not the variance. Every run ships stock 31B adapters and a manifest of file hashes; a validator gate runs in-notebook before any GPU is burned.

### 2.2 Pre-registration discipline

Each shot is a separate pre-registered arm: frozen task list, frozen harness, a written pass bar, one fire. The log records five pre-W1 arms (a diagnostic, two main-shape arms, a targeted calibration arm, and two injection generations) totalling 0/47. Then: W1 (first 10 of a 32-task WIN slice), W2A (remaining 22), W2B (W1's 10 with verify-fix instructions appended — an A/B with no other variable), G2v2 (5 tasks with the target function named in the issue text), FAM (12 never-before-run tasks), and the running Best-of-K probe (5 tasks x 3 rollouts at T=0.75). Re-runs are new arms by rule; nothing in this paper is a re-fire to chase noise.

### 2.3 The surgical-green filter and the zero-GPU audit

A fix the agent cannot reproduce from its own trace is not training data. We audited all 44 landed traces without spending GPU: for each trace we extracted every edited file, every submit, every CONFIDENCE line, and every pytest invocation. The harvest filter keeps a trace only if all of these hold: exit code 0 with resolved=true, patch 200–1200 characters, a CONFIDENCE line quoting a passing test, exactly one changed source file, and zero touches to test files, docs, or config. The filter is a script (`scratch/gemma/harvest_phase_b.py`), not a judgment call. Ten traces survive. The cuts are findings in their own right (Section 3.4): two resolved traces edited test files, one edited only a docs page, four exit-2 failures are all semantic (which kills a plausible guard hypothesis), one resolved-set member submitted an empty patch after 28 read-only calls, and one run died to sandbox latency at 9 calls.

### 2.4 The held-out family probe (FAM)

WIN-slice numbers are public-set bounds: task selection itself was calibrated on earlier readouts. FAM is the antidote — 12 tasks never used in W1, W2A, W2B, or G2, split 4/4/4 into rich-small (<600-char golds), rich-large (800–1100), and non-rich (FastAPI plus one requests task), same frozen harness, no injection. Its pre-registration names three mutually exclusive readings: rich-small high plus rich-large collapse means a size cliff (generality hypothesis G1); rich groups high with non-rich low means family prior (G3); flat low means the WIN streak was slice luck. FAM ran unblinded but untouched: no prompt change, no selection after the fact.

## 3. Experiments

### 3.1 The calibration ladder: 0/47 to 14/32

| Stage | Resolved | Note |
|---|---|---|
| Baseline (5 arms) | 0/47 | DIAG 0/11 + H1v6 0/10 + ARM2 0/10 + CAL 0/3 + CALv3 0/3 + G2 0/5 + G2v2 0/5 |
| W1 (first 10 WIN) | 5/10 | fastapi_14786, rich_2943, rich_3882, fastapi_13537, rich_3518 |
| W2A (rest of WIN) | 9/22 | 7 rich/fastapi greens incl. 5077 (3803 chars); 7309 fails with a 687-char patch |
| WIN combined | 14/32 (43.8%) | W1 + W2A, same frozen harness |
| FAM (held-out) | 3/12 (25.0%) | 1/4 x 3 groups; 3/11 (27.3%) ex-harness-kill |
| MAIN10 (early arms) | 0/10 | pre-calibration slice, both early configurations |

The ladder's step is selection plus procedure, not intelligence: the same agent that scores 0/10 on an uncalibrated slice scores 14/32 on single-file, adds-test tasks under 1200 characters with an 8-minute, 28-call envelope. The honest reading cuts both ways — calibration is a real, large, reproducible effect, and every WIN number is a public-set upper bound, which is why FAM exists.

### 3.2 Two controlled nulls

**Verification does not convert (W2B: 0/5, 5/10 identical set).** Appending VERIFY-FIX instructions (re-run the failing behavior after the edit, confirm the fix on the reproducer) to the W1 prompt changed the outcome on exactly zero tasks: the same five resolve, the same five miss. Patches moved on three of the five misses (rich_3296: 539 to 1737 chars; requests_7328: 884 to 3127; fastapi_14258: 2308 to 1085) while outcomes stayed put — the agent worked differently and failed identically. Verification prompting changes behavior without changing results.

**Specification does not convert (G2v2: 0/5).** Appending the exact target function and file to the issue text ("The change required is in `_parse_content_type_header` in `src/requests/utils.py`. Other functions with similar patterns are out of scope.") converted zero of three probe tasks, with two negative-control tasks also failing as predicted. The sharpest datum is requests_7309: the agent produced the same 687-character patch with the target named as it did without it. Naming the locus does not move the weights. One scope caveat is recorded and bounds the claim: the injection named `_parse_content_type_header` and fenced "other functions" as out of scope, but the gold patch touches the sister function `parse_dict_header` — so G2v2 falsifies "naming the function converts," not "complete specification converts." A follow-up arm (CALv3) moved the 7309 patch from the 687-char attractor to a 1326-char two-hunk patch whose second hunk goes in the gold direction (drops the bare `None`), and still failed — specification-adjacent information moves the patch without converting the outcome.

### 3.3 The flat held-out (FAM: 3/12, perfect 1/4 x 3)

| Group | Resolved | Members |
|---|---|---|
| rich-small | 1/4 | rich_3470 resolves; 3471, 3472, 3063 fail |
| rich-large | 1/4 | rich_3480 resolves; 3064, 3535, 3506 fail |
| non-rich | 1/4 | fastapi_14492 resolves; 14360, 5624, 7502 fail |

A perfectly flat profile falsifies both structural hypotheses at once: there is no size cliff (small and large rich groups tie) and no family prior (rich and non-rich tie). The three corrections sharpen rather than soften the picture. rich_3506 died at 9 tool calls with exit -1 — a sandbox-latency wall death, not a repair failure — so the capability rate is arguably 3/11 (27.3%). rich_3472 spent all 28 calls reading, edited nothing, and submitted an empty patch: the honest empty-submit failure mode, worth an explicit ban in all future prompts. fastapi_14492 edited a docs page (`docs_src/...`) alongside nothing else: it counts in the 3/12 as a harness-graded resolve but is excluded from any training pool. FAM's own verdict, pre-registered: the WIN streak does not generalize as a rate, and neither size nor family explains the residual.

### 3.4 What the audit found in 44 traces

Ten of fourteen WIN-resolved traces are source-only surgical greens; the audit's cuts are the section's real content. Two resolved traces edited test files (fastapi_9425 touched `test_annotation.py`; rich_3067 touched `tests/test_highlighter.py` across 13 edits) and are out of the training pool by rule. Four exit-code-2 failures (fastapi_14360, 14303, 14430, 11355) are all semantic — wrong module, wrong API, wrong site — with zero syntax errors anywhere, which rejects the cheap hypothesis that a `py_compile` gate would save them. Only 3 of 44 traces ever invoked pytest (two errored on collection, one selected 48 tests and still failed), and 11 of 12 FAM traces verified exclusively through hand-written `repro.py` scripts: the agent's confidence is calibrated on its own theater, with CONFIDENCE lines at 2–3 quoting reproducer output character-for-character and never quoting an exit code or a PASS count. The training pool this justifies is small (10 traces) and that is the point — everything else is either contaminated, untested, or both.

### 3.5 In flight: the sampling probe (TBD at draft time)

At T=0.2 the agent is near-deterministic, so every null above measures one basin per task. The running probe re-fires five tasks (attractor stratum: 7205/7315/7309; multi-modal stratum: 7328/3296) for K=3 rollouts each at T=0.75 under the frozen harness, ranking non-empty diffs by existing repo tests with ties broken toward smaller patches, and aborting a stratum if candidates come back byte-identical. Pre-registered gates: stratum A >=1/3 converts means basins are escapable by sampling; 0/3 means attractor basins are weight-locked; stratum B >=1/2 means execution-ranked Best-of-K harvests rollout variance. Positive adds a "sampling-basin escape" subsection; negative adds one line. Either way the paper finalizes before the fuse.

## 4. Related work

Repair-agent literature clusters around three explanations for why small models fail: they cannot find the fault (localization), they cannot tell a fix from theater (verification), or they have never seen the pattern (prior). Locate-then-repair decompositions attack the first; our traces obey the locate mandate (exact symbol names read from the issue, `grep -rn` before edits) and still fail at the same rate, so mechanism compliance without outcome movement is itself a datum. Completion-vs-correctness studies attack the second; our CONFIDENCE audit is a field instance — every trace ends certain, a quarter of them rightly. Curriculum and graph-context work attacks the third; our lane notes record a localization split (roughly mid-twenty percent when localized vs low single digits otherwise) consistent with our residual pattern that single-file sub-1200-character tasks convert and nothing else does. The honest contrast with headline field numbers (public-board top near 0.24 at the time of writing, from an unverified snapshot of 285 teams and 302 submissions; reported large-model rates far above ours) is selection and scale: our rates are small-model, single-harness, public-set bounds with every miss published. That is the contribution — a complete falsification chain at a scale anyone can afford to reproduce, where the nulls are the results.

## 5. Limitations

Single model family and one harness lineage: method generalizes, rates do not. Group comparisons are n=4 per bucket — powered for cliffs, blind to moderate effects. Verification ran on reproducers, not the hidden suite, so "verified" means "verified by the agent's own theater" unless stated. Contamination is unadjusted: public tasks may sit in pretraining, which makes our misses more interesting and our hits less so. The 14258 injection variant was never tested (saying so beats implying otherwise). The sampling probe was in flight at draft time.

## 6. Conclusion

Calibration is real: freezing one procedure and selecting solvable-shaped tasks took a small agent from 0/47 to 14/32. Verification prompting and specification injection are insufficient: two controlled probes moved behavior and patches while converting nothing. No task group predicts success on held-out data: 1/4 x 3 kills size and family stories together. What remains is weights and selection — attractor basins that swallow information (same 687-char patch with and without the answer's address on it) and a locality boundary around small single-file fixes. The next converter, if one exists, lives in sampling or in the weights, not in the prompt. That is a falsifiable sentence, and the probe running now decides the first half of it.

## References

- Falsification-chain log: `scratch/gemma/` fire tables (`fire-w1-table.json`, `fire-w2a-table.json`, `fire-w2b-table.json`, `fire-g2-v2-table.json`, `fire-fam-table.json`), `zero-gpu-audit.json`, `harvest_phase_b.py`, `probe-bestofk-prereg.md` (this repo).
- Pre-registration and paper plan: `paper/PRE-REG.md`, `paper/PAPER-PLAN.md`, `paper/DRAFT.md` (Nov-08 lane).
- Harness and BLUEPRINT lineage: `HARNESS_README.md`, `BLUEPRINT-V2.md`, `W2-DESIGN.md`, `RUN-LOG.md` (Nov-07/08 lane).
- Public competition context: Gemma 4 Developer Agent, paper track (Writeup, 3000 words, fuse 2026-11-12 23:59 UTC); main track fuse 2026-12-02.
- Related systems (directional contrast; exact figures as reported in lane notes, not re-verified here): SWE-bench-Live-style live-task benchmarks; Code Graphs-style localization splits; FrogNano-style curriculum training.

## Appendix A — Honesty notes

1. WIN-slice rates are public-set upper bounds; FAM is the only held-out rate in this paper.
2. fastapi_14492's resolve touched a docs page; excluded from training, kept in the 3/12 with this footnote.
3. rich_3506 is a harness wall-death (exit -1, 9 calls); capability rate 3/11 = 27.3% is the fairer denominator.
4. G2v2's scope caveat (sister function `parse_dict_header` vs named `_parse_content_type_header`) bounds the "specification" claim to naming.
5. Best-of-K was in flight at draft time; Section 3.5 reports whichever way it lands, or one line if it misses the fuse.

## Appendix B — Artifact index (all under `scratch/gemma/`)

Tables: `fire-w1-table.json`, `fire-w2a-table.json`, `fire-w2b-table.json`, `fire-g2-v2-table.json`, `fire-fam-table.json`, `fire-bestofk-table.json` (on readout). Traces: `w2a-traces/`, `fam-traces/` (22 + 12), output zips `fire-*-output.zip`. Audit: `zero-gpu-audit.json`, `harvest_phase_b.py`, `fire-bestofk-audit.json`. Pre-regs: `probe-bestofk-prereg.md`, `paper/PRE-REG.md`. Kernels: 144573461 (W2A), 144573665 (W2B), 144590816 (G2v2), 144590833 (FAM), 144607478 (BestofK).
