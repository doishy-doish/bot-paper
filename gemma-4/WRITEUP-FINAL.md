# Calibration Is Real, Verification Is Not Enough
**Subtitle:** A pre-registered falsification chain on small-model code repair — frozen 31B agent, 47 failed baseline attempts, 14/32 on a calibrated slice, two controlled nulls, and a flat held-out probe

**Track:** Google — The Gemma 4 Developer Agent (Paper Track)
**Author:** Andrale Misquitta

---

## Abstract

We ask a narrow question: what actually converts failures into fixes for a small repair agent, once the harness is frozen? Starting from 0/47 resolved attempts across seven baseline arms on two checkpoints (0/37 E4B-era, 0/10 same-checkpoint 31B), a frozen 31-billion-parameter agent (`gemma-4-31b-it-qat-w4a16-ct`) resolves 14/32 tasks (43.8%, Wilson 95% CI [28.2%, 60.7%]; +43.8 pp, Fisher p = 0.017) on a calibrated single-file slice — 5/10 on the first ten, 9/22 on the held-out remainder — then scores a flat 3/12 (25.0%; 3/11 = 27.3% after removing a harness-killed run) on twelve brand-new tasks, exactly 1/4 in each of three groups (`rich-small`, `rich-large`, `non-rich`). Two controlled interventions both score clean nulls: adding verify-then-fix instructions converts 0/5 misses (`W2B`, 5/10 with the identical resolved set, p=9.8e-4 under rollout independence), and injecting the exact target function name converts 0/5 (`G2v2`), with the same 687-character wrong patch recurring on `requests_7309` with and without the injection. A 44-trace zero-GPU audit cuts the clean training pool to 9 source-only surgical greens, rejects a compile-guard hypothesis (all four exit-2 failures are semantic, zero are syntax), and finds only 3/44 traces ever invoked `pytest`. Calibration is real and large; verification prompting and symbol specification are insufficient; no task group predicts success out-of-sample; and the remaining levers live in sampling/weights (attractor basins) and task selection (locality). The contribution is the falsification chain, not a headline score.

---

## 1. Introduction

Small models are the economic frontier of software repair: if a 31B agent can fix real bugs on commodity multi-GPU hardware, every team can run it offline. The literature reports headline numbers on large models, but the decision a practitioner faces is narrower — given a frozen harness and a fixed budget, which change actually converts a miss into a fix? Prompt tweaks are cheap to propose and expensive to believe, because every run costs GPU hours and every post-hoc story fits the last readout.

We ran this lane the boring way. One agent shape, frozen early and never re-tuned. Every subsequent run was pre-registered with its own tasks, its own pass/fail bar, and a hard stop on silent re-fires: a re-run is a new arm, never a second attempt. Seven baseline arms failed first (0/47 combined: 0/37 on E4B, 0/10 on the 31B submission checkpoint), giving us an unvarnished floor instead of a selected story. From there we tested three candidate converters — verification discipline, specification injection, and family/size structure — and one zero-GPU measurement-hygiene program that decides what, if anything, is clean enough to train on.

The headline result is a split, not a slope. Calibration (picking well-scoped single-file tasks and freezing the procedure) moved the solve rate from 0% to 43.8% on 32 tasks. Prompt-level interventions moved nothing: 0/5 and 0/5 on controlled probes. A held-out family probe then came back perfectly flat (1/4 × 3), killing both of our structural hypotheses at once. What survives is a residual with two regimes: a handful of tasks where the agent converges to the same wrong patch regardless of extra information, and a locality pattern where single-file, sub-1200-character fixes succeed and multi-file or bloated diffs fail. A sampling probe (`Best-of-3` at T=0.75) tests whether the attractor tasks are escapable basins or locked weights.

---

## 2. Methods

### 2.1 The Frozen Agent
All runs use one agent: a single-root loop on the 31B instruction-tuned base (`gemma-4-31b-it-qat-w4a16-ct`), nine active tools (`run`, `read`, `edit`, `write`, four code-navigation tools, and `submit`; three graph tools declared in the schema but disabled by prompt instruction), and a fixed execution envelope — 28 tool calls, 80 turns, 480 s (8-minute) task wall-clock limit, 240 s per-command timeout, 300 s verification timeout, `EventsCompactionConfig` (`compaction_interval=15`, `overlap_size=2`, `token_threshold=14336`, `event_retention_size=5`), offline sandbox, served on `4 × L4` GPUs via `vLLM` (`tensor_parallel_size=4`).

The system prompt enforces the "winner-shape" procedure: `SPEC` first (three lines: trigger, expected, acceptance), locate-then-read, best-guess edit by call 12, verify against a targeted test or reproducer, never touch test or config files, and finish with `git diff` then `submit`. Sampling is locked at `temperature=0.2`, `top_p=0.95`, fixed `max_output_tokens`, and `thinking_budget=4096` — verified three ways (sampling config, notebook assertions, and kernel boot logs, which also record `top_k` as unset). At T=0.2, the agent acts as a near-deterministic function of the task, so repeated runs measure basin stability rather than sampling variance. Every run ships stock 31B adapters and a file-hash manifest; an automated validator gate runs in-notebook before any GPU time is spent.

### 2.2 Pre-Registration Discipline
Each shot is a separate pre-registered arm with a frozen task list, frozen harness, written pass bar, and single execution:
* **Baseline & Injection Floor (7 arms, 0/47 = 0/37 E4B + 0/10 31B):** Five E4B-era arms (`DIAG` 0/11, `H1v6` 0/10, `ARM2` 0/10, `CAL` 0/3, `CALv3` 0/3) plus two 31B-era injection arms (`G2` 0/5, `G2v2` 0/5). The same-checkpoint 31B floor is 0/10 on attractor tasks.
* **`W1` (n=10) & `W2A` (n=22):** A 32-task calibrated `WIN-SLICE` (single-file, `adds-test`, gold diff 271–1194 chars), split into the first 10 tasks (`W1`) and the held-out 22 tasks (`W2A`, pre-registered generalization gate >= 6/22).
* **`W2B` (n=10):** `W1`'s exact 10 tasks with `VERIFY-FIX` instructions appended — an A/B test with no other variable.
* **`G2v2` (n=5):** Three variable/scope misses (`requests_7309`, `requests_7328`, `rich_3296`) and two fixed-attractor controls (`requests_7205`, `requests_7315`) with the target function named in the issue text (counted in the 0/47 floor above; gate: >= 1/3 probe conversions).
* **`FAM` (n=12):** Twelve never-before-run tasks balanced 4/4/4 across repository and size strata.
* **`Best-of-3` Probe (n=5 × 3):** High-temperature (T=0.75) multi-rollout probe on the five `G2v2` failure tasks.

### 2.3 The Surgical-Green Filter and Zero-GPU Audit
A fix the agent cannot cleanly reproduce from its own trace is unsafe training data. We audited all 44 unique landed traces locally without spending GPU: for each trace we extracted every edited file, `submit` call, `CONFIDENCE` line, and `pytest` invocation. The programmatic harvest filter keeps a trace only if all six conditions hold: (1) exit code 0 with `resolved=true`, (2) patch length 200–1200 characters, (3) a `CONFIDENCE` line quoting a passing check, (4) `files_changed == 1`, (5) zero touches to test files, and (6) zero touches to docs or config files. Nine traces survive across the entire dataset (10 before removing a docs-only solve in `FAM`). The cuts themselves are empirical findings (Section 3.4).

### 2.4 The Held-Out Family Probe (`FAM`)
`WIN-slice` numbers are public-set upper bounds because task selection itself was calibrated on earlier readouts. `FAM` is the out-of-sample check — 12 tasks never used in `W1`, `W2A`, `W2B`, or `G2`, split 4/4/4 into `rich-small` (<600-char golds), `rich-large` (800–1100 chars), and `non-rich` (FastAPI plus one `requests` task) under the same frozen harness with no injection. Its pre-registration defines three mutually exclusive readings: `rich-small` high with `rich-large` collapse supports a size cliff (**G1**); both `rich` groups high with `non-rich` low supports a repository-family prior (**G3**); a flat low profile rejects both in favor of calibration selection.

---

## 3. Experiments

### 3.1 The Calibration Ladder: 0/47 to 14/32

| Stage | Resolved | Rate | Wilson 95% CI | Note |
| :--- | :---: | :---: | :---: | :--- |
| Baseline (7 arms) | 0/47 | 0.0% | [0.0%, 7.6%] | `DIAG` 0/11 + `H1v6` 0/10 + `ARM2` 0/10 + `CAL` 0/3 + `CALv3` 0/3 + `G2` 0/5 + `G2v2` 0/5 (0/37 E4B + 0/10 31B) |
| `MAIN10` (early arms) | 0/10 | 0.0% | [0.0%, 27.8%] | Pre-calibration slice on the E4B checkpoint (`gold` up to 19 KB), both early configurations |
| `W1` (first 10 `WIN`) | 5/10 | 50.0% | [23.7%, 76.3%] | `fastapi_14786`, `rich_2943`, `rich_3882`, `fastapi_13537`, `rich_3518` |
| `W2B` (`W1` + `VERIFY-FIX`) | 5/10 | 50.0% | [23.7%, 76.3%] | Identical 5/5 pass/fail set (10/10 agreement; 0/5 miss conversion) |
| `W2A` (rest of `WIN`) | 9/22 | 40.9% | [23.3%, 61.3%] | `7/10` on `gold <= 569B`; `2/12` on `gold >= 629B` (`14372`, `5077`); `7309` fails at `687B` |
| `WIN` combined | **14/32** | **43.8%** | **[28.2%, 60.7%]** | `W1 + W2A`, same frozen harness |
| `FAM` (held-out) | **3/12** | **25.0%** | **[8.9%, 53.2%]** | Flat `1/4 × 3` groups; `3/11` (27.3%) ex-harness-kill |

The ladder's jump from 0% to 43.8% (+43.8 pp; Fisher `p = 0.017`) comes from task selection plus procedural discipline, not weight changes. Within `WIN-SLICE`, tasks with `gold <= 600` chars solved at 12/20 (60.0%) versus 2/12 (16.7%) above 600 chars (OR = 7.50, two-sided Fisher exact `p = 0.028`), and `Textualize/rich` solved 8/9 (88.9%) versus 5/13 (38.5%) in `fastapi` and 1/10 (10.0%) in `psf/requests`. `FAM` tests whether those within-slice splits survive out-of-sample.

### 3.2 Two Controlled Nulls

**Verification does not convert (`W2B`: 0/5 misses converted, 5/10 identical set).**
Appending `VERIFY-FIX` instructions (re-run the failing behavior after the edit and confirm the fix on the reproducer) to the `W1` prompt changed binary outcomes on zero tasks: the exact same five resolve and the exact same five miss (`Cohen's kappa = 1.00`; `p = 0.5^10 = 9.8e-4` under independent rollouts; 95% Wilson upper bound on miss conversion is `43.4%` for `W2B` alone and `27.8%` across all `0/10` `W2B + G2v2` single-rollout probes). Patches moved substantially on three of the five misses (`rich_3296`: 539 to 1,737 chars; `requests_7328`: 884 to 3,127 chars, shifting its redirect guard from `len(hist)-1 >= max` to `len(hist) > max`; `fastapi_14258`: 2,308 to 1,085 chars), while `requests_7205` and `requests_7315` converged to the exact same flawed guards (`if _netrc and any(_netrc)` and deleting the `//` collapse block). Verification prompting changes trajectory behavior without changing results.

**Specification does not convert (`G2v2`: 0/5 converted).**
Appending the target function and file to the issue text (*"The change required is in `_parse_content_type_header` in `src/requests/utils.py`. Other functions with similar patterns are out of scope."*) converted zero of three probe tasks (`requests_7309`, `requests_7328`, `rich_3296`), with both attractor controls (`requests_7205`, `requests_7315`) also failing. The sharpest datum is `requests_7309`: the agent emitted the identical 687-character patch (16 calls) with the target named as it had in `W2A` (21 calls) without it. One scope caveat bounds this claim: on `7309`, the gold patch also touches the sister function `parse_dict_header`, which `G2v2` fenced out — so `G2v2` falsifies *"naming the primary function converts"*, not *"complete specification converts"*. A follow-up arm (`CALv3`) providing both functions and behavioral facts moved `7309` from the 687-char attractor to a 1,326-char two-hunk patch whose second hunk moves in the gold direction (dropping the bare `None`), and still failed. Specification-adjacent information moves the patch without converting the outcome.

### 3.3 The Flat Held-Out Probe (`FAM`: 3/12, Perfect 1/4 × 3)

| Group | Resolved | Members |
| :--- | :---: | :--- |
| `rich-small` | 1/4 | `rich_3470` resolves (2970B, 28c); `3471` (1026B, 28c), `3472` (0B, 28c), `3063` (1212B, 25c) fail |
| `rich-large` | 1/4 | `rich_3480` resolves (2242B, 21c); `3064` (2074B, 28c), `3535` (798B, 24c), `3506` (exit -1, 0B, 9c) fail |
| `non-rich` | 1/4 | `fastapi_14492` resolves (388B, 18c); `14360` (exit 2, 634B, 24c), `5624` (2094B, 28c), `7502` (2633B, 28c) fail |

A perfectly flat profile (`1/4` vs. `1/4` vs. `1/4`, exact `p = 1.00`, per-stratum 95% CI `[4.6%, 69.9%]`) falsifies both structural hypotheses at once: `rich-small` and `rich-large` tie (rejecting **G1**), and `rich` and `non-rich` tie (rejecting **G3**). Comparing `Textualize/rich` inside `WIN-SLICE` (8/9, 88.9%) against fresh `rich` tasks in `FAM` (2/8, 25.0%) yields a significant selection gap (two-sided Fisher exact `p = 0.015`), proving the apparent `rich` advantage was an artifact of `WIN` calibration rather than repository simplicity. Three trace findings sharpen this picture:
1. `rich_3506` died at 9 tool calls (482.2 s, `exit=-1`, `patch=0`) due to sandbox command latency (a 396 s `grep` and 250 s one-liners), making the model capability rate **3/11 (27.3%)**.
2. `rich_3472` spent all 28 calls reading (165.1 s), edited nothing, and submitted an empty patch — motivating an explicit **empty-submit ban** and per-command read time-boxing.
3. `fastapi_14492` resolved by editing a docs file (`docs_src/metadata/tutorial001_1_py39.py`): it counts in `3/12` as a harness-graded resolve, but is excluded from source-only training data.

### 3.4 What the Audit Found in 44 Traces
Nine of fourteen `WIN`-resolved traces survive as source-only surgical greens (10 across all 44 traces before cutting `fastapi_14492`'s docs edit); the audit's cuts are the section's real content:
* **Size and test/docs cuts:** Across `WIN-SLICE`, patches `<= 1,200` chars solved at 11/17 (64.7%) versus 3/15 (20.0%) above 1,200 chars (OR = 7.33, two-sided Fisher exact `p = 0.016`). Three `WIN` solves exceed 1,200 chars (`rich_3882` 1,475B, `fastapi_14372` 2,110B, `fastapi_5077` 3,803B), and two modified test files (`fastapi_9425` touched `test_annotation.py`; `rich_3067` made 13 edits to `tests/test_highlighter.py`). In `FAM`, `3470` (2,970B) and `3480` (2,242B, timed out before explicit `submit`) exceed 1,200B, and `14492` edited `docs_src/...`.
* **Semantic `exit-2` failures refute a syntax guard:** All four `exit-2` failures across 44 traces (`fastapi_14360`, `14303`, `14430`, `11355`) are semantic assertion or custom reproducer errors (`14360` asserted HTTP 200, `14303` printed `"Test failed"` after running `py_compile`, `14430` printed union-typing mismatches, and `11355` raised a `TypeError` inside its own `repro.py`) with zero syntax errors anywhere — refuting the hypothesis that a post-edit `py_compile` gate would rescue them.
* **Self-verification theater:** Only 3 of 44 traces ever invoked `pytest` (two errored on collection, one selected 48 tests and still failed), and 11 of 12 `FAM` traces verified exclusively through hand-written `repro.py` scripts. `CONFIDENCE` lines quoted reproducer stdout verbatim without exit codes or suite pass counts. Quarantining small-N (n=9) LoRA training follows directly: fine-tuning on 9 trajectories skewed toward one library (`rich`) risks out-of-distribution negative transfer while frozen 31B already achieves 27.3% on fresh tasks.

### 3.5 High-Temperature Sampling Probe (Best-of-3, T=0.75)
At T=0.2, the agent is near-deterministic, so every null above measures one basin per task. We evaluate five tasks across two strata — **Stratum A (Attractors):** `requests_7205`, `requests_7315`, `requests_7309`; **Stratum B (Multi-Modal):** `requests_7328`, `rich_3296` — for K=3 rollouts each at T=0.75 under the frozen harness, ranking non-empty diffs by existing repository tests with ties broken toward smaller patches, and aborting a stratum if candidates come back byte-identical. Pre-registered gates: Stratum A >= 1/3 converts means basins are escapable by sampling (0/3 means attractor basins are weight-locked); Stratum B >= 1/2 means execution-ranked Best-of-K harvests rollout variance.

> *Readout (kernel `144607478`, landed):* Stratum A resolved `0/3` (`7205`: F, `7315`: F, `7309`: F — the identical 687-char diff in all three reps); Stratum B resolved `0/2` (`7328`: F, `3296`: F — distinct diffs, 2,289–5,940 and 503–4,261 chars, all failing). Elevated temperature with execution-ranked Best-of-3 converts nothing (`0/15` rollouts resolved): attractor basins are weight-locked, and sampling variance does not convert the non-attractor misses either.

---

### 3.6 Census-Guards Probe (pre-registered, kernel 144655910)
Freezing the winner-shape harness and baking four census guards into system.md (edit-miss recovery, 60s read time-box, empty-submit ban, 1,500-char thrash-stop), we ran 6 valid-miss challenges (baseline 0/6 by construction) + 2 valid-resolve controls on the same 31B shape.

| Arm | Task | Resolved | Patch | Calls | Time |
| :--- | :--- | :---: | :--- | :---: | :--- |
| challenge | fastapi_14258 | F | 1513B | 21 | 491s |
| challenge | fastapi_14487 | F | 1148B | 28 | 181s |
| challenge | fastapi_15589 | F | 3059B | 28 | 334s |
| challenge | fastapi_5624 | T | 2880B | 28 | 303s |
| challenge | rich_3471 | F | 753B | 25 | 485s |
| challenge | rich_3506 | F | 575B | 10 | 485s |
| control | rich_3518 | T | 528B | 26 | 92s |
| control | rich_3470 | F | 2445B | 28 | 297s |

Result (challenges 1/6, controls 1/2): Weak signal, needs larger n -- fastapi_5624 converted but control rich_3470 regressed, and rich_3506 still died submit-less under the read time-box.

---

## 4. Related Work

Repair-agent literature clusters around three explanations for why small models fail: fault localization, verification distinguishability, and domain priors. Bellam (2026) ran a pre-registered scale study of structural tooling: nine open-weight models (`Qwen3` 0.6B–32B, `Gemma 4` E2B–12B) on 500 held-out SWE-bench tasks, finding code-graph and embedding tools raise function Acc@10 by +0.06–0.10 for small models but barely at 32B — with the embedding index, not the code graph, carrying the gain, and a placebo arm showing random tool output harms small models. Our traces consistently obey the prompt's `SPEC` and `grep` localization mandate on exact target symbols and still fail at identical rates, showing that procedural localization compliance without synthesis precision does not move outcomes.

Completion-versus-correctness and live-benchmark studies examine evaluation inflation and self-verification. Zhang et al. (2025) built `SWE-bench-Live` (1,319 tasks across 93 repositories, dedicated Docker images, automated curation) precisely because static-benchmark scores overstate live-task performance — the same direction as our calibrated-to-fresh drop (43.8% → 25.0%) — while our `CONFIDENCE` audit documents single-rollout self-verification theater in the field. Finally, post-training work on compact coding agents shows why harness design and data scale dominate small-N tuning: Kim et al. (2026) (`FrogNano`) post-train a 4B agent exclusively via RL on ~1,500 synthetic-task environments, generating tasks at the current checkpoint's learnability frontier — the strongest external confirmation that capability exists when the task distribution matches the model's frontier.

---

## 5. Limitations

1. **Single model family and harness lineage:** Methodology generalizes across agent architectures, while absolute solve rates are specific to `gemma-4-31b-it-qat-w4a16-ct` and the ADK sandbox.
2. **Stratum sample size (n=4 per `FAM` bucket):** Powered to detect large cliffs (>= 50 percentage-point gaps) and falsify dominant group predictors, but blind to modest (10–15 pp) covariate effects: at n=4 per cell, Fisher exact reaches p < 0.05 only at 4/4 vs 0/4 (p = 0.029); 1/4 vs 1/4 gives p = 1.00 (95% CI [4.6%, 69.9%]).
3. **Reproducer-based confidence:** Agent self-checks ran on custom `repro.py` scripts rather than hidden grading suites; "verified" in trace logs denotes passing the agent's own reproducer unless noted.
4. **Public repository exposure:** `Textualize/rich`, `fastapi/fastapi`, and `psf/requests` are public repositories potentially present in pretraining data, reinforcing why `WIN-SLICE` (`43.8%`) is an upper bound.
5. **Partial injection coverage:** `fastapi_14258` was evaluated in `W1` and `W2B` but not included in `G2v2`.
6. **Cross-checkpoint floor:** five of the seven baseline arms ran the E4B checkpoint (37 runs); the 31B-only floor is 0/10 on attractor tasks, so the same-checkpoint calibration step is 0/10 → 14/32. The E4B floor is prior-art context for the near-zero literature, not a controlled same-model comparison.

---

## 6. Conclusion

Calibration is real: freezing one disciplined procedure and selecting single-file, well-scoped tasks took a 31B quantized agent from 0/10 same-checkpoint (0/47 all-checkpoint) to 14/32 (43.8%) and 3/11 (27.3% ex-latency) on fresh held-out tasks. Verification prompting (0/5) and target-function naming (0/5) are insufficient, plus high-temperature sampling (0/5): three controlled probes moved trajectory behavior and patch lengths while converting nothing. No task group predicts success on held-out data: 1/4 × 3 on `FAM` falsifies size and repository-family predictors together. What remains is weights and selection — attractor basins that swallow information (the identical 687-char patch with and without the target function named) and a locality boundary around compact single-file fixes. When small-N post-training data cannot support out-of-distribution generalization, distribution-invariant harness invariants (empty-submit bans, per-command read time-boxing, and existing-suite candidate selection) are the highest-leverage engineering controls.

---

## References

1. **Gemma Team (2026).** *Gemma 4 Technical Report & Developer Agent Benchmark.* Google DeepMind / Kaggle Competition (`gemma-4-developer-agent`, `gemma-4-developer-agent-paper`).
2. **Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024).** *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* ICLR 2024.
3. **Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O. (2024).** *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering.* NeurIPS 2024.
4. **Zhang, L., He, S., Zhang, C., et al. (2025).** *SWE-bench Goes Live!* (`arXiv:2505.23419`).
5. **Kim, M., Shi, Z., Penaloza, E., Cui, C., Castanyer, R. C., Hashemzadeh, M., et al. (2026).** *FrogNano: Training a 4B Coding Agent via Online Task Synthesis.* (`arXiv:2609.07925`).
6. **Bellam, S. (2026).** *How Much Do Structural Tools Help Small Coding Agents? A Pre-Registered Scale Study of Code Localization.* (`arXiv:2610.33958`).

---

## Appendix A — Honesty Notes & Reproducible Kaggle Artifacts

* **Public-set upper bounds vs. held-out rates:** `WIN-slice` (`14/32 = 43.8%`) is a calibrated public-set upper bound; `FAM` (`3/12 = 25.0%`; `3/11 = 27.3%` excluding the 482.2 s sandbox-latency kill on `rich_3506`) is the uncalibrated held-out rate.
* **Documentation-only resolve (`fastapi_14492`):** Counted in `FAM`'s `3/12` graded score (`exit=0`) because the issue's target edit resided in `docs_src/metadata/tutorial001_1_py39.py`, and excluded from the 9-trace source-only surgical green pool.
* **Specification scope caveat (`requests_7309`):** `G2v2` named `_parse_content_type_header` while fencing out sister functions (`parse_dict_header`), bounding `G2v2`'s claim to primary-function naming; `CALv3` supplied both functions and moved the diff (`687B -> 1326B`) without converting the outcome.
* **Kaggle Kernel Execution Manifest:** `144573461` (`W2A`), `144573665` (`W2B`), `144590816` (`G2v2`), `144590833` (`FAM`), and `144607478` (`Best-of-3` temperature probe).
