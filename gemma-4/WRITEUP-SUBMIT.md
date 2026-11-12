# Calibration Is Real, Verification Is Not Enough
**Subtitle:** A frozen 31B agent goes from 0/10 to 14/32 (43.8%) through task calibration, while verification prompts, function hints, and T=0.75 Best-of-3 sampling convert 0/5 each

**Track:** Google — The Gemma 4 Developer Agent (Paper Track)  
**Author:** Andrale Misquitta
**Code & Artifacts:** https://github.com/doishy-doish/bot-paper (Apache 2.0) | **Kaggle Kernels:** `144573461`, `144573665`, `144590816`, `144590833`, `144607478`  

---

## Abstract

We ask a narrow question: once a coding agent's harness is frozen, what actually turns a failed bug fix into a working one? Starting from `0/47` resolved runs across seven baseline arms on two checkpoints (`0/37` on Gemma 4 E4B, `0/10` on our 31B submission checkpoint), a frozen 31-billion-parameter agent (`gemma-4-31b-it-qat-w4a16-ct`) solves **`14/32` tasks (`43.8%`**, 95% CI `[28.2%, 60.7%]`; `+43.8` pp, Fisher `p = 0.017`) on a calibrated single-file slice (`5/10` on the first ten, `9/22` on the held-out remainder). On twelve brand-new tasks (`FAM`), it then scores a flat **`3/12` (`25.0%`**; `3/11 = 27.3%` after removing one run killed by sandbox lag)—exactly **`1/4`** in each of three buckets (`rich-small`, `rich-large`, `non-rich`).

Three controlled interventions all score clean nulls (`0` conversions across `30` probe runs on `6` unique miss tasks):
1. **Verification prompting (`W2B`):** Telling the agent to verify its fix before submitting converts `0/5` misses (`5/10` overall, with the exact same five tasks passing and failing; `p = 9.8e-4` under independent rollouts).
2. **Target-function naming (`G2v2`):** Naming the exact function to edit in the issue text converts `0/5` tasks, reproducing the identical `687`-character wrong patch on `requests_7309` with and without the hint.
3. **High-temperature sampling (`Best-of-3` at `T=0.75`):** Running three independent rollouts per task at elevated temperature converts `0/5` tasks (`0/15` runs) on the same five `G2v2` tasks, with `requests_7309` emitting the exact same `687`-character SHA-256 diff in all three rollouts.

A zero-GPU audit of all `44` unique traces cuts our clean training pool down to **`9` source-only surgical fixes**, rules out a syntax-guard fix (all four `exit-2` crashes in `fastapi` are logic/assertion failures, not syntax errors), and shows that only `3/44` traces ever ran `pytest`. We release the audit as `confidence-audit`, a zero-GPU tool that reproduces every verdict below from the banked traces in one command. Calibration is real and large; verification prompts, function hints, and high-temperature sampling do not convert misses; no task group predicts success on unseen tasks; and residual failures come down to weight-locked wrong-patch habits and edit size. The contribution is the falsification chain, not a headline score.

---

## 1. Introduction

Small open-weight models are the practical frontier for automated bug fixing: if a `31B` quantized agent can patch real repositories on a single multi-GPU node, any engineering team can run it privately. Papers usually report headline pass rates on giant models, but a working engineer faces a narrower choice: **given one frozen harness and a fixed GPU budget, which change actually converts a failed task into a pass?** Prompt tweaks are cheap to invent and expensive to trust, because every run burns GPU hours and it is easy to tell a convincing story after seeing the numbers.

We ran this lane the boring way. One agent setup, frozen early and never re-tuned. Every run was pre-registered with a locked task list, a pass/fail bar, and a ban on silent retries. Seven baseline arms failed first (`0/47` combined: `0/37` on E4B, `0/10` on the `31B` checkpoint), giving us an honest floor instead of a cherry-picked win. From that floor, we tested four plausible levers—**verification prompting**, **naming the target function**, **repository/size grouping**, and **high-temperature multi-rollout sampling**—plus a zero-GPU trace audit to decide whether any of our wins were clean enough for fine-tuning.

The result is a split, not a smooth curve. **Calibration**—picking well-scoped single-file bugs and freezing a disciplined workflow—lifted the solve rate from `0%` to `43.8%` across `32` tasks. **Interventions** moved nothing: `0/5` on verification prompts, `0/5` on target-function hints, and `0/5` (`0/15` runs) on `Best-of-3` sampling at `T=0.75`. A fresh `12`-task holdout then came back completely flat (`1/4` in every bucket), knocking out both repository-family and patch-size rules at once. What remains are two concrete failure patterns: a few bugs where the model locks onto the exact same wrong edit even at `T=0.75`, and a locality boundary where compact single-file patches succeed while multi-file or sprawling edits fail.

---

## 2. Methods

### 2.1 The Frozen Agent
Every run uses the same setup: a single-agent loop on the `31B` instruction-tuned quantized model (`gemma-4-31b-it-qat-w4a16-ct`) offline on `4 × L4` GPUs via `vLLM`. The agent has nine active tools inside fixed limits:
* **Budgets:** `28` tool calls, `80` turns, `480 s` task limit, `240 s` per-command timeout, `300 s` grading timeout, compaction every `15` steps.
* **Workflow:** Write a three-line `SPEC`, read targets before editing, best-guess edit by call `12`, test it, check `git diff`, `submit`.
* **Locked Sampling:** `temperature=0.2`, `top_p=0.95`, fixed `max_output_tokens`, `thinking_budget=4096`. At `T=0.2` the agent is nearly deterministic, so repeats test bad habits, not noise. A pre-flight script validates hashes and schemas first.

### 2.2 Pre-Registered Experiments
Every arm was defined on paper with a locked task list and pass/fail rule before running:
* **Baseline Floor (`7` arms, `0/47` total = `0/37` E4B + `0/10` 31B):** Five early E4B-checkpoint runs (`DIAG` `0/11`, `H1v6` `0/10`, `ARM2` `0/10`, `CAL` `0/3`, `CALv3` `0/3`) plus two `31B` hint runs (`G2` `0/5`, `G2v2` `0/5`). The same-checkpoint `31B` floor is `0/10`.
* **Calibrated Slice (`W1`, `n=10`, and `W2A`, `n=22`):** `32` single-file bugs (`adds-test`, gold `271–1194` chars): first `10` (`W1`) plus held-out `22` (`W2A`, bar `>= 6/22`).
* **Verification-Prompt Test (`W2B`, `n=10`):** Re-running `W1`'s exact `10` tasks with explicit `VERIFY-FIX` instructions added to the prompt—nothing else changed.
* **Target-Function Hint Test (`G2v2`, `n=5`, counted in the `0/47` floor above):** Adding the exact target function name to the issue text on three misses (`requests_7309`, `requests_7328`, `rich_3296`) plus two stubborn controls (`requests_7205`, `requests_7315`), requiring `>= 1/3` probe conversions.
* **Fresh Stratified Holdout (`FAM`, `n=12`):** Twelve never-before-seen tasks split `4/4/4` into `rich-small` (gold `<600` chars), `rich-large` (`800–1100` chars), and `non-rich` (`fastapi` and `requests`).
* **High-Temperature Sampling Probe (`Best-of-3`, `n=5 × 3 = 15` runs):** Three rollouts per task at `T=0.75` on the five `G2v2` failures (no hints), testing high-temperature escape from wrong-patch habits.

### 2.3 Zero-GPU Trace Audit & Clean-Data Filter
We inspected all `44` landed traces locally (`0` GPU hours), checking every modified file, `submit` call, `CONFIDENCE` line, and `pytest` command. A trace counts as a **surgical green** only if all six rules pass: (1) `resolved=true` (`exit=0`), (2) patch length `200–1200` chars, (3) `CONFIDENCE` line citing a passing check, (4) `files_changed == 1`, (5) zero edits to test files, and (6) zero edits to docs or config files. Only **`9` traces** survive across the entire dataset (`10` before cutting one docs-only fix in `FAM`).

---

## 3. Experiments

### 3.1 The Calibration Ladder: `0/47` to `14/32`

| Arm / Stage | What Was Tested | Solved | Rate | Wilson 95% CI | Notes |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline (`7` arms)** | Uncalibrated & hint arms | `0/47` | `0.0%` | `[0.0%, 7.6%]` | `0/37` E4B (`DIAG`, `H1v6`, `ARM2`, `CAL`, `CALv3`) + `0/10` 31B (`G2`, `G2v2`) |
| **`MAIN10`** | Pre-calibration E4B slice | `0/10` | `0.0%` | `[0.0%, 27.8%]` | Subset of `H1v6`/`ARM2` with multi-file diffs up to `19 KB` (not extra runs) |
| **`W1`** | First `10` calibrated tasks | `5/10` | `50.0%` | `[23.7%, 76.3%]` | `fastapi_14786`, `rich_2943`, `rich_3882`, `fastapi_13537`, `rich_3518` |
| **`W2B`** | `W1` + `VERIFY-FIX` prompt | `5/10` | `50.0%` | `[23.7%, 76.3%]` | Exact same `5` wins and `5` losses (`10/10` match; `0/5` miss conversion) |
| **`W2A`** | Held-out `22` calibrated tasks | `9/22` | `40.9%` | `[23.3%, 61.3%]` | `7/10` on gold `<= 569B`; `2/12` on gold `>= 629B` (`14372`, `5077`) |
| **`WIN` Combined** | `W1 + W2A` calibrated slice | **`14/32`** | **`43.8%`** | **`[28.2%, 60.7%]`** | `+43.8` pp over the `0/10` 31B floor (two-sided Fisher exact `p = 0.017`) |
| **`FAM` (Holdout)** | `12` brand-new tasks (`4 × 3`) | **`3/12`** | **`25.0%`** | **`[8.9%, 53.2%]`** | Flat `1/4 × 3` (`3/11 = 27.3%` excluding `rich_3506` sandbox lag kill) |
| **`Best-of-3` (`T=0.75`)** | `5` misses × `3` high-temp rollouts | **`0/15`** | **`0.0%`** | **`[0.0%, 20.4%]`** | `0/5` tasks converted (`0/3` attractors, `0/2` variable-patch tasks) |
| **`E4B` replication** | Same `32` calibrated tasks, E4B checkpoint | `6/32` | `18.8%` | `[8.9%, 35.3%]` | `14786`, `6644`, `3518`, `3006`, `3067`, `3454` (kernel `144641971`) |

Calibration lifted the model from `0/10` to `14/32` (`43.8%`, `p = 0.017`) with zero weight updates. Two patterns looked strong:
* **Patch size:** Tasks with gold diffs `<= 600` chars solved at `12/20` (`60.0%`) vs. `2/12` (`16.7%`) above `600` chars (`OR = 7.50`, two-sided Fisher exact `p = 0.028`).
* **Repository:** `Textualize/rich` solved `8/9` (`88.9%`) vs. `5/13` (`38.5%`) in `fastapi` and `1/10` (`10.0%`) in `psf/requests` (exploratory; Bonferroni-adjusted `p = 0.084` across three repositories).

Our fresh holdout (`FAM`) tested whether either pattern actually holds on new bugs.

### 3.2 Controlled Prompt & Hint Nulls

**1. Verification prompting converts nothing (`W2B`: `0/5` misses converted, `5/10` identical set).**  
Adding explicit `VERIFY-FIX` instructions to `W1` changed the pass/fail result on **zero tasks**: the same five passed and the same five failed (`Cohen's kappa = 1.00`; `0.5^10 = 9.8e-4` under independent rollouts; 95% Wilson upper bound `43.4%` in `W2B`, `27.8%` across all `10` single-rollout probes). Patches still shifted on three misses—`rich_3296` `539`→`1,737` chars, `requests_7328` `884`→`3,127` chars (`len(hist)-1 >= max` to `len(hist) > max`), `fastapi_14258` `2,308`→`1,085` chars—while `requests_7205` and `requests_7315` repeated the same wrong edits (`if _netrc and any(_netrc)`, deleting the `//` collapse). Verification prompts change patch size, not outcomes.

**2. Naming the target function converts nothing (`G2v2`: `0/5` converted).**  
Appending the target function and file path to the issue text (*"The change required is in `_parse_content_type_header` in `src/requests/utils.py`. Other functions with similar patterns are out of scope."*) converted `0/3` probes plus `0/2` controls. On `requests_7309`, the agent wrote the **same `687`-character wrong patch** (`16` calls) with the hint as without it (`21` calls). Caveat: the true fix also edits sister function `parse_dict_header`, which our hint ruled out—so `G2v2` proves *naming the main function is not enough*, not that *full specification is useless*. In `CALv3` (an earlier multi-function specification probe), naming both functions grew `7309`'s diff `687`→`1,326` chars with the second hunk moving the right way—and still failed. Extra specification shifts patches without converting results.

### 3.3 The Flat Fresh Holdout (`FAM`: `3/12`, Exact `1/4 × 3`)

| Bucket (`n=4` each) | Solved | Winning Task (`patch`, `calls`) | Failed Tasks (`exit`, `patch`, `calls`) |
| :--- | :---: | :--- | :--- |
| **`rich-small`** | `1/4` (`25.0%`) | `rich_3470` (`2970B`, `28c`) | `rich_3471` (`1`, `1026B`, `28c`), `rich_3472` (`1`, `0B`, `28c`), `rich_3063` (`1`, `1212B`, `25c`) |
| **`rich-large`** | `1/4` (`25.0%`) | `rich_3480` (`2242B`, `21c`) | `rich_3064` (`1`, `2074B`, `28c`), `rich_3535` (`1`, `798B`, `24c`), `rich_3506` (`-1`, `0B`, `9c`) |
| **`non-rich`** | `1/4` (`25.0%`) | `fastapi_14492` (`388B`, `18c`) | `fastapi_14360` (`2`, `634B`, `24c`), `fastapi_5624` (`1`, `2094B`, `28c`), `requests_7502` (`1`, `2633B`, `28c`) |

A flat `1/4 × 3` (`exact p = 1.00`, 95% CI `[4.6%, 69.9%]` per bucket) rejects both group predictors: `rich-small` ties `rich-large` (no size cliff), `rich` ties `non-rich` (no repo advantage). `Textualize/rich` in `WIN-SLICE` (`8/9 = 88.9%`) vs fresh `rich` in `FAM` (`2/8 = 25.0%`) gaps sharply (Fisher `p = 0.015`): the `88.9%` came from `WIN` selection, not an easy codebase. Three trace details:
1. **`rich_3506` died from sandbox lag (`3/11 = 27.3%` ex-timeout):** `482.2 s`, `exit=-1`, `0` edits after `9` calls — one `grep` took `396 s`.
2. **`rich_3472` burned `28` read-only calls (`165.1 s`) and submitted an empty diff:** A simple **empty-submit ban** and per-command read timeout prevents both zero-diff losses without touching model weights.
3. **`fastapi_14492` edited a docs file (`docs_src/metadata/tutorial001_1_py39.py`):** Valid for the `3/12` benchmark score, but excluded from our source-only training pool.

### 3.4 What the 44-Trace Audit Found
Only **`9` of `14`** `WIN` solves are clean source-only surgical fixes (`10` of `44` before cutting `fastapi_14492`'s docs edit). The audit ships as `tools/confidence-audit.py` in the public repo, with all 44 traces and banked tables:
* **Compact diffs win (`<= 1,200` chars):** Across `WIN-SLICE`, patches `<= 1,200` chars solved at **`11/17` (`64.7%`)** vs. **`3/15` (`20.0%`)** above `1,200` chars (`OR = 7.33`, two-sided Fisher exact `p = 0.016`). Three `WIN` wins exceeded `1,200` chars (`rich_3882` `1,475B`, `fastapi_14372` `2,110B`, `fastapi_5077` `3,803B`), and two edited test files (`fastapi_9425` touched `test_annotation.py`; `rich_3067` made `13` edits to `tests/test_highlighter.py`).
* **All four `exit-2` crashes are logic errors, not syntax errors:** Every `exit-2` across all `44` traces (`fastapi_14360`, `14303`, `14430`, `11355`) is semantic, so a post-edit `py_compile` guard would save **`0/4`**: `14360` failed an HTTP `200` assertion, `14303` printed `"Test failed"` after already running `py_compile`, `14430` printed union-typing mismatches, and `11355` hit a `TypeError` in its own reproducer script.
* **Self-verification theater:** Only `3/44` traces ever ran `pytest` (`2` crashed during test collection, `1` ran `48` tests and failed), while `11/12` `FAM` traces relied entirely on agent-written `repro.py` scripts, each ending with a confident `CONFIDENCE` line quoting its own output. Fine-tuning on `9` `rich`-skewed traces risks overfitting one library while frozen `31B` scores `27.3%` on fresh tasks.

### 3.5 High-Temperature Sampling Probe (`Best-of-3`, `T=0.75`)
Because baseline runs at `T=0.2` are nearly deterministic, a single miss could just be a bad roll of the dice. We tested the five `G2v2` failure tasks across two groups—**Attractors (`requests_7205`, `requests_7315`, `requests_7309`)** and **Multi-Modal Misses (`requests_7328`, `rich_3296`)**—running **`K=3` independent rollouts at `T=0.75`** (`15` runs total, no issue hints) and hashing every final diff (`SHA-256`).

| Group | Task | Rollout 1 (`patch`, `calls`, `hash`) | Rollout 2 (`patch`, `calls`, `hash`) | Rollout 3 (`patch`, `calls`, `hash`) | Converted |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Attractor** | `requests_7205` | `3498B`, `28c` (`d3fb4e5f6136`) | `502B`, `23c` (`c0a47bd2c1b4`) | `502B`, `25c` (`c0a47bd2c1b4`) | `0/3` |
| **Attractor** | `requests_7315` | `521B`, `21c` (`8a947f41a86f`) | `521B`, `14c` (`8a947f41a86f`) | `1352B`, `28c` (`64af87569ac6`) | `0/3` |
| **Attractor** | `requests_7309` | `687B`, `13c` (`600a2da88433`) | `687B`, `12c` (`600a2da88433`) | `687B`, `18c` (`600a2da88433`) | `0/3` |
| **Multi-Modal** | `requests_7328` | `4419B`, `28c` (`a7a17b132a4a`) | `5940B`, `28c` (`a7f82bf6a9bc`) | `2289B`, `28c` (`9f146015d241`) | `0/3` |
| **Multi-Modal** | `rich_3296` | `539B`, `26c` (`51f248c709d3`) | `503B`, `28c` (`69c9aaf817b0`) | `4261B`, `28c` (`6e290e963a20`) | `0/3` |

**Result (`0/15` runs, `0/5` tasks converted; 95% Wilson upper bound `20.4%`):**
* **Attractors (`0/3`):** At `T=0.75`, `requests_7309` emitted the **byte-identical `687B` diff (hash `600a2da88433`) in `3/3` rollouts**. `7205` and `7315` repeated identical wrong hashes in `2/3` rollouts each. The attractors are weight-locked, not sampling noise.
* **Multi-Modal Misses (`0/2`):** Six distinct diffs across six rollouts (`0/6` solved); all three `requests_7328` runs bloated to `2.3–5.9 KB` at the `28`-call cap. With `W2B` (`0/5`), `G2v2` (`0/5`) and `Best-of-3` (`0/5`), zero conversions across `30` runs on `6` unique miss tasks rules out prompts, hints, and sampling on frozen `31B` weights.

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

Result (challenges 1/6, controls 1/2): Weak signal with a control regression -- fastapi_5624 converted, but baseline-resolved rich_3470 (2,970-char solve) failed here (2,445 chars, exit 1), and rich_3506 still died submit-less. Net conversion is zero-to-negative: the guards change behavior (no empty submits, no 5KB bloat) while costs offset gains, consistent with the thesis that residual failures are weight-locked, not harness-shaped. Candidate mechanism for the regression is the 1,500-char thrash-stop or 60s read time-box cutting a task that needed a large diff, though the trace alone does not isolate which guard bit; needs larger n to size the trade-off.

---

## 4. Related Work

Small-model failures trace to three causes: finding code (**localization**), judging fixes (**verification**), and training breadth (**scale**). Bellam (2026) found code-graph and embedding search lift function `Acc@10` by `+0.06–0.10` on tiny models across `500` SWE-bench tasks (`0.6B–32B`, incl. Gemma 4 `E2B–12B`), with ~zero gain at `32B`—and random tool output hurts. Our traces confirm the flip side: the agent finds the right symbols with `grep` yet fails at the same rate, because locating a line is not writing the logic.

Holdout pessimism and our no-fine-tuning call match the literature. Zhang et al. (2025) built *SWE-bench-Live* (`1,319` tasks, `93` repos) because static scores (`43.2%`) overstate live pass rates (`19.3%`)—mirroring our `43.8%`→`25.0%` calibrated-to-fresh drop. Kim et al. (2026) (*FrogNano*) show weight updates work with large difficulty-matched task pools (`~1,500` synthetic envs for a `4B` agent)—not `9` single-library traces.

---

## 5. Limitations

1. **One model and sandbox:** Our methodology transfers to any agent setup, while exact pass rates reflect `gemma-4-31b-it-qat-w4a16-ct` inside this ADK shell environment.
2. **Small bucket size in `FAM` (`n=4` per group):** At `n=4` per bucket, Fisher's exact test only reaches `p < 0.05` for huge gaps (`4/4` vs. `0/4`, `p = 0.029`; `1/4` vs. `1/4` gives `p = 1.00`, 95% CI `[4.6%, 69.9%]`). `FAM` catches big cliffs, not small `10–15%` differences.
3. **Self-written reproducers:** Most traces verified edits with custom `repro.py` scripts rather than the hidden grading test suite.
4. **Public repositories:** `Textualize/rich`, `fastapi/fastapi`, and `psf/requests` are public codebases that may appear in pretraining data, making `WIN-SLICE` (`43.8%`) an upper bound.
5. **Cross-checkpoint baseline:** Five of the seven baseline arms ran on Gemma 4 E4B (`37` runs); the `31B`-only floor is `0/10` on attractor tasks (`0/10 -> 14/32`, `p = 0.017`).
6. **Cross-checkpoint replication (E4B `6/32 = 18.8%`, 95% CI `[8.9%, 35.3%]`, kernel `144641971`):** calibration transfers across checkpoints at a rate-scaled level.

---

## 6. Conclusion

Calibration works: freezing a clean workflow and choosing well-scoped single-file tasks moved a `31B` quantized agent from `0/10` same-checkpoint (`0/47` all-checkpoint) to `14/32` (`43.8%`, `p = 0.017`) and `3/11` (`27.3%` ex-latency) on fresh tasks. Verification prompting (`0/5`), target-function hints (`0/5`), and `T=0.75` `Best-of-3` sampling (`0/5`, `0/15` runs) all failed to convert a single miss (95% upper bound `20.4%`). No repository or size bucket predicted success on unseen tasks (`1/4 × 3` on `FAM`). What remains are weight-locked wrong-patch habits (`requests_7309` emitting the exact same `687B` diff across all three `T=0.75` rollouts) and a locality boundary around compact single-file edits (`<= 1,200` chars: `64.7%` vs. `20.0%`, `p = 0.016`). With only `9` clean traces, simple harness guards (empty-submit bans, read time-boxes, stopping `>1,500B` thrash runs) are the most reliable improvements to ship; a pre-registered guards probe on the same shape converted 1/6 misses (controls 1/2, weak signal, S3.6).

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

* **Public-set upper bounds vs. held-out rates:** `WIN-slice` (`14/32 = 43.8%`) is a calibrated public-set upper bound; `FAM` (`3/12 = 25.0%`; `3/11 = 27.3%` excluding the `482.2 s` sandbox-latency kill on `rich_3506`) is the uncalibrated held-out rate.
* **Docs-only solve (`fastapi_14492`):** Counted in `FAM`'s `3/12` graded score (`exit=0`) because the target edit lived in `docs_src/metadata/tutorial001_1_py39.py`, and excluded from the `9`-trace source-only training pool.
* **Hint scope caveat (`requests_7309`):** `G2v2` named `_parse_content_type_header` while ruling out sister functions (`parse_dict_header`), bounding `G2v2` to main-function naming; `CALv3` named both functions and shifted the diff (`687B -> 1326B`) without passing tests.
* **Kaggle Kernel Execution Manifest:** `144573461` (`W2A`), `144573665` (`W2B`), `144590816` (`G2v2`), `144590833` (`FAM`), and `144607478` (`Best-of-3` temperature probe).
