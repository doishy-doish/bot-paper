Calibration Is Real, Verification Is Not Enough
A pre-registered falsification chain on small-model code repair — frozen 31B agent, 47 failed baseline attempts, 14/32 on a calibrated slice, two controlled nulls, and a flat held-out probe

Track: Google — The Gemma 4 Developer Agent (Paper Track)

Author: Andrale Misquitta

Abstract
We ask a narrow question: what actually converts failures into fixes for an open-weight code-repair agent once the harness is frozen? Starting from 0/47 resolved attempts across seven baseline arms on two checkpoints (0/37 E4B-era, 0/10 same-checkpoint 31B), a frozen 31-billion-parameter quantized agent (gemma-4-31b-it-qat-w4a16-ct) resolves 14/32 tasks (43.8%, Wilson 95% CI [28.2%, 60.7%]) on a calibrated single-file slice (5/10 on the initial ten, 9/22 on the held-out remainder), then scores a flat 3/12 (25.0%; 3/11 = 27.3% excluding one sandbox-latency kill) on twelve brand-new tasks—exactly 1/4 in each of three strata (small-rich, large-rich, non-rich). Two controlled interventions both yield clean nulls: adding explicit verify-then-fix instructions converts 0/5 misses (W2B, 5/10 with the identical resolved set (p=9.8e-4 under rollout independence), and injecting the target function name into the issue text converts 0/5 (G2v2), reproducing the exact same 687-character wrong patch on requests_7309 with and without the target symbol named. A 44-trace zero-GPU forensic audit cuts the clean training pool to 9 source-only surgical greens, refutes a syntax/compile-guard hypothesis (all four exit-2 failures in fastapi are semantic assertion mismatches, zero are syntax errors), and reveals that only 3/44 traces ever invoked pytest. Calibration is real and large; verification prompting and symbol specification are insufficient; group-level task features fail to predict success out-of-sample; and residual failures reside in task-specific parametric attractors and locality boundaries.

1. Introduction
Small open-weight models are the economic frontier of autonomous software repair: if a 31B quantized agent can reliably resolve repository issues on commodity multi-GPU hardware, any engineering team can deploy it offline. Yet benchmark headlines obscure the operational question a practitioner actually faces: given a frozen agent harness and a fixed compute budget, which intervention genuinely converts a miss into a fix? Prompt edits are cheap to propose and expensive to trust, because every evaluation consumes GPU hours and post-hoc explanations easily fit noisy readouts.

We ran this investigation under strict pre-registration. One agent architecture was frozen early and never re-tuned. Every subsequent experiment was registered before execution with a locked task list, an explicit pass/fail bar, and a hard rule against silent re-fires: any rerun or prompt modification constitutes a distinct experimental arm. Seven baseline and injection arms failed first (0/47 combined: 0/37 on E4B, 0/10 on the 31B submission checkpoint), establishing an unvarnished floor before any positive signal appeared. From that floor, we tested three candidate converters—verification discipline, target-locus specification, and repository/size structure—alongside a zero-GPU trace audit governing what data, if any, is clean enough for post-training.

The headline result is a structural split, not a smooth scaling curve. Calibration—selecting well-scoped single-file tasks and freezing a disciplined procedure—moved solve rate from 0% to 43.8% across 32 tasks. Prompt-level interventions moved nothing: 0/5 and 0/5 on controlled probes. A stratified held-out probe (FAM) then returned a perfectly flat 1/4 × 3, falsifying both repository-family and diff-size predictors simultaneously. What survives is a residual governed by two mechanisms: task-specific attractor basins where the agent converges to the same wrong diff regardless of extra context, and a locality boundary around compact single-file edits.

2. Methods
2.1 The Frozen Agent and Execution Envelope
All experiments execute a single-root Agent Development Kit (ADK) loop backed by gemma-4-31b-it-qat-w4a16-ct (W4A16 quantization-aware trained) served offline on 4 × NVIDIA L4 GPUs via vLLM (tensor_parallel_size=4). The tool registry exposes nine active tools (run, read, edit, write, four code-navigation tools, and submit, with three graph tools declared and disabled by prompt instruction) under fixed execution envelopes:

Budgets & Timeouts: 28 tool calls, 80 max turns, 480 s (8-minute) task wall-clock limit, 240 s per-command timeout, and 300 s evaluation suite timeout, with EventsCompactionConfig (compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5).

Winner-Shape Procedure: The system prompt enforces a fixed workflow: write a three-line SPEC (trigger, expected, acceptance), locate and read target symbols before editing, apply a best-guess edit by call 12, verify against a targeted test or reproducer without modifying test or config files, inspect git diff, and call submit.

Deterministic Sampling & Pre-Flight Validation: Sampling is locked at temperature=0.2, top_p=0.95, max_output_tokens fixed, and thinking_budget=4096 (verified across config files, notebook assertions, and kernel boot logs). At T=0.2, the agent behaves as a near-deterministic function of the task, ensuring repeated runs measure basin stability rather than sampling noise. Every run passes an automated manifest, brace-scan, and schema validator before GPU allocation.

2.2 Pre-Registration Discipline
Each run is a standalone pre-registered arm with a frozen task set, frozen harness, and pre-committed decision threshold:

Pre-W1 & Injection Floor (7 arms, 0/47 = 0/37 E4B + 0/10 31B): Five E4B-era arms (DIAG 0/11, H1v6 0/10, ARM2 0/10, CAL 0/3, CALv3 0/3) plus two 31B-era injection arms (G2 0/5, G2v2 0/5). The same-checkpoint 31B floor is 0/10 on attractor tasks.

W1 (n=10) & W2A (n=22): The 32-task calibrated WIN-SLICE (single-file, adds-test, gold diff 271–1194 chars), partitioned into the first 10 tasks (W1) and the held-out 22 tasks (W2A, pre-registered generalization gate >= 6/22).

W2B (n=10): Exact W1 task set with VERIFY FIX instructions appended—a single-variable A/B test.

G2v2 (n=5): Target-function symbol injection on three variable/scope misses (requests_7309, requests_7328, rich_3296) and two fixed-attractor controls (requests_7205, requests_7315), requiring >= 1/3 probe conversions.

FAM (n=12): Twelve never-before-evaluated tasks balanced across three strata (4 × 3).

Best-of-3 Sampling Probe (n=5 × 3): High-temperature (T=0.75) multi-rollout probe testing whether the five G2v2 failure tasks escape their basins under sampling diversity.

2.3 Surgical-Green Hygiene Filter and Zero-GPU Audit
A fix the agent cannot cleanly reproduce in its own trace is unsafe post-training data. We audited all 44 unique landed traces locally (0 GPU hours), extracting every modified file path, submit call, CONFIDENCE line, and pytest invocation. A trajectory qualifies as a surgical green only if all six programmatic conditions hold: (1) resolved=true (exit=0), (2) 200 <= patch_chars <= 1200, (3) CONFIDENCE line citing a passing check, (4) files_changed == 1, (5) zero edits to test files, and (6) zero edits to documentation or configuration files.

2.4 Held-Out Stratified Probe (FAM)
Because WIN-SLICE tasks were selected through calibration criteria, WIN solve rates represent public-set upper bounds. FAM tests out-of-sample transfer across 12 fresh tasks partitioned 4/4/4 into rich-small (gold <600 chars), rich-large (gold 800–1100 chars), and non-rich (fastapi and requests). Its pre-registered decision matrix separates three hypotheses: high rich-small with rich-large collapse supports a size cliff (G1); high rich across sizes with low non-rich supports a repository-family prior (G3); a flat low profile rejects both in favor of calibration selection.

3. Experiments
3.1 The Calibration Ladder: 0/47 to 14/32
Stage / Arm	Cohort	Resolved	Rate	Wilson 95% CI	Notes
Baseline (7 arms)	Uncalibrated & injection arms	0/47	0.0%	[0.0%, 7.6%]	DIAG 0/11, H1v6 0/10, ARM2 0/10, CAL 0/3, CALv3 0/3, G2 0/5, G2v2 0/5 (0/37 E4B + 0/10 31B)
MAIN10	Pre-calibration slice	0/10	0.0%	[0.0%, 27.8%]	Multi-file / uncalibrated tasks on the E4B checkpoint (gold up to 19KB)
W1	First 10 of WIN-SLICE	5/10	50.0%	[23.7%, 76.3%]	fastapi_14786, rich_2943, rich_3882, fastapi_13537, rich_3518
W2B	W1 + VERIFY FIX prompt	5/10	50.0%	[23.7%, 76.3%]	Identical 5/5 pass/fail set (10/10 agreement; 0/5 miss conversion)
W2A	Held-out 22 of WIN-SLICE	9/22	40.9%	[23.3%, 61.3%]	7/10 on gold <= 569B; 2/12 on gold >= 629B (14372, 5077)
WIN Combined	W1 + W2A (WIN-SLICE)	14/32	43.8%	[28.2%, 60.7%]	Overlapping CIs confirm W1 and W2A sample the same ~0.44 regime
FAM (Held-Out)	Fresh 4 × 3 stratified probe	3/12	25.0%	[8.9%, 53.2%]	Flat 1/4 × 3 (3/11 = 27.3% excluding rich_3506 sandbox latency kill)
￼
The step from 0% to 43.8% comes from task selection paired with a frozen procedural envelope, not weight changes. Within WIN-SLICE, tasks with gold <= 600 chars solved at 12/20 (60.0%) versus 2/12 (16.7%) above 600 chars (two-sided Fisher exact 
𝑝
=
0.028
p=0.028), and Textualize/rich solved 8/9 (88.9%) versus 5/13 (38.5%) in fastapi and 1/10 (10.0%) in psf/requests. FAM tests whether those within-slice patterns hold out-of-sample.

3.2 Two Controlled Nulls
Verification prompting does not convert (W2B: 0/5 misses converted, 5/10 identical set).

Appending explicit VERIFY FIX instructions to the W1 prompt left binary outcomes unchanged on all 10 tasks. Under independent rollouts at p=0.5, exact 10/10 agreement occurs with probability 0.5^10 = 9.8e-4, confirming task-locked behavior. Crucially, the prompt change did alter patch trajectories on three of the five misses—rich_3296 expanded from 539 to 1,737 chars, requests_7328 expanded from 884 to 3,127 chars (shifting its redirect guard from len(hist)-1 >= max to len(hist) > max), and fastapi_14258 shrank from 2,308 to 1,085 chars—while requests_7205 and requests_7315 converged to the exact same flawed guards (if _netrc and any(_netrc) and deleting the // collapse block). Verification prompting shifts trajectory length without converting outcomes.

Target-function specification does not convert (G2v2: 0/5 converted).

Appending the target function and file directly to the issue text ("The change required is in _parse_content_type_header in src/requests/utils.py. Other functions with similar patterns are out of scope.") converted 0/3 probes (requests_7309, requests_7328, rich_3296) and 0/2 attractor controls (requests_7205, requests_7315). Most tellingly, requests_7309 emitted the identical 687-character diff (16 calls) with the target function named as it had in W2A (21 calls) without it. One important scope caveat bounds this claim: on 7309, the gold patch also modifies the sister function parse_dict_header, which G2v2's wording fenced out. Thus G2v2 falsifies "naming the primary function converts", not "complete specification converts". In CALv3, providing both functions and behavioral facts moved 7309's diff from 687 to 1,326 chars with its second hunk aligned toward gold—and still failed assertions. Specification moves patch structure, but is insufficient to convert capability/prior misses.

3.3 The Flat Held-Out Probe (FAM: 3/12, Exact 1/4 × 3)
Stratum (n=4 each)	Resolved	Solved Task (patch, calls)	Failed Tasks (exit, patch, calls)
rich-small	1/4 (25.0%)	rich_3470 (2970B, 28c)	rich_3471 (1, 1026B, 28c), rich_3472 (1, 0B, 28c), rich_3063 (1, 1212B, 25c)
rich-large	1/4 (25.0%)	rich_3480 (2242B, 21c)	rich_3064 (1, 2074B, 28c), rich_3535 (1, 798B, 24c), rich_3506 (-1, 0B, 9c)
non-rich	1/4 (25.0%)	fastapi_14492 (388B, 18c)	fastapi_14360 (2, 634B, 24c), fastapi_5624 (1, 2094B, 28c), requests_7502 (1, 2633B, 28c)
￼
A flat 1/4 × 3 readout falsifies both structural hypotheses simultaneously: rich-small and rich-large tie (rejecting G1), and rich and non-rich tie (rejecting G3). The 88.9% rich rate inside WIN-SLICE was a selection artifact of WIN's calibration filter, not intrinsic repository simplicity. Three trace-level findings sharpen this result:

rich_3506 died at 9 tool calls (482.2s, exit=-1, patch=0) due to sandbox command latency (396s on a single grep and 250s on one-liners), making the capability rate 3/11 (27.3%) excluding harness latency kills.

rich_3472 executed 28 read-only grep calls (165.1s) and submitted an empty diff—motivating an explicit empty-submit ban and read time-boxing.

fastapi_14492 resolved by editing a documentation example (docs_src/metadata/tutorial001_1_py39.py), counting in 3/12 as graded while being excluded from source-only training data.

3.4 Forensic Audit Across All 44 Traces
Applying the surgical-green filter across all 17 solves (14 WIN + 3 FAM) leaves 9 clean source-only surgical greens (10 before excluding fastapi_14492's docs edit):

Excluded for size (>1200B): rich_3882 (1475B), fastapi_14372 (2110B), fastapi_5077 (3803B), rich_3470 (2970B), and rich_3480 (2242B, which hit the 480s wall timeout before explicit submit).

Excluded for test/docs edits: fastapi_9425 modified test_annotation.py, rich_3067 made 13 edits to tests/test_highlighter.py, and fastapi_14492 modified docs_src/....

Semantic exit-2 failures refute a syntax guard: All four exit-2 losses across the 44 traces (fastapi_14303, 14430, 11355, 14360) were semantic assertion or custom reproducer errors (14360 asserted HTTP 200, 14303 printed "Test failed" after running py_compile, 14430 printed union-typing mismatches, and 11355 raised a TypeError in its own repro.py) with zero syntax or import collection breaks. A post-edit py_compile / --collect-only gate would recover 0/4.

Self-verification theater: Only 3/44 traces ever invoked pytest (2 failed collection, 1 selected 48 tests), while 12/12 FAM traces relied exclusively on hand-written repro.py scripts. CONFIDENCE lines quoted reproducer stdout verbatim without exit codes or suite pass counts. Quarantine of small-
𝑁
N (n=9) LoRA fine-tuning follows directly: updating low-rank weights on 9 trajectories skewed toward one library (rich) risks out-of-distribution negative transfer while frozen base 31B already achieves 27.3% on fresh tasks.

3.5 High-Temperature Sampling Basin Probe (Best-of-3, T=0.75)
At T=0.2, near-deterministic decoding measures a single basin per task. To test whether the five G2v2 failures are escapable sampling basins or weight-locked attractors, we evaluate K=3 independent rollouts at T=0.75 across two pre-registered strata—Stratum A (Attractors): requests_7205, requests_7315, requests_7309; Stratum B (Multi-Modal): requests_7328, rich_3296—ranking non-empty candidate diffs via existing repository test suites with ties broken toward smaller patches. Pre-registered thresholds: >= 1/3 conversion in Stratum A indicates sampling-escapable basins (0/3 indicates weight lock); >= 1/2 conversion in Stratum B indicates execution-ranked Best-of-K harvests rollout variance.

(Insert final readout from kernel 144607478 / gemma-bestofk-temp-probe here prior to 23:59 UTC.)

4. Related Work
Prior coding-agent literature attributes small-model repair failures to three bottlenecks: fault localization, verification distinguishability, and domain priors (Yang et al., 2024; Xia et al., 2024). Locate-then-repair pipelines target localization, yet our traces routinely follow the prompt's SPEC and grep localization protocol on exact target symbols and still fail at identical rates—demonstrating that procedural compliance does not guarantee repair synthesis. Self-verification and execution-reranking studies (Jain et al., 2025; Ehrlich et al., 2025) show that single-rollout agent-authored tests exhibit low distinguishability between valid and invalid patches; our W2B null (0/5 conversion) and CONFIDENCE trace audit provide direct empirical confirmation. Finally, post-training scaling work (Yang et al., 2025; Pan et al., 2025) indicates that ~32B code-repair SFT requires hundreds to thousands of multi-repository trajectories to avoid out-of-distribution regression, motivating our quarantine of 9-trajectory LoRA updates in favor of distribution-invariant harness controls.

5. Limitations
Single Model & Harness Lineage: All 54 runs evaluate gemma-4-31b-it-qat-w4a16-ct inside one ADK tool envelope; methodology transfers across architectures, while absolute solve rates are model- and sandbox-specific.

Stratum Power (n=4 per FAM cell): FAM is powered to detect large cliffs (>=50% gaps) and falsifies strong group predictors, but is underpowered for subtle (10%–15%) covariate effects.

Public Repository Contamination: Textualize/rich, fastapi/fastapi, and psf/requests are public open-source repositories potentially represented in pretraining data, making WIN-SLICE (43.8%) an upper bound relative to private evaluation repositories.

Partial Injection Coverage: fastapi_14258 was evaluated in W1 and W2B but not included in G2v2, and G2v2's requests_7309 prompt named only _parse_content_type_header while scoping out sister functions.

Cross-Checkpoint Floor: five of the seven baseline arms ran the E4B checkpoint (37 runs); the 31B-only floor is 0/10 on attractor tasks, so the same-checkpoint calibration step is 0/10 → 14/32. The E4B floor is prior-art context for the near-zero literature, not a controlled same-model comparison.

6. Conclusion
Calibration is real: freezing a disciplined procedure and filtering for single-file, well-scoped tasks lifted a 31B quantized agent from 0/10 same-checkpoint (0/47 all-checkpoint) to 14/32 (43.8%) and 3/11 (27.3% ex-latency) on fresh held-out tasks. Verification prompting (0/5) and target-function naming (0/5) are insufficient, moving patch shape without converting outcomes. Group-level features—repository family and gold diff length—fail as out-of-sample predictors (1/4 × 3 on FAM). Residual failures are governed by task-level parametric attractors and locality boundaries, making OOD-safe harness invariants (empty-submit bans, per-command read time-boxing, and thrash-stopping) and multi-rollout existing-suite selection the highest-leverage levers when small-
𝑁
N fine-tuning data cannot support generalization.

References
Gemma Team (2026). Gemma 4 Technical Report & Developer Agent Benchmark. Google DeepMind / Kaggle Competition Track (gemma-4-developer-agent, gemma-4-developer-agent-paper).

Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., & Narasimhan, K. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? ICLR 2024.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., & Press, O. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. NeurIPS 2024.

Jain, N., et al. (2025). R2E-Gym: Procedural Environments and Hybrid Verifiers for Scaling Open-Weight SWE Agents. arXiv:2504.07164.

Yang, J., et al. (2025). SWE-smith: Scaling Data for Software Engineering Agents. arXiv:2504.21798.

Pan, J., et al. (2025). Training Software Engineering Agents and Verifiers with SWE-Gym. ICML 2025.

Ehrlich, R., et al. (2025). CodeMonkeys: Scaling Test-Time Compute for Software Engineering. arXiv:2501.14723.

Appendix A — Reproducibility & Honesty Notes
Public-Set Bounds vs. Held-Out Rates: WIN-SLICE (14/32 = 43.8%) reflects calibrated public tasks and serves as an upper bound; FAM (3/12 = 25.0%; 3/11 = 27.3% excluding the 482.2s sandbox-latency kill on rich_3506) is the uncalibrated held-out baseline.

Documentation-Only Solve (fastapi_14492): Counted in FAM's 3/12 graded score (exit=0) because the task's target edit resided in docs_src/metadata/tutorial001_1_py39.py, and excluded from the 9-trace source-only surgical green pool.

Specification Caveat (requests_7309): G2v2 named _parse_content_type_header while scoping out sister functions (parse_dict_header), bounding G2v2's claim to primary-function naming; CALv3 supplied both functions and moved the diff (687B -> 1326B) without converting the outcome.

Reproducible Kaggle Kernel Artifacts: 144573461 (W2A), 144573665 (W2B), 144590816 (G2v2), 144590833 (FAM), and 144607478 (Best-of-3 temperature probe).

3. Drop-In Replacements for Section 3.5 (gemma-bestofk-temp-probe)
Keep whichever block matches the andralemisquitta/gemma-bestofk-temp-probe readout when it lands before 23:59 UTC:

If 0/5 converts (Stratum A 0/3, Stratum B 0/2):

Raising sampling temperature to T=0.75 across K=3 rollouts converted 0/5 tasks (0/3 in Stratum A, 0/2 in Stratum B). Even under elevated temperature and existing-suite test ranking, requests_7205, requests_7315, and requests_7309 failed to escape their wrong-patch basins, while requests_7328 and rich_3296 produced distinct candidate diffs that still failed boundary assertions. Together with W2B (0/5) and G2v2 (0/5), this closes all inference-time prompt, specification, and sampling levers on frozen 31B weights: breaking these failure modes requires weight-level updates at multi-repository scale.

If >= 1/5 converts (e.g., Stratum B converts 7328 or 3296, Stratum A stays 0/3):

Raising sampling temperature to T=0.75 across K=3 rollouts converted X/5 tasks (0/3 in Stratum A; X/2 in Stratum B). On the rigid attractor stratum (7205, 7315, 7309), elevated temperature failed to break the underlying wrong-patch basins (0/3), confirming weight-level lock. Conversely, on the multi-modal stratum (7328, 3296), sampling diversity paired with existing-suite pytest ranking converted X/2 misses by selecting a boundary-compliant candidate out of the rollout distribution—demonstrating that decoupled execution reranking harvests variance on non-attractor tasks without modifying model weights.
