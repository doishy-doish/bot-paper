> **License:** halt-r/ is proprietary — (c) 2026 Andrale Misquitta, All Rights Reserved. See [LICENSE](./LICENSE). Not covered by the repo-wide Apache-2.0.

# HALT-R: Nobel-grounded recursive reasoning, CPU/WASM-first
Date: 2026-11-12. Status: reference staged, WASM port + Actions next.

## The bet
TRM refines answers by fixed unrolling with BPTT (GPU-hungry, slow to train).
Energy-based associative memory does the SAME shape (iterate to fixed point)
with physics behind it — and storage is Hebbian one-shot, no gradients at all.

## Nobel grounding (not decoration — each maps to a mechanism)
1. Hopfield / Hinton, Physics 2024 (neural-network associative memory):
   Modern Hopfield update xi <- X softmax(beta X^T xi). Recall = energy descent.
   This IS the recursion loop, with convergence guarantees TRM lacks.
2. Parisi, Physics 2021 (spin glasses, complex energy landscapes):
   rugged landscapes with many metastable states = why fixed unrolling gets
   stuck and why adaptive halting + temperature annealing helps. Storage-capacity
   scaling laws come from here.
3. Boltzmann / contrastive learning (Hinton lineage):
   local contrastive-Hebbian updates (positive-phase vs negative-phase) replace
   BPTT. CPU-friendly, no backward graph, exact fit for WASM + Actions runners.

## Architecture (tiny, CPU-first)
- Store: patterns via Hebbian outer product (classic) or X matrix directly (modern).
  Zero gradient steps. Milliseconds on CPU.
- Recall: iterate Hopfield update until energy delta < eps (adaptive halt).
  Easy queries exit in 2-3 steps, hard ones run full depth. This is the TRM
  efficiency win, with a Lyapunov guarantee.
- Learn (only part that trains): a SMALL encoder (16->16 linear + sign, or 2-layer
  MLP) trained by contrastive-Hebbian on CPU in minutes, or skipped entirely
  (random projections work for the toy). No BPTT through recursion. Ever.

## Why WASM + GitHub Actions
- Recall loop = matvec + softmax. ~100 lines of Rust, no deps, compiles to
  <50KB wasm. Runs in browser, node, and Actions runners identically.
- Training (Hebbian + tiny contrastive) = seconds on CPU. Actions free runners
  (2-core) finish the full store+recall+verify in under 2 minutes. Zero GPU,
  zero quota, reproducible on every push.

## Measured (toy: 8 patterns x 64-bit, 8 flips, 200 trials, CPU ms)
- Classic Hopfield: 95.0% retrieval, 2.17 avg steps (max 12, halt on fixed point).
- Modern Hopfield (beta=8): 100% retrieval, 3.13 avg steps (max 12, halt on tol).
- Storage cost: zero gradient steps (Hebbian one-shot). Steps saved vs fixed unroll: ~75%.

## Layout
- ref-hopfield.mjs — pure-JS reference (classic + modern update, adaptive halt,
  energy trace). Runs anywhere, no build.
- rust/ (next) — no-dep Rust recall core -> wasm-pack -> halt_r.wasm.
- .github/workflows/halt-r.yml (next, staged on approval) — build wasm,
  run recall benchmark, verify retrieval rate, publish artifact.

## Roadmap
1. ref (done) -> verify retrieval + halt savings on toy patterns.
2. Rust core + wasm build (local file, no push needed to author).
3. Actions workflow: ONLY on explicit push approval (repo rules).
4. Encoder: contrastive-Hebbian tiny MLP, still CPU-minutes.
5. Scale: ARC-mini symbols as binary patterns; report steps-saved vs fixed unroll.

## Results log (all CPU, worker sandbox + Actions runner, zero GPU)
Toy: 8x64-bit patterns, 8 flips, 200 trials unless noted. Reasoning: 4x4 binary grids.

- Retrieval (seen patterns): classic 95.0%/2.17 steps; modern b=8 100%/3.13 steps.
  Storage = Hebbian one-shot, zero grad steps. Halt saves ~75% vs fixed 12.
- Capacity (D=64, 8 flips): classic 8->95%, 16->20%, 32->0% (textbook ~0.14D collapse).
  Modern b=8: 8->100%/3.1, 16->100%/4.0, 32->100%/4.8. Modern is the scalable substrate.
- Memory != reasoning: nearest-memory rule transfer: mirror 0%, move 100%, fill 100%
  clean (tiny spaces memorize); +10% noise: mirror 0%, move 92%, fill 88%.
- Operator learning (least squares, closed form, CPU ms): mirror 500 pairs -> 100%
  exact on unseen (K=20 suffices); move 100%, fill 100% clean. Noisy10: bit 89-95%,
  exact 16-61% (linear can't denoise+map; Hopfield cleanup of unseen targets: 0% -
  negative result, associative cleanup needs the target stored).
- Causal (diamond 0->1,0->2,1->3,2->3, N=300, exhaustive BIC over 543 DAGs):
  obs-only SHD=2 (effect err 0.57, edge flipped = Markov equivalence);
  +do(X2) SHD=2 (other edge flips = single-target insufficiency, Hauser-Buehlmann);
  +do(X0)+do(X2) SHD=0, map exact, effect 1.132 vs 1.10 (err 0.03).
  The causal reasoning model: discovery + multi-target interventions + do-calculus.

Frontier: multi-target interventional discovery on larger graphs (PC/GES scale-up),
nonlinear SEMs (fill-rule needs hidden layer), encoder contrastive learning for
noise-robust operators. Math-dataset fallback NOT needed - causal loop closed.

## Rung results 2 (all CPU closed-form, no gradients)
- Hebbian stack (denoise->operate vs direct, mirror): noise2%: direct 74.3%/98.1%
  vs stacked 67.7%/97.7% (ceiling 72.4%); noise10%: 21.3%/90.3% vs 23.3%/90.4%
  (ceiling 18.5%). Verdict: single map already AT the noise ceiling; stacking adds
  nothing on redundancy-free data (denoiser~=identity). Bug caught mid-run
  (involution mislabel) and corrected. Depth without structure is dead scaffolding.
- Masked pretraining (3 bits of 16): random grids 62.5%/27% (~prior, learns nothing);
  2x2-block grids 100%/100%. Self-supervision works iff the data has structure.
- Systematicity (train d1+d2 shifts, test held-out d3): seen 68%/61%, held-out 24%.
  Monolithic map can't compose; mixing primitives degrades even seen directions.
  Fodor's challenge stands = the marked frontier. Next: factorized operators + router.

## Kolmogorov shootout (accuracy per byte — smallest machine booting its territory wins)
| model | acc | bytes | acc/KB |
|---|---|---|---|
| causal DAG diamond | 1.0 | 96 | 10.67 |
| classic Hopfield 8x64b | 0.95 | 264 | 3.68 |
| modern Hopfield b=8, 32 pats | 1.0 | 456 | 2.25 |
| linear operator | 1.0 | 2048 | 0.50 |
| ELM H=256 readout | 0.84 | 32768 | 0.03 |
Float64 accounting; toy scale. Ordering is the claim: the frontier model is the
tiniest one that regenerates its territory. ELM's 32KB readout is the bloat the
thrash-stop rule predicts. (Full table: kolmogorov.json.)

## Scramble protocol (public compute, private numbers)
Same protocol as src/bot/SCRAMBLE.md. Synthetic toys stay plaintext (no private
numbers in them). Anything real (datasets, private evals) encrypts-before-push:
wrapper {enc:"AES-256-GCM-v1", nonce, data, tag, sha256, bytes, date}, quiet logs
(counts + hashes), no pull_request triggers. Decrypt after pull:
node scripts/scramble.mjs decrypt <file> --key-file <vault key> (writes .dec,
never overwrites wrapper — never commit decrypted files).
Key lives in exactly 3 places (values never printed): scratch/vault/keys/scramble.key,
private halt-r keys/scramble.key (storage only, no Actions), public repo secret SCRAMBLE.
Details: SCRAMBLE.md.

## Router rung (factorized compositional generalization)
Ops library trained once on primitives (mirror, shiftA, shiftB — never composed).
Held-out composition B-then-M, router picks pipeline by 5 support pairs:
monolithic map scored 20-24% on the same held-out; factorized pipeline scores
100% exact / 100% bit on 200 queries. Footnote: support tied M>B vs B>M (2x2
block is mirror-symmetric, both pipelines coincide) — tie-break needs asymmetric
probes; accuracy unaffected. The contrast is the finding: factorization, not
capacity, is what crosses the composition gap.

## Router asym probe (non-commuting orders; corrects the tie footnote)
M (mirror L-R) commutes with S-shift B (different axes), so B>M vs M>B tie on
ALL inputs — structural, not a symmetry artifact. The earlier footnote blaming
2x2-block symmetry was wrong; corrected here. Non-commuting probe A>M
(east-shift then mirror) vs M>A: 5/5 asym supports diverge, router picks A>M
uniquely (next best err 4 vs 0), 200/200 queries 100% exact / 100% bit, 0 ties.
Symmetric 2x2 blocks already separate A-orders (A>M err 0 vs next 2), as expected.
(Rung file: rung-router-asym.json.)

## Nonlinear factorized pipeline (learned ELM ops, held-out composition)
Primitives M (mirror) + N (dilate-east OR, nonlinear), one ELM H=256 each on 400
random pairs (CPU closed-form): M 100%/100%, N 77% exact / 98.1% bit on 100
held-out. Composition N-then-M held-out (asymmetric only): router picks N>M by
5 support, factorized ELM pipeline 78% exact / 98.4% bit on 200 queries —
composition adds ~no error beyond the N-op bottleneck. Monolithic ELM trained on
primitives only, tested on the same composition: 0% exact / 66.5% bit.
Factorization, not capacity, crosses the gap — even with imperfect learned ops.
(Rung file: rung-factor-nonlinear.json.)

## Contrastive-Hebbian encoder rung (online rule, no backprop, no matrix inverse)
Single-layer contrastive-Hebbian (clamped-vs-free co-occurrence, perceptron-form
online updates, CPU seconds): mirror rule 100% exact / 100% bit on unseen
grids at K=500 pairs (98.7%/99.9% at K=200) — the online rule reaches the
least-squares closed-form ceiling with enough pairs. This proves the HALT-R
learn step for the linear encoder, the actual architecture target.
Negative control behaves correctly: single-layer scores 0% exact on xor-neighbor
(a task needing hidden units). The 2-layer hidden CHL as implemented learns
nothing (59.7% bit vs 60.6% single-layer control) — hidden-layer credit
assignment without backprop stays OPEN; ELM closed-form covers hidden layers
in the meantime (78%/98% factorized). (Rung file: rung-contrastive.json.)

## Causal scale-up rung (5 nodes, exhaustive BIC, CPU ~0.1s)
Random 5-node DAG (6 edges), linear-Gaussian N=500. Exhaustive BIC over all
29,281 DAGs (120 orders x 1024 subsets, cached OLS, ~0.1s CPU) recovers the
exact graph obs-only (SHD 0); interventional arms (+do(X2), +do(X0)) confirm
stability (SHD 0 throughout) — no ladder headroom on this instance
(orientation already identified; a near-unfaithful instance is next for the
ladder). Path-traced total effect X0->X4 on learned weights: -2.443 vs -2.346
truth (err 0.098), obs data only. Two self-caught bugs fixed on record:
self-parent mask (RSS=0 degenerate) and an SHD scorer iterating key-sets as
edges (phantom SHD 12 on an exact recovery). (Rung file: rung-causal-scale.json.)

## Near-unfaithful rung (exact cancellation; only intervention sees the edge)
Triangle 0->1 (2), 1->2 (2), 0->2 (-4) + 2->3 (1.5): the direct edge exactly
cancels the indirect path, so population Corr(X0,X2)=0. Seed 7, N=500.
Obs marginal screening (Fisher-z, the PC-skeleton first step): r=-0.060,
p=0.18 — edge DROPPED, as theory predicts. Exhaustive BIC prune-free ALSO
drops it (skeleton 3/4): it prefers a 3-edge collider model 0->1<-2, 2->3
(score 154.1 vs truth 158.5) that fits better with fewer edges. Exact
cancellation defeats score-based discovery too — not just constraint-based.
do(X1=+-2.5) breaks the cancellation: within-arm corr -0.61 (z=-15.9),
OLS X2~{X0,X1} nails w02=-3.97, w12=1.97 model-free. Ladder, honestly scored
on skeleton recall: obs-marginal MISS -> obs-BIC MISS -> intervention HIT.
Method corrections on record: (1) branch-and-bound prune REMOVED everywhere —
partial BIC can exceed best-so-far yet later negative terms win, so the prune
silently cut valid branches (caught when two sample sizes returned identical
graphs); 4-5 node exhaustive needs no pruning (1536/122880 evals, cached OLS,
ms). (2) BIC ties inside a Markov class resolve by enumeration order, so
DAG-SHD is reported only where v-structures compel orientations; otherwise
skeleton recall. (3) seeds + weights recorded in-file from here on. The 5-node
scale-up optimum was re-run prune-free (seed 11, weights in-file): SHD 0
exact — the earlier SHD-0 stands, method note corrected.
(Rung file: rung-causal-nearfaith.json.)

## Nonlinear SEM rung (linear discovery blind, quadratic expansion recovers)
X0,X1 ~ N(0,1); X2 = X0^2 + X1 + e. Seed 21, N=500. Linear-Gaussian exhaustive
BIC: skeleton {1-2} only — the quadratic edge 0-2 is invisible (linear fit of
X0^2 on symmetric X0 is 0). Quadratic expansion (nodes X0,X1,X0^2 with X0^2
root-fixed, X2): recovers 0sq->2 and 1->2 with correct v-structure
orientations, plus one disclosed artifact edge (0sq->0, the deterministic
feature read backwards). Verdict: linear discovery is blind past the linear
frontier; polynomial/GAM expansion is the next rung.
(Rung file: rung-causal-nonlinear.json.)

## WASM recall core rung (Rust, zero deps, Actions-verified bit-identical)
rust-recall/src/lib.rs (~70 lines, no dependencies): modern Hopfield step
(x <- X^T softmax(beta X x)) + C ABI `recall()` on caller-owned buffers.
GitHub Actions (ubuntu-latest, free runner) builds wasm32-unknown-unknown
and runs node test-recall.mjs: recall.wasm = 20,675 bytes (under the 50KB
claim), 12/12 iters deterministic, maxAbsDiff vs JS reference 0.00e+0
(exact: +-1 arithmetic is exact in f32 and f64, winner-take-all normalizes
identically), retrieval nearest=3 hamDist 0.0 — 3/3 PASS, run 43314419634
green first try. The same hosted bytes (halt-r/rust-recall/recall.wasm, committed
from the CI artifact) were then fetched and instantiated in a live browser page:
12 iters, nearest=3, hamDist 0 — one source file, three substrates (Rust/wasm,
node, browser). CPU-only, zero GPU, zero quota.
(Rung file: rung-wasm.json.)

## Greedy scale-up rung (10 nodes; hill-climb SHD 3, regime-aware interventions close it)
Random 10-node DAG (10 edges, seed 31, weights +-1..2.5, weights in-file),
N=1000 obs. Greedy hill-climb BIC (add/delete/reverse + cycle check, best of
3 restarts): skeleton recall 1.0, SHD 3 (adds 1->5, 2->4; flip 5->4), ~0.1s
CPU, 473 cached fits. Exhaustive is impossible here (~4.2e18 DAGs). All obs
errors touch node 4/5, so do(X4=+-3): naive pooling (obs+int, N=1500) lands in
the SAME local optimum, SHD 3 — pooled single-DAG BIC is mixture-misspecified
(honest negative, kept on record). Regime-aware instead: hill-climb on the
do(X4) data alone (N=500, X4 exogenous) recovers the mutilated graph EXACTLY
(SHD 0: 4->5 oriented, both spurious edges gone); the 2 edges into X4 come
from the obs run. Union = full truth SHD 0. Lesson: at scale, interventions
must be scored as regimes (GIES-style joint scoring is the next method rung),
never pooled.
(Rung files: rung-causal-greedy.json, rung-causal-greedy-int.json,
rung-causal-greedy-intonly.json.)

## GIES-lite rung (joint regime-aware scoring: full truth SHD 0, one run)
Same 10-node instance and data (obs N=1000 + do(X4) arms). Score(par) =
BIC_obs(all nodes) + BIC_arm(all nodes except intervened X4, whose incoming
edges are mutilated away). Same hill-climb, best of 3 cold restarts: SHD 0
exact — add 0, del 0, flip 0. The complete causal ladder on one instance:
obs-only SHD 3 -> naive pooling SHD 3 (misspecified, negative) ->
regime-split mutilated-exact -> joint regime-aware SHD 0. Interventions work
at scale iff scored as regimes. CPU ~0.1-0.2s throughout.
(Rung file: rung-causal-gieslite.json.)

## Polynomial auto-discovery rung (BIC picks the degree; no human in the loop)
Same nonlinear SEM (X2 = X0^2 + X1 + e, seed 21, N=500). Per-target best-subset
BIC over polynomial feature pools: deg1 (linear) score 613.1, selects {X1}
only — the quadratic edge invisible; deg2 score 31.6, selects {X1, X0^2};
deg3 score 31.6, selects the same {X1, X0^2} (penalty correctly rejects the
cubic extras). Winner: degree 2, exact feature set, ~580 BIC margin — discovery
needs no human-picked degree. The remaining frontier: cross-term/auto-basis
selection (GAM-sieve) and orientation under expansion.
(Rung file: rung-causal-polyauto.json.)

## EP hidden-layer rung (credit assignment without backprop works; trails backprop)
Equilibrium propagation (free vs nudged settling, contrastive update, symmetric
W, tanh). Recipe correction on record: the earlier naive-CHL 59.7% failure was
under-settling (12 steps), not a principled wall — settling to the true fixed
point (25-50 steps), init +-0.3, beta 0.5-1.0. 2-bit XOR: EP 4/4 in 8/9 configs.
8-bit xor-neighbor (K=300/200 test): EP H=16 76.5%/10%, H=32 73.6%/11.5% vs
backprop H=16 84%/23% (63ms) and H=32 87.7%/29%. EP learns (>> chance 50%),
trails backprop ~10 pts, bigger-H hurts (optimization brittleness), ~100x slower
(settling: 1.6-6.2s vs 63ms). Architecture consequence: ELM closed-form covers
hidden layers now (instant); EP is the demonstrated principled online rule,
CPU-feasible. Gap-closing stays open.
(Rung file: rung-ep-hidden.json.)

## Greedy-20 rung (scaling curve: 4 -> 5 -> 10 -> 20 nodes)
Random 20-node DAG (p=0.15, 25 edges, seed 41), N=2000 obs. Greedy hill-climb
BIC, best of 3: SHD 6 (add 4, flip 2, del 0), skeleton recall 1.0, 1.45s CPU,
3362 cached fits. Curve: 4 nodes SHD 0 (1536 evals) -> 5 nodes SHD 0 (122880
evals) -> 10 nodes SHD 3 (~0.1s) -> 20 nodes SHD 6 (~1.5s). Greedy never
DELETES a true edge at this density (del 0 throughout 10- and 20-node runs);
all errors are over-adds + flips — the regime interventions already fix.
(Rung file: rung-causal-greedy20.json.)

## ELM noise-robustness rung (bigger H hurts under noise)
8-bit 1D mirror + xor-neighbor, ELM tanh, K=500, ridge 1e-3, noise on inputs
train+test. Mirror clean: H=64 and H=256 both 100%/100%. Noise10: H=64
90.5%/44.5% vs H=256 84.1%/27.5% — SMALLER net more robust (capacity overfits
noise; ridge 1e-3 insufficient at H=256). Xor-neighbor clean: H=256 99.6%/97%
vs H=64 85.4%/29.5%; noise10 ~71%/10-20%; noise20 ~57-58% (near chance) —
nonlinearity + noise compound. Lesson: size H to the noise, not the task; or
raise ridge with noise (untested, noted).
(Rung file: rung-elm-noise.json.)
