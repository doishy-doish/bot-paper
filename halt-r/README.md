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
