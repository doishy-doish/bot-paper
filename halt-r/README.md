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
