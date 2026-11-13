/-
graft5-verdict.lean — machine-checked verdict arithmetic for HALT-R graft5.

What this IS: the pre-reg decision rule as a formal predicate, with every number
in the committed graft5 verdict (halt-r/graft5.json + graft5.md, run 43460562247)
machine-certified by the Lean kernel. Plain core Lean 4 only, placeholder-free.

What this IS NOT: a proof that graft5 works. The measurements (interpPpl, b/c
discordant counts, lam*) are empirical CI data — no prover can observe the world.
Scope, stated exactly: given the REPORTED numbers (rounded as in collect),
the verdicts follow from the bar. The bench numerics are out of scope.

The framing (this is the "mathematical framing" work): the bar
  lam* > 0 ∧ d > 0.2 ∧ p < 0.05
is encoded over INTEGERS so the kernel checks everything by computation:
  d > 0.2  ⟺  d100 > 20            (d100 = 100·(raw−graft) in reported hundredths)
  p < 0.05  ⟺  40·tail < 2^n       (p = 2·tail/2^n, exact McNemar two-sided;
                                    tail = Σ_{k≤min(b,c)} C(n,k), n = b+c)
McNemar matches collect: mcP nests (1/2)^n · tail · 2, capped at 1.

Row map (b/c from collect logs; ppl from graft5.json, ×100 rounded = md text):
  ts W8 graft5     raw 11.03 vs 8.55,  b/c 19/2, lam* 0.25 → WIN
  ts W8 graft2     raw 11.03 vs 9.62,  b/c 5/0,  lam* 0.1  → lean, fails p only
  shakes W8 graft5 raw 13.49 vs 13.45, b/c 4/3,  lam* 0.1  → fails d and p
  shakes W16 all   lam* = 0                              → dead (no retrieval)
  ts W16 graft5    raw 11.53 vs 11.60 (graft WORSE), b/c 1/2 → fails d and p
Bound sanity (the math-backtrack floor, log10 reading): 512 < 594.
-/

-- Binomial coefficient (Pascal recursion; tiny arguments only).
def choose : Nat → Nat → Nat
  | _, 0 => 1
  | 0, _ + 1 => 0
  | n + 1, k + 1 => choose n k + choose n (k + 1)

-- One-sided McNemar tail: Σ_{k ≤ min(b,c)} C(b+c, k).
def tailSum (n lo : Nat) : Nat :=
  (List.range (lo + 1)).foldl (fun acc k => acc + choose n k) 0

-- Pre-reg bar over integers. lamPos = lam* > 0 (from config).
def winsNat (lamPos : Bool) (d100 tail twoPowN : Nat) : Bool :=
  lamPos && decide (d100 > 20) && decide (40 * tail < twoPowN)

/- Helpers: the exact arithmetic the verdicts rest on. -/
example : tailSum 21 2 = 232 := rfl
example : tailSum 5 0 = 1 := rfl
example : tailSum 7 3 = 64 := rfl
example : tailSum 3 1 = 4 := rfl
example : (2 ^ 21 : Nat) = 2097152 := rfl
example : (2 ^ 5 : Nat) = 32 := rfl
example : (2 ^ 7 : Nat) = 128 := rfl
example : (2 ^ 3 : Nat) = 8 := rfl
example : 1103 - 855 = 248 := rfl
example : 1103 - 962 = 141 := rfl
example : 1349 - 1345 = 4 := rfl
example : decide (1153 > 1160) = false := rfl
example : decide (512 < 594) = true := rfl

/- Verdicts: each row's bar application, kernel-checked. -/
-- ts W8 graft5: lam 0.25>0, d=2.48>0.2, p=464/2097152≈0.00022<0.05 → WIN.
example : winsNat true 248 232 2097152 = true := rfl
-- ts W8 graft2: lam ok, d=1.41 ok, p=2/32=0.0625 NOT <0.05 → lean stands (fails p only).
example : winsNat true 141 1 32 = false := rfl
-- shakes W8 graft5: d=0.04 (4≯20), p=128/128=1.0 → fails d and p.
example : winsNat true 4 64 128 = false := rfl
-- shakes W16 (all projs): lam*=0 → dead regardless of the rest.
example : winsNat false 0 1 1 = false := rfl
-- ts W16 graft5: graft worse (1153≯1160, d≤0), p=8/8=1.0 → fails d and p.
example : winsNat true 0 4 8 = false := rfl
