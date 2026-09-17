# Protocol — Two-Lock Asymmetric Task (v5.4)

Status: **executed, negative result.**

## Motivation

The symmetric two-lock task (v5.3) does not distinguish a capability
gate from a target selection. Redesigning the reward so that
z_u determines whether the agent may act, and z_w determines what
to do if acting, yields an analytically clean asymmetry in expected
return.

## Environment

Identical to v5.3, except:

- Action space: {0, 1, 2} = {wait, a_0, a_1}.
- Reward:
  - z_u = 0: success iff a = wait
  - z_u = 1: success iff a = 1 + z_w
- Reward value: +1 success, −1 failure.

## Analytical prediction

- A, D, E, H: EV = 0
- B (aux predicts z_u): EV = +0.5
- C (aux predicts z_w): EV ≤ 0

Predicted gap: return(B) − return(C) ≈ +0.5.

## Preregistered tests

- R1: return(B) − return(D) > 0 (Wilcoxon one-sided, α = 0.0125)
- R2: return(B) − return(C) > 0 (the asymmetry test)
- R5: return(B) − return(A) > 0
- R6: return(B) − return(H) > 0
- R3: TOST |return(C) − return(D)| < 0.15
- R4: TOST |return(A) − return(E)| < 0.15
- G4: permuted own-bit balanced accuracy ≤ 0.55 for B and C

## Execution

Two exploration schemes tested.

**v5.4a.** ent_coef = 0.01, no exploration bonus. Ten seeds, six
variants, 200k environment steps each.

**v5.4b.** ent_coef = 0.10, per-action exploration bonus
+0.05 · max(0, 1 − step/150k) for a ≠ 0. Same budget.

## Result

**v5.4a.** All variants: return ≈ 0.007. No separation.
`acting_frac` for C = 0.000 on all 10 seeds.
R1, R2, R5, R6 FAIL. R3, R4 trivially PASS (equality).

**v5.4b.** All variants: return ≈ 0.000. No separation.
`acting_frac` for C = 0.000 on all 10 seeds. `task_acc` = 0.50 for
all. R1, R2, R5, R6 FAIL. R3, R4 trivially PASS.

## Failure mode

Single-step safe-action tasks trap PPO.

- EV(wait) = 0
- EV(act) = 0.25 · (+1) + 0.75 · (−1) = −0.5

Wait is a strict local maximum in the unconditional policy space.
Once π_wait → 1, all sampled actions are wait, advantage on act-actions
is identically zero, and the policy gradient on non-wait actions
vanishes. Entropy bonus 0.10 and per-action bonus 0.05 are
insufficient to break the fixed point.

The trap is not a defect of the head or the architecture: auxiliary
heads do learn z_u and z_w (aux_u ≈ 0.98, aux_w ≈ 0.98), but the
policy never reaches them because the sampled action distribution
collapses to wait before any learning signal on act-actions exists.

## Interpretation

This is a **negative result for the design under PPO exploration.**
The asymmetric test remains analytically valid; the current
exploration scheme cannot reach its optimum. Alternative schemes
(behavioral cloning warm-up, removal of the wait action, or
substantial per-action bonus > 0.5) are outside the scope of this
preregistration and would constitute a new experiment.

We do not report an asymmetric test outcome. We report that the
asymmetric test cannot be executed with the current method.

## Artifacts

- `v5_4_train.py` — v5.4a and v5.4b training code.
- `v5_4_results.json`, `v5_4b_results.json` — full per-seed logs.
- `v5_4_checkpoints/`, `v5_4b_checkpoints/` — all model checkpoints.
