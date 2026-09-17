# Functional Dependence of Reinforcement Learning Policies on Auxiliary Latent Predictions: A Pre-Registered Testbed with Positive and Negative Results

**Evgeny V. Mosunov** (Мосунов Евгений Васильевич)

*Preprint, September 2026*

---

## Abstract

We ask whether a reinforcement learning policy develops **functional
dependence** on an auxiliary prediction of a hidden task-relevant
state — that is, whether the policy's action distribution changes when
the auxiliary prediction changes, holding the observation fixed.

We design a pre-registered testbed in three stages.

**Stage 1 (partially observable gridworld).** Three hidden latents
evolve under identical dynamics and are predicted from history with
matched accuracy (oracle MSE/Var ≈ 0.53 vs 0.83 without history).
Six PPO variants (A: no auxiliary channel; B: predict s; C: predict w;
D: predict v; E: zero channel; H: noise channel) are trained for 10
seeds. We obtain a clean negative result: return and total-variation
distance between action distributions under matched-swap are
indistinguishable across variants. The observation contains a
sufficient statistic for a reactive policy, so accurate auxiliary
prediction is functionally redundant.

**Stage 2 (two-lock symmetric).** We strip the task-relevant bits
from observation. Two hidden bits u, w ∈ {0,1} are communicated only
through a noisy private channel c ∈ R³. Correct action is a
deterministic function of (u, w). Across 10 seeds and two sensor
noise regimes (σ_c = 0.005 and σ_c = 0.010, SNR = 29 and 14.5), the
policy functionally depends on auxiliary predictions of u and w:
return(B) − return(D) = +0.313 (p = 0.00195, Wilcoxon), TV(B) − TV(D)
= +0.237 (p = 0.00195), and permutation of the causal channel collapses
own-bit balanced accuracy from 0.96 to ≤ 0.53 (criterion ≤ 0.55).
Capability and target bits are statistically equivalent
(TOST, δ = 0.15). The effect is driven by **causal relevance to task
success**, not by "self-reference."

**Stage 3 (two-lock asymmetric, attempted).** We redesign the reward
so that u is a *capability gate* and w a *target selection*. The
analytical prediction is return(B) ≈ +0.5 vs return(C) ≈ 0, a
structural asymmetry. Two variants of exploration (entropy bonus,
per-action exploration bonus) fail to break a single-step exploration
trap: the safe action `wait` is a strict local maximum in the
unconditional policy space, and PPO cannot escape it. This is reported
as a negative result with a documented failure mode.

We do not claim to demonstrate a self-model in any philosophical sense.
We claim a reusable methodological contribution: a testbed that
separates *accuracy* of auxiliary prediction from *functional use*, a
two-lock design that removes the observation shortcut, and a
pre-registered suite of statistical tests and permutation controls.

**Keywords:** reinforcement learning, auxiliary prediction, functional
dependence, pre-registration, negative results, self-model.

---

## 1. Introduction

A natural operational test for whether an RL agent uses a
representation of its own state or capabilities is to ask: does the
policy's behavior change when that representation changes, holding the
observation fixed? High auxiliary-head accuracy alone does not
establish this. If observation already contains a sufficient statistic
for the task, an accurate head can be entirely redundant.

We present a three-stage pre-registered study.

**Stage 1** constructs a gridworld in which three hidden latents are
predicted from history with matched accuracy. We report a negative
result: the auxiliary channel has no functional effect, because the
observation is sufficient for the reactive policy.

**Stage 2** removes the observation shortcut by moving the
task-relevant bits out of observation and into a noisy private
channel. The auxiliary head becomes the only route through which those
bits can reach the policy. Across two sensor noise regimes, the policy
functionally depends on auxiliary predictions of causally-relevant
bits and not of a causal-null bit or noise. A permutation control
confirms the causal direction.

**Stage 3** attempts to break the symmetry between "capability" and
"target" bits via a structurally asymmetric reward. The design is
analytically clean (gap of +0.5 in expected return between B and C),
but PPO fails to learn it due to an exploration trap in single-step
safe-action tasks. We document the failure and its cause.

**Contributions.**

1. A testbed design methodology that separates accuracy from
   functional use of auxiliary prediction.
2. A positive result on causal-relevance-driven functional dependence
   in the two-lock symmetric task, reproduced across two sensor noise
   regimes, with permutation control.
3. A negative result on a natural gridworld design with an analytical
   explanation (observation shortcut).
4. A negative result on the asymmetric extension with a documented
   exploration trap.
5. A pre-registered protocol with all statistical tests frozen before
   execution.

---

## 2. Related work

Auxiliary prediction has a long history in reinforcement learning —
UNREAL (Jaderberg et al., 2017), curiosity-driven exploration (Pathak
et al., 2017), self-predictive representations (Schwarzer et al.,
2021), contrastive predictive coding (van den Oord et al., 2018). Our
contribution is not a new auxiliary task but a design that makes the
distinction between *accurate representation* and *functionally used
representation* measurable.

World models (Ha & Schmidhuber, 2018; Hafner et al., 2020) also train
internal predictors of hidden state. We do not compare our design to
world-model methods directly; we focus on a minimal testbed in which
the observation shortcut can be removed by construction.

We note in passing that pre-registration is not standard in RL
benchmarking (Henderson et al., 2018; Agarwal et al., 2021). We use it
here to prevent post-hoc selection of metrics after viewing results.

---

## 3. Environment specification (frozen v4.2)

### 3.1 Latents

s, w, v ∈ [0,1]⁴; common slow context q ∈ [0,1]⁴.

### 3.2 Dynamics

    q_{t+1} = clip(0.98 · q_t + 0.01 + η_t, 0, 1),      η_t ~ N(0, 0.02² I)
    x_{t+1} = clip(0.20 + 0.20 · x_t + B · φ(a_t) + 0.40 · q_t + ε_t, 0, 1)
                                                          x ∈ {s, w, v}
    with ε_t ~ N(0, 0.035² I).

### 3.3 Matrices

    A = 0.20 · I₄

    B = [[-0.025, -0.010,  0.000,  0.015,  0.005],
         [-0.0075,-0.0225,  0.010,  0.000,  0.0125],
         [ 0.000, -0.0125, -0.020,  0.010,  0.005],
         [-0.010,  0.000, -0.0125, 0.0075, -0.0175]]

    C = 0.40 · I₄
    d = [0.20, 0.20, 0.20, 0.20]ᵀ

    M = [[1, 0, 1, 0, 0],
         [0, 1, 0, 1, 0],
         [1, 1, 0, 0, 0],
         [0, 0, 1, 1, 0]]   (wait column = 0)

### 3.4 Execution

    c_s(a) = clip(⟨M_{:,a}, s⟩ / 4, 0, 1)
    c_w(a) = clip(⟨M_{:,a}, w⟩ / 4, 0, 1)
    p_exec(a) = σ(8 · (c_s(a) − 0.25) + 8 · (c_w(a) − 0.25))

v does not enter p_exec.

### 3.5 Observation

    y_q = clip(q_t + N(0, 0.20² I), 0, 1)   (4 dims)
    pos_x/GRID, pos_y/GRID, (goal_x − pos_x)/GRID, (goal_y − pos_y)/GRID
    min_obstacle_distance / (2 · GRID)
    signal_flag
    task one-hot (4 dims) for RL.  Total policy input: 14 dims.

### 3.6 Environment validation (E1–E4)

- **E1** residual unpredictability: 0.25 ≤ U ≤ 0.75
- **E2** matched predictability: |G_i − G_j| / mean ≤ 0.10
- **E3** causal relevance: E_s > 0, E_w > 0, E_v < 0.05
- **E4** partial observability: 0.25·Var < MSE_current < 0.90·Var
  and MSE_hist ≤ 0.75·MSE_current

All criteria pass in-sample (seeds 1–5) and out-of-sample (seeds 6–10).
External oracle predictor: MSE/Var = 0.53 with history, 0.83 without.

### 3.7 RL wrapper

Action space: 5 discrete actions (N, S, W, E, wait).
Task family: reach, find, avoid, wait; 7 train / 5 test configurations.
Horizon 100. Reward +1 goal, −0.01 per step.

---

## 4. Experiment 1 — Gridworld (negative result)

Six variants with PPO, GRU backbone, aux head MLP [64, 64, 4],
λ_aux = 300, 10 seeds, 200k environment steps per variant.

### 4.1 Results (median over 10 seeds)

| Variant | Return | TV |
|---------|--------|----|
| A (no aux)          | −0.018 | 0.0045 |
| B (predict s)       | −0.724 | 0.0029 |
| C (predict w)       | −0.763 | 0.0033 |
| D (predict v)       | −0.735 | 0.0030 |
| E (zero channel)    | −0.018 | 0.0000 |
| H (noise target)    | −0.765 | 0.0000 |

### 4.2 Analysis

Observation contains position and goal coordinates. A reactive policy
("move toward the goal") is near-optimal. Even at MSE/Var ≈ 0.58 for
the auxiliary head, the policy optimizer has no gradient incentive to
give the auxiliary channel nonzero weight. The negative result is
structural: any environment whose observation is sufficient for the
task cannot distinguish functional dependence on auxiliary prediction
from redundancy.

### 4.3 Implication

Reports of "self-model accuracy" in such environments do not establish
that the policy uses the self-model. Functional dependence must be
tested in an environment where observation is not sufficient for
optimal behavior.

---

## 5. Experiment 2 — Two-Lock Symmetric (positive result)

### 5.1 Design

Three hidden bits u, w, v ∈ {0,1} with P = 0.5, each communicated
through a noisy private channel:

    c_u = clip(u + N(0, σ_c² I), 0, 1)      (4 dims)
    c_w = clip(w + N(0, σ_c² I), 0, 1)
    c_v = clip(v + N(0, σ_c² I), 0, 1)

Observation includes c_u, c_w, c_v and task-feature vector
o_task ∈ R⁴. Actions: {0, 1, 2, 3}. Correct action:

    a* = 0 if z_u = 0 else 1 + z_w,
    z_u = 1[u_0 > u_1],  z_w = 1[w_0 > w_1]

Reward: +2 if both bits correct, 0 if one, −2 if none.

### 5.2 Architecture

- `gru_vis` produces hv (24 dim) from o (8 dim).
- `gru_priv` produces hp (8 dim) from the same o.
- Auxiliary head input: [o, c, hp] (8 + 4 + 8 = 20 dim), output 4 dim
  sigmoid. This is the private path.
- Policy actor input: [hv, aux] (24 + 4 = 28 dim).
- v_c is a *private* channel: it is not part of observation. The
  auxiliary head is the only way any c can influence policy.

### 5.3 Pre-registration

Wilcoxon signed-rank, one-sided, α = 0.0125 (Bonferroni for 4 tests):

- R1: return(B) − return(D) > 0
- R2: TV(B) − TV(D) > 0
- R5a: return(B) − return(A) > 0
- R5b: return(B) − return(H) > 0
- R6: return(B) − return(E) > 0

TOST equivalence, δ = 0.15, α = 0.05:

- R3: |return(B) − return(C)| < 0.15
- R4: |TV(B) − TV(C)| < 0.15

Permutation control (G4): own-bit balanced accuracy with c-channel
permuted across episodes must be ≤ 0.55 for B and C.

### 5.4 Results at σ_c = 0.005 (SNR = 29)

Median over 10 seeds:

| Variant | Return [min, max] | TV [min, max] |
|---------|-------------------|----------------|
| A | −0.018 [−0.038, +0.021] | 0.0045 [0.0025, 0.0112] |
| B | **+0.304** [+0.264, +0.335] | **0.253** [0.219, 0.275] |
| C | **+0.305** [+0.240, +0.358] | **0.242** [0.219, 0.287] |
| D | −0.018 [−0.053, +0.033] | 0.017 [0.012, 0.021] |
| E | +0.004 [−0.048, +0.037] | 0.0037 [0.0027, 0.0059] |
| H | −0.019 [−0.047, +0.034] | 0.0050 [0.0026, 0.0093] |

Preregistered tests:

    R1  return(B) − return(D) = +0.313   p = 0.00195   PASS
    R2  TV(B) − TV(D)         = +0.237   p = 0.00195   PASS
    R3  |return(B) − return(C)| = 0.014  TOST PASS (δ=0.15)
    R4  |TV(B) − TV(C)|         = 0.0038 TOST PASS (δ=0.15)
    R5a return(B) − return(A) = +0.321   p = 0.00195   PASS
    R5b return(B) − return(H) = +0.312   p = 0.00195   PASS
    R6  return(B) − return(E) = +0.301   p = 0.00195   PASS
    G4  permuted own-bit bacc ≤ 0.55                     PASS

Permutation control (G4) at σ_c = 0.005, median over 10 seeds:

| Variant | Own-bit normal | Own-bit permuted | Drop |
|---------|----------------|------------------|------|
| B (u)   | 0.9565 | 0.5001 | +0.4611 |
| C (w)   | 0.9465 | 0.4976 | +0.4424 |
| D       | 0.4994 | 0.5000 | +0.0000 |

### 5.5 Results at σ_c = 0.010 (SNR = 14.5)

Second regime to test sensor-noise stability. All preregistered tests
PASS. Numbers are within seed noise of σ_c = 0.005:

    return(B) median: +0.305 (vs +0.304 at σ_c = 0.005)
    return(C) median: +0.306 (vs +0.305)
    TV(B) median:     0.2548 (vs 0.2533)
    aux_B → z_u:      0.9666 (vs 0.9752)
    G4 perm max:      0.5311 (vs 0.5279; criterion ≤ 0.55)

The mechanism is stable to a 2× increase in sensor noise.

### 5.6 Interpretation

The policy functionally uses auxiliary predictions of u and w. It does
not use v or noise. B and C are statistically equivalent (R3/R4).
Causal relevance to task success drives functional dependence;
"self-relevance" is not distinguishable from "external relevance" in
this symmetric design.

---

## 6. Experiment 3 — Two-Lock Asymmetric (negative result, attempted)

### 6.1 Design

Same observation and channel as Experiment 2. New action space
{0, 1, 2} = {wait, a_0, a_1}. Reward:

    if z_u = 0:  success iff a = wait
    if z_u = 1:  success iff a = 1 + z_w

u is a *capability gate*: it determines whether the agent may act.
w is a *target*: it selects which action to take if acting.

### 6.2 Analytical prediction

- A, D, E, H: optimal EV = 0
- B (aux predicts z_u): EV = 0.5 · 1 + 0.5 · 0.5 = **+0.5**
- C (aux predicts z_w): EV ≤ 0

Predicted gap: return(B) − return(C) ≈ +0.5.

### 6.3 Attempted two exploration fixes

**v5.4a.** ent_coef = 0.01, no exploration bonus. Result: all variants
return ≈ 0.007, no separation. `acting_frac` for C = 0.000 on all
10 seeds.

**v5.4b.** ent_coef = 0.10, per-action exploration bonus
+0.05 · (1 − step/150k) for a ≠ wait. Result: all variants
return ≈ 0.000, no separation. `acting_frac` for C = 0.000 on all
10 seeds. `task_acc` = 0.50 for all.

### 6.4 Failure mode

PPO in a single-step task with a safe action is trapped. Analytically:

- EV(wait) = 0
- EV(act) = 0.25 · (+1) + 0.75 · (−1) = −0.5

Wait is a strict local maximum in unconditional policy space. The
policy concentrates π_wait → 1. All sampled actions are wait, so
advantage for act-actions is identically zero, so the policy gradient
on non-wait actions vanishes. Entropy bonus 0.10 and per-action bonus
0.05 are insufficient to overcome this fixed point. Breaking the trap
requires either bonus > 0.5 (which changes the reward landscape
substantially) or removal of the wait action entirely (which changes
the task).

### 6.5 Interpretation

This is a genuine negative result for the *design*, not for the
hypothesis. The asymmetric test remains analytically clean, but the
current PPO exploration scheme cannot reach its optimum. We do not
report an asymmetric test outcome; we report that the asymmetric test
cannot be executed with the current method.

---

## 7. Formal model (Tier 1 v1.0)

We distinguish six levels of informational support for the claim "the
policy uses latent z":

1. **Availability**: z is present in the agent's environment.
2. **Accessibility**: z is reachable by some sensor.
3. **Representation**: some head of the agent encodes z with accuracy
   above chance.
4. **Attainability**: a policy could in principle achieve high reward
   on this environment.
5. **Functional use**: policy action distribution depends causally on
   the encoded representation.
6. **Utility**: functional use yields reward improvement over a
   matched no-channel baseline.

The gridworld experiment shows levels 1–4 do not imply 5. The
two-lock experiment shows level 5 can be established when the
observation shortcut is removed.

We do not claim these levels form a strict hierarchy for all
environments; they are diagnostic axes for this study.

---

## 8. Discussion

Two findings stand out.

**First**, accurate auxiliary prediction is not the same as
functionally used auxiliary prediction. In any environment where a
reactive policy solves the task, an accurate auxiliary head can be
entirely redundant. We recommend that "self-model accuracy" claims be
accompanied by a functional-dependence test in an environment where
observation is not sufficient for optimal behavior.

**Second**, in a two-lock design that avoids the shortcut, policy
behavior depends causally on auxiliary predictions of
causally-relevant bits but not of a causal-null bit or noise. The
effect is stable to a 2× change in sensor noise. In the symmetric
variant, capability and target bits are treated equally. The
asymmetric variant, which would distinguish them, is analytically
clean but exploration-trap-infeasible under PPO with the tested
hyperparameters.

We do not claim to have demonstrated a self-model. What we claim is a
methodology: a testbed that separates accuracy from functional use, a
two-lock design that removes the shortcut, and a pre-registered suite
of tests plus permutation control.

---

## 9. Limitations

1. The symmetric two-lock task does not distinguish "self" from
   "external." The asymmetric design is attempted but failed at the
   exploration level.
2. Single-step episodes (T = 1) in the two-lock task.
3. Scalar bits (z ∈ {0,1}); continuous latents may differ.
4. PPO only.
5. No comparison to alternative methods for measuring self-model
   functional dependence.
6. The asymmetric test requires a different exploration scheme
   (behavioral cloning warm-up, remove-safe-action design, or
   substantial exploration bonus); we did not run any of these.
7. The interpretation of "capability = self" is one of several
   possible readings and is not empirically required by the design.

---

## 10. Reproducibility

All experiments use fixed seeds (0–9). Environment parameters are
frozen in `ENVIRONMENT_SPEC_v4_2.md`. Protocols are frozen in
`PROTOCOL_v5_3_symmetric.md` and `PROTOCOL_v5_4_asymmetric.md`.
Code files are indexed in `CODE_INDEX.md`. Preregistered statistical
tests are specified in the protocol documents before execution.

Full reproduction: approximately 20 hours on a single modern CPU.

---

## 11. References

See `REFERENCES.md`.
