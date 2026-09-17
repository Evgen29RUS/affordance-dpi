# Preregistration — v5.3

Дата фиксации критериев: до запуска полного прогона.
Режим: two-lock symmetric, T = 1, приватные каналы c_u, c_w, c_v.

---

## §1. Гипотеза

**H1.** При наличии приватного сенсорного канала `c_x ∉ o` auxiliary-
супервизия на self-relevant латент делает возможной функциональную
зависимость policy от этого латента.

**H0.** `tv(B) = tv(D)`, `return(B) ≈ return(D)`.

---

## §2. Требуемые результаты

При обучении PPO с фиксированными гиперпараметрами на 10 seeds:

| ID | Условие |
|---|---|
| R1 | `return(B) − return(D) > 0`, значимо (Wilcoxon, p < 0.0125) |
| R2 | `tv(B) − tv(D) > 0`, значимо (Wilcoxon, p < 0.0125) |
| R3 | `return(B) ≈ return(C)` (TOST, δ = 0.15, α = 0.05) |
| R4 | `tv(B) ≈ tv(C)` (TOST, δ = 0.15, α = 0.05) |
| R5a | `return(B) − return(A) > 0`, значимо (Wilcoxon, p < 0.0125) |
| R5b | `return(B) − return(H) > 0`, значимо (Wilcoxon, p < 0.0125) |
| R6 | `return(B) − return(E) > 0`, значимо (Wilcoxon, p < 0.0125) |

---

## §3. Варианты агентов

| Вариант | gru_priv | head | target | aux в policy |
|---|---|---|---|---|
| A | нет | нет | — | — |
| B | да | [o, c_u, hp] | u_0 | aux |
| C | да | [o, c_w, hp] | w_0 | aux |
| D | да | [o, c_v, hp] | v_0 | aux |
| E | нет | нет | — | 0₄ |
| H | да | [o, noise, hp] | noise_tgt | aux |

---

## §4. Pre-PPO gates

### Step A′ — сенсорные каналы

| Тест | Условие |
|---|---|
| A′-0 | P(z_u=1), P(z_w=1) ∈ [0.45, 0.55] |
| A′-1 | c_u → z_u ≥ 0.95, c_w → z_w ≥ 0.95 |
| A′-2 | c_u → z_w ≤ 0.55, c_w → z_u ≤ 0.55 |
| A′-3 | o_0 → z_u ≤ 0.60, o_0 → z_w ≤ 0.60 |
| A′-4 | c_v → z_u ≤ 0.55, c_v → z_w ≤ 0.55 |
| A′-5 | gap (own − history) ≥ 0.30 |

### Step B′ — head-only

| Вариант | Условие |
|---|---|
| B | aux → z_u ≥ 0.90; shuffle(c) → z_u ≤ 0.60; aux → z_w ≤ 0.60 |
| C | aux → z_w ≥ 0.90; shuffle(c) → z_w ≤ 0.60; aux → z_u ≤ 0.60 |
| D | aux → z_u ≤ 0.60; aux → z_w ≤ 0.60 |

**Если хотя бы один pre-PPO gate FAIL — полный прогон не запускается.**

---

## §5. Sanity gate (bug-check only)

20k шагов, seed 0, варианты B, C, D:

| Проверка | Условие |
|---|---|
| B: representation | aux_u → z_u ≥ 0.85 |
| B: isolation | hv → z_u ≤ 0.60 |
| C: representation | aux_w → z_w ≥ 0.85 |
| C: isolation | hv → z_w ≤ 0.60 |
| D: control | aux_u → z_u ≤ 0.60, aux_w → z_w ≤ 0.60 |
| TV matcher | max skip ≤ 0.05 |
| Stability | нет NaN |

Sanity gate — **только bug-check**. Performance gate — отдельно,
после полного прогона.

---

## §6. G4 permutation control

**Процедура.** Перемешать `c_x` между эпизодами; сохранить `o` и `hp`
исходного эпизода. Измерить argmax action → z.

| Вариант | Порог |
|---|---|
| B | permuted own-bit bacc ≤ 0.55 |
| C | permuted own-bit bacc ≤ 0.55 |
| D | permuted own-bit bacc ≤ 0.55 (контроль) |

**Falsification.** Если permuted own-bit bacc > 0.55 хотя бы на одном
seed — это означает утечку через `o` или `hv`, и H1 не подтверждена.

---

## §7. Критерии PASS / FAIL / INVALID

**PASS.** Все R1–R6 PASS + G4 PASS + все pre-PPO gates PASS.

**FAIL.** `return(B) − return(D) ≤ 0.10` при `aux → z_own ≥ 0.90`.
Это означает: representation работает, policy игнорирует aux.

**INVALID.** 
- `aux → z_own < 0.85` — head не обучилась.
- `hv → z_* > 0.70` — visual leak.
- TV matcher skip > 5%.

---

## §8. STOPPING RULES

1. Если pre-PPO gates FAIL — полный прогон не запускается.
2. Если sanity (bug-check) FAIL — полный прогон не запускается.
3. Если PPO не сходится (`return(B) < 0.15` при `aux → z_own > 0.90`)
   — диагностика, не продолжение.

---

## §9. Что НЕ будет считаться фальсификацией

- Низкий `return(B)` при низком `return(D)` — PPO не сошёлся.
- Разрыв argmax vs stochastic — следствие `ent_coef = 0.01`.
- Разница σ_c = 0.005 vs 0.010 в пределах seed-шума — robustness.
- Прогноз `return(B) ≥ +1.0` не подтвердился (+0.31) — неполная
  сходимость стохастической policy, не провал H1.