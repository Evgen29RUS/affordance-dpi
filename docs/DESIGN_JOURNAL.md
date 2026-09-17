# Design Journal

Хронология ключевых решений при построении эксперимента v5.3.

---

## День 1. Постановка задачи

Задача: построить среду + протокол агентов, в которых PPO с
фиксированными гиперпараметрами достигает функциональной зависимости
policy от self-relevant скрытого латента.

Требования: R1–R6, D1–D4 (§7 хендоффа).

---

## День 2–3. Отбраковка внешних идей

13 внешних идей отвергнуты как уже существующие:
Physical AI OS, Reality Compiler, Physical Commit, Evidence
Independence, Confidence Laundering, Semantic Laundering, Epistemic
Handover, Experience Graph, Failure Capital, Causal Reality Testing,
Identifiability-gated commit, Independence Budget, Active experiment
selection.

Выжил один кандидат: функциональная зависимость policy от
self-relevant латента при matched предсказуемости с causal-null
контролем.

---

## День 4. Agent-based simulation (v1–v3)

Попытка построить симуляцию с потребностями, навыками, священными
текстами. Отвергнуто: это Sugarscape (Epstein & Axtell, 1996).

---

## День 5. EIGEA / Reality Runtime

Протокол авторизации физических действий. Coupling выведен из аксиом,
5 тестов PASSED. Отвергнуто: coupling вписан в код, все предсказания
выполняются по построению. Ablation → tv = 0.

---

## День 6. Среда v4.0

`q, s, w, v` как независимые AR(1). Провал: E2_sv структурно
недостижим (causal-null не может быть matched с causal-positive).

---

## День 7. Среда v4.2

Введение q_t как общего контекста. Ключевое изменение:
causal-null ≠ statistically-unpredictable. PASSED all E1–E4.

---

## День 8. Среда v4.3 OOS

OOS-валидация на seeds 6–10. PASSED. Среда заморожена.

---

## День 9–15. Агенты v5.3.1–v5.3.6

Шесть архитектурных итераций, все дали `tv ≈ 0.001`.

**v5.3.1.** Shared GRU, λ_aux=1. Head не училась.
**v5.3.2.** Sigmoid head, λ_aux=300, H-target noise. Head училась
частично.
**v5.3.3.** BPTT через aux-loss. Head достигла oracle.
**v5.3.4.** Split GRU. Head училась, tv ≈ 0.003.
**v5.3.5.** Удалён h_vis. return(A) упал.
**v5.3.6.** Reward scaling, ent=0.005. Обучение сломано.

---

## День 16. Аналитический прорыв

Формально установлен DPI-барьер:
aux = F(o_{0:t}) ⇒ I(aux; Z) ≤ I(o_{0:t}; Z)

text

Если Z не в истории — aux не может его нести.

---

## День 17. classify_probe

Проба на v4.3:
hist_full → XOR = 0.8691 (порог < 0.55)

text

Провал. XOR в истории.

---

## День 18. Kalman residual

Попытка построить residual target.
H_full → sign(r_s) = 0.9899 (порог < 0.55)

text

Провал. Residual предсказуем из истории.

---

## День 19. Введение приватного канала

Дизайн-фикс:
aux = F(o, c_x, hp), c_x ∉ o

text

DPI обойдён по построению.

---

## День 20. Step A′

Pre-PPO gates на сенсорные каналы. Все PASS в двух σ_c-режимах.

---

## День 21. Step B′

Head-only training. Все PASS.

---

## День 22. Sanity

Bug-check на 20k. Все PASS.

---

## День 23–25. Полный прогон

10 seeds × 6 вариантов × 200k × 2 σ_c. Все R1–R6 PASS в обоих
режимах.

---

## День 26. G4 permutation control

Drop ≥ +0.44 в обоих режимах. PASS.

---

## Итог

H1 подтверждена для T = 1, z не зависит от action, SNR ∈ [14.5, 29].