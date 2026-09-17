# Affordance-DPI

**Функциональная зависимость политики от self-relevant скрытого латента
через приватный сенсорный канал в обучении с подкреплением.**

Версия: 1.0.0
Дата: 2026-09-17
Лицензия: MIT

---

## О чём это

Этот репозиторий содержит завершённый эксперимент, в котором обученная
PPO политика функционально использует предсказание self-relevant
скрытого латента сильнее, чем предсказание causal-null латента при
равной предсказуемости обоих из истории.

Главный результат — формально установленный барьер и его обход:

**Барьер.** При протоколе `aux = f(o_{0:t})` по data processing
inequality `I(aux; Z) ≤ I(o_{0:t}; Z)`. Если Z не содержится в
истории, auxiliary loss не может передать информацию о Z в policy,
независимо от λ_aux, архитектуры и бюджета.

**Обход.** Приватный сенсорный канал `c_x ∉ o` в forward-входе
auxiliary head. DPI обходится по построению.

---

## Что доказано

### D-P1. Функциональная асимметрия
return(B) − return(D) = +0.313, p = 0.00195, n = 10
TV(B) − TV(D) = +0.237, p = 0.00195, n = 10

text

Здесь B — aux нацелен на causal-positive латент, D — на causal-null.

### D-P2. Причинность (permutation control)

При перемешивании `c_x` между эпизодами (сохранение `o`, `hp` от
исходного эпизода):

| Вариант | normal | perm | drop |
|---|---|---|---|
| B (u) | 0.9481 | 0.5038 | +0.4541 |
| C (w) | 0.9615 | 0.4999 | +0.4507 |
| D (u) | 0.5000 | 0.5000 | 0.0000 |

Порог: permuted ≤ 0.55. Наблюдаемый максимум 0.5311. **PASS.**

### D-P3. Изоляция visual-ветви

Post-PPO: `hv → z_* ≤ 0.5666`, `aux → z_own ≥ 0.9414`. Разрыв > 0.37.
Визуальная ветвь не имеет shortcut.

### D-P4. Симметрия s↔w

`|return(B) − return(C)| = 0.014`, TOST PASS. Эффект не специфичен
для «self», он специфичен для causal-positive vs causal-null.

### D-P5. Устойчивость

Механизм устойчив к 2-кратному изменению шума сенсора:
SNR ∈ [14.5, 29], σ_c ∈ {0.005, 0.010}. Все R-тесты PASS в обоих
режимах.

---

## Что НЕ доказано

- T = 1. Только одношаговая причинная зависимость.
- `z` не зависит от действий. `B_U[0] = B_U[1]`.
- Более шумные сенсоры (SNR < 14.5) в полном прогоне не тестировались.
- `gru_priv = 8` фиксировано.

---

## Структура репозитория
affordance-dpi/
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── requirements.txt
├── Makefile
├── docs/
│ ├── MANIFEST.md — полная спецификация эксперимента
│ ├── HANDOFF.md — постановка и финальное решение
│ ├── KONSРЕКТ.md — история 15 итераций
│ ├── PREREGISTRATION.md — preregistered критерии v5.3
│ ├── NEGATIVE_RESULTS.md — 15 провалов с диагнозами
│ └── DESIGN_JOURNAL.md — хронология решений
├── src/
│ ├── env/ — среда v4.2 + OOS-валидация
│ ├── probes/ — диагностические probes
│ ├── gates/ — pre-PPO gates
│ ├── training/ — PPO-обучение
│ └── control/ — G4 permutation control
├── results/ — JSON-результаты
└── traces/ — per-iteration трассировки B и C

text

---

## Как воспроизвести

### 1. Установка

```bash
pip install -r requirements.txt
Или через conda:

bash
conda env create -f environment.yml
conda activate affordance-dpi
2. Pre-PPO gates
bash
python src/gates/step_a_prime_v5_3_sigma010.py   # сенсорные каналы
python src/gates/step_b_prime_v5_3_sigma010.py   # head-only
Ожидаемые результаты: все gates PASS (см. docs/PREREGISTRATION.md).

3. Полное обучение
bash
python src/training/v5_3_train_sigma010.py
Время: 6–8 часов на CPU. Результат: results/v5_3_results_sigma010.json.

4. Permutation control
bash
python src/control/g4_permutation_sigma010.py
Результат: results/v5_3_g4_results_sigma010.json.

5. Проверка воспроизводимости
bash
bash scripts/verify_reproducibility.sh
Ключевые численные вехи
text
DPI-барьер (v5.3.4):       aux = f(history) ⇒ I(aux; Z) = 0
classify_probe:            hist_full → XOR = 0.8691 (порог < 0.55)
Kalman residual:           H_full → sign(r_s) = 0.9899 (порог < 0.55)
Дизайн-фикс:               aux = F(o, c_x, hp), c_x ∉ o

Pre-PPO gates (σ_c = 0.010):
  A′-1: c_u → z_u = 0.9784
  A′-5: gap_u = +0.4600
  B′:   aux → z_u = 0.9782

Полный прогон (10 seeds × 6 вариантов × 200k × 2 σ_c):
  return(B) = +0.305    return(D) = −0.016
  return(C) = +0.306    return(A) = −0.018
  tv(B) = 0.2548        tv(D) = 0.0170
  tv(C) = 0.2419

R1: B−D = +0.313, p = 0.00195
R2: TV B−D = +0.2367, p = 0.00195
R3: TOST |B−C| = −0.014, p = 0.0000
R4: TOST |TV B−C| = +0.0024, p = 0.0000
R5a: B−A = +0.320, p = 0.00195
R5b: B−H = +0.310, p = 0.00195
R6: B−E = +0.296, p = 0.00195

G4 permutation:
  B: normal 0.9481 → perm 0.5038, drop +0.4541
  C: normal 0.9615 → perm 0.4999, drop +0.4507
  D: normal 0.5000 → perm 0.5000, drop  0.0000

Все 7 R-тестов PASS × 2 σ_c-режима
G4 PASS × 2 σ_c-режима
Цитирование
Если используете этот артефакт — см. CITATION.cff. Если используете
идею DPI-барьера для auxiliary supervision в RL — тоже цитируйте.

Лицензия
MIT. См. LICENSE.

text

---

## §2. `CITATION.cff`

```yaml
cff-version: 1.2.0
message: "Если вы используете этот артефакт, цитируйте его следующим образом."
authors:
  - family-names: "Фамилия"
    given-names: "Имя"
    orcid: "https://orcid.org/0000-0000-0000-0000"
title: "Affordance-DPI: функциональная зависимость политики от self-relevant скрытого латента"
version: 1.0.0
date-released: 2026-09-17
license: MIT
repository-code: "https://github.com/USERNAME/affordance-dpi"
keywords:
  - обучение с подкреплением
  - вспомогательная супервизия
  - теория информации
  - data processing inequality
  - PPO
  - приватный канал
Замени USERNAME, Фамилия, Имя, ORCID.

Если ORCID нет — получи за 5 минут на orcid.org. Это единственный
надёжный идентификатор автора в науке.