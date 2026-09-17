# Changelog

Все значимые изменения фиксируются в этом файле.
Формат: [Keep a Changelog](https://keepachangelog.com/).
Версионирование: [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-09-17

### Добавлено

- Первый релиз эксперимента v5.3.
- Спецификация среды v4.2 (frozen), в том числе:
  - Динамика s, w, v, q.
  - Задача two-lock symmetric, T = 1.
  - Приватные каналы c_u, c_w, c_v.
- Архитектура агента:
  - `gru_vis`, `gru_priv`, head(o, c, hp), actor([hv, aux]).
- Pre-PPO gates (Step A′ и Step B′) в двух σ_c-режимах.
- PPO-обучение: 10 seeds × 6 вариантов × 200k шагов × 2 σ_c.
- G4 permutation control.
- Полные результаты в JSON.
- Документы: MANIFEST, HANDOFF, KONSРЕКТ, PREREGISTRATION,
  NEGATIVE_RESULTS, DESIGN_JOURNAL.

### Доказано

- `return(B) − return(D) = +0.313`, p = 0.00195, n = 10.
- G4 permutation drop ≥ +0.44.
- `hv → z_*` ≤ 0.5666 (нет visual leak).
- Устойчивость в SNR ∈ [14.5, 29].

### Не доказано

- T > 1.
- Action-dependent z.
- SNR < 14.5.
- Другие размерности `gru_priv`.