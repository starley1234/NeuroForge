# NEXUS-Capital — архитектурные детали реализации

В этом документе описано, как разделы спецификации отражены в коде.

## Честность и валидация (важные оговорки)

Чтобы архитектурные компоненты не создавали иллюзию точности, внесены
следующие инженерные гарантии:

1. **xVal для чисел.** Вместо случайных RFF используется мультипликативный
   xVal (`core/layers.py`): один обучаемый вектор `<num>`, масштабируемый
   `sign·log1p(|x|)`. Это сохраняет алгебраический масштаб и знак.
2. **FastWeightMemory (а не «настоящий TTT»).** Слой переименован из
   `TTTRegimeMemory` в `FastWeightMemory`; это аппроксимация TTT-Linear
   (Sun et al. 2024) с одношаговым обновлением и обучаемым per-head lr.
   Полноценный TTT требует кастомных ядер. Алиас `TTTRegimeMemory` сохранён.
3. **Калибровка Neural SDE обязательна.** `calibrate_to_returns()` подгоняет
   `cal_mu/cal_sigma` и степени свободы Стьюдента (толстые хвосты) под
   реальные возвраты. Пока флаг `calibrated=False`, VaR/ES в выдаче помечены
   как surrogate. В тренировочном скрипте калибровка вызывается перед
   обучением.
4. **Хронологический сплит.** `RealMarketDataset.chronological_split()`
   режет train/val по времени, без `random_split` — иначе утечка будущего.
5. **Бейзлайны.** `training/baselines.py` считает zero/mean/AR(1) на
   валидации — без сравнения с наивом заявлять о преимуществе нельзя.
6. **Активный L_balance.** `FieldReconstructionHead` (`utility/balance.py`)
   ПРЕДСКАЗЫВАЕТ поля из латентности, и балансовые тождества
   `GP = Rev - COGS`, `Assets = L + E` проверяются по предсказаниям, а не
   по уже поданным на вход числам.
7. **Cross-attention фьюжн.** `MultimodalFusion` больше не усредняет
   разнородные модальности; learnable query аккумулирует их через
   multihead cross-attention.

## Уровень 1. Непрерывные экономические энкодеры

| Модальность | Файл | Класс | Ключевая идея |
|---|---|---|---|
| Микроструктура L2/L3, тики | `encoders/orderbook.py` | `OrderBookEncoder`, `SSMHiPPO` | Цены/объёмы не токенизируются; последовательность тиков проходит через диагонализованный SSM (S4-подобный) с HiPPO-инициализацией, O(T) и константной памятью на шаг |
| Корпоративная отчётность / ERP | `encoders/structured.py` | `StructuredEncoder` | Каждое поле получает field-embedding + `sign·log1p(|x|)` проекцию значения; self-attention по полям; маскированный mean-pool |
| Графы поставок / B2B | `encoders/graph.py` | `GraphRiskOperator` | Диффузия риска по рёбрам через GraphSAGE с edge-фичами (CDS, задержки); голова `default_prob` |
| Аудио конференций | `encoders/audio.py` | `AudioStressEncoder` | Свёрточная пирамида по мел-спектрограмме + голова стресса (микротремор/паузы) |
| Новости / текст | `encoders/text.py` | `FinancialTextEncoder` | Числа в тексте детектируются регэкспом и проецируются в непрерывный эмбеддинг через Random Fourier Features, а не как BPE-токены |

Все энкодеры возвращают векторы размерности `d_value` и сливаются через
`MultimodalFusion` с обучаемым gating.

## Уровень 2. Unified Value Bus

`core/value_bus.py` — стопка `NexusCoreBlock`, каждый из которых
последовательно применяет:

1. **TTT-Regime Memory** (`core/ttt_memory.py`) — быстрая матрица
   $\mathcal{M}_t$ с замкнутым обновлением. Поддерживает low-rank
   факторизацию `A·B` для экономии VRAM и pull-back к базовым весам, чтобы
   состояние не улетало. Состояние детэчится на каждом шаге, поэтому
   стриминг 100k тиков не растит граф.
2. **Local Financial Attention** (`core/attention.py`) — слайдинговое окно
   `W=2048`, относительное позиционное смещение и `value_gate`, усиливающий
   экономически важные позиции.
3. **Financial Sparse MoE** (`core/moe.py`) — top-2 маршрутизация среди 8
   экспертов, балансировочный aux-loss (Switch Transformer) и общий
   остаточный SwiGLU-эксперт.

Поверх стека `EconomicInvariantHead` считывает CF, σ, дисконт и эластичность
— они используются как вспомогательный обучающий сигнал и как входы в loss.

## Уровень 3. Latent Economic Workspace

`workspace/neural_sde.py` реализует `dx = μ(x,t)dt + σ(x,t)dW` с
положительной волатильностью через softplus. Симуляция векторизована по
батчу и по путям (`paths=10000`).

`workspace/monte_carlo.py`:
- `value_at_risk` / `expected_shortfall` — эмпирические квантили P&L;
- `default_probability` — доля путей с P&L < порога;
- дисконтирование через `e^{-rt}`.

`workspace/game_theory.py`:
- `nash_bargaining` — деление излишка по переговорной силе;
- `bertrand_price` — оптимальная цена в логит-модели через L-BFGS;
- `find_nash_equilibrium` — fictitious play для матричных игр,
  возвращает ε-зазор равновесия.

`Workspace.reflect_loop` несколько раз прогоняет Монте-Карло и впрыскивает
риск-метрики обратно в латентность — это и есть «размышление о рисках».

## Уровень 4. Dual Output Engine

`outputs/engine.py`:
- `DiscreteOutputHead` — логиты по словарю для документов/решений;
- `ContinuousEconomicHead` — цена, веса портфеля (symплекс через softmax),
  вероятность дефолта, спред ликвидности и `UtilityLayer`.

## Функция полезности и совокупный loss

`utility/utility_layer.py`:
$$U = \mathbb{E}[R] - \tfrac{\gamma}{2}\mathrm{Var}(R) - \mathrm{Cost}$$

`utility/losses.py` собирает:
```
L_total = L_pred
        - λ_utility · E[U]
        + λ_var · VaR_α
        + λ_balance · L_balance
        + 0.01 · L_aux(MoE)
```
`L_balance` проверяет балансовые тождества и тождество
`Gross Profit = Revenue − COGS` в log-чувствительном пространстве.

## Data Flywheel

- `data/edgar_parser.py` — регэкспы по 17 полям отчётности (включая русский
  язык), парсер убытка в скобках `(120.4)`, и синтетический генератор
  отчётов, **соблюдающий балансовые тождества**.
- `data/synthetic_market.py` — 7 шаблонов шоков (инфляция, разрыв поставок,
  рост ставки, санкции, демпинг, бум спроса, кризис ликвидности) со
  случайной амплитудой.
- `data/self_play.py` — B2B переговоры (концессии к резерву) и
  Market-Maker против Арбитражёра; каждый эпизод даёт кортеж
  `{контекст, последовательность офферов, P&L, риск}`.

### Реальные открытые данные (`data/real/`)

- `binance.py` — прямые ZIP с `data.binance.vision` (klines 1s..1mo,
  aggTrades) без API-ключа; `klines_to_tick_features` формирует 8-мерный
  тик-признак; `synthetic_book_from_klines` восстанавливает L2 вокруг mid.
- `sec_edgar.py` — поквартальные XBRL bulk-выгрузки SEC (public domain),
  маппинг US-GAAP-тегов на внутренние поля юнит-экономики, контроль
  балансового тождества.
- `fred.py` — макро-ряды FRED (DFF, DGS10, T10Y2Y, CPIAUCSL, VIXCLS,
  HY-спред, USD-индекс, UNRATE) через бесплатный API; при отсутствии ключа —
  детерминированный оффлайн-профиль реальных средних.
- `real_dataset.py` — `RealMarketDataset`/`collate_real`, объединяющий все
  три источника в батчи для модели, с автоматическим фоллбэком на
  оффлайн-OHLCV при недоступности сети.

Процедура обучения на этих данных — `scripts/train_real_data.py` (см. README,
раздел «Обучение на реальных открытых данных»).

## Обучение

`training/trainer.py` — поддерживает AMP и опционально 8-bit AdamW через
`bitsandbytes`. `training/grpo.py` реализует Group Relative Policy
Optimization с экономическими наградами (+2.0 P&L, +1.5 VaR, +1.0 баланс,
−5.0 дефолт) и group-normalized advantage.

## Соответствие бюджету VRAM

Бюджет считается в `NexusConfig.estimate_params()`. Активации при стриминге
тиков не накапливаются, потому что TTT-состояние отвязывается от графа на
каждом шаге (`state.detach()`), поэтому поток в 100k+ событий укладывается в
константную память.
