# NEXUS-Capital (VALUEX)

**Value-Augmented Latent Universal Exchange & Economic Reasoning Engine**

Экономически заземлённая мультимодальная архитектура ИИ для финансов и
бизнеса. В отличие от обычных LLM, которые воспринимают `"$1 000 000"` как
последовательность токенов, NEXUS-Capital проецирует деньги, риск,
себестоимость и время в **непрерывное латентное пространство стоимости**
$\mathbb{R}^{d_{\text{value}}}$ и оптимизирует непосредственно
**функцию полезности** (ожидаемая маржа − штраф за риск − издержки).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-30%20passed-brightgreen.svg)](#тесты)

---

## 1. Системные тупики LLM в финансах и как они решены

| Проблема классических моделей | Решение в NEXUS-Capital |
|---|---|
| **Токенизация чисел** — модель не понимает масштаба и вероятностей | Числовые и векторные поля кодируются в **непрерывные тензоры стоимости** (`ContinuousProjector`, Random Fourier Features поверх `log1p`) |
| **Отсутствие ценности** — генерация без маржи, ROI и риска | **Функция полезности** фон Неймана–Моргенштерна встроена в ядро и функцию потерь: `L = L_pred − λ₁·Utility + λ₂·VaR` |
| **Слепота к смене режима** — веса помнят полугодовую давность | **Linear Fast-Weight Memory (TTT)**: быстрая матрица `M_t` обновляется на лету замкнутым градиентом, константная память на тик |
| **Медленный CoT для сценариев** — тысячи токенов на stress-test | **Latent Monte-Carlo** на дифференцируемом **Neural SDE**: тысячи симуляций в скрытом пространстве + расчёт равновесий Неша |

---

## 2. Архитектура: 4 уровня

```
 УРОВЕНЬ 1: Непрерывные экономические энкодеры
   ┌──────────────┬──────────────┬──────────────┬──────────────┐
   │ OrderBook SSM│ Structured   │ Graph Neural │ Audio-Stress │  + Text
   │ L2/L3 тики   │ Tabular/AST  │ Operator     │ Encoder      │
   └──────┬───────┴──────┬───────┴──────┬───────┴──────┬───────┘
          └──────────────┴──────┬───────┴──────────────┘
                                ▼
 УРОВЕНЬ 2: Unified Value Bus (R^d_value)
   • TTT Market Regime Memory   • Local Financial Attention (W=2048)
   • Financial Sparse MoE (8 экспертов, активны 2)
   • Инварианты: CF, σ, e^{-rt}, E_d
                                ▼
 УРОВЕНЬ 3: Latent Economic Workspace
   • Neural SDE (dx = μ dt + σ dW)
   • Латентный Монте-Карло: VaR, Expected Shortfall, P(default)
   • Теория игр: Nash bargaining, Бертран, ε-Nash
                                ▼
 УРОВЕНЬ 4: Dual Output Engine
   [Дискретные документы/решения]   [Непрерывные экон. векторы]
   меморандумы, смарт-контракты      цены, веса портфеля, PD, спред
```

---

## 3. Структура репозитория

```
NEXUS-Capital/
├── nexus_capital/
│   ├── core/                  # Конфиг, TTT, внимание, MoE, Value Bus
│   │   ├── config.py          #   дата-класс + пресеты small/rtx5060
│   │   ├── layers.py          #   RMSNorm, SwiGLU, ContinuousProjector
│   │   ├── ttt_memory.py      #   Linear Fast-Weight Memory
│   │   ├── attention.py       #   Local Financial Attention (W=2048)
│   │   ├── moe.py             #   Sparse MoE (8 экспертов)
│   │   ├── value_bus.py       #   стек ядра + экономические инварианты
│   │   └── loader.py          #   загрузка YAML-конфигов
│   ├── encoders/              # Уровень 1: 5 модальностей
│   │   ├── orderbook.py       #   SSM-энкодер стакана L2/L3
│   │   ├── structured.py      #   отчётность, юнит-экономика, ERP
│   │   ├── graph.py           #   контрагенты, цепочки поставок, CDS
│   │   ├── audio.py           #   стресс по речи earnings calls
│   │   └── text.py            #   новости + внедрение чисел в эмбеддинги
│   ├── workspace/             # Уровень 3
│   │   ├── neural_sde.py      #   дифференцируемый суррогат рынка
│   │   ├── monte_carlo.py     #   VaR, ES, P(default)
│   │   ├── game_theory.py     #   Nash, Бертран, fictitious play
│   │   └── workspace.py       #   контур рефлексии о рисках
│   ├── utility/               # Функция полезности и совокупный loss
│   ├── outputs/               # Dual Output Engine
│   ├── data/                  # Data Flywheel
│   │   ├── orderbook_stream.py#   синтетика микроструктуры
│   │   ├── edgar_parser.py    #   парсер 10-K/10-Q + синтетические отчёты
│   │   ├── synthetic_market.py#   генератор рыночных стрессов
│   │   └── self_play.py       #   мультиагентный self-play (B2B, MM/арбитраж)
│   ├── models/nexus_model.py  # Сборка всей модели
│   └── training/              # Тренер + GRPO (Value-RL) + чекпоинты
├── configs/                   # small.yaml, rtx5060.yaml
├── examples/                  # demo_pipeline.py, demo_monte_carlo.py
├── scripts/train.py           # Цикл предобучения
└── tests/                     # 30 тестов
```

---

## 4. Быстрый старт

```bash
cd NEXUS-Capital
pip install -r requirements.txt

# Полный прогон всех 4 уровней на CPU (синтетические данные)
python examples/demo_pipeline.py

# Латентный Монте-Карло: распределение P&L, VaR, ES
python examples/demo_monte_carlo.py

# Цикл обучения (Фазы 1–3) на синтетике
python scripts/train.py --steps 100 --batch-size 4

# Тесты
pytest -q
```

## 5. Обучение на реальных открытых данных

Полный список источников со ссылками и лицензиями — в
[`nexus_capital/data/sources.md`](nexus_capital/data/sources.md).
Из коробки поддерживаются три бесплатных источника:

| Модальность | Источник | Ключ | Коннектор |
|---|---|---|---|
| Микроструктура (тики/OHLCV) | Binance Public Data `data.binance.vision` | не нужен | `data/real/binance.py` |
| Корпоративная отчётность | SEC EDGAR XBRL bulk (public domain) | не нужен | `data/real/sec_edgar.py` |
| Макро (ставки, CPI, VIX, спреды, FX) | FRED | бесплатный ключ `FRED_API_KEY` | `data/real/fred.py` |

### Простая процедура

```bash
# 1. (опционально) бесплатный FRED-ключ с https://fred.stlouisfed.org/docs/api/api_key.html
export FRED_API_KEY=ваш_ключ

# 2. Минимальный запуск на CPU: BTCUSDT 1m, Q1 2024, без SEC
python scripts/train_real_data.py \
    --symbol BTCUSDT --interval 1m \
    --start-month 2024-01 --end-month 2024-03 \
    --steps 100 --batch-size 4 --mc-paths 64

# 3. С реальной отчётностью SEC (10-K/10-Q) за 2023–2024
python scripts/train_real_data.py --with-sec --sec-years 2023 2024 \
    --steps 300 --batch-size 4

# 4. На GPU с конфигом под RTX 5060
python scripts/train_real_data.py --config configs/rtx5060.yaml \
    --device cuda --batch-size 2 --mc-paths 256 --steps 1000
```

Скрипт делает следующее:
1. **Качает** klines Binance (ZIP-CSV прямым HTTPS, кэш в `data/raw/binance`),
   при `--with-sec` — поквартальные XBRL-выгрузки SEC, и макро FRED;
2. **Готовит** батчи: последовательность тиков + L2-стакан (восстанавливается
   вокруг mid из OHLCV) + поля юнит-экономики + макро-вектор;
3. **Обучает** ядро с `L_total = L_pred − λ₁·Utility + λ₂·VaR + L_aux`;
4. **Оценивает** на hold-out: MSE прогноза доходности, sign-accuracy, Sharpe,
   средний VaR₅%;
5. **Сохраняет** чекпоинт в `checkpoints/nexus_real/`.

Если сеть недоступна (например, в изолированной среде), коннектор Binance
автоматически переключается на детерминированный оффлайн-OHLCV, а FRED — на
оффлайн AR(1)-профиль реальных средних, поэтому процедура обучения
запускается в любом случае. Для настоящей торговли/исследований
используйте реальные данные и при необходимости настоящие L2/L3 через
Tardis.dev или биржевые веб-сокеты (см. sources.md).

### Целевая задача
По умолчанию модель учится предсказывать **следующую доходность** как
непрерывную экономическую величину, одновременно максимизируя ожидаемую
полезность и штрафуя VaR. В режиме `--task pricing` это соответствует
динамическому ценообразованию; в `--task invariants` — восстановлению
экономических инвариантов (CF, σ, дисконт, эластичность).

### Двуязычное обучение (русский + английский) и предобученный словарь

Словарь **не обучается заново** — используется предобученный
**SentencePiece BPE из XLM-RoBERTa** (`xlm-roberta-base`, ~250 000 токенов),
который из коробки покрывает и русский, и английский общим алфавитом.
Обучается только embedding-матрица (размерность `text_embed_dim=256` с
проекцией в `d_value`) и остальная модель.

```bash
# XLM-R скачается один раз при первом запуске (нужен интернет).
# Двуязычный LM-этап на финансовом корпусе RU+EN:
python scripts/train_real_data.py \
    --languages en ru --text-steps 300 --text-batch-size 8 \
    --steps 200 --batch-size 4

# Принудительно встроенный BPE (без загрузки XLM-R, для оффлайна/CI):
python scripts/train_real_data.py --offline-tokenizer --text-steps 100
```

Корпус (`nexus_capital/data/real/corpus.py`):
- **EN** — встроенные финансовые фразы + при `--with-sec-text` реальный
  текст SEC 10-K/10-Q (Apple, Microsoft, Alphabet, Amazon, Meta),
  public domain;
- **RU** — встроенный корпус балансов, P&L, ставок ЦБ, рыночных новостей;
  пользовательские `.txt` можно положить в `data/raw/corpus/ru/`.

На текстовом этапе считается next-token loss, а также экономические члены
(Utility/VaR/MoE) по пулированному представлению предложения. После него
идёт рыночный этап на тиках Binance/синтетике.

### Минимальный пример

```python
import torch
from nexus_capital import small_config, build_model
from nexus_capital.data.orderbook_stream import collate_book_ticks
from nexus_capital.data.edgar_parser import (
    synthetic_filing, fields_to_tensor, DEFAULT_UNIT_FIELDS)
import numpy as np

cfg = small_config()
model = build_model(cfg).eval()

bt = collate_book_ticks(batch_size=2, n_levels=16, T=32, n_features=8)
rng = np.random.default_rng(0)
fields, masks = [], []
for _ in range(2):
    f, m = fields_to_tensor(synthetic_filing(rng),
                            DEFAULT_UNIT_FIELDS[:16])
    fields.append(f); masks.append(m)

with torch.no_grad():
    out = model(
        book=bt["book"], ticks=bt["ticks"],
        fields=torch.stack(fields), field_mask=torch.stack(masks),
        tokens=torch.randint(0, cfg.vocab_size, (2, 32)),
        workspace_mode="risk",
    )

print(out["workspace"]["risk"]["var"])            # VaR_5%
print(out["economic"]["portfolio_weights"].sum()) # веса шортрат=1
```

---

## 6. Ключевая математика

### Функция полезности (в ядре и loss)
$$U(w,\text{Risk}) = \mathbb{E}[R(w)] - \frac{\gamma}{2}\operatorname{Var}(R(w)) - \operatorname{Cost}(w)$$

### TTT быстрая память рыночного режима
$$\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} - \eta_t\nabla_{\mathcal{M}}\|\mathcal{M}_{t-1}k_{\text{market}}-v_{\text{spread}}\|^2$$
Градиент имеет замкнутую форму $q(M^Tq-v)^T$ — обновление O(d_mem·d_value) без обратного прохода графа.

### Neural SDE (суррогат рынка)
$$dx = \mu(x,t)\,dt + \sigma(x,t)\,dW, \qquad \text{Эйлер–Маруяма, батч по 10 000 путей}$$

### Совокупная функция потерь
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{pred}} - \lambda_1\,\text{Utility}(x) + \lambda_2\,\text{VaR}_\alpha(x) + \mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{aux}}$$

`L_balance` штрафует за нарушение балансовых тождеств:
`Assets = Liabilities + Equity` и `Gross Profit = Revenue − COGS`.

### Награды Value-RL (GRPO)
| Событие | Награда |
|---|---|
| Рост маржи/P&L | **+2.0** |
| Снижение VaR | **+1.5** |
| Точность баланса | **+1.0** |
| Дефолт | **−5.0** |

---

## 7. Бюджет для RTX 5060 (16 ГБ)

Целевая конфигурация в `configs/rtx5060.yaml`:

| Компонент | Конфигурация | VRAM |
|---|---|---|
| Активное ядро | d_value=1536, 24 слоя | ~2.8 ГБ |
| Sparse MoE | 8 экспертов, активны 2 | ~4.4 ГБ |
| Neural SDE | 20M параметров | ~0.1 ГБ |
| 8-bit AdamW/GaLore | состояние оптимизатора | ~2.8 ГБ |
| Активации/буфер стакана | batch 2–4, streaming | ~3.4 ГБ |
| **Итого** | | **~13.5–14.5 ГБ** |

---

## 8. Дорожная карта (Фазы MVP)

1. **TTT для финансовых потоков** — `ttt_memory.py`, стриминг 100k событий без роста VRAM ✅
2. **Data Engine & SDE** — парсер EDGAR, синтетика стрессов, Neural SDE ✅
3. **Pre-train & Latent Workspace** — предобучение ядра, итерации рефлексии ✅
4. **Value-RL (GRPO)** — награды на P&L/риск/баланс/дефолт ✅

Прикладные возможности: динамическое ценообразование B2B/e-commerce, агент
закупок с равновесием Неша, стресс-тестирование портфелей и бизнеса.

---

## 9. Тесты

```bash
pytest -q
# 37 passed
```

Покрытие: TTT-память, локальное внимание, маршрутизация MoE, инварианты Value
Bus, Neural SDE (с backward), Монте-Карло VaR/ES, Nash bargaining, Бертран,
ε-Nash, полный forward/backward модели, Data Flywheel, utility и GRPO,
а также коннекторы реальных данных (Binance/SEC/FRED) в оффлайн-режиме.

---

## Лицензия

MIT
