# NEXUS-Engine

**N**on-linear **E**pisodic e**X**tensible **U**nified **S**ystem for Physical Intelligence & Engineering — референсная реализация архитектуры **UniPhysical-Latent Framework**: непрерывные динамические энкодеры, физически инвариантное латентное ядро с $O(1)$ памятью и двухрежимный вывод «код + поля».

Проект работает **на CPU из коробки** и масштабируется до профиля обучения на одной RTX 5060 16 ГБ.

```bash
./nexus.sh setup        # окружение + самопроверка
./nexus.sh quickstart   # данные → токенизатор → обучение → приёмка (~20 с)
./nexus.sh serve        # HTTP API на http://localhost:8000
```

Пошагово и с разбором проблем — [docs/QUICKSTART.md](docs/QUICKSTART.md).

```
Уровень 1  Continuous Dynamic Encoders   AST-BPE · SSM-Audio/Video · B-Rep GNO · Point-SSM · Event-ODE · Neural-ODE био
Уровень 2  Physical World Core           Unified Latent Bus (t, xyz, масса, силы) → TTT → Sliding Attention → Sparse MoE → Latent Reasoning + FNO-критик
Уровень 3  Dual Output Engine            дискретно: OpenSCAD / текст / код   ·   непрерывно: поля FEM/CFD, траектории, G-код
```

---

## 1. Что здесь реально работает

| Тезис спецификации | Реализация | Где посмотреть |
| :-- | :-- | :-- |
| **Tokenization Tax** — 90 % вычислений на статический шум | Непрерывные SSM/ODE-энкодеры по реальному $\Delta t$ + «дельта новизны» вместо патчей | `nexus/encoders/base.py::GatedDeltaSSM`, `novelty_delta` |
| **Потеря причинности** | Непрерывная ось времени $t$ в секундах, а не индекс токена; шина сортирует события по физическому времени | `nexus/layers/common.py::ContinuousTimeEmbedding`, `nexus/bus.py` |
| **Слепота к 3D и физике** | SDF/CSG-ядро: объём, масса, центр масс, тензор инерции, полости, толщина стенки, свесы, достижимость фрезой; B-Rep/CSG граф; **настоящий МКЭ** на гексаэдрах (matrix-free CG), проверенный на аналитике | `nexus/geometry/*`, `nexus/scad/parser.py`, `nexus/fem/hex_fem.py` |
| **Квадратичный KV-кэш** | Linear Fast-Weights (TTT): состояние фиксированного размера — на 64k контекста ×1710 компрессии против KV при recall 0.45 даже у необученной памяти | `nexus/layers/ttt.py`, `nexus/eval/needle.py` |
| **Многословный CoT** | Latent Reasoning Loop с Adaptive Pondering и физическим зондом FNO | `nexus/reasoning/workspace.py` |
| **OpenSCAD Data Flywheel** | Генератор вариаций → headless-рендер → аудит → пакетный FEM → обучающий кортеж | `nexus/scad/generator.py`, `nexus/data/flywheel.py` |
| **Physics-RL без критика-сети** | GRPO с наградами +1.0 компиляция / +1.5 manifold / +2.0 прочность | `nexus/training/grpo.py`, `nexus/training/rewards.py` |

Внешние зависимости — только `torch` и `numpy`. `openscad` и `CalculiX (ccx)` подхватываются автоматически, **если установлены**; иначе используются встроенные CSG- и load-path-движки, поэтому маховик данных крутится и в CI без GUI.

---

## 2. Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # или: pip install -r requirements.txt

pytest -q                        # 78 тестов, ~40 с на CPU
python -m nexus.cli demo         # сквозная демонстрация всех трёх уровней
```

Сквозной демо-прогон (`nexus demo`) печатает примерно следующее:

```
── 1. Инженерный контур: OpenSCAD → CSG → аудит → FEM ──
   деталь: bearing_block, масса 210.4 г, σmax 12.7 МПа, запас 16.9, награда +4.50
── 2. Мультимодальный вход: 5 классов сигналов на одну шину ──
   пакеты: {'brep': 7, 'audio': 20, 'gaussian': 24, 'event': 32, 'ecg': 16, 'stress': 8}
── 3. Ядро + латентное рассуждение + двухрежимный вывод ──
   латентных шагов рассуждения: 2, выход: logits (1, 538, 512), поле (1, 1, 8, 8, 8)
── 4. O(1) память (Needle-in-a-Haystack) ──
   контекст 8192: состояние 0.041 МБ против 9.0 МБ у KV-кэша (×219.4)
── 5. Маховик данных ──
   {'total': 10, 'compiled': 10, 'manifold': 9, 'fem_passed': 9, 'yield_rate': 0.9}
── 6. Бюджет VRAM для RTX 5060 (16 ГБ) ──
   {'core_gb': 2.54, 'moe_gb': 6.75, 'optimizer_gb': 2.54, 'total_gb': 12.21, 'fits_16gb': True}
```

### CLI

```bash
nexus doctor                             # диагностика окружения и самопроверка
nexus quickstart --scale small           # всё сразу: данные → токенизатор → обучение → приёмка
nexus info --preset rtx5060 --build      # конфиг, число параметров, бюджет VRAM, доступные бэкенды
nexus train-tokenizer --source dir:./corpus --vocab-size 8192   # свой BPE
nexus train-lm --source dir:./corpus     # обучение на стандартном датасете
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode logit   # дистилляция из LLM
nexus eval --ref latest --baseline production --gate     # приёмочные тесты + промоушен
nexus registry list | history | promote | rollback | prune
nexus serve --host 0.0.0.0 --port 8000   # HTTP API: инференс + обучение фоном
nexus analyze examples/scad/flange.scad --force 0 0 -600 --material alu6061 --stl out.stl
nexus reward examples/scad/thin_plate_bad.scad          # физическая награда как в RL
nexus flywheel -n 256 --out artifacts/flywheel          # датасет { ТЗ → SCAD → 3D → FEM }
nexus train-fno --data artifacts/flywheel --epochs 30   # фаза 2: FNO-суррогат
nexus pretrain  --data artifacts/flywheel --preset tiny # фаза 3: ядро
nexus rl --steps 10                                     # фаза 4: GRPO с физическими наградами
nexus needle --lengths 1024,8192,32768                  # O(1) память
nexus vram --preset rtx5060                             # бюджет VRAM
```

(эквивалентно `python -m nexus.cli <команда>`; `make help` — список целей)

---

## 3. Эксплуатация: обучение, версии, API

Полные инструкции — [docs/training.md](docs/training.md),
[docs/operations.md](docs/operations.md), [docs/api.md](docs/api.md).

### 3.1 Одна команда: `nexus quickstart`

| Масштаб | Деталей | Шагов | Время (CPU) |
| :-- | --: | --: | :-- |
| `nano` | 8 | 8 | ~8 с |
| `small` | 48 | 60 | ~20 с |
| `medium` | 256 | 400 | ~10 мин |
| `gpu` | 2000 | 4000 | часы на RTX 5060 |

Проверяет окружение → строит датасет SCAD→3D→FEM → обучает BPE-токенизатор →
обучает ядро → прогоняет приёмочные тесты → при успехе помечает версию
`production`. Пример вывода `--scale small`: `val_ppl 942` против случайного
базлайна `2048`, gate пройден, всё за 20 секунд.

### 3.2 Обучение на стандартном датасете

```bash
nexus train-lm --source builtin:engineering            # офлайн-корпус, работает сразу
nexus train-lm --source dir:./corpus --seq-len 1024 --grad-accum 16 --device cuda --amp
nexus train-lm --source hf:wikitext/wikitext-2-raw-v1:train --limit 20000
nexus train-lm --source flywheel:artifacts/flywheel --resume latest    # дообучение
```

Источники: `builtin:` · `dir:` · `file:` · `jsonl:PATH#field` · `flywheel:` ·
`hf:NAME[/CONFIG][:SPLIT]#field`. Общий тренер: grad-accum, косинусный LR с
прогревом, AMP, 8-bit AdamW, ранняя остановка, автосохранение версии.

Токенизатор: `nexus train-tokenizer` обучает байтовый BPE на своём корпусе
(в стиле minbpe: регексное пред-разбиение, lossless, JSON-формат). На инженерном
корпусе он даёт ~4.1 байта/токен против ~2.5 у ручного AST-BPE — то есть
последовательности короче, а обучение и инференс дешевле. Токенизатор
сохраняется **вместе с версией модели** в реестре, так что перепутать их нельзя.

### 3.3 Дистилляция из существующей LLM

```bash
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode logit     # KL по логитам + CE
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode cached --top-k 64   # логиты на диск
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode sequence  # чёрный ящик: корпус от учителя
nexus distill --teacher-nexus core --teacher-ref production --mode logit  # самодистилляция
```

Потери `alpha·KL(student‖teacher)·T² + (1−alpha)·CE`; в режиме `logit` студент
автоматически переводится на словарь учителя. Учитель может быть любым классом
с `logits()` и `generate_text()`.

### 3.4 Реестр версий — ничего не теряется

```
artifacts/registry/core/
  v0001/{model.pt, meta.json}   v0002/…   v0003/…
  tags.json  {"latest": 3, "production": 2}
  tags.log   журнал переключений тегов
```

* новая версия — **новый каталог**, перезапись запрещена на уровне API;
* запись атомарна (tmp → `os.replace`), в `meta.json` пишутся конфиг, метрики,
  родительская версия, датасет, стадия, git-коммит и sha256;
* `load()` сверяет sha256; `prune` не трогает версии под тегами;
* откат — переключение тега: `nexus registry rollback --tag production`.

### 3.5 Автотесты дообученной модели (quality gate)

```bash
nexus eval --model-name core --ref latest --baseline production \
           --max-ppl 40 --min-compile 0.3 --max-regression 1.05 --gate
```

Проверки: конечность логитов, перплексия, вырожденность генерации, детерминизм
greedy, латентность, постоянство размера TTT-состояния, доля компилируемых
SCAD-генераций, регрессия к базовой версии. Тег `production` переключается
**только** при полном прохождении; код возврата пригоден для CI.

### 3.6 HTTP API

```bash
nexus serve --host 0.0.0.0 --port 8000 --ref production
curl -s -X POST localhost:8000/v1/design -d '{"spec":"<task>кронштейн 350 Н","material":"alu6061"}'
curl -s -X POST localhost:8000/v1/jobs   -d '{"args":["train-lm","--epochs","1"]}'
```

Без внешних веб-зависимостей (стандартная библиотека). `GET /` — мини-панель.
Эндпойнты: `/health`, `/metrics` (Prometheus), `/v1/models`, `/v1/generate`,
`/v1/generate/stream` (SSE), `/v1/generate/batch`, `/v1/design`, `/v1/analyze`,
`/v1/reward`, `/v1/reload`, `/v1/registry/promote|rollback`, `/v1/eval`,
`/v1/jobs`. Обучение идёт фоновой задачей, инференс при этом не встаёт.
Защита: `NEXUS_API_KEY=secret` включает `Authorization: Bearer`, плюс
rate-limit по IP (`--rate-limit`). Ускорение: `--compile` (torch.compile).

### 3.7 Docker

```bash
docker compose --profile train up train   # данные + первая версия модели
docker compose up -d                      # API на :8000, реестр в ./artifacts
```

CI (`.github/workflows/ci.yml`): установка, `doctor`, полный `pytest`,
сквозной `quickstart --scale nano` и выгрузка отчёта приёмки артефактом.

---

## 4. Архитектура

### 4.1 Уровень 1 — непрерывные динамические энкодеры

| Класс данных | Энкодер | Модуль |
| :-- | :-- | :-- |
| 1. Базовые дискретные (текст, код, AST, математика) | AST-BPE Embedder, обратимый токенизатор с инженерными мёржами | `encoders/text_ast.py`, `data/tokenizer.py` |
| 2. Аудио / визуальные | Spatio-Temporal SSM по реальному $\Delta t$ (переменный FPS допустим) | `encoders/audio_visual.py` |
| 3. **Инженерия и наука (приоритет)** | B-Rep/CSG Graph Neural Operator + энкодер тензорных полей FEM/CFD | `encoders/scad_brep.py` |
| 4. Пространство и физика (Embodied AI) | Point-SSM (3DGS), Event-ODE (DVS), проприоцепция/тактильность | `encoders/spatial.py` |
| 5. Человек и биофизика | Neural-ODE фильтр по ЭКГ/ЭЭГ/ЭМГ/GSR с неравномерной дискретизацией | `encoders/spatial.py::BiophysicsEncoder` |

Все энкодеры выдают `LatentPacket` — латенты плюс **физические инварианты**: координаты в метрах, время в секундах, масса, вектор силы.

### 4.2 Уровень 2 — физическое инвариантное ядро

**Unified Latent Bus** (`bus.py`) сливает пакеты в единый поток, упорядоченный по физическому времени, и добавляет инварианты к каждому токену.

**NEXUS Deep Layer Stack** (`layers/block.py`), 24 слоя:

1. **Linear Fast-Weight Memory (TTT)** — `layers/ttt.py`

   $$\mathcal{M}_t = (1-\alpha_t)\mathcal{M}_{t-1} - \eta_t \nabla \mathcal{L}_t,\qquad \mathcal{L}_t = \tfrac12\lVert \mathcal{M}_{t-1}k_t - v_t\rVert^2$$

   $\alpha_t,\eta_t$ предсказываются сетью по токену (data-dependent gating). Обработка чанками длины $L{=}128$: внутри чанка всё векторизовано, между чанками — строгая рекуррентность. Тест `test_ttt_chunking_equivalent_to_streaming` проверяет совпадение чанкового и потокового режимов.

2. **Local Sliding-Window Attention** — `layers/sliding_attention.py`, GQA + RoPE, окно $W = 1024\ldots2048$, KV-кэш жёстко обрезается по окну.

3. **Fine-Grained Sparse MoE** — `layers/moe.py`, 1 общий + 4 активных эксперта из 32, балансировка нагрузки и z-loss роутера.

**Latent Reasoning Workspace** (`reasoning/workspace.py`): скрытое состояние циркулирует $k$ раз без генерации токенов; на каждом шаге можно дёрнуть **FNO-суррогат** (`surrogates/fno.py`) и получить мгновенную оценку прочности/аэродинамики; Adaptive Pondering останавливает цикл по накопленной вероятности остановки.

### 4.3 Уровень 3 — двухрежимный вывод

`heads.py`: дискретная голова (OpenSCAD, текст, Python/C++) и непрерывная (траектории/G-код `action_dim`, тензорное поле $G^3$, скаляры «масса / $\sigma_{max}$ / запас прочности»).

---

## 5. OpenSCAD Data Flywheel

```
Базовые скрипты (examples/scad) → Domain Randomization (5 шаблонов, 6–8 параметров каждый)
      → Headless-рендер: OpenSCAD или встроенный CSG/SDF → воксели, B-Rep граф, STL
      → Геометрический аудит: manifold, компоненты, полости, стенка, свесы, ЧПУ-достижимость
      → Пакетный FEM: CalculiX или load-path решатель → тензор σ
      → Кортеж { ТЗ + нагрузки → OpenSCAD код → 3D-форма → FEM тензор }
```

```bash
nexus flywheel -n 512 --out artifacts/flywheel --grid 24 --fem-grid 16
# dataset.jsonl (ТЗ, код, масса, аудит, FEM) + fields.npz (occupancy, σ) + stats.json
```

Выход маховика на 24 сэмплах: 100 % компилируется, 92 % проходит аудит manifold, все валидные детали получают FEM-тензор.

Точность встроенной геометрии проверяется тестами: куб 20 мм даёт массу 20.7 г из аналитических 21.6 г (сетка 40³), толщина стенки восстанавливается как 3.04 мм для плиты 3 мм и 2.17 мм для трубы со стенкой 2 мм.

**Три FEM-бэкенда** (`--fem auto|hex|loadpath|calculix`):

* `hex` — **настоящий МКЭ**: трилинейные гексаэдры C3D8, matrix-free CG с якобиевым
  предобуславливателем, одна матрица жёсткости на все воксели (`fem/hex_fem.py`).
  Валидация на аналитике: брус 10×10×40 мм под 500 Н даёт σ 4.45 МПа против
  5.00 расчётных и δ 0.00266 мм против 0.0029 — сходимость за 130 итераций, 0.2 с;
  концентрация напряжений у отверстия воспроизводится (проверено тестом).
* `loadpath` — быстрый суррогат переноса силового потока (для массового прогона наград).
* `calculix` — внешний `ccx`, если установлен.

---

## 6. Бюджет VRAM для RTX 5060 (16 ГБ)

`nexus vram --preset rtx5060` (BF16, 8-bit оптимизатор, 50 % экспертов на CPU-offload, $L=128$, batch 2):

| Компонент | Конфигурация | VRAM |
| :-- | :-- | --: |
| Базовое ядро (active) | 1.36B, $d_{latent}=1536$, 24 слоя | 2.54 ГБ |
| Sparse MoE | 32 эксперта, 7.7B весов, половина на CPU | 6.75 ГБ |
| FNO-критик FEM | ~32M параметров | 0.06 ГБ |
| 8-bit оптимизатор | состояние по активным весам | 2.54 ГБ |
| Активации + chunked prefix | $L=128$, batch 2 | 0.32 ГБ |
| Состояние TTT | $O(1)$, не зависит от контекста | 0.003 ГБ |
| **Итого** | | **12.21 ГБ** (запас 3.8 ГБ) |

Пресет `rtx5060-compact` (14 экспертов, ~3.6B суммарных весов MoE, как в таблице ТЗ) — **8.41 ГБ**.

Подробный разбор расхождений с таблицей исходной спецификации: [docs/vram_budget.md](docs/vram_budget.md).

---

## 7. План разработки и текущий статус

| Фаза | Содержание | Статус |
| :-- | :-- | :-- |
| **1. Ядро памяти** (нед. 1–2) | TTT-слой + Sliding Attention, тест на длинный контекст | ✅ `layers/`, `nexus needle` |
| **2. Data Engine & FNO** (нед. 3–4) | Генератор вариаций, headless-рендер, пакетный FEM, обучение FNO | ✅ `scad/`, `fem/`, `nexus train-fno` |
| **3. Pre-train & Reasoning** (нед. 5–6) | Обучение ядра на коде/AST, включение Latent Workspace | ✅ пайплайн `nexus pretrain` (масштабный прогон — за пользователем и его GPU) |
| **4. Physics-RL (GRPO)** (нед. 7–8) | RL без критика-сети, физические награды | ✅ `nexus rl` |
| **5. Новые модальности** | Заморозка ядра, обучение лёгких энкодеров 10–50M | ✅ API `model.freeze_core()` + `model.register_encoder()` |

```python
model = NexusEngine(NexusConfig.rtx5060())
model.freeze_core()                      # ядро 1.3B заморожено
model.register_encoder("audio", AudioSSMEncoder(model.cfg.d_latent))   # учится только энкодер
```

---

## 8. Структура репозитория

```
nexus/
  config.py                 пресеты tiny / rtx5060 / rtx5060-compact
  bus.py                    Unified Latent Bus + физические инварианты
  model.py                  сборка трёх уровней, loss, generate, отчёт по параметрам
  heads.py                  Dual Output Engine
  demo.py                   сквозная демонстрация
  cli.py                    единый CLI
  layers/                   ttt.py · sliding_attention.py · moe.py · block.py · common.py
  encoders/                 text_ast · audio_visual · scad_brep · spatial · base(SSM/ODE)
  reasoning/workspace.py    латентный контур + Adaptive Pondering
  surrogates/fno.py         FNO-3D и FEM-критик
  geometry/                 csg.py (SDF) · voxel.py (масса, аудит) · brep.py (граф, STL)
  scad/                     parser.py · generator.py · render.py
  fem/                      hex_fem.py (МКЭ, matrix-free CG) · solver.py · calculix.py
  data/                     tokenizer.py · bpe.py (обучаемый BPE) · corpora.py
                            flywheel.py · dataset.py
  quickstart.py             сквозной сценарий «одной командой»
  registry.py               версии моделей, теги, откат, sha256
  training/                 trainer.py (общий цикл) · train_lm.py · distill.py
                            pretrain.py · train_fno.py · grpo.py · rewards.py
  serve/                    service.py (инференс) · app.py (HTTP API) · jobs.py
  eval/                     suite.py (quality gate) · needle.py · vram.py
tests/                      78 тестов: геометрия, ядро, пайплайн, реестр, обучение,
                            API, BPE, МКЭ, quickstart
nexus.sh                    единая точка запуска (setup / quickstart / serve / …)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
examples/                   quickstart.py · multimodal.py · scad/*.scad
docs/                       QUICKSTART.md · architecture.md · training.md · api.md
                            operations.md · vram_budget.md · roadmap.md
```

---

## 9. Ограничения и чего не хватает

* Веса не обучены: репозиторий даёт **архитектуру, данные и контур обучения**, а не готовую модель. На случайной инициализации GRPO ожидаемо получает награду −1 (сгенерированный текст не компилируется) — сначала фаза 3 на реальном объёме данных, затем RL.
* Встроенный FEM — физически мотивированный суррогат, а не полноценный МКЭ; для сертификационных расчётов подключайте CalculiX.
* Парсер OpenSCAD покрывает подмножество языка (примитивы, булевы операции, трансформации, переменные, арифметика); `hull`/`minkowski` аппроксимируются содержимым.
* Энкодеры аудио/видео/3DGS/DVS/биометрии реализованы как рабочие модули уровня 1 с корректной непрерывной динамикой, но обучающих корпусов для них в репозитории нет.

### Что закрыто в этой итерации

| Было | Стало |
| :-- | :-- |
| ручной токенизатор | обучаемый байтовый BPE (`nexus train-tokenizer`), хранится вместе с версией модели |
| суррогатный FEM | настоящий МКЭ на гексаэдрах (matrix-free CG), сверен с аналитикой |
| API без защиты | ключ `NEXUS_API_KEY`, rate-limit, SSE-стриминг, батч-инференс, `/metrics` |
| нет запуска «из коробки» | `./nexus.sh setup/quickstart/serve/all`, `nexus doctor`, Docker, CI |
| стартовый loss ~40 | GPT-2-инициализация: старт ≈ ln(V), обучение реально сходится |
| парсер без циклов | поддержаны `for (i = [a:step:b])` и списки |

### Чего ещё не хватает

| # | Чего нет | Почему важно | Оценка |
| :-- | :-- | :-- | :-- |
| 1 | **Обученных весов и большого корпуса** | инфраструктура готова, нужен сбор 10–100 ГБ кода/CAD/документации и прогон на GPU | недели GPU |
| 2 | **Ускоренных ядер TTT/MoE** (Triton/CUDA) | чистый PyTorch; фьюзинг чанкового скана и группировка экспертов дадут 3–10× | 1–2 недели |
| 3 | **Multigrid-предобуславливателя для МКЭ** | сейчас Jacobi+CG: сетки крупнее 48³ считаются секундами, а не миллисекундами | 3–5 дней |
| 4 | **Marching cubes и STEP/IGES** | воксельный STL груб для производства; нужен OpenCascade для B-Rep | 1 неделя |
| 5 | **Датасетов для остальных модальностей** | энкодеры аудио/видео/3DGS/DVS/биометрии работают, учить не на чем | зависит от домена |
| 6 | **Распределённого обучения (FSDP/DeepSpeed)** | сейчас одна карта; для 1.3B+ на нескольких GPU нужен шардинг | 3–5 дней |
| 7 | **Веб-UI поверх API** | сейчас только curl и мини-панель; конструктор ТЗ → деталь → отчёт был бы нагляднее | 1 неделя |

Лицензия: MIT.
