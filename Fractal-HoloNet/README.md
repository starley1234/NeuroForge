# Fractal-HoloNet: Multimodal Continuous Signal & Language Model

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Continuous Modality](https://img.shields.io/badge/Modality-Continuous%20Signals%20%2B%20Text-purple.svg)](https://github.com/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)](https://onnxruntime.ai/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

**Fractal-HoloNet** — универсальная мультимодальная архитектура ИИ с линейной вычислительной сложностью $\mathcal{O}(N)$ и константной памятью $\mathcal{O}(1)$. 

Благодаря **непрерывной комплексно-фазовой природе**, модель способна напрямую принимать, обрабатывать и прогнозировать **непрерывные аналоговые сигналы (сырой звук, ЭКГ, датчики IoT, телеметрию, видеопотоки) без дискретизации в тяжелые токены** с нулевой потерей физической динамики.

---

## 🌊 Мультимодальность для непрерывных сигналов

### Почему это уникально:
* **Традиционный подход (Whisper / VQ-VAE)**: вынужден нарезать непрерывную волну на дискретные токены через квантование, теряя тонкие фазовые гармоники и раздувая размер контекста.
* **Fractal-HoloNet**: проецирует сырой сигнал $x(t) \in \mathbb{R}^C$ напрямую в комплексное фазовое пространство $\mathbb{C}^d$ через обучаемый банк вейвлет-фильтров Фурье. Фазовый аккумулятор естественным образом моделирует физические колебания, гармоники и тренды.

### Поддерживаемые непрерывные модальности:
1. **Биомедицинские сигналы**: ЭКГ (электрокардиограмма), ЭЭГ мозга, пульсоксиметрия.
2. **Акустика и звук**: сырые аудио-волны 16kHz/44.1kHz без спектрограммного огрубления.
3. **Промышленный IoT и телеметрия**: вибрации турбин, датчики давления, температуры, сетевой трафик.
4. **Финансовые потоки**: высокочастотные тиковые данные (HFT).

---

## 📊 Результаты прогнозирования непрерывного сигнала

Модель протестирована на прогнозировании 64 шагов сложного непрерывного квазипериодического сигнала с детекцией аномалий в реальном времени:
* **MSE прогнозирования**: `0.0274`
* **Точность детекции аномалий (BCE)**: `0.00019`
* **Задержка прогнозирования (64 шага)**: `39.5 ms` ($\mathcal{O}(1)$ шаговый стриминг)

![Multimodal Signal Forecast](artifacts/multimodal_signal_forecast.png)

---

## 📂 Структура репозитория

```text
├── fractal_holonet/          # 📦 Пакет: ядро, мультимодальность, токенизаторы, сервер
│   ├── core.py               #   🧠 Ядро v1: RMSNorm, CRAC, O(1) генератор, config
│   ├── multimodal.py         #   🌊 Мультимодальное ядро: ContinuousSignalEncoder, AnomalyHead
│   ├── tokenizer.py          #   🔌 Байтовый + byte-level BPE токенизаторы, inference pipeline
│   ├── distillation.py       #   🧠 Ядро дистилляции знаний (Teacher API клиент)
│   ├── self_train.py         #   🤖 Автономное самообучение (eval-гейт, daemon)
│   ├── registry.py           #   📒 Реестр обученных моделей (метаданные, метрики, активация)
│   ├── serve.py              #   🌐 REST API (генерация, сигналы, /v1/distill, /v1/self-train, /v1/models)
│   └── datasets.py           #   📚 Учебные корпуса и ByteDataset
├── scripts/                  # 🛠️ CLI-скрипты
│   ├── train.py              #   📝 Базовое обучение (Causal LM)
│   ├── train_from_scratch.py #   🆕 Каноническое обучение с нуля + регистрация в реестре
│   ├── train_tokenizer.py    #   🔤 Обучение byte-level BPE токенизатора
│   ├── train_russian.py      #   🇷🇺 Обучение русскому языку
│   ├── train_multimodal.py   #   🏋️ Обучение на непрерывных сигналах (ЭКГ, Аудио, IoT)
│   ├── train_benchmark.py    #   🎭 Обучение на эталонном корпусе (TinyShakespeare)
│   ├── train_v2_lm.py        #   🧬 Обучение ELAST-HOLO v2 (LM)
│   ├── distill.py            #   🚀 CLI дистилляции (Teacher endpoint + ключ)
│   ├── register_checkpoint.py#   📒 Импорт существующих чекпоинтов в реестр
│   ├── export_onnx.py        #   📦 Экспорт в ONNX и валидация через ONNX Runtime
│   └── verify_all.py         #   🔍 Сквозная проверка всех компонентов (End-to-End)
├── tests/                    # 🧪 PyTest (v1 + v2 + tokenizers + registry + self-train)
├── examples/                 # 💡 Демо-скрипты (генерация, русский инференс)
├── research/                 # 🔬 Эксперименты, прототипы, ELAST-HOLO v2, бенчмарки
├── data/                     # 📊 Корпуса (TinyShakespeare, русские слова)
├── registry/                 # 📒 Реестр моделей + обученные токенизаторы
├── checkpoints/              # 💾 Легаси-чекпоинты (text + multimodal + v2)
├── exports/                  # 📦 ONNX-экспорт
├── artifacts/                # 🖼️ Графики и результаты прогонов
├── .github/workflows/ci.yml  # 🔁 CI: тесты + сквозная проверка
├── pyproject.toml            # 📌 Пакет, зависимости, конфигурация pytest
├── requirements.txt          # 📌 Зафиксированные зависимости
├── Dockerfile                # 🐳 Production Dockerfile
└── docker-compose.yml        # 🚀 Оркестрация Uvicorn
```

---

## ⚡ Использование на практике

### 1. Сквозная проверка всех модулей системы
```bash
python3 scripts/verify_all.py
```

### 2. Обучение
```bash
# Обучение на текстах
python3 scripts/train.py

# Обучение на непрерывных сигналах (ЭКГ, аудио, датчики)
python3 scripts/train_multimodal.py
```

### 3. Использование в Python для IoT / Биомедицины
```python
import torch
from fractal_holonet.multimodal import MultimodalFractalHoloNet

# Загрузка обученной мультимодальной модели
model = MultimodalFractalHoloNet.from_pretrained("./checkpoints/fractal_holonet_multimodal")

# Сырой непрерывный сигнал с датчика (1 поток, 64 отсчета)
raw_stream = torch.randn(1, 64, 1)

# 1. Прогноз на 32 шага вперед без токенизации
forecast = model.forecast_stream(raw_stream, forecast_steps=32)
print("Форма прогноза:", forecast.shape) # (1, 32, 1)

# 2. Мгновенная детекция аномалий
_, anomaly_scores, _ = model.forward_continuous(raw_stream)
print("Оценка аномальности по шагам:", anomaly_scores[0, :, 0])
```

### 4. Запуск через REST API
```bash
uvicorn fractal_holonet.serve:app --host 0.0.0.0 --port 8000
```

#### Эндпоинт дистилляции знаний (Teacher API -> Student):
`POST /v1/distill`
```bash
curl -X POST http://localhost:8000/v1/distill \
  -H "Content-Type: application/json" \
  -d '{
    "teacher_endpoint": "https://api.openai.com/v1",
    "teacher_model": "gpt-4o-mini",
    "teacher_api_key": "sk-your-key",
    "prompts": [
      "Explain O(N) context complexity in Fractal-HoloNet",
      "How does holographic phase resonance work?"
    ],
    "epochs": 5
  }'
```

#### Эндпоинт прогнозирования и скоринга аномалий:
`POST /v1/signal/forecast`
```bash
curl -X POST http://localhost:8000/v1/signal/forecast \
  -H "Content-Type: application/json" \
  -d '{
    "signal_history": [[0.12], [0.35], [0.89], [1.45], [0.80], [0.20]],
    "forecast_steps": 16
  }'
```

### 5. Запуск всех тестов
```bash
python3 -m pytest
```

---

## 🔤 Токенизация: байты и byte-level BPE

| Токенизатор | Словарь | Кириллица | Комментарий |
|---|---|---|---|
| `SimpleProductionTokenizer` | 260 (4 спец + 256 байт) | 2 байта/символ | легаси-режим, универсальное байтовое покрытие |
| `BpeTokenizer` | 8192 (`registry/tokenizers/bpe-8192/`) | **×4.33 сжатие** (52 токена против 225 байт) | обучен на `data/russian_words.txt` (cp1251→UTF-8) + `data/data_benchmark.txt` |

Правила, закреплённые в коде:
* `vocab_size` конфига **берётся из токенизатора** (`build_config_for_tokenizer`) —
  рассинхрон «конфиг 300 vs словарь 260» исключён;
* pad/bos/eos всегда читаются из токенизатора (`tokenizer.pad_token_id` и т.д.),
  нигде не зашиты константами;
* в шаблонах дистилляции используется **настоящий eos-токен**, а не строка `<eos>`;
* пайплайн сам определяет формат `tokenizer.json` (легаси-словарь или HF BPE).

Обучение токенизатора:
```bash
python scripts/train_tokenizer.py --vocab-size 8192
```

---

## 📒 Реестр обученных моделей

Все обученные модели живут в `registry/` с уникальным id, метаданными
(`registry.json`), метриками и своими весами — **чекпоинты никогда не
перезаписываются**. Каждый запуск обучения создаёт новую запись.

```bash
# список моделей
python -c "import fractal_holonet.registry as r; [print(m['id'], m['status'], m['name']) for m in r.list_models()]"

# импорт существующего чекпоинта
python scripts/register_checkpoint.py

# обучение С НУЛЯ (регистрация, cosine LR, val-сплит, метрики)
python scripts/train_from_scratch.py \
    --tokenizer registry/tokenizers/bpe-8192 \
    --epochs 5 --batch-size 64 --block-size 96

# обслуживание конкретной модели из реестра
FH_MODEL_ID=<model-id> uvicorn fractal_holonet.serve:app --host 0.0.0.0 --port 8000

# REST: список, информация, активация
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/models/<model-id>
curl -X POST http://localhost:8000/v1/models/activate -H "Content-Type: application/json" \
  -d '{"model_id": "<model-id>"}'
```

---

## 🧬 ELAST-HOLO: архитектура v2 (research)

Новое поколение ядра: **Elastic-Time Holographic Associative Machine**
(дизайн-документ: `research/ARCHITECTURE_V2.md`, код: `research/arch_v2_core.py`).

Ключевые механизмы:
* **M1 Упругое время** — сеть управляет темпом собственных фазовых часов
  (`dtheta` на токен): контекстно-зависимые шкалы времени и нативная
  обработка нерегулярных/событийных потоков через интервалы `dt`.
* **M3 Комплексная дельта-запись на матричном состоянии** — rank-1 запись
  с фазово-корректированным стиранием `S ← rot⊙S + β(k vᵀ − k(kᴴS))`
  (нормированные ключи → стабильный нерасширяющий проектор). Решает
  ассоциативный recall (MQAR), недоступный диагональной рекурренции v1.
* **M4 Двойная память fast/slow** с гейтом консолидации по метрике сюрприза.
* **M5 Итеративное ассоциативное чтение** — K fixed-point шагов уточнения.
* **M2/M7** — циркулянтное смешивание (FFT) и аналитический (Гильберт)
  сигнальный фронтенд.

Измерено (CPU-масштаб, равные параметры и бюджет):
* **MQAR** (L=24, 3 пары): v2 **0.260** vs v1 0.155; экстраполяция L=48:
  **0.180** vs 0.100 (chance 0.0625) — `research/benchmarks/mqar_benchmark.py`.
* **Нерегулярное время**: elastic clock даёт **+11.8%** MSE против
  dt-blind при равных параметрах — `research/benchmarks/irregular_time_bench.py`.

```bash
python research/benchmarks/mqar_benchmark.py          # v1 vs v2 на MQAR
python research/benchmarks/irregular_time_bench.py    # выигрыш M1
python scripts/train_v2_lm.py                         # чекпоинт v2 LM (checkpoints/fractal_holonet_v2)
python3 -m pytest                                     # все тесты (v1 + v2 + self-train)
```

---

## 🤖 Автономное самообучение (развёрнутая LLM обучает модель сама)

`fractal_holonet/self_train.py` + REST-эндпоинты `/v1/self-train*`: цикл, в котором внешняя
LLM (любой OpenAI-совместимый эндпоинт) генерирует обучающие данные, а
студент дообучается **с eval-гейтом** — раунд принимается только при
улучшении holdout-лосса, иначе веса откатываются (защита чекпоинта).

* Учебная программа (curriculum) + студент сам генерирует пробные промпты.
* Без API-ключа работает встроенный синтетический учитель (offline).

```bash
# CLI: синхронно
python -m fractal_holonet.self_train --rounds 3

# CLI: фоновый демон каждые 300 секунд
TEACHER_API_KEY=sk-... python -m fractal_holonet.self_train --interval 300 --model gpt-4o-mini

# REST: один раунд
curl -X POST http://localhost:8000/v1/self-train \
  -H "Content-Type: application/json" \
  -d '{"teacher_endpoint":"https://api.openai.com/v1","teacher_model":"gpt-4o-mini",
       "teacher_api_key":"sk-your-key","epochs":4}'

# REST: фоновый цикл + статус
curl -X POST http://localhost:8000/v1/self-train/start -H "Content-Type: application/json" \
  -d '{"interval_sec":300,"epochs":4,"teacher_api_key":"sk-your-key"}'
curl http://localhost:8000/v1/self-train/status
curl -X POST http://localhost:8000/v1/self-train/stop
```
