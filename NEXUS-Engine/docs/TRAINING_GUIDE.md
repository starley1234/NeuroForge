# Регламент обучения

Откуда в обучении берётся физика (текст, инварианты, награды, фильтр) —
[PHYSICS_EXPLAINED.md](PHYSICS_EXPLAINED.md). Дообучение готовой открытой модели
(LoRA на Qwen3-Coder и др.) — [LORA.md](LORA.md): это самый быстрый путь к
рабочей модели, если своих токенов пока меньше 300M.

Коротко: **данные → холдаут → короткий пробный прогон → длинный прогон с ранней
остановкой → приёмка с гейтом → промоушен → откат при регрессии.** Каждый шаг —
одна команда, каждая модель — новая версия в реестре.

---

## 1. Три способа запустить обучение

| Способ | Команда | Когда |
| :-- | :-- | :-- |
| **Всё сразу** | `nexus quickstart --scale gpu --device cuda` | первый реальный прогон: данные, токенизатор, обучение, приёмка |
| **По шагам** | `nexus flywheel` → `nexus train-tokenizer` → `nexus train-lm` → `nexus eval` | когда нужен контроль над каждым этапом |
| **Фоном через API** | `POST /v1/jobs {"args":["train-lm","--epochs","1"]}` | обучение на сервере без остановки инференса |

Масштабы `quickstart` (любой параметр переопределяется `--samples --steps --math
--vocab --seq-len --preset --device`):

| Масштаб | Деталей | Математики | Модель | Шагов | Время на RTX 5060 Ti |
| :-- | --: | --: | :-- | --: | :-- |
| `nano` / `small` | 8 / 48 | 0 / 200 | tiny 1M | 8 / 60 | секунды — **проверка пайплайна, не обучение** |
| `medium` | 500 | 5 000 | small 200M | 1 500 | минуты |
| `gpu` | 5 000 | 50 000 | small 200M | 20 000 | часы — первая осмысленная модель |
| `gpu-large` | 20 000 | 200 000 | 1.4B активных | 60 000 | сутки |

---

## 2. Правильный порядок (то, что стоит делать всегда)

### Шаг 1. Данные и честный холдаут

```bash
nexus flywheel -n 5000 --out artifacts/flywheel --grid 26 --fem-grid 16
nexus flywheel -n 300  --out artifacts/holdout  --grid 26 --fem-grid 16 --seed 999
nexus gen-math -n 50000 --out artifacts/math/train.jsonl --seed 1
nexus gen-math -n 2000  --out artifacts/math/val.jsonl   --seed 777
```

Валидация должна быть из **другого сида и других деталей**, иначе случайный сплит
внутри одного корпуса покажет заниженную перплексию (соседние блоки одного и того
же скрипта попадут и в train, и в val).

### Шаг 2. Токенизатор — один раз на корпус

```bash
nexus train-tokenizer --source "mix:flywheel:artifacts/flywheel=0.7,jsonl:artifacts/math/train.jsonl#text=0.3" \
                      --vocab-size 16384 --out artifacts/tokenizer/bpe.json
```

Токенизатор сохраняется внутрь каждой версии модели, так что перепутать пару
«веса ↔ словарь» невозможно. Меняете словарь — начинайте новую линию моделей
(`--model-name core-v2`), дообучать поверх старого нельзя.

### Шаг 3. Пробный прогон на 200 шагов

```bash
nexus train-lm --source "mix:flywheel:artifacts/flywheel=0.7,jsonl:artifacts/math/train.jsonl#text=0.3" \
               --val-source "flywheel:artifacts/holdout" \
               --tokenizer artifacts/tokenizer/bpe.json \
               --preset small --seq-len 1024 --batch-size 8 --grad-accum 4 \
               --max-steps 200 --eval-every 50 --device cuda --amp \
               --model-name core-probe
```

Смотрите на первую строку лога: `loss` должен стартовать около `ln(vocab)`
(для 16k это ≈ 9.7) и падать. Если стоит на месте — слишком маленький LR или
битые данные; если улетает в NaN — слишком большой LR или нет AMP-скейлера.

### Шаг 4. Длинный прогон с ранней остановкой

```bash
nexus train-lm --source "mix:flywheel:artifacts/flywheel=0.7,jsonl:artifacts/math/train.jsonl#text=0.3" \
               --val-source "flywheel:artifacts/holdout" \
               --tokenizer artifacts/tokenizer/bpe.json \
               --preset small --seq-len 1024 --batch-size 8 --grad-accum 4 \
               --max-steps 20000 --eval-every 500 --patience 4 \
               --lr 3e-4 --device cuda --amp --model-name core
```

Что происходит под капотом:

* смешанная точность выбирается автоматически: **bfloat16** на картах, которые
  его поддерживают (включая RTX 50xx), fp16 только как запасной вариант —
  у bf16 диапазон как у fp32, поэтому NaN не возникает;
* NaN-шаг не роняет прогон: он пропускается с предупреждением, но 20 подряд
  останавливают обучение с внятной ошибкой;
* каждые `--eval-every` шагов считается валидация и печатается строка
  `val_loss=… ppl=… train_ce=… разрыв=+0.12`;
* лучшие веса запоминаются; в конце они **восстанавливаются**, даже если
  последние шаги были хуже (`restored_best: 1` в метриках);
* если валидация не улучшалась `--patience` проверок подряд — обучение
  останавливается досрочно;
* метрики `train_ce`, `val_loss`, `val_ppl`, `overfit_gap`, `best_step` пишутся
  в `meta.json` версии.

### Шаг 5. Приёмка и промоушен

```bash
nexus eval --model-name core --ref latest --baseline production \
           --source "flywheel:artifacts/holdout" \
           --max-ppl 60 --max-overfit 1.0 --max-regression 1.05 \
           --json artifacts/eval/core_latest.json --gate
```

Тег `production` переключается **только** если прошли все проверки. Код возврата
`0/1` — готов для CI и для `&&` в скриптах.

### Шаг 6. Если стало хуже — откат

```bash
nexus registry compare --model-name core --ref v0007 --tag v0008   # что изменилось
nexus registry rollback --model-name core --tag production          # вернуть прошлую
```

---

## 2.1 Как понять, что модель ещё ничего не умеет

Смотрите на `val_ppl` относительно размера словаря:

| val_ppl | Что это значит |
| :-- | :-- |
| ≈ размер словаря | модель угадывает случайно — обучения фактически не было |
| 0.3–0.7 × словаря | «шевелится», но текст будет бессвязным |
| 20–60 (словарь 16k) | синтаксис OpenSCAD выучен, детали иногда компилируются |
| < 15 | рабочий диапазон, есть смысл включать RL-фазу |

Пример из реального прогона: `vocab 2048`, `val_ppl 1665`, `steps 50` —
это 20 % улучшения над случайностью, вывод обязан быть мусором. API теперь сам
пишет об этом в поле `quality_hint` ответа `/v1/generate`.

Дополнительно: `scad_compile_rate` в приёмке — доля сгенерированных деталей,
которые вообще компилируются. Пока она 0, говорить о качестве рано.

## 2.2 Хватает ли данных: главная причина «модель ничего не выучила»

Перед стартом тренер печатает бюджет:

```
[lm] бюджет: 20000 шагов = 30 эпох по 667 шагов; корпус ≈ 5.5M токенов, пройдём 164M
[lm] модель 237M параметров → 0.02 токенов на параметр (для обучения с нуля нужно ≈20)
[lm] ВНИМАНИЕ: данных мало для модели такого размера
```

Читается так:

| Показатель | Что означает |
| :-- | :-- |
| токенов на параметр < 1 | модель физически не может выучить язык — нужен корпус больше или модель меньше |
| 1–5 | выучит синтаксис, но не смысл |
| ≥ 20 | нормальный режим обучения с нуля |
| эпох > 10 | после 3–5 проходов начинается заучивание примеров |

Сколько токенов дают источники (порядок величин):

| Источник | Токенов |
| :-- | --: |
| `nexus flywheel -n 5000` | ~7M |
| `nexus gen-math -n 500000` | ~60M |
| база сайта, 10 000 скриптов | ~30M |
| CAD-Coder (Apache-2.0), 250k | ~200M |

Практический вывод: для пресета `small` (237M) реалистичный минимум — **100M+
токенов**, и это уже совместная работа маховика, математики и своей базы. Если
столько данных нет, честнее взять `--preset tiny` либо дообучать готовую
открытую модель, а не учить с нуля.

## 3. Как не переобучить

| Признак | Что видно | Что делать |
| :-- | :-- | :-- |
| `разрыв` в логе растёт (> 0.5) | val_loss уходит вверх, train_ce вниз | включить `--patience 3`, уменьшить число шагов, добавить данных |
| `overfit_gap` в приёмке > 1.0 | проверка `overfit_gap` красная | больше данных / меньше модель (`--preset tiny`) / сильнее `weight_decay` |
| val_ppl хорошая, а `scad_compile_rate` = 0 | модель зубрит токены, а не язык | добавить долю математики и документации в микс |
| перплексия падает подозрительно быстро | скорее всего утечка: валидация из того же источника | использовать `--val-source` с отдельным сидом |
| `loss = NaN` с первого шага | переполнение fp16 в смешанной точности | по умолчанию берётся bfloat16; принудительно: `--amp-dtype bf16`, либо убрать `--amp`, либо снизить `--lr` |

Практические границы: данных должно быть **минимум ~20 токенов на параметр**.
Для `small` (200M) это ≈ 4 млрд токенов в идеале; на 50–100 млн токенов модель
всё ещё учится, но 2–3 эпохи — потолок, дальше начинается зубрёжка. Для
контроля держите `--eval-every` таким, чтобы проверок за прогон было 20–40.

---

## 4. Как хранятся версии

```
artifacts/registry/core/
  v0001/{model.pt, meta.json, tokenizer.json}
  v0002/…                       ← новая версия, старые не трогаются
  tags.json   {"latest": 7, "production": 5}
  tags.log    журнал переключений тегов
```

* Номер версии монотонный, перезапись запрещена на уровне API.
* Запись атомарна: временный каталог → `os.replace`.
* В `meta.json`: конфиг модели, все метрики, родительская версия, датасет,
  стадия, git-коммит, sha256 весов. При загрузке контрольная сумма сверяется.
* `nexus registry prune --keep 5` чистит старое, **никогда** не трогая версии
  под тегами.

Полезные команды:

```bash
nexus registry list                                  # все модели и теги
nexus registry history --model-name core             # метрики каждой версии
nexus registry compare --model-name core --ref v0007 --tag v0008
nexus registry show --model-name core --ref production
```

---

## 5. Дообучение и дистилляция поверх готовой версии

```bash
# дообучить production на новых данных (родитель фиксируется в метаданных)
nexus train-lm --resume production --source "jsonl:artifacts/collected/dataset.jsonl#text" \
               --model-name core --lr 5e-5 --max-steps 3000 --eval-every 200 --patience 3

# дистилляция из внешней LLM
nexus distill --teacher Qwen/Qwen2.5-Coder-7B-Instruct --mode logit --alpha 0.7

# самоулучшение: собрать данные своей же моделью с физической проверкой
nexus collect --teacher nexus --ref production -n 2000 --attempts 3
```

При дообучении берите LR в 3–10 раз меньше исходного и обязательно ставьте
`--patience`: на маленьком новом корпусе модель забывает старое за сотни шагов
(катастрофическое забывание). Альтернатива — заморозить ядро и учить только
энкодер модальности: `model.freeze_core()` + `model.register_encoder(...)`.

---

## 6. Шпаргалка «сделай так»

```bash
# 1. данные + отдельный холдаут
nexus flywheel -n 5000 --out artifacts/flywheel
nexus flywheel -n 300 --out artifacts/holdout --seed 999
nexus gen-math -n 50000 --out artifacts/math/train.jsonl

# 2. словарь
nexus train-tokenizer --source "mix:flywheel:artifacts/flywheel=0.7,jsonl:artifacts/math/train.jsonl#text=0.3" --vocab-size 16384

# 3. обучение с защитой от переобучения
nexus train-lm --source "mix:flywheel:artifacts/flywheel=0.7,jsonl:artifacts/math/train.jsonl#text=0.3" \
  --val-source flywheel:artifacts/holdout --tokenizer artifacts/tokenizer/bpe.json \
  --preset small --seq-len 1024 --batch-size 8 --grad-accum 4 \
  --max-steps 20000 --eval-every 500 --patience 4 --lr 3e-4 --device cuda --amp

# 4. приёмка с гейтом, иначе откат
nexus eval --ref latest --baseline production --source flywheel:artifacts/holdout --gate \
  || nexus registry rollback --tag production

# 5. в бой
nexus serve --ref production --tokenizer artifacts/tokenizer/bpe.json
```
