# Обучение

Откуда брать данные — [DATA_PLAN.md](DATA_PLAN.md) (каталог открытых датасетов,
свой генератор математики, MCP-сервер, дистилляция с верификацией).

**Пошаговый регламент с защитой от переобучения — [TRAINING_GUIDE.md](TRAINING_GUIDE.md).**

Быстрый путь — `nexus quickstart --scale small`: он сам построит данные,
обучит токенизатор и ядро, прогонит приёмку и назначит `production`.
Ниже — ручной режим по шагам.

## 0. Токенизатор

```bash
nexus train-tokenizer --source dir:./corpus --vocab-size 8192 \
                      --out artifacts/tokenizer/bpe.json
nexus train-lm --source dir:./corpus --tokenizer artifacts/tokenizer/bpe.json
```

Байтовый BPE обучается на своём корпусе, обратим (`decode(encode(x)) == x`) и
сохраняется в каталоге версии модели, поэтому пара «веса ↔ токенизатор» не
может разъехаться.

Три сценария, один и тот же тренер (`nexus/training/trainer.py`), один и тот же
реестр версий.

## 1. Стандартный датасет (LM)

```bash
nexus train-lm --source builtin:engineering --preset tiny --epochs 2
nexus train-lm --source dir:./corpus --seq-len 1024 --batch-size 2 --grad-accum 16
nexus train-lm --source hf:wikitext/wikitext-2-raw-v1:train --limit 20000 --device cuda --amp
nexus train-lm --source flywheel:artifacts/flywheel --resume latest   # дообучение
```

Источники (`nexus/data/corpora.py`):

| Спецификация | Что берёт |
| :-- | :-- |
| `builtin:engineering` | встроенный мини-корпус, работает офлайн |
| `dir:PATH` | `*.txt/.md/.scad/.py/.c/.cpp/.h/.json` рекурсивно |
| `file:PATH` | один файл |
| `jsonl:PATH#field` | JSONL, поле `field` (по умолчанию `text`) |
| `flywheel:PATH` | артефакты маховика: ТЗ + код + сводка FEM |
| `hf:NAME[/CONFIG][:SPLIT]#field` | HuggingFace `datasets` (нужен пакет `datasets`) |

Тексты склеиваются и режутся на блоки `seq_len` (обычная LM-упаковка).
`--resume latest|production|v0007` продолжает обучение с версии из реестра,
родитель фиксируется в метаданных.

## 2. Дистилляция из существующей LLM

```bash
# онлайн по логитам (студент переходит на словарь учителя)
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode logit --alpha 0.7 --temperature 2.0

# логиты учителя считаются один раз и кладутся на диск (учитель потом не нужен)
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode cached --top-k 64

# чёрный ящик: учитель генерирует корпус, студент учится на нём
nexus distill --teacher Qwen/Qwen2.5-0.5B --mode sequence

# самодистилляция из своей же версии в реестре (офлайн)
nexus distill --teacher-nexus core --teacher-ref production --mode logit
```

Функция потерь: `alpha·KL(student‖teacher)·T² + (1−alpha)·CE`.
`--mode cached` хранит top-k логитов (`float16` + `int32`), что снижает объём
кэша примерно в `vocab/k` раз.

Учителя: `HFTeacher` (пакет `transformers`), `NexusTeacher` (чекпойнт из
реестра). Свой учитель — любой класс с методами `logits()` и `generate_text()`.

## 3. Инженерные фазы

```bash
nexus flywheel -n 512               # данные SCAD → 3D → FEM
nexus train-fno --epochs 30         # суррогат прочности
nexus pretrain --data artifacts/flywheel
nexus rl --steps 50                 # GRPO с физическими наградами
```

## Память и скорость

| Флаг | Эффект |
| :-- | :-- |
| `--grad-accum N` | эффективный батч без роста VRAM |
| `--amp` | bf16/fp16 автокаст на CUDA |
| `--eight-bit` | 8-битный AdamW (нужен `bitsandbytes`) |
| `--eval-every N --patience K` | ранняя остановка по валидации |
| `--max-steps N` | жёсткий лимит шагов (для smoke-прогонов) |
