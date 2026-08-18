# Быстрый старт

## 1. Установка и первый запуск (3 команды)

### Linux / macOS

```bash
git clone <repo> && cd NEXUS-Engine
./nexus.sh setup          # venv + зависимости + самопроверка (~3 мин)
./nexus.sh quickstart     # данные → токенизатор → обучение → приёмка (~20 с)
./nexus.sh serve          # HTTP API на http://localhost:8000
```

Одной кнопкой всё сразу: `./nexus.sh all`.

### Windows (PowerShell)

`nexus.sh` — bash-скрипт, PowerShell его не выполняет (команда просто молча
завершается). Используйте `nexus.ps1`:

```powershell
cd C:\github\NeuroForge\NEXUS-Engine
.\nexus.ps1 setup
.\nexus.ps1 quickstart
.\nexus.ps1 serve
```

Если PowerShell блокирует выполнение скриптов (`... не удается загрузить, так как
выполнение сценариев отключено`), любой из вариантов:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # только на эту сессию
# или
powershell -ExecutionPolicy Bypass -File .\nexus.ps1 setup
# или через обёртку для cmd.exe
.\nexus.cmd setup
```

Совсем без скриптов (работает и в conda-окружении):

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\pip install -e ".[dev]"
.\.venv\Scripts\python -m nexus.cli doctor
.\.venv\Scripts\python -m nexus.cli quickstart --scale small
.\.venv\Scripts\python -m nexus.cli serve --host 0.0.0.0 --port 8000
```

Масштаб прогона: `NEXUS_SCALE=nano|small|medium|gpu ./nexus.sh quickstart`

| Масштаб | Деталей | Шагов | Время (CPU) | Зачем |
| :-- | --: | --: | :-- | :-- |
| `nano` | 8 | 8 | ~8 с | проверить, что всё живо |
| `small` | 48 | 60 | ~20 с | демо и разработка (по умолчанию) |
| `medium` | 256 | 400 | ~10 мин | осмысленные метрики |
| `gpu` | 2000 | 4000 | часы на RTX 5060 | реальное обучение |

Что делает `quickstart`: проверяет окружение → генерирует детали OpenSCAD и
считает по ним FEM → обучает BPE-токенизатор → обучает ядро → прогоняет
приёмочные тесты и, если они прошли, помечает версию как `production`.

## 2. Проверка, что работает

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/analyze \
  -d '{"code":"difference(){cube([40,40,6],center=true); cylinder(h=20,r=5,center=true);}",
       "material":"alu6061","force":[0,0,-400]}'
curl -s -X POST localhost:8000/v1/generate -d '{"prompt":"<task>кронштейн 300 Н","max_new_tokens":64}'
```

Браузером: `http://localhost:8000/` — панель со списком эндпойнтов и статусом.

## 3. Без API, из командной строки

```bash
./nexus.sh doctor                                   # что установлено и что работает
./nexus.sh analyze examples/scad/l_bracket.scad --force 0 0 -400 --material alu6061
./nexus.sh demo                                     # все три уровня архитектуры
./nexus.sh test                                     # 70+ тестов
./nexus.sh cli --help                               # полный список команд
```

## 4. Docker

```bash
docker compose --profile train up train    # подготовить данные и первую модель
docker compose up -d                       # поднять API на :8000
```

## 5. Своё обучение

```bash
./nexus.sh cli train-tokenizer --source dir:./corpus --vocab-size 8192
./nexus.sh cli train-lm --source dir:./corpus --tokenizer artifacts/tokenizer/bpe.json \
                        --seq-len 1024 --epochs 3 --device cuda --amp
./nexus.sh cli eval --ref latest --baseline production --gate
```

Не понравился результат — откат одной командой:

```bash
./nexus.sh cli registry rollback --model-name core --tag production
```

## 6. Если что-то пошло не так

| Симптом | Что делать |
| :-- | :-- |
| `окружение не готово` | `./nexus.sh setup` (Windows: `.\nexus.ps1 setup`) |
| В PowerShell `./nexus.sh` ничего не делает | это bash-скрипт; используйте `.\nexus.ps1` или `.\nexus.cmd` |
| `выполнение сценариев отключено` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| Windows: кракозябры в консоли | скрипт сам ставит `PYTHONUTF8=1`; вручную: `$env:PYTHONUTF8=1` |
| Python 3.13+ и torch не ставится | возьмите Python 3.11/3.12 (`py -3.11 -m venv .venv`) |
| torch ставится очень долго | это ~800 МБ; для GPU-сборки `NEXUS_GPU=1 ./nexus.sh setup` |
| `quickstart` завершился с «ПРОВАЛЕНО» | норма для `nano`: слишком мало шагов; берите `small` и выше |
| порт занят | `NEXUS_PORT=8100 ./nexus.sh serve` |
| нужен закрытый API | `NEXUS_API_KEY=secret ./nexus.sh serve`, затем заголовок `Authorization: Bearer secret` |
| мало памяти при обучении | `--batch-size 1 --grad-accum 16 --seq-len 256` |
