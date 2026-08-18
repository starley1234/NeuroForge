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

### WSL / Git Bash на диске C:

Если репозиторий склонирован под Windows (`core.autocrlf=true`), скрипт получает
окончания строк CRLF, и bash падает:

```
env: $'bash\r': No such file or directory
```

Лечится один раз:

```bash
git pull                       # в репозитории теперь есть .gitattributes с eol=lf
sed -i 's/\r$//' nexus.sh      # если файл уже лежал с CRLF
chmod +x nexus.sh
./nexus.sh setup
```

Совсем «в лоб», без скрипта:

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/python -m nexus.cli quickstart --scale small
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

| Масштаб | Деталей | Математика | Модель | Шагов | Время |
| :-- | --: | --: | :-- | --: | :-- |
| `nano` | 8 | — | tiny (1M) | 8 | ~8 с CPU — проверить, что всё живо |
| `small` | 48 | 200 | tiny (1M) | 60 | ~20 с CPU — демо пайплайна (по умолчанию) |
| `medium` | 500 | 5 000 | small (200M) | 1 500 | минуты на GPU / ~час CPU |
| `gpu` | 5 000 | 50 000 | small (200M) | 20 000 | часы на RTX 5060 Ti — **первая осмысленная модель** |
| `gpu-large` | 20 000 | 200 000 | 1.4B активных | 60 000 | сутки на RTX 5060 Ti |

Любой параметр переопределяется:

```powershell
.\nexus.ps1 cli quickstart --scale gpu --samples 8000 --steps 30000 --math 100000 `
                           --vocab 16384 --seq-len 1024 --preset small --device cuda
```

`nano` и `small` — это **демо пайплайна**: 48 деталей и 60 шагов физически не могут
научить модель языку, они лишь доказывают, что весь конвейер (данные → токенизатор →
обучение → приёмка → реестр) работает. Осмысленный минимум — `--scale medium`,
реальный — `--scale gpu`.

Что делает `quickstart`: проверяет окружение → генерирует детали OpenSCAD и
считает по ним FEM → обучает BPE-токенизатор → обучает ядро → прогоняет
приёмочные тесты и, если они прошли, помечает версию как `production`.

## 1.1 Видеокарта NVIDIA (важно!)

`pip install torch` тянет с PyPI **CPU-сборку**, поэтому на машине с RTX всё
молча считается на процессоре. Для карт 50-й серии (RTX 5060/5070/5080/5090,
архитектура Blackwell, sm_120) нужны колёса **cu128** и PyTorch ≥ 2.7.

`setup` теперь сам находит `nvidia-smi` и ставит правильную сборку. Если torch
уже стоит в CPU-варианте — одна команда:

```powershell
.\nexus.ps1 gpu       # Windows
./nexus.sh gpu        # Linux / WSL
```

Вручную то же самое:

```powershell
.\.venv\Scripts\pip uninstall -y torch
.\.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Именно `--index-url`, а не `--extra-index-url`: иначе pip снова возьмёт CPU-колесо
с PyPI. Переопределить выбор: `$env:NEXUS_CUDA = "cu126"` (40xx/30xx) или `"cpu"`.

Проверка: `.\nexus.ps1 doctor` → `"cuda_available": true`, `"gpu_capability": "sm_120"`.

Дальше запускайте с GPU-масштабом:

```powershell
$env:NEXUS_SCALE = "gpu"
.\nexus.ps1 quickstart          # либо: .\nexus.ps1 cli quickstart --scale gpu --device cuda
```

Для 16 ГБ VRAM (5060 Ti): пресет `rtx5060-compact` (~8.4 ГБ) или `rtx5060` (~12.2 ГБ),
AMP включается автоматически при обучении на CUDA.

## 2. Проверка, что работает

Проще всего — открыть в браузере **http://localhost:8000/docs**: Swagger UI со всеми
методами, примерами тел запросов и кнопкой «Try it out». Машинная спецификация —
`/openapi.json`, альтернативный вид — `/redoc`.

> В PowerShell `curl` — это алиас `Invoke-WebRequest`, и одинарные кавычки ломают JSON.
> Используйте `curl.exe` или встроенный командлет:
> ```powershell
> curl.exe -s localhost:8000/v1/analyze -H "Content-Type: application/json" `
>   -d "{\"code\":\"cube([20,20,4],center=true);\",\"force\":[0,0,-300]}"
> # либо
> Invoke-RestMethod -Uri localhost:8000/v1/analyze -Method Post -ContentType application/json `
>   -Body (@{ code = "cube([20,20,4],center=true);"; force = @(0,0,-300) } | ConvertTo-Json)
> ```

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
| PowerShell: `Отсутствует закрывающий знак "}"` | старый `nexus.ps1` без BOM; сделайте `git pull` (файл теперь ASCII + BOM + CRLF) |
| WSL: `env: $'bash\r'` | CRLF-окончания: `sed -i 's/\r$//' nexus.sh` после `git pull` |
| Python 3.13+ и torch не ставится | возьмите Python 3.11/3.12 (`py -3.11 -m venv .venv`) |
| Есть RTX, но `CUDA: False` | стоит CPU-сборка torch: `git pull`, затем `.\nexus.ps1 gpu` (RTX 50xx → cu128) |
| PowerShell: `POST /v1/analyze → 400` | `curl` там алиас `Invoke-WebRequest`; используйте `curl.exe`, `Invoke-RestMethod` или `/docs` |
| «Модель ничего не выучила» | масштаб `nano`/`small` — это демо; берите `--scale medium` или `gpu`. Ориентир: `val_ppl` должна быть много меньше размера словаря |
| В ответе `/v1/generate` поле `quality_hint` | сервис сам предупреждает, что версия недообучена |
| `loss=nan` на GPU | было в ревизиях до `e264ad6`; обновитесь (`git pull`) — теперь bfloat16 и защита от NaN |
| Ошибка `index is on cpu ... cuda:0` | тоже исправлено: приёмка берёт устройство модели |
| `sm_120 is not compatible` | старое колесо CUDA: нужен cu128 и torch ≥ 2.7 |
| torch ставится очень долго | это ~800 МБ; для GPU-сборки `NEXUS_GPU=1 ./nexus.sh setup` |
| `quickstart` завершился с «ПРОВАЛЕНО» | норма для `nano`: слишком мало шагов; берите `small` и выше |
| порт занят | `NEXUS_PORT=8100 ./nexus.sh serve` |
| нужен закрытый API | `NEXUS_API_KEY=secret ./nexus.sh serve`, затем заголовок `Authorization: Bearer secret` |
| мало памяти при обучении | `--batch-size 1 --grad-accum 16 --seq-len 256` |
