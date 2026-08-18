# AURA-Micro

Сверхмалая сеть для носимого **треугольника из 3 микрофонов**. Один вход в прод:

```bash
cd AURA-Micro
python -m aura_micro train          # обучение (синтез; ESC-50 если скачан)
python -m aura_micro serve          # консоль теста в браузере
```

Или `./run.sh train` / `./run.sh serve`.

## Режимы

| Команда | Что делает |
| --- | --- |
| `python -m aura_micro download` | качает **ESC-50** (2000 клипов, [karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50), CC BY-NC) |
| `python -m aura_micro train` | 60 шагов, ~минута на CPU. `--source auto\|synth\|esc50`, `--download` |
| `python -m aura_micro infer` | одна сцена в stdout |
| `python -m aura_micro serve` | UI + `/api/scene` на `0.0.0.0:8080` |
| `python -m aura_micro info` | Flash/RAM бюджет и статус данных |

Обучение **не требует** полевых записей: сухой звук (процедурный или ESC-50) прогоняется через физический пространственный рендер (ITD, Доплер, воздух, реверб, окклюзия). Классы ESC-50 мапятся на 30 голов AURA (`aura_micro/catalog.py`).

UrbanSound8K / FSD50K / AudioSet можно подложить так же: сухой моно-WAV + id класса → `render_scene(..., dry=)`.

## Железо

INT8 Flash ≤ 500 КБ, состояние FastGRNN = 64 числа, latency 25–40 мс, каскад Wake-on-Sound.

```
3×16 kHz → log-Mel + GCC-PHAT → Sinc stem → D-DS-CNN → FastGRNN → 4 heads
```

## Тесты

```bash
pip install -e ".[dev]"
pytest -q
```
