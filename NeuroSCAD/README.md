# NeuroSCAD

**Typed intent → deterministic solid code.** Работающий vertical slice преобразования инженерного ТЗ в параметрический OpenSCAD.

> `v0.1 prototype`: сейчас распознаётся семейство разрезных хомутов/кронштейнов. Нейросеть, STEP и FEM намеренно не имитируются заглушками.

## Уже работает

- русское/английское ТЗ → параметризованный CSG-IR;
- проверка ссылок, арности, циклов, диапазонов и размеров;
- детерминированный OpenSCAD и Customizer sliders;
- опциональный render OpenSCAD/Manifold и Trimesh-анализ;
- CLI, optional FastAPI и zero-dependency core.

## Быстрый старт

```bash
cd NeuroSCAD
python3 -m neuroscad.cli generate \
  'Кронштейн для камеры на трубу 25 мм с фиксацией винтом М4' \
  --ir examples/camera_bracket.json --scad examples/camera_bracket.scad
python3 -m unittest discover -s tests -v
python3 -m neuroscad.cli validate examples/camera_bracket.json
```

Настоящий STL render (нужен системный OpenSCAD):

```bash
pip install -e '.[validation]'
python3 -m neuroscad.cli validate examples/camera_bracket.json --render
```

API:

```bash
pip install -e '.[api]'
uvicorn neuroscad.api:app --host 0.0.0.0 --port 8000
curl -X POST http://localhost:8000/v1/generate -H 'content-type: application/json' \
  -d '{"prompt":"хомут на трубу 25 мм, винт М4"}'
```

## Структура

```text
neuroscad/ir.py          typed IR, expressions, validation
neuroscad/compiler.py    deterministic OpenSCAD compiler
neuroscad/templates.py   полезный pre-model baseline
neuroscad/validator.py   layered IR → geometry → mesh report
neuroscad/api.py         optional FastAPI adapter
schema/                  machine-readable IR contract
docs/ARCHITECTURE.md     решения, reality checks, roadmap
docs/RESEARCH.md         prior art и ссылки
examples/                воспроизводимый bracket IR/SCAD
```

`valid: true, level: ir` означает лишь корректный IR — **не** доказанную водонепроницаемость или прочность. `geometry` требует render; `mesh` — проверки сетки. Прочность без материала, закреплений, нагрузки и FEM не заявляется.

Подробнее: [архитектура](docs/ARCHITECTURE.md) и [обзор наработок](docs/RESEARCH.md).
