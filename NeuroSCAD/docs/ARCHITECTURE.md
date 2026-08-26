# Архитектура NeuroSCAD: решение после проверки гипотез

## Главный вывод

Нейросеть не должна быть геометрическим ядром и не должна сразу генерировать SCAD-строки. Её задача — предложить **типизированное намерение** (`CSG-IR`), а детерминированные компоненты обязаны проверить грамматику, собрать геометрию и сообщить уровень доказанной валидности.

```text
текст / изображение / ограничения
              │
      intent extractor (MVP: шаблоны; позже: VLM)
              ▼
       typed CSG-IR + параметры
          │          │
  static validator   └── OpenSCAD compiler ── Customizer
          │                         │
          └──────── OpenSCAD/Manifold render ── mesh metrology
```

## Что изменено относительно исходной идеи

1. **IR — граф с явными ссылками и арностью, а не плоские пары `class + R⁹`.** У операций разная размерность; единый R⁹ теряет семантику и единицы. Выражения хранят связи вроде `outer_d = tube_d + 2*wall`, поэтому слайдер сохраняет параметричность.
2. **Грамматика и геометрическая валидность разделены.** Корректный AST не гарантирует непустой solid, manifold mesh или соответствие ТЗ. Отчёт указывает достигнутый уровень: `ir`, `geometry`, `mesh`.
3. **OpenSCAD и STEP — разные ветви.** STL — сетка. Настоящий STEP требует B-Rep ядра (OCCT/build123d/CadQuery). В MVP реализован честный SCAD/STL путь; B-Rep backend запланирован отдельно.
4. **Скорость измеряется по стадиям.** `<50 ms` реалистично для небольшого decode или шаблонного extraction, но не гарантируется для полного quality-render.
5. **FEM не заменён PyBullet.** PyBullet — rigid-body dynamics, не расчёт напряжений. Для прочности нужны материал, закрепления, сетка, нагрузки и CalculiX/Code_Aster.
6. **«100% watertight» не является аксиомой.** Manifold даёт manifold-result для допустимых manifold-inputs, но входы, пустые boolean и технологичность всё равно проверяются.

## Почему MVP полезен

- воспроизводимый путь `prompt → IR JSON → SCAD`;
- параметры автоматически становятся OpenSCAD Customizer sliders;
- модель не исполняет произвольный shell/Python/SCAD код;
- IR служит форматом датасета и constrained decoding;
- тестируемое ядро не зависит от CUDA, OpenSCAD или FastAPI.

## Следующие этапы

### M1 — CAD-компилятор
- sketch/extrude/revolve, chamfer/fillet, pattern;
- dimensional types (`Length`, `Angle`, scalar);
- OpenSCAD/Manifold backend для STL и OCCT backend для STEP;
- property-based/fuzz tests и regression corpus.

### M2 — данные и модель
- DeepCAD/Text2CAD и Fusion 360 Gallery с проверкой лицензий;
- семейства реальных деталей вместо равномерно случайных деревьев;
- topology/geometry dedup и split по семействам против leakage;
- baseline retrieval + parameter fitting и Qwen LoRA;
- целевая [Hierarchical CAD Flow v2](MODEL_ARCHITECTURE_V2.md): categorical topology diffusion + continuous rectified flow;
- visual feedback после representation/SFT обучения, не вместо точных метрик.

### M3 — инженерная проверка
- minimum wall, clearance, disconnected shells, overhang/bridging;
- материал и профиль принтера как обязательные входы;
- FEM с convergence checks;
- human approval gate: система предлагает, но не сертифицирует.

## Метрики

| Слой | Метрика |
|---|---|
| Синтаксис | IR parse rate, grammar-valid rate |
| Исполнение | compile rate, non-empty solid rate |
| Геометрия | watertight, components, volume, Chamfer/IoU |
| Параметричность | slider success по диапазону, preservation constraints |
| Семантика | dimension accuracy, feature F1, human task completion |
| Производство | min wall, unsupported area, tolerance violations |
| Скорость | p50/p95 отдельно для decode, compile, validation |

## Security boundary

API принимает JSON закрытой схемы, ограничивает число узлов/глубину/диапазоны и запускает OpenSCAD в контейнере без сети, с read-only filesystem и CPU/RAM/time limits. Нельзя исполнять код модели непосредственно на хосте.
