# Обзор наработок — актуализировано 2026-08-20

## Модели и представления

- **Text2CAD (NeurIPS 2024)** подтвердил text → construction sequence и выпустил многоуровневые аннотации DeepCAD. Полный набор около 605 GB и лицензирован CC-BY-NC-SA-4.0, поэтому для коммерческих weights он исключён.
- **CAD-Recode (ICCV 2025)** использует Qwen2-1.5B, point-cloud encoder и миллион процедурных CadQuery-программ. Результат поддерживает идею небольшой специализированной модели, но опубликованный dataset — CC-BY-NC-4.0.
- **Zero-to-CAD (2026)** публикует миллион читаемых executable CadQuery sequences, STEP/STL и multi-view renders под Apache-2.0. Это наиболее интересный новый источник для расширения operation vocabulary.
- **CADFS (CVPR 2026)** предоставляет text/multiview → FeatureScript и около 90.5 GB данных под CC-BY-4.0; полезен после появления безопасного FeatureScript adapter.
- **CADEvolve (2026)** даёт примерно 1.3M развитых CadQuery programs под Apache-2.0.

## Геометрический runtime

- **Manifold** ориентирован на гарантированно manifold boolean output при корректных manifold inputs, имеет C++/Python/JS-WASM bindings и применяется OpenSCAD.
- **OpenSCAD Playground / openscad-wasm** подтверждают возможность локального browser rendering. Серверный CLI проще для текущего deployment, WASM остаётся путём к offline-first UI.
- **OpenSCAD Customizer** уже задаёт sliders комментариями; NeuroSCAD IR является источником этих параметров.
- Для точного STEP нужен **OpenCASCADE/B-Rep backend**. Mesh CSG и B-Rep имеют общий engineering intent, но разные компиляторы и критерии качества.

## Датасеты с ограничениями

- **Fusion 360 Gallery** — 8,625 реальных sketch/extrude histories, но условия использования non-commercial research.
- **SketchGraphs** — 15M constraints sketches; код MIT, однако copyright исходных sketches остаётся у авторов Onshape-документов.
- **CAD-Coder** заявляет Apache-2.0 для 250K text/CadQuery pairs, но указывает происхождение от Text2CAD; перед коммерческим использованием требуется отдельный provenance review.

Машинно-читаемое решение по каждому источнику: `data/catalog.json`.

## Ссылки

1. https://huggingface.co/datasets/SadilKhan/Text2CAD
2. https://cad-recode.github.io/
3. https://huggingface.co/datasets/ADSKAILab/Zero-To-CAD-1m
4. https://huggingface.co/datasets/VladPyatov/CADFS
5. https://huggingface.co/datasets/kulibinai/cadevolve
6. https://huggingface.co/datasets/gudo7208/CAD-Coder
7. https://www.autodesk.com/research/publications/fusion-360-gallery
8. https://github.com/PrincetonLIPS/SketchGraphs
9. https://manifoldcad.org/docs/html/
10. https://github.com/openscad/openscad-playground
11. https://files.openscad.org/documentation/manual/Customizer.html
