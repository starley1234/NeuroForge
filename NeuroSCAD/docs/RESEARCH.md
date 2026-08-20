# Обзор наработок — актуализировано 2026-08-20

## Модели и представления

- **Text2CAD (NeurIPS 2024)** подтвердил text → construction sequence и выпустил многоуровневые аннотации DeepCAD. Полный набор около 605 GB и лицензирован CC-BY-NC-SA-4.0, поэтому для коммерческих weights он исключён.
- **CAD-Recode (ICCV 2025)** использует Qwen2-1.5B, point-cloud encoder и миллион процедурных CadQuery-программ. Результат поддерживает идею небольшой специализированной модели, но опубликованный dataset — CC-BY-NC-4.0.
- **Zero-to-CAD (2026)** публикует миллион читаемых executable CadQuery sequences, STEP/STL и multi-view renders под Apache-2.0. Это наиболее интересный новый источник для расширения operation vocabulary.
- **CADFS (CVPR 2026)** предоставляет text/multiview → FeatureScript и около 90.5 GB данных под CC-BY-4.0; полезен после появления безопасного FeatureScript adapter.
- **CADEvolve (2026)** даёт примерно 1.3M развитых CadQuery programs под Apache-2.0.

## Практические выводы для недорогого решения

- **Text-to-CadQuery** показал, что direct code generation использует уже имеющиеся code priors: Qwen2.5-3B дал лучший semantic score среди их открытых SFT-моделей (69.3%), тогда как более крупная модель не всегда была лучше по всем метрикам.
- **CAD-Coder (NeurIPS 2025)** применил SFT + executable filtering + geometric reward. Важный для нас результат: отобранные 8k high-quality примеров дали лучший результат, чем более крупный шумный набор.
- **Zero-to-CAD (2026)** масштабировал именно agentic generate → execute → inspect → repair pipeline, выпустив 1M Apache-2.0 CadQuery programs и curated 100k subset.
- **CADSmith (2026)** сочетает точные kernel metrics и multi-view VLM judge в двух correction loops; это сильнее чистого self-critique и подтверждает необходимость одновременно численного и визуального feedback.
- **CADTests/Text2CAD-Bench/MUSE** показывают failure cascade: исполняемый код ещё не означает правильную, производимую и функциональную конструкцию. Поэтому compile rate — только первый gate.
- **Don’t Mesh with Me** обучил 1.3B code model на 37,220 CSG parts и подтвердил, что небольшая code model может выучить CSG completion, хотя text control и сложность данных остаются ограничениями.

Практическое решение NeuroSCAD зафиксировано в [LOW_COST_DISTILLATION.md](LOW_COST_DISTILLATION.md): сначала RAG и teacher distillation на 500 собственных OpenSCAD, затем 3B LoRA, execution filtering и DPO; diffusion остаётся дальнейшим экспериментом.

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
12. https://arxiv.org/html/2505.06507v1
13. https://arxiv.org/html/2505.19713
14. https://arxiv.org/html/2604.24479
15. https://arxiv.org/html/2603.26512v1
16. https://arxiv.org/html/2411.15279v1
17. https://arxiv.org/html/2605.07807v1
18. https://arxiv.org/html/2605.18430v1
