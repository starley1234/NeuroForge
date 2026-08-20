# Краткий обзор наработок (проверено 2026-08-20)

- **Text2CAD (NeurIPS 2024)** генерирует construction sequences из многоуровневых описаний; его аннотации DeepCAD дают хороший baseline, но не доказывают production reliability.
- **Fusion 360 Gallery**: 8,625 человеческих sketch/extrude histories и gym; полезная реалистичная выборка, однако лицензия исследовательская.
- **Manifold** ориентирован на гарантированно manifold boolean output при корректных manifold inputs, имеет C++/Python/JS-WASM bindings и применяется OpenSCAD.
- **OpenSCAD Playground / openscad-wasm** подтверждают browser rendering. Для MVP CLI проще; WASM логичен далее для приватности и меньшей серверной нагрузки.
- **OpenSCAD Customizer** уже задаёт sliders комментариями. IR должен быть источником этого формата, а не дублировать его вручную.
- Для точного STEP нужен **OpenCASCADE** backend. Mesh CSG и B-Rep должны иметь общий intent IR, но разные компиляторы.

## Ссылки

1. https://sadilkhan.github.io/text2cad-project/
2. https://proceedings.neurips.cc/paper_files/paper/2024/file/0e5b96f97c1813bb75f6c28532c2ecc7-Paper-Conference.pdf
3. https://www.autodesk.com/research/publications/fusion-360-gallery
4. https://manifoldcad.org/docs/html/
5. https://github.com/openscad/openscad-playground
6. https://files.openscad.org/documentation/manual/Customizer.html
7. https://github.com/BelfrySCAD/BOSL2/wiki/Tutorial-Attachment-Attach
