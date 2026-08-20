# Dataset decision record — 2026-08-20

Полный реестр находится в `data/catalog.json`.

| Dataset | Масштаб | Лицензия/риск | Решение |
|---|---:|---|---|
| Owner OpenSCAD corpus | ~500 | права владельца, фиксировать по файлам | основной источник distillation/RAG и golden benchmark |
| NeuroSCAD Synthetic v2 | управляемый | Apache-2.0 | тестирование CSG-IR pipeline |
| Zero-to-CAD 1M | 1M | Apache-2.0 | выборочный источник разнообразия после проверки pipeline |
| CADFS | ~100K, 90.5 GB | CC-BY-4.0 | кандидат для multimodal после FeatureScript adapter |
| CADEvolve | ~1.3M | Apache-2.0 | кандидат для CadQuery diversity |
| CAD-Coder | 250K | заявлен Apache-2.0, upstream provenance требует проверки | legal review |
| Text2CAD | 605 GB | CC-BY-NC-SA-4.0, gated | только research benchmark |
| Fusion 360 Gallery | 8,625 | non-commercial research | только research benchmark |
| CAD-Recode | 1M | CC-BY-NC-4.0 | только research benchmark |
| SketchGraphs | 15M | код MIT, права на sketches у авторов | legal review |

## Правила ingestion

- Никакого исполнения внешнего CadQuery/FeatureScript/Python на API-хосте.
- Import worker без сети, с timeout/RAM/CPU limit.
- Каждый пример получает `source`, `license`, original ID и transform version.
- Сначала parse в IR, затем static validation и render; невалидные примеры карантинируются.
- Дедупликация по canonical IR hash и geometry hash.
- Split по семейству/генератору, не случайно по строкам.
- NonCommercial данные физически отделены и не входят в production training mix.
