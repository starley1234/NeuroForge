# Обучение NeuroSCAD

## Рекомендуемая стратегия

Текущий `training/train.py` — простой Qwen LoRA text-to-IR baseline, а не окончательная архитектура. Целевая модель описана в [MODEL_ARCHITECTURE_V2.md](MODEL_ARCHITECTURE_V2.md): pretrained semantic encoder + hierarchical CAD autoencoder + categorical topology diffusion + continuous rectified flow.

Для RTX 5060 Ti 16 GB практичнее:

1. Доказать pipeline на собственном синтетическом CSG-IR.
2. Обучить и измерить простой code-model LoRA baseline.
3. Отдельно обучить CAD autoencoder и проверить точную реконструкцию.
4. Затем обучать coarse-to-fine topology diffusion и geometry flow.
5. Применить grammar-constrained decoding и статический validator.
6. Добавлять внешние данные только через parser → IR → render → license gate.
7. Продвигать checkpoint только после family-held-out и geometry regression тестов.

## Подготовка собственных данных

```bash
python3 -m training.prepare --output data/processed --samples 10000 --seed 42
```

Получаются `train.jsonl`, `validation.jsonl`, `test.jsonl` и manifest. Synthetic v2 равномерно смешивает хомуты, монтажные пластины и втулки, варьирует размеры и формулировки на русском/английском. Каждый target сначала проходит IR/constraint validation. Split определяется SHA-256 семейства и полного набора геометрических параметров: парафразы одной детали не оказываются одновременно в train и test.

## LoRA SFT

```bash
pip install -e '.[training]'
cp training/config.example.json training/config.local.json
python3 -m training.train --config training/config.local.json
```

Профиль рассчитан на CUDA и 16 GB VRAM: batch 1, gradient accumulation 16, BF16/FP16, gradient checkpointing, LoRA rank 16. Фактическая память зависит от версии CUDA, optimizer и длины примеров — её надо измерять, а не обещать заранее.

## Evaluation и promotion

Inference job должен записать JSONL с полями `prediction` и, желательно, `target`:

```bash
python3 -m training.evaluate outputs/predictions.jsonl
```

Минимальный автоматический gate сейчас: `IR valid rate >= 99%`. Для production дополнительно обязательны:

- compile/non-empty/watertight rates;
- точность размеров и feature F1;
- held-out split по семействам деталей;
- slider sweep по min/default/max;
- сравнение p95 latency и памяти с текущей моделью;
- ручная проверка опасных инженерных сценариев.

Checkpoint **не подключается к API автоматически**. Это защищает production от случайного ухудшения после обучения.

## Внешние датасеты

Политика и лицензии записаны машинно-читаемо в `data/catalog.json`. Для коммерческого продукта приоритетны собственная синтетика и источники Apache-2.0/CC-BY с сохранением provenance. NonCommercial-наборы допускаются только как изолированные исследовательские benchmarks и не смешиваются с production weights.
