# Обучение NeuroSCAD

## Рекомендуемая стратегия

Не начинать с 180M transformer «с нуля». Для RTX 5060 Ti 16 GB практичнее:

1. Доказать pipeline на собственном синтетическом CSG-IR.
2. Fine-tune небольшую code model 1.5B через LoRA.
3. Применить grammar-constrained decoding и статический validator.
4. Добавлять внешние данные только через parser → IR → render → license gate.
5. Продвигать checkpoint только после family-held-out и geometry regression тестов.

## Подготовка собственных данных

```bash
python3 -m training.prepare --output data/processed --samples 10000 --seed 42
```

Получаются `train.jsonl`, `validation.jsonl`, `test.jsonl` и manifest. Split определяется SHA-256 полного набора геометрических параметров: парафразы одной детали не оказываются одновременно в train и test.

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
