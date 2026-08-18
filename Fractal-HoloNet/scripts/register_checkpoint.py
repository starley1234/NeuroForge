"""
Импорт существующих чекпоинтов в реестр моделей (одноразовая миграция).

Запуск: python scripts/register_checkpoint.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fractal_holonet.registry as registry


def main():
    print("=" * 70)
    print("  ИМПОРТ СУЩЕСТВУЮЩИХ ЧЕКПОИНТОВ В РЕЕСТР")
    print("=" * 70)

    entries = [
        dict(
            model_id="legacy-base-ru-v1",
            name="Fractal-HoloNet v1 (byte tokenizer, ru corpus)",
            checkpoint_dir=str(ROOT / "checkpoints" / "fractal_holonet_base"),
            architecture="fractal-holonet-v1",
            tokenizer_type="byte",
            tokenizer_vocab=260,
            config_summary={"d_model": 128, "n_layers": 4, "d_ff": 384, "vocab_size": 300},
            metrics={"params": 1023104, "greedy_acc_corpus": 0.92},
            notes="Обучен train_russian.py (легаси-скрипт). Рассинхрон vocab: конфиг 300 vs словарь 260.",
        ),
        dict(
            model_id="legacy-signal-v1",
            name="Fractal-HoloNet v1 multimodal (continuous signals)",
            checkpoint_dir=str(ROOT / "checkpoints" / "fractal_holonet_multimodal"),
            architecture="fractal-holonet-multimodal",
            tokenizer_type="",
            tokenizer_vocab=0,
            config_summary={"d_model": 128, "n_layers": 4, "d_ff": 384, "input_signal_dim": 1},
            metrics={"params": 997729, "forecast_mse_1step": 0.0274, "anomaly_bce": 0.00019,
                     "forecast_mse_64step": 0.50},
            notes="Обучен train_multimodal.py на синтетике (ЭКГ-подобные сигналы).",
        ),
        dict(
            model_id="elast-holo-v2-lm",
            name="ELAST-HOLO v2 (ElasticHoloNet)",
            checkpoint_dir=str(ROOT / "checkpoints" / "fractal_holonet_v2"),
            architecture="elast-holo-v2",
            tokenizer_type="byte",
            tokenizer_vocab=260,
            config_summary={"d_model": 64, "n_layers": 4, "d_ff": 192, "vocab_size": 300,
                            "n_read_iters": 2, "use_slow_memory": True},
            metrics={"params": 430660, "greedy_acc_corpus": 0.922,
                     "mqar_acc_L24_v2": 0.260, "mqar_acc_L24_v1": 0.155,
                     "mqar_extrap_L48_v2": 0.180, "mqar_extrap_L48_v1": 0.100},
            notes="ELAST-HOLO v2 (M1-M5, M7). См. research/ARCHITECTURE_V2.md.",
        ),
    ]

    for e in entries:
        mid = e["model_id"]
        if _exists(mid):
            print(f"  [skip] {mid} уже в реестре")
            continue
        registry.import_existing(**e)
        print(f"  [ok]   {mid} импортирован ({e['checkpoint_dir']})")

    print("\nРеестр моделей:")
    for m in registry.list_models():
        print(f"  - {m['id']:<24} {m['status']:<10} {m['name'][:50]}")


def _exists(model_id: str) -> bool:
    try:
        registry.get_model(model_id)
        return True
    except KeyError:
        return False


if __name__ == "__main__":
    main()
