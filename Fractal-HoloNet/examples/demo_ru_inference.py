import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractal_holonet.tokenizer import FractalHoloNetInferencePipeline

pipe = FractalHoloNetInferencePipeline(str(ROOT / "checkpoints" / "fractal_holonet_base"))

prompts = [
    "Искусственный интеллект — это",
    "Архитектура Fractal-HoloNet построена на",
    "Мороз и солнце; день чудесный!",
    "У лукоморья дуб зеленый;"
]

print("=" * 70)
print("  🇷🇺 ТЕСТИРОВАНИЕ РУССКОЯЗЫЧНОЙ ГЕНЕРАЦИИ FRACTAL-HOLONET")
print("=" * 70)

for p in prompts:
    res = pipe.generate(p, max_new_tokens=120, temperature=0.4, top_k=20, top_p=0.85)
    print(f"\n[ПРОМПТ]: {p}")
    print(f"[ПРОДОЛЖЕНИЕ]: {res['generated_text']}")
    print(f"[ИТОГОВЫЙ ТЕКСТ]:\n{res['full_text']}")
    print("-" * 70)
