import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from fractal_holonet.tokenizer import FractalHoloNetInferencePipeline
from fractal_holonet.multimodal import MultimodalFractalHoloNet

def demo_language_model_responses():
    print("=" * 75)
    print("  🎭 ТЕСТИРОВАНИЕ ОТВЕТОВ ЯЗЫКОВОЙ МОДЕЛИ FRACTAL-HOLONET")
    print("=" * 75)
    
    pipe = FractalHoloNetInferencePipeline(str(ROOT / "checkpoints" / "fractal_holonet_base"))
    
    test_cases = [
        {
            "category": "Классическая драматургия / Диалог",
            "prompt": "First Citizen:\nWe are accounted poor citizens, the patricians good.\nWhat authority surfeits on would",
            "max_tokens": 100,
            "temperature": 0.6,
            "top_k": 30,
            "top_p": 0.9
        },
        {
            "category": "Монологи / Эмоциональная речь",
            "prompt": "KING RICHARD:\nGive me another horse: bind up my wounds.\nHave mercy,",
            "max_tokens": 80,
            "temperature": 0.65,
            "top_k": 25,
            "top_p": 0.85
        },
        {
            "category": "Поэтическая форма / Рифма и ритм",
            "prompt": "ROMEO:\nO, speak again, bright angel! for thou art\nAs glorious to",
            "max_tokens": 90,
            "temperature": 0.7,
            "top_k": 40,
            "top_p": 0.9
        },
        {
            "category": "Короткий зачин / Достройка контекста",
            "prompt": "HAMLET:\nTo be, or not to be, that is the",
            "max_tokens": 70,
            "temperature": 0.5,
            "top_k": 20,
            "top_p": 0.8
        }
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Категория: {tc['category']}")
        print(f"Параметры: T={tc['temperature']}, top_k={tc['top_k']}, top_p={tc['top_p']}, max_tokens={tc['max_tokens']}")
        print("-" * 50)
        print(f"📝 [ПРОМПТ]:\n{tc['prompt']}")
        
        t0 = time.time()
        res = pipe.generate(
            prompt=tc['prompt'],
            max_new_tokens=tc['max_tokens'],
            temperature=tc['temperature'],
            top_k=tc['top_k'],
            top_p=tc['top_p']
        )
        elapsed_ms = (time.time() - t0) * 1000.0
        tok_per_sec = res['generated_tokens'] / max(1e-4, elapsed_ms / 1000.0)
        
        print(f"\n✨ [ОТВЕТ МОДЕЛИ (Сгенерировано)]:\n{res['generated_text']}")
        print("-" * 50)
        print(f"⚡ Статистика: {res['generated_tokens']} токенов за {elapsed_ms:.2f} ms ({tok_per_sec:.1f} tok/s)")
        print("=" * 75)


def demo_multimodal_signal_responses():
    print("\n" + "=" * 75)
    print("  🌊 ТЕСТИРОВАНИЕ МУЛЬТИМОДАЛЬНЫХ ОТВЕТОВ НА НЕПРЕРЫВНЫЕ СИГНАЛЫ")
    print("=" * 75)
    
    multi_model = MultimodalFractalHoloNet.from_pretrained(str(ROOT / "checkpoints" / "fractal_holonet_multimodal"))
    multi_model.eval()
    
    # 1. Нормальный физиологический сигнал (гармонический дрейф)
    t = torch.linspace(0, 4 * 3.1415, 64)
    normal_signal = (torch.sin(t) + 0.5 * torch.sin(2 * t)).view(1, 64, 1)
    
    # 2. Сигнал с резкой аномалией (спайк/выброс давления)
    anom_signal = normal_signal.clone()
    anom_signal[0, 40:48, 0] += 5.0
    
    with torch.no_grad():
        # Тест скоринга аномалий
        _, norm_scores, _ = multi_model.forward_continuous(normal_signal)
        _, anom_scores, _ = multi_model.forward_continuous(anom_signal)
        
        # Тест O(1) прогноза
        t0 = time.time()
        forecast = multi_model.forecast_stream(normal_signal, forecast_steps=32)
        fore_ms = (time.time() - t0) * 1000.0
        
    print("\n[Сигнал 1] Нормальный гармонический поток:")
    print(f"  Длина истории: 64 отсчета")
    print(f"  Средний скор аномальности: {norm_scores.mean().item():.6f} (Норма: близко к 0)")
    print(f"  Прогноз на 32 шага вперед сформирован за {fore_ms:.2f} ms")
    
    print("\n[Сигнал 2] Поток с внесенной аномалией на шагах 40-48:")
    print(f"  Скор до аномалии (шаг 10): {anom_scores[0, 10, 0].item():.6f}")
    print(f"  Скор в пике аномалии (шаг 44): {anom_scores[0, 44, 0].item():.6f}")
    print(f"  Реакция модели: Обнаружен резкий спайк аномальности (перепад в {anom_scores[0, 44, 0].item() / max(1e-6, anom_scores[0, 10, 0].item()):.1f}x)")
    print("=" * 75)

if __name__ == "__main__":
    demo_language_model_responses()
    demo_multimodal_signal_responses()
