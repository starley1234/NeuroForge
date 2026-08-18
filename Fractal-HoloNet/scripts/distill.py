import os
import sys
import time
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fractal_holonet.distillation import TeacherAPIClient, FractalHoloNetDistiller
from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline

def run_distillation_cli():
    print("=" * 75)
    print("  🧠 KNOWLEDGE DISTILLATION: TEACHER LLM -> FRACTAL-HOLONET STUDENT")
    print("=" * 75)
    
    # 1. Параметры Teacher API (через переменные окружения или аргументы)
    endpoint = os.getenv("TEACHER_ENDPOINT", "https://api.openai.com/v1")
    api_key = os.getenv("TEACHER_API_KEY", "")
    model_name = os.getenv("TEACHER_MODEL", "gpt-4o-mini")
    
    # Обучающие промпты для дистилляции сложных рассуждений и знаний
    distill_prompts = [
        "What is Fractal-HoloNet and how does it achieve O(N) context complexity?",
        "Explain the advantage of complex phase resonance over standard self-attention.",
        "How can AI process continuous raw audio and ECG without tokenization?",
        "Why is constant O(1) inference state memory critical for autonomous agents?",
        "Write a concise explanation of holographic memory in neural networks."
    ]
    
    tokenizer = SimpleProductionTokenizer()
    ckpt_dir = str(ROOT / "checkpoints" / "fractal_holonet_base")
    
    if os.path.exists(os.path.join(ckpt_dir, "pytorch_model.pt")):
        student = ProductionFractalHoloNet.from_pretrained(ckpt_dir)
    else:
        config = FractalHoloNetConfig(vocab_size=300, d_model=128, n_layers=4, d_ff=384)
        student = ProductionFractalHoloNet(config)
        
    if not api_key:
        print("\n⚠️ Внимание: TEACHER_API_KEY не задан в окружении.")
        print("💡 Запускаем демонстрационную дистилляцию на локально сгенерированных ответах эксперта (Synthetic CoT)...")
        
        synthetic_pairs = [
            {
                "prompt": "What is Fractal-HoloNet?",
                "response": "Fractal-HoloNet is a next-generation neural architecture that uses complex phase resonance in C^d space to achieve linear O(N) context scaling and O(1) constant inference memory."
            },
            {
                "prompt": "Explain complex phase resonance.",
                "response": "Complex phase resonance accumulates key-value vectors onto oscillating phase angles e^(i*omega*t), enabling continuous non-decaying associative memory."
            },
            {
                "prompt": "Why is O(1) streaming state crucial?",
                "response": "O(1) streaming memory ensures zero latency growth and constant RAM usage regardless of how many thousands of tokens or sensor timesteps are processed."
            }
        ] * 10
        
        distiller = FractalHoloNetDistiller(student_model=student, tokenizer=tokenizer, lr=2e-3)
        res = distiller.distill_from_pairs(synthetic_pairs, epochs=6, batch_size=4, save_dir=ckpt_dir)
    else:
        print(f"\n📡 Подключение к Teacher API: {endpoint} (Модель: {model_name})")
        teacher = TeacherAPIClient(endpoint=endpoint, api_key=api_key, model_name=model_name)
        distiller = FractalHoloNetDistiller(student_model=student, tokenizer=tokenizer, teacher_client=teacher, lr=1.5e-3)
        res = distiller.distill_from_teacher_api(distill_prompts, epochs=5, batch_size=4, save_dir=ckpt_dir)
        
    print("\n--- 🌟 Тест Student модели Fractal-HoloNet после дистилляции ---")
    pipe = FractalHoloNetInferencePipeline(ckpt_dir)
    test_res = pipe.generate("What is Fractal-HoloNet?", max_new_tokens=40, temperature=0.6)
    print(f"Промпт: 'What is Fractal-HoloNet?'")
    print(f"Ответ Student-модели: '{test_res['generated_text']}'")

if __name__ == "__main__":
    run_distillation_cli()
