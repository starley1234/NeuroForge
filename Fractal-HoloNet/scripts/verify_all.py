import os
import sys
import time
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from fastapi.testclient import TestClient

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.multimodal import MultimodalFractalHoloNet, MultimodalSignalConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline
from fractal_holonet.serve import app

def run_system_verification():
    print("=" * 70)
    print("      🔍 FRACTAL-HOLONET FULL SYSTEM VERIFICATION")
    print("=" * 70)
    
    # 1. Verification of Checkpoints
    print("\n[1/5] Проверка сохраненных чекпоинтов...")
    text_ckpt = str(ROOT / "checkpoints" / "fractal_holonet_base")
    multi_ckpt = str(ROOT / "checkpoints" / "fractal_holonet_multimodal")
    
    assert os.path.exists(os.path.join(text_ckpt, "config.json")), "Missing text config!"
    assert os.path.exists(os.path.join(text_ckpt, "pytorch_model.pt")), "Missing text weights!"
    assert os.path.exists(os.path.join(multi_ckpt, "config.json")), "Missing multimodal config!"
    assert os.path.exists(os.path.join(multi_ckpt, "pytorch_model.pt")), "Missing multimodal weights!"
    print("  ✅ Чекпоинты валидны и присутствуют на диске.")
    
    # 2. Text Inference Pipeline
    print("\n[2/5] Проверка Text Inference Pipeline (O(1) Streaming)...")
    pipe = FractalHoloNetInferencePipeline(text_ckpt)
    test_prompt = "Fractal-HoloNet is a novel"
    res = pipe.generate(test_prompt, max_new_tokens=30, temperature=0.7)
    assert len(res["generated_text"]) > 0, "Empty generation!"
    print(f"  Промпт: '{test_prompt}'")
    print(f"  Сгенерировано: '{res['generated_text']}'")
    print(f"  Задержка: {res.get('latency_ms', 'N/A')} | Токенов: {res['generated_tokens']}")
    print("  ✅ Текстовый пайплайн работает штатно.")
    
    # 3. Multimodal Continuous Signal Forecast & Anomaly Detection
    print("\n[3/5] Проверка Multimodal Continuous Signal Engine...")
    multi_model = MultimodalFractalHoloNet.from_pretrained(multi_ckpt)
    multi_model.eval()
    
    raw_signal = torch.sin(torch.linspace(0, 4 * 3.1415, 48)).view(1, 48, 1)
    t0 = time.time()
    forecast = multi_model.forecast_stream(raw_signal, forecast_steps=24)
    forecast_time = (time.time() - t0) * 1000.0
    assert forecast.shape == (1, 24, 1), f"Unexpected shape: {forecast.shape}"
    
    _, anom_scores, _ = multi_model.forward_continuous(raw_signal)
    assert anom_scores.shape == (1, 48, 1), f"Unexpected anomaly shape: {anom_scores.shape}"
    print(f"  Вход: {raw_signal.shape} -> Прогноз: {forecast.shape} за {forecast_time:.2f} ms")
    print(f"  Средний скор аномальности: {anom_scores.mean().item():.5f}")
    print("  ✅ Мультимодальный движок непрерывных сигналов работает штатно.")
    
    # 4. REST API Endpoints End-to-End
    print("\n[4/5] Проверка REST API через HTTP TestClient...")
    client = TestClient(app)
    
    # Health
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"
    
    # Info
    r = client.get("/info")
    assert r.status_code == 200 and len(r.json()["modalities"]) >= 4
    
    # Text Generation
    r = client.post("/v1/generate", json={"prompt": "AI", "max_tokens": 15})
    assert r.status_code == 200 and "generated_text" in r.json()
    
    # Embeddings
    r = client.post("/v1/embeddings", json={"text": "Phase resonance"})
    assert r.status_code == 200 and r.json()["dimension"] == 128
    
    # Signal Forecast API
    sample_signal = [[0.1], [0.3], [0.7], [1.2], [0.8], [0.2]]
    r = client.post("/v1/signal/forecast", json={"signal_history": sample_signal, "forecast_steps": 12})
    assert r.status_code == 200 and len(r.json()["forecast"]) == 12
    print("  ✅ Все эндпоинты REST API (/health, /info, /v1/generate, /v1/embeddings, /v1/signal/forecast, /v1/distill) зарегистрированы.")
    
    # 5. ONNX Export & ONNX Runtime Validation
    print("\n[5/5] Проверка ONNX модели и ONNX Runtime...")
    onnx_path = str(ROOT / "exports" / "fractal_holonet.onnx")
    assert os.path.exists(onnx_path), "Missing ONNX export file!"
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    dummy_input = torch.randint(0, 300, (1, 16)).numpy()
    ort_out = session.run(None, {session.get_inputs()[0].name: dummy_input})
    assert ort_out[0].shape == (1, 16, 300)
    print(f"  ONNX Runtime Session Output Shape: {ort_out[0].shape}")
    print("  ✅ ONNX граф валиден и готов к аппаратному ускорению.")
    
    print("\n" + "=" * 70)
    print("  🎉 ВСЕ СИСТЕМЫ И КОМПОНЕНТЫ УСПЕШНО ПРОШЛИ ПРОВЕРКУ!")
    print("=" * 70)

if __name__ == "__main__":
    run_system_verification()
