import os
import pytest
import torch
from fastapi.testclient import TestClient

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.multimodal import MultimodalFractalHoloNet, MultimodalSignalConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline
from fractal_holonet.serve import app

@pytest.fixture
def test_checkpoint_dir(tmp_path):
    checkpoint_dir = str(tmp_path / "model_test")
    config = FractalHoloNetConfig(vocab_size=300, d_model=64, n_layers=2, d_ff=128)
    model = ProductionFractalHoloNet(config)
    model.save_pretrained(checkpoint_dir)
    
    tokenizer = SimpleProductionTokenizer()
    tokenizer.save(os.path.join(checkpoint_dir, "tokenizer.json"))
    return checkpoint_dir

def test_text_model_forward_and_generate(test_checkpoint_dir):
    model = ProductionFractalHoloNet.from_pretrained(test_checkpoint_dir)
    model.eval()
    
    # Sequence forward pass
    inp = torch.randint(0, 300, (2, 16))
    logits, states = model(inp)
    assert logits.shape == (2, 16, 300)
    assert len(states) == 2
    
    # Step forward pass
    step_inp = torch.randint(0, 300, (2, 1))
    step_logits, next_states = model(step_inp, states=states, use_step=True)
    assert step_logits.shape == (2, 1, 300)

def test_multimodal_continuous_signal():
    config = MultimodalSignalConfig(
        input_signal_dim=2,
        output_signal_dim=2,
        patch_size=1,
        d_model=64,
        n_layers=2,
        d_ff=128,
        use_learnable_fourier_filter=True,
        num_fourier_filters=16,
        vocab_size=0
    )
    model = MultimodalFractalHoloNet(config)
    model.eval()
    
    # 1. Forward continuous pass
    raw_signal = torch.randn(2, 32, 2) # (B, T, channels)
    pred_sig, anom_scores, states = model.forward_continuous(raw_signal)
    assert pred_sig.shape == (2, 32, 2)
    assert anom_scores.shape == (2, 32, 1)
    
    # 2. O(1) Real-time forecasting
    history = torch.randn(1, 16, 2)
    forecast = model.forecast_stream(history, forecast_steps=10)
    assert forecast.shape == (1, 10, 2)

def test_fastapi_multimodal_endpoints():
    client = TestClient(app)
    
    # Health
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    
    # Info
    r = client.get("/info")
    assert r.status_code == 200
    assert "raw_audio" in r.json()["modalities"]
    
    # Text Generation
    r = client.post("/v1/generate", json={"prompt": "Fractal", "max_tokens": 10})
    assert r.status_code == 200
    assert "generated_text" in r.json()
    
    # Continuous Signal Forecasting & Anomaly Detection
    sample_signal = [[0.1], [0.5], [1.2], [0.9], [0.2], [0.05]]
    r = client.post("/v1/signal/forecast", json={"signal_history": sample_signal, "forecast_steps": 8})
    assert r.status_code == 200
    res = r.json()
    assert res["forecast_steps"] == 8
    assert len(res["anomaly_scores"]) == 6
    assert "latency_ms" in res

def test_distillation_pipeline(monkeypatch, tmp_path):
    from fractal_holonet.distillation import TeacherAPIClient, FractalHoloNetDistiller

    # Мок генерации ответов Teacher API для независимого теста
    def mock_generate(self, prompt, system_prompt=None, max_tokens=256, temperature=0.7):
        return f"Distilled answer for: {prompt}"

    monkeypatch.setattr(TeacherAPIClient, "generate_completion", mock_generate)

    tokenizer = SimpleProductionTokenizer()
    config = FractalHoloNetConfig(vocab_size=300, d_model=64, n_layers=2, d_ff=128)
    student = ProductionFractalHoloNet(config)
    teacher = TeacherAPIClient(endpoint="https://api.openai.com/v1", api_key="sk-test", model_name="gpt-4o-mini")

    distiller = FractalHoloNetDistiller(student_model=student, tokenizer=tokenizer, teacher_client=teacher, lr=1e-3)
    # Дистилляция пишет во временный каталог, а НЕ в прод-чекпоинт
    res = distiller.distill_from_teacher_api(
        prompts=["What is Fractal-HoloNet?", "Explain phase resonance."],
        epochs=2,
        batch_size=2,
        save_dir=str(tmp_path),
    )
    assert res["status"] == "success"
    assert res["epochs"] == 2
    assert res["final_loss"] > 0.0
    assert os.path.exists(os.path.join(str(tmp_path), "pytorch_model.pt"))
