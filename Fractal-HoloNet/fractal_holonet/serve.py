import os
import time
import threading
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import torch

from fractal_holonet.core import ProductionFractalHoloNet, FractalHoloNetConfig
from fractal_holonet.multimodal import MultimodalFractalHoloNet, MultimodalSignalConfig
from fractal_holonet.tokenizer import SimpleProductionTokenizer, FractalHoloNetInferencePipeline
from fractal_holonet.distillation import TeacherAPIClient, FractalHoloNetDistiller
from fractal_holonet.self_train import SelfTrainLoop, SelfTrainService, SyntheticTeacher
import fractal_holonet.registry as registry

TEXT_MODEL_DIR = os.getenv("TEXT_MODEL_DIR", "./checkpoints/fractal_holonet_base")
SIGNAL_MODEL_DIR = os.getenv("SIGNAL_MODEL_DIR", "./checkpoints/fractal_holonet_multimodal")

pipeline: Optional[FractalHoloNetInferencePipeline] = None
multimodal_model: Optional[MultimodalFractalHoloNet] = None
self_train_service: Optional[SelfTrainService] = None
active_model_id: Optional[str] = None

# Сериализация мутаций чекпоинта (дистилляция и self-training)
train_lock = threading.Lock()


def resolve_text_model_dir() -> str:
    """Модель для текстового пайплайна: FH_MODEL_ID из реестра или TEXT_MODEL_DIR."""
    model_id = os.getenv("FH_MODEL_ID", "")
    if model_id:
        if model_id == "active":
            model_id = registry.get_active()
            if not model_id:
                raise RuntimeError("FH_MODEL_ID=active, но активная модель не задана в реестре")
        return str(registry.model_dir(model_id))
    return TEXT_MODEL_DIR


def init_services():
    global pipeline, multimodal_model
    # 1. Text Pipeline (реестр моделей или файловый каталог)
    model_dir = resolve_text_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        config = FractalHoloNetConfig(vocab_size=300, d_model=128, n_layers=4, d_ff=384)
        model = ProductionFractalHoloNet(config)
        model.save_pretrained(model_dir)
        tok = SimpleProductionTokenizer()
        tok.save(os.path.join(model_dir, "tokenizer.json"))
    pipeline = FractalHoloNetInferencePipeline(model_dir=model_dir, device="cpu")
    
    # 2. Multimodal Continuous Signal Model
    if os.path.exists(SIGNAL_MODEL_DIR):
        multimodal_model = MultimodalFractalHoloNet.from_pretrained(SIGNAL_MODEL_DIR, map_location="cpu")
    else:
        sig_cfg = MultimodalSignalConfig(
            input_signal_dim=1,
            output_signal_dim=1,
            patch_size=1,
            d_model=128,
            n_layers=4,
            d_ff=384,
            vocab_size=0
        )
        multimodal_model = MultimodalFractalHoloNet(sig_cfg)
        multimodal_model.save_pretrained(SIGNAL_MODEL_DIR)
    multimodal_model.eval()

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_services()
    yield

app = FastAPI(
    title="Fractal-HoloNet Multimodal & Distillation AI Service",
    description="Universal continuous signal & text intelligence with Knowledge Distillation from Teacher LLMs",
    version="2.1.0",
    lifespan=lifespan
)

# Text Schemas
class GenerationRequest(BaseModel):
    prompt: str = Field(..., json_schema_extra={"example": "Fractal-HoloNet is a novel"})
    max_tokens: int = Field(default=50, ge=1, le=1024)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)

class GenerationResponse(BaseModel):
    prompt: str
    generated_text: str
    full_text: str
    prompt_tokens: int
    generated_tokens: int
    latency_ms: float

class EmbeddingRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Signal embedding"})

class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimension: int

# Continuous Signal Schemas
class SignalForecastRequest(BaseModel):
    signal_history: List[List[float]] = Field(
        ...,
        description="Continuous raw multi-channel signal timesteps [[ch1, ch2...], ...]",
        json_schema_extra={"example": [[0.12], [0.35], [0.89], [1.45], [0.80], [0.20]]}
    )
    forecast_steps: int = Field(default=32, ge=1, le=512)

class SignalForecastResponse(BaseModel):
    forecast: List[List[float]]
    forecast_steps: int
    channels: int
    anomaly_scores: List[float]
    latency_ms: float

# Distillation Schemas
class DistillationAPIRequest(BaseModel):
    teacher_endpoint: str = Field(..., description="OpenAI-совместимый URL (e.g. https://api.openai.com/v1, https://openrouter.ai/api/v1)", json_schema_extra={"example": "https://api.openai.com/v1"})
    teacher_model: str = Field(..., description="Название модели Teacher (e.g. gpt-4o-mini, deepseek-chat, mistral-7b)", json_schema_extra={"example": "gpt-4o-mini"})
    teacher_api_key: str = Field(..., description="API-ключ доступа к Teacher модели")
    prompts: List[str] = Field(..., min_length=1, description="Список задач/промптов для генерации обучающего датасета учителем")
    system_prompt: Optional[str] = Field(default="You are an expert AI providing concise and accurate reasoning.")
    epochs: int = Field(default=5, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=1e-3, ge=1e-5, le=1e-2)

class DistillationResponse(BaseModel):
    status: str
    teacher_model: str
    samples_distilled: int
    epochs: int
    final_loss: float
    duration_sec: float
    checkpoint_dir: str

# Self-training (autonomous teacher-driven learning) Schemas
class SelfTrainRequest(BaseModel):
    teacher_endpoint: Optional[str] = Field(default="https://api.openai.com/v1")
    teacher_model: Optional[str] = Field(default="gpt-4o-mini")
    teacher_api_key: Optional[str] = Field(default="")
    prompts: Optional[List[str]] = Field(default=None, description="Учебные промпты; если пусто — встроенная учебная программа")
    epochs: int = Field(default=4, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    curriculum: bool = Field(default=True, description="Студент сам генерирует дополнительные промпты")

class SelfTrainResponse(BaseModel):
    round: int
    accepted: bool
    loss_before: float
    loss_after: float
    best_loss: float
    duration_sec: float
    samples: int
    teacher_mode: str
    checkpoint_dir: str

class SelfTrainServiceRequest(BaseModel):
    teacher_endpoint: Optional[str] = Field(default="https://api.openai.com/v1")
    teacher_model: Optional[str] = Field(default="gpt-4o-mini")
    teacher_api_key: Optional[str] = Field(default="")
    interval_sec: float = Field(default=300.0, ge=5.0, le=86400.0)
    epochs: int = Field(default=4, ge=1, le=50)
    batch_size: int = Field(default=4, ge=1, le=64)
    curriculum: bool = Field(default=True)

class ModelInfoResponse(BaseModel):
    architecture: str
    modalities: List[str]
    features: List[str]
    device: str
    status: str

@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "healthy", "service": "Fractal-HoloNet Multimodal & Distillation Inference", "timestamp": time.time()}

@app.get("/info", response_model=ModelInfoResponse)
def model_info():
    return {
        "architecture": "Multimodal Fractal Gated Holographic Resonance Network",
        "modalities": ["text", "raw_audio", "ecg_biomedical", "iot_telemetry", "continuous_streams"],
        "features": ["O(N) context", "O(1) step inference", "continuous signal forecasting", "knowledge distillation"],
        "device": "cpu",
        "status": "ready"
    }

@app.post("/v1/generate", response_model=GenerationResponse)
def generate_text(req: GenerationRequest):
    global pipeline
    if pipeline is None:
        init_services()
    t0 = time.time()
    res = pipeline.generate(
        prompt=req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p
    )
    latency_ms = (time.time() - t0) * 1000.0
    return {
        "prompt": res["prompt"],
        "generated_text": res["generated_text"],
        "full_text": res["full_text"],
        "prompt_tokens": res["prompt_tokens"],
        "generated_tokens": res["generated_tokens"],
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def get_embeddings(req: EmbeddingRequest):
    global pipeline
    if pipeline is None:
        init_services()
    emb = pipeline.get_embeddings(req.text)
    return {"embedding": emb, "dimension": len(emb)}

@app.post("/v1/signal/forecast", response_model=SignalForecastResponse)
def forecast_signal(req: SignalForecastRequest):
    global multimodal_model
    if multimodal_model is None:
        init_services()
        
    t0 = time.time()
    sig_tensor = torch.tensor([req.signal_history], dtype=torch.float32)
    
    with torch.no_grad():
        _, anom_scores_tensor, _ = multimodal_model.forward_continuous(sig_tensor)
        anom_scores = anom_scores_tensor[0, :, 0].cpu().tolist()
        forecast_tensor = multimodal_model.forecast_stream(sig_tensor, forecast_steps=req.forecast_steps)
        forecast_list = forecast_tensor[0].cpu().tolist()
        
    latency_ms = (time.time() - t0) * 1000.0
    
    return {
        "forecast": forecast_list,
        "forecast_steps": len(forecast_list),
        "channels": len(forecast_list[0]) if forecast_list else 1,
        "anomaly_scores": anom_scores,
        "latency_ms": round(latency_ms, 2)
    }

@app.post("/v1/distill", response_model=DistillationResponse)
def distill_model(req: DistillationAPIRequest):
    """
    Эндпоинт дистилляции знаний: обращается к Teacher модели через endpoint, model и api_key,
    генерирует эталонные ответы/рассуждения и обучает локальную модель Fractal-HoloNet.
    """
    global pipeline
    if pipeline is None:
        init_services()
        
    with train_lock:
        try:
            teacher_client = TeacherAPIClient(
                endpoint=req.teacher_endpoint,
                api_key=req.teacher_api_key,
                model_name=req.teacher_model
            )
            distiller = FractalHoloNetDistiller(
                student_model=pipeline.model,
                tokenizer=pipeline.tokenizer,
                teacher_client=teacher_client,
                lr=req.learning_rate,
                device="cpu"
            )
            result = distiller.distill_from_teacher_api(
                prompts=req.prompts,
                system_prompt=req.system_prompt,
                epochs=req.epochs,
                batch_size=req.batch_size,
                save_dir=TEXT_MODEL_DIR
            )

            # Перезагружаем пайплайн со свежими весами
            pipeline = FractalHoloNetInferencePipeline(model_dir=TEXT_MODEL_DIR, device="cpu")

            return {
                "status": "success",
                "teacher_model": req.teacher_model,
                "samples_distilled": len(req.prompts),
                "epochs": result["epochs"],
                "final_loss": result["final_loss"],
                "duration_sec": result["duration_sec"],
                "checkpoint_dir": result["save_dir"]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Distillation failed: {str(e)}")


@app.post("/v1/self-train", response_model=SelfTrainResponse)
def self_train_once(req: SelfTrainRequest):
    """
    Один раунд автономного самообучения: развёрнутая LLM (teacher) генерирует
    обучающие данные, студент дообучается, а eval-гейт принимает раунд только
    при улучшении holdout-лосса (иначе веса откатываются).
    """
    global pipeline
    if pipeline is None:
        init_services()

    with train_lock:
        teacher = (
            TeacherAPIClient(endpoint=req.teacher_endpoint, api_key=req.teacher_api_key, model_name=req.teacher_model)
            if req.teacher_api_key
            else SyntheticTeacher()
        )
        loop = SelfTrainLoop(
            student_model=pipeline.model,
            tokenizer=pipeline.tokenizer,
            teacher_client=teacher,
            save_dir=TEXT_MODEL_DIR,
        )
        result = loop.run_round(
            prompts=req.prompts, epochs=req.epochs, batch_size=req.batch_size, curriculum=req.curriculum
        )
        if result["accepted"]:
            pipeline = FractalHoloNetInferencePipeline(model_dir=TEXT_MODEL_DIR, device="cpu")
        return {
            "round": result["round"],
            "accepted": result["accepted"],
            "loss_before": result["loss_before"],
            "loss_after": result["loss_after"],
            "best_loss": result["best_loss"],
            "duration_sec": result["duration_sec"],
            "samples": result["samples"],
            "teacher_mode": result["teacher_mode"],
            "checkpoint_dir": result["save_dir"],
        }


@app.post("/v1/self-train/start")
def self_train_start(req: SelfTrainServiceRequest):
    """Запуск фонового цикла самообучения (daemon): LLM-учитель периодически
    дообучает модель с eval-гейтом."""
    global self_train_service, pipeline
    if pipeline is None:
        init_services()

    with train_lock:
        if self_train_service is not None and self_train_service.is_running():
            return {"status": "already_running", **self_train_service.status()}

        def _reload_pipeline(_result):
            global pipeline
            pipeline = FractalHoloNetInferencePipeline(model_dir=TEXT_MODEL_DIR, device="cpu")

        self_train_service = SelfTrainService(
            checkpoint_dir=TEXT_MODEL_DIR,
            save_dir=TEXT_MODEL_DIR,
            teacher_endpoint=req.teacher_endpoint,
            teacher_api_key=req.teacher_api_key,
            teacher_model=req.teacher_model,
            interval_sec=req.interval_sec,
            epochs=req.epochs,
            batch_size=req.batch_size,
            curriculum=req.curriculum,
            on_accepted=_reload_pipeline,
        )
        self_train_service.start()
    return {"status": "started", **self_train_service.status()}


@app.post("/v1/self-train/stop")
def self_train_stop():
    global self_train_service
    if self_train_service is not None:
        self_train_service.stop()
    return {"status": "stopped", "rounds": self_train_service.rounds if self_train_service else 0}


@app.get("/v1/self-train/status")
def self_train_status():
    if self_train_service is None:
        return {"running": False, "rounds": 0, "teacher_mode": "none", "last_result": None}
    return self_train_service.status()


# ---------------------------------------------------------------------------
# Реестр обученных моделей
# ---------------------------------------------------------------------------
class ActivateModelRequest(BaseModel):
    model_id: Optional[str] = Field(default=None, description="id модели; None = сброс активации")


@app.get("/v1/models")
def registry_list():
    """Список обученных моделей из реестра (метаданные + метрики)."""
    try:
        return {"active": registry.get_active(), "models": registry.list_models()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Registry error: {str(e)}")


@app.get("/v1/models/{model_id}")
def registry_get(model_id: str):
    try:
        return registry.get_model(model_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")


@app.post("/v1/models/activate")
def registry_activate(req: ActivateModelRequest):
    """Переключает активную модель и перезагружает текстовый пайплайн."""
    global pipeline, active_model_id
    with train_lock:
        try:
            registry.activate(req.model_id)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"Unknown model: {req.model_id}")
        if req.model_id:
            try:
                pipeline = FractalHoloNetInferencePipeline(
                    model_dir=str(registry.model_dir(req.model_id)), device="cpu"
                )
                active_model_id = req.model_id
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        else:
            pipeline = FractalHoloNetInferencePipeline(model_dir=TEXT_MODEL_DIR, device="cpu")
            active_model_id = None
    return {"status": "activated", "model_id": req.model_id}
