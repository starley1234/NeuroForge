"""Сервис инференса: загрузка моделей из реестра, генерация, инженерный анализ."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch

from ..config import NexusConfig
from ..data.tokenizer import DEFAULT_TOKENIZER
from ..fem.solver import solve
from ..model import NexusEngine
from ..registry import ModelRegistry, ModelVersion
from ..scad.render import render
from ..training.rewards import score_scad


@dataclass
class LoadedModel:
    model: NexusEngine
    version: Optional[ModelVersion]
    key: str
    loaded_at: float
    tokenizer: Any = None


class InferenceService:
    """Потокобезопасный кэш моделей + прикладные операции.

    Если запрошенной версии нет в реестре, поднимается свежая модель из пресета
    (режим «пустой инсталляции»): API остаётся работоспособным до первого
    обучения, но помечает ответ флагом `untrained: true`.
    """

    def __init__(self, registry_root: str = "artifacts/registry",
                 default_model: str = "core", default_ref: str = "production",
                 device: str = "auto", preset: str = "tiny", max_cached: int = 2,
                 tokenizer_path: Optional[str] = None, compile_model: bool = False):
        self.registry = ModelRegistry(registry_root)
        self.default_model = default_model
        self.default_ref = default_ref
        from ..runtime import pick_device
        self.device = pick_device(device)
        self.preset = preset
        self.max_cached = max_cached
        self.compile_model = compile_model
        from ..data.bpe import load_tokenizer
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.latency_ms: List[float] = []
        self._cache: Dict[str, LoadedModel] = {}
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests = 0

    # ------------------------------------------------------------- загрузка
    def _preset_config(self) -> NexusConfig:
        return {"tiny": NexusConfig.tiny, "rtx5060": NexusConfig.rtx5060,
                "rtx5060-compact": NexusConfig.rtx5060_compact}[self.preset]()

    def get_model(self, name: Optional[str] = None, ref: Optional[str] = None) -> LoadedModel:
        name = name or self.default_model
        ref = ref or self.default_ref
        key = f"{name}:{ref}"
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            try:
                from ..training.trainer import load_model
                model, mv = load_model(self.registry, name, ref, self.device)
                from ..data.bpe import load_tokenizer
                tok = load_tokenizer(mv.tokenizer_path) if mv.tokenizer_path else self.tokenizer
                loaded = LoadedModel(model, mv, key, time.time(), tok)
            except Exception:
                model = NexusEngine(self._preset_config()).to(self.device).eval()
                loaded = LoadedModel(model, None, key, time.time(), self.tokenizer)
            if self.compile_model:
                try:
                    loaded.model = torch.compile(loaded.model)  # type: ignore[assignment]
                except Exception as exc:                        # pragma: no cover
                    print(f"[serve] torch.compile недоступен: {exc}")
            if len(self._cache) >= self.max_cached:
                oldest = min(self._cache.values(), key=lambda m: m.loaded_at)
                self._cache.pop(oldest.key, None)
            self._cache[key] = loaded
            return loaded

    def reload(self, name: Optional[str] = None, ref: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            self._cache.clear()
        lm = self.get_model(name, ref)
        return {"loaded": lm.key, "version": lm.version.to_dict() if lm.version else None}

    # ------------------------------------------------------------- операции
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.8,
                 top_k: int = 40, model: Optional[str] = None,
                 ref: Optional[str] = None) -> Dict[str, Any]:
        self.requests += 1
        lm = self.get_model(model, ref)
        tok = lm.tokenizer or self.tokenizer
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=self.device)
        t0 = time.time()
        out = lm.model.generate(ids, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_k=top_k)
        text = tok.decode(out[0, ids.shape[1]:].tolist())
        self.latency_ms.append((time.time() - t0) * 1000 / max(max_new_tokens, 1))
        del self.latency_ms[:-500]
        return {
            "text": text, "prompt": prompt,
            "tokens_generated": int(out.shape[1] - ids.shape[1]),
            "ms": round((time.time() - t0) * 1000, 2),
            "model": lm.key, "untrained": lm.version is None,
            "version": lm.version.to_dict() if lm.version else None,
        }

    def analyze(self, code: str, material: str = "pla",
                force: Tuple[float, float, float] = (0.0, 0.0, -200.0),
                fixture: str = "base", grid: int = 24,
                calculix: bool = False) -> Dict[str, Any]:
        self.requests += 1
        res = render(code, resolution=grid, material=material)
        if not res.ok:
            return {"ok": False, "error": res.error}
        fem = solve(res.voxels, tuple(force), fixture, material, prefer_calculix=calculix)
        return {**res.summary(), "fem": fem.to_dict()}

    def reward(self, code: str, material: str = "pla",
               force: Tuple[float, float, float] = (0.0, 0.0, -200.0),
               fixture: str = "base", required_sf: float = 2.0,
               grid: int = 20) -> Dict[str, Any]:
        self.requests += 1
        return score_scad(code, tuple(force), fixture, material,
                          required_sf=required_sf, resolution=grid).to_dict()

    def design(self, spec: str, max_new_tokens: int = 96, material: str = "pla",
               force: Tuple[float, float, float] = (0.0, 0.0, -200.0),
               model: Optional[str] = None, ref: Optional[str] = None) -> Dict[str, Any]:
        """Сквозной сценарий: ТЗ → код → геометрия → FEM → награда."""
        gen = self.generate(spec + "<scad>", max_new_tokens=max_new_tokens,
                            model=model, ref=ref)
        analysis = self.analyze(gen["text"], material=material, force=force)
        return {"generation": gen, "analysis": analysis,
                "reward": self.reward(gen["text"], material, force)}

    @torch.no_grad()
    def generate_stream(self, prompt: str, max_new_tokens: int = 64,
                        temperature: float = 0.8, top_k: int = 40,
                        model: Optional[str] = None,
                        ref: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Потоковая генерация: по токену за шаг (для SSE)."""
        self.requests += 1
        lm = self.get_model(model, ref)
        tok = lm.tokenizer or self.tokenizer
        ids = torch.tensor([tok.encode(prompt, bos=True)], device=self.device)
        produced: List[int] = []
        for i in range(max_new_tokens):
            ids = lm.model.generate(ids, max_new_tokens=1, temperature=temperature,
                                    top_k=top_k)
            token = int(ids[0, -1])
            produced.append(token)
            yield {"index": i, "token": token,
                   "text": tok.decode([token]),
                   "done": i == max_new_tokens - 1}
        yield {"index": max_new_tokens, "text": "", "done": True,
               "full_text": tok.decode(produced), "model": lm.key}

    @torch.no_grad()
    def generate_batch(self, prompts: List[str], max_new_tokens: int = 64,
                       temperature: float = 0.8, top_k: int = 40,
                       model: Optional[str] = None,
                       ref: Optional[str] = None) -> Dict[str, Any]:
        """Батч-инференс: один прогон модели на несколько промптов."""
        self.requests += len(prompts)
        lm = self.get_model(model, ref)
        tok = lm.tokenizer or self.tokenizer
        encoded = [tok.encode(p, bos=True) for p in prompts]
        width = max(len(e) for e in encoded)
        pad = tok.pad_id
        batch = torch.tensor([[pad] * (width - len(e)) + e for e in encoded],
                             device=self.device)
        t0 = time.time()
        out = lm.model.generate(batch, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_k=top_k)
        texts = [tok.decode(row[width:].tolist()) for row in out]
        return {"texts": texts, "count": len(texts), "model": lm.key,
                "ms": round((time.time() - t0) * 1000, 2),
                "ms_per_prompt": round((time.time() - t0) * 1000 / max(len(prompts), 1), 2)}

    def metrics(self) -> str:
        """Метрики в текстовом формате Prometheus."""
        lat = sorted(self.latency_ms)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
        lines = [
            "# HELP nexus_requests_total Обработано запросов",
            "# TYPE nexus_requests_total counter",
            f"nexus_requests_total {self.requests}",
            "# HELP nexus_uptime_seconds Время работы сервиса",
            "# TYPE nexus_uptime_seconds gauge",
            f"nexus_uptime_seconds {round(time.time() - self.started, 1)}",
            "# HELP nexus_models_loaded Загруженных моделей в кэше",
            "# TYPE nexus_models_loaded gauge",
            f"nexus_models_loaded {len(self._cache)}",
            "# HELP nexus_generation_latency_ms Латентность генерации на токен",
            "# TYPE nexus_generation_latency_ms summary",
            f'nexus_generation_latency_ms{{quantile="0.5"}} {round(p50, 3)}',
            f'nexus_generation_latency_ms{{quantile="0.95"}} {round(p95, 3)}',
        ]
        for m in self._cache.values():
            version = m.version.version if m.version else 0
            lines.append(f'nexus_model_version{{model="{m.key}"}} {version}')
        return "\n".join(lines) + "\n"

    # -------------------------------------------------------------- сервисное
    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "uptime_s": round(time.time() - self.started, 1),
            "device": self.device,
            "requests": self.requests,
            "loaded": [
                {"key": m.key, "version": m.version.version if m.version else None,
                 "untrained": m.version is None}
                for m in self._cache.values()
            ],
            "registry": self.registry.root,
        }

    def models(self) -> Dict[str, Any]:
        return self.registry.summary()
