"""Сервис инференса: загрузка моделей из реестра, генерация, инженерный анализ."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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


class InferenceService:
    """Потокобезопасный кэш моделей + прикладные операции.

    Если запрошенной версии нет в реестре, поднимается свежая модель из пресета
    (режим «пустой инсталляции»): API остаётся работоспособным до первого
    обучения, но помечает ответ флагом `untrained: true`.
    """

    def __init__(self, registry_root: str = "artifacts/registry",
                 default_model: str = "core", default_ref: str = "production",
                 device: str = "cpu", preset: str = "tiny", max_cached: int = 2):
        self.registry = ModelRegistry(registry_root)
        self.default_model = default_model
        self.default_ref = default_ref
        self.device = device
        self.preset = preset
        self.max_cached = max_cached
        self.tokenizer = DEFAULT_TOKENIZER
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
                loaded = LoadedModel(model, mv, key, time.time())
            except Exception:
                model = NexusEngine(self._preset_config()).to(self.device).eval()
                loaded = LoadedModel(model, None, key, time.time())
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
        ids = torch.tensor([self.tokenizer.encode(prompt, bos=True)], device=self.device)
        t0 = time.time()
        out = lm.model.generate(ids, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_k=top_k)
        text = self.tokenizer.decode(out[0, ids.shape[1]:].tolist())
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
