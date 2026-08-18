"""Реестр моделей: версионирование, теги, откат.

Правила хранения (защита от потери работы):

* каждая новая версия пишется в **новый** каталог `vNNNN` — перезапись
  существующей версии запрещена на уровне API;
* запись атомарна: сначала во временный каталог, затем `os.replace`;
* к весам всегда прикладывается `meta.json` (конфиг, метрики, родитель,
  датасет, git-коммит, sha256 файла);
* теги (`production`, `staging`, …) — это указатели на версию, их можно
  переключать и откатывать, история переключений пишется в `tags.log`.

Раскладка на диске::

    artifacts/registry/
      <model_name>/
        v0001/{model.pt, meta.json}
        v0002/{model.pt, meta.json}
        tags.json          {"production": 1, "latest": 2}
        tags.log           журнал переключений
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

DEFAULT_ROOT = os.environ.get("NEXUS_REGISTRY", "artifacts/registry")


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


@dataclass
class ModelVersion:
    name: str
    version: int
    path: str
    created: str
    kind: str = "core"                       # core | fno | adapter
    metrics: Dict[str, float] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)
    parent: Optional[int] = None
    dataset: str = ""
    stage: str = ""                          # pretrain | lm | distill | rl | fno
    notes: str = ""
    tokenizer: str = ""                      # имя файла токенизатора в каталоге версии
    git_commit: str = ""
    sha256: str = ""
    size_bytes: int = 0

    @property
    def weights(self) -> str:
        return os.path.join(self.path, "model.pt")

    @property
    def tokenizer_path(self) -> Optional[str]:
        if not self.tokenizer:
            return None
        path = os.path.join(self.path, self.tokenizer)
        return path if os.path.exists(path) else None

    @property
    def tag(self) -> str:
        return f"{self.name}:v{self.version:04d}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RegistryError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, root: str = DEFAULT_ROOT):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # ------------------------------------------------------------- служебное
    def _model_dir(self, name: str) -> str:
        return os.path.join(self.root, name)

    def _version_dir(self, name: str, version: int) -> str:
        return os.path.join(self._model_dir(name), f"v{version:04d}")

    def _tags_path(self, name: str) -> str:
        return os.path.join(self._model_dir(name), "tags.json")

    def models(self) -> List[str]:
        if not os.path.isdir(self.root):
            return []
        return sorted(d for d in os.listdir(self.root)
                      if os.path.isdir(os.path.join(self.root, d)))

    def versions(self, name: str) -> List[int]:
        d = self._model_dir(name)
        if not os.path.isdir(d):
            return []
        out = []
        for entry in os.listdir(d):
            if entry.startswith("v") and entry[1:].isdigit():
                if os.path.exists(os.path.join(d, entry, "meta.json")):
                    out.append(int(entry[1:]))
        return sorted(out)

    def next_version(self, name: str) -> int:
        vs = self.versions(name)
        return (vs[-1] + 1) if vs else 1

    # ------------------------------------------------------------- сохранение
    def save(
        self,
        name: str,
        state_dict: Dict[str, torch.Tensor],
        config: Dict[str, Any],
        *,
        kind: str = "core",
        metrics: Optional[Dict[str, float]] = None,
        parent: Optional[int] = None,
        dataset: str = "",
        stage: str = "",
        notes: str = "",
        tokenizer_path: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> ModelVersion:
        """Сохранить новую версию. Существующие версии никогда не затираются."""
        version = self.next_version(name)
        target = self._version_dir(name, version)
        if os.path.exists(target):
            raise RegistryError(f"версия {name}:v{version:04d} уже существует")
        os.makedirs(self._model_dir(name), exist_ok=True)

        tmp = tempfile.mkdtemp(prefix=f".{name}.v{version:04d}.", dir=self._model_dir(name))
        try:
            weights = os.path.join(tmp, "model.pt")
            payload = {"model": state_dict, "config": config, "kind": kind}
            if extra:
                payload.update(extra)
            torch.save(payload, weights)
            tokenizer_name = ""
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer_name = "tokenizer.json"
                shutil.copyfile(tokenizer_path, os.path.join(tmp, tokenizer_name))
            mv = ModelVersion(
                name=name, version=version, path=target,
                created=time.strftime("%Y-%m-%dT%H:%M:%S"), kind=kind,
                metrics=dict(metrics or {}), config=config,
                parent=parent if parent is not None else (self.versions(name)[-1] if self.versions(name) else None),
                dataset=dataset, stage=stage, notes=notes, tokenizer=tokenizer_name,
                git_commit=_git_commit(), sha256=_sha256(weights),
                size_bytes=os.path.getsize(weights),
            )
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(mv.to_dict(), fh, indent=2, ensure_ascii=False)
            os.replace(tmp, target)
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        self._set_tag(name, "latest", version, reason="save")
        return mv

    # ---------------------------------------------------------------- чтение
    def resolve(self, name: str, ref: Any = "latest") -> int:
        """Разрешить ссылку (число, 'latest', 'production', 'v0003', 'name:v3')."""
        if isinstance(ref, int):
            version = ref
        else:
            ref = str(ref)
            if ":" in ref:
                ref = ref.split(":", 1)[1]
            if ref.startswith("v") and ref[1:].isdigit():
                version = int(ref[1:])
            elif ref.isdigit():
                version = int(ref)
            else:
                tags = self.tags(name)
                if ref not in tags:
                    raise RegistryError(f"тег {ref!r} не найден для модели {name!r}")
                version = int(tags[ref])
        if version not in self.versions(name):
            raise RegistryError(f"версия {name}:v{version:04d} не найдена")
        return version

    def get(self, name: str, ref: Any = "latest") -> ModelVersion:
        version = self.resolve(name, ref)
        with open(os.path.join(self._version_dir(name, version), "meta.json"),
                  encoding="utf-8") as fh:
            return ModelVersion(**json.load(fh))

    def load(self, name: str, ref: Any = "latest", map_location: str = "cpu",
             verify: bool = True) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any], ModelVersion]:
        mv = self.get(name, ref)
        if verify and mv.sha256 and _sha256(mv.weights) != mv.sha256:
            raise RegistryError(f"контрольная сумма {mv.tag} не совпала — файл повреждён")
        blob = torch.load(mv.weights, map_location=map_location, weights_only=False)
        return blob["model"], blob.get("config", mv.config), mv

    def history(self, name: str) -> List[ModelVersion]:
        return [self.get(name, v) for v in self.versions(name)]

    # ------------------------------------------------------------------ теги
    def tags(self, name: str) -> Dict[str, int]:
        path = self._tags_path(name)
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    def _set_tag(self, name: str, tag: str, version: int, reason: str = "") -> None:
        tags = self.tags(name)
        previous = tags.get(tag)
        tags[tag] = version
        with open(self._tags_path(name), "w", encoding="utf-8") as fh:
            json.dump(tags, fh, indent=2, ensure_ascii=False)
        with open(os.path.join(self._model_dir(name), "tags.log"), "a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "tag": tag,
                "from": previous, "to": version, "reason": reason,
            }, ensure_ascii=False) + "\n")

    def promote(self, name: str, ref: Any = "latest", tag: str = "production",
                reason: str = "manual") -> ModelVersion:
        version = self.resolve(name, ref)
        self._set_tag(name, tag, version, reason)
        return self.get(name, version)

    def rollback(self, name: str, tag: str = "production", steps: int = 1) -> ModelVersion:
        """Откатить тег на предыдущую (по журналу) версию."""
        current = self.tags(name).get(tag)
        if current is None:
            raise RegistryError(f"тег {tag!r} не установлен для {name!r}")
        candidates = [v for v in self.versions(name) if v < current]
        if len(candidates) < steps:
            raise RegistryError("нет предыдущей версии для отката")
        target = candidates[-steps]
        self._set_tag(name, tag, target, reason=f"rollback from v{current:04d}")
        return self.get(name, target)

    # ---------------------------------------------------------------- уборка
    def prune(self, name: str, keep: int = 5, dry_run: bool = False) -> List[int]:
        """Удалить старые версии, сохранив последние `keep` и все помеченные тегами."""
        protected = set(self.tags(name).values())
        versions = self.versions(name)
        keep_set = set(versions[-keep:]) | protected
        removed = []
        for v in versions:
            if v in keep_set:
                continue
            removed.append(v)
            if not dry_run:
                shutil.rmtree(self._version_dir(name, v), ignore_errors=True)
        return removed

    def verify(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Проверить целостность хранилища: файлы на месте, sha256 совпадают.

        Полезно после переноса реестра, отката бэкапа или падения диска —
        одна команда вместо ручной сверки.
        """
        report: Dict[str, Any] = {"ok": True, "checked": 0, "problems": []}
        for model_name in ([name] if name else self.models()):
            for version in self.versions(model_name):
                report["checked"] += 1
                try:
                    mv = self.get(model_name, version)
                except Exception as exc:
                    report["problems"].append(
                        {"model": model_name, "version": version,
                         "problem": f"meta.json не читается: {exc}"})
                    continue
                if not os.path.exists(mv.weights):
                    report["problems"].append(
                        {"model": model_name, "version": version,
                         "problem": "нет файла весов model.pt"})
                    continue
                if mv.sha256 and _sha256(mv.weights) != mv.sha256:
                    report["problems"].append(
                        {"model": model_name, "version": version,
                         "problem": "sha256 не совпадает — файл повреждён"})
                if mv.tokenizer and not mv.tokenizer_path:
                    report["problems"].append(
                        {"model": model_name, "version": version,
                         "problem": "в метаданных указан токенизатор, но файла нет"})
            for tag, version in self.tags(model_name).items():
                if version not in self.versions(model_name):
                    report["problems"].append(
                        {"model": model_name, "version": version,
                         "problem": f"тег {tag} указывает на несуществующую версию"})
        report["ok"] = not report["problems"]
        return report

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"root": os.path.abspath(self.root), "models": {}}
        for name in self.models():
            versions = self.versions(name)
            if not versions:
                continue
            out["models"][name] = {
                "versions": versions,
                "tags": self.tags(name),
                "latest": self.get(name, versions[-1]).to_dict(),
            }
        return out
