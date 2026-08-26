import json
import os

import pytest
import torch

from nexus.config import NexusConfig
from nexus.model import NexusEngine
from nexus.registry import ModelRegistry, RegistryError


def _model():
    return NexusEngine(NexusConfig.tiny(), with_critic=False)


def test_versions_never_overwrite(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    v1 = reg.save("core", m.state_dict(), m.cfg.to_dict(), metrics={"val_loss": 3.0})
    v2 = reg.save("core", m.state_dict(), m.cfg.to_dict(), metrics={"val_loss": 2.0})
    assert (v1.version, v2.version) == (1, 2)
    assert reg.versions("core") == [1, 2]
    assert v1.parent is None and v2.parent == 1
    assert os.path.exists(v1.weights) and os.path.exists(v2.weights)
    assert reg.tags("core")["latest"] == 2


def test_load_verifies_checksum(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    mv = reg.save("core", m.state_dict(), m.cfg.to_dict())
    state, cfg, meta = reg.load("core", "latest")
    assert meta.version == mv.version and cfg["d_latent"] == m.cfg.d_latent
    assert set(state) == set(m.state_dict())

    with open(mv.weights, "ab") as fh:      # портим файл
        fh.write(b"\x00")
    with pytest.raises(RegistryError):
        reg.load("core", "latest")


def test_promote_and_rollback(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    for _ in range(3):
        reg.save("core", m.state_dict(), m.cfg.to_dict())
    reg.promote("core", 2, "production")
    assert reg.tags("core")["production"] == 2
    reg.promote("core", 3, "production")
    assert reg.get("core", "production").version == 3

    rolled = reg.rollback("core", "production")
    assert rolled.version == 2
    assert reg.get("core", "production").version == 2
    log = open(os.path.join(str(tmp_path), "core", "tags.log"), encoding="utf-8").read()
    assert "rollback" in log


def test_prune_keeps_tagged_and_recent(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    for _ in range(5):
        reg.save("core", m.state_dict(), m.cfg.to_dict())
    reg.promote("core", 1, "production")
    removed = reg.prune("core", keep=2)
    assert removed == [2, 3]                     # v1 защищена тегом, v4/v5 свежие
    assert reg.versions("core") == [1, 4, 5]


def test_resolve_reference_forms(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    reg.save("core", m.state_dict(), m.cfg.to_dict())
    reg.save("core", m.state_dict(), m.cfg.to_dict())
    reg.promote("core", 1, "production")
    assert reg.resolve("core", "latest") == 2
    assert reg.resolve("core", "production") == 1
    assert reg.resolve("core", "v0001") == 1
    assert reg.resolve("core", 2) == 2
    assert reg.resolve("core", "core:v2") == 2
    with pytest.raises(RegistryError):
        reg.resolve("core", "v0099")


def test_summary_serializable(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = _model()
    reg.save("core", m.state_dict(), m.cfg.to_dict(), metrics={"val_ppl": 12.3})
    json.dumps(reg.summary())
