import json
import threading
import urllib.request

import pytest

from nexus.config import NexusConfig
from nexus.model import NexusEngine
from nexus.registry import ModelRegistry
from nexus.serve.app import NexusAPI, create_server
from nexus.serve.service import InferenceService


@pytest.fixture()
def api(tmp_path):
    reg = ModelRegistry(str(tmp_path / "registry"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict(), metrics={"val_ppl": 5.0})
    reg.save("core", model.state_dict(), model.cfg.to_dict(), metrics={"val_ppl": 4.0})
    reg.promote("core", 1, "production")
    service = InferenceService(str(tmp_path / "registry"), "core", "production")
    return NexusAPI(service)


def test_health_and_models(api):
    code, health = api.dispatch("GET", "/health", {})
    assert code == 200 and health["status"] == "ok"
    code, models = api.dispatch("GET", "/v1/models", {})
    assert models["models"]["core"]["versions"] == [1, 2]


def test_generate_uses_registry_version(api):
    code, out = api.dispatch("POST", "/v1/generate",
                             {"prompt": "<task>кронштейн", "max_new_tokens": 4})
    assert code == 200
    assert out["tokens_generated"] == 4
    assert out["untrained"] is False
    assert out["version"]["version"] == 1        # production, а не latest


def test_analyze_and_reward_endpoints(api):
    code, res = api.dispatch("POST", "/v1/analyze", {
        "code": "difference(){cube([30,30,6],center=true); cylinder(h=20,r=4,center=true);}",
        "force": [0, 0, -300], "material": "alu6061", "grid": 12})
    assert res["ok"] and res["fem"]["safety_factor"] > 0
    assert res["audit"]["manifold"] is True

    code, bad = api.dispatch("POST", "/v1/analyze", {"code": "cube([1,1,1)"})
    assert bad["ok"] is False and bad["error"]

    code, rw = api.dispatch("POST", "/v1/reward",
                            {"code": "cube([20,20,5],center=true);", "grid": 10})
    assert rw["compile"] == 1.0 and "total" in rw


def test_design_pipeline(api):
    code, out = api.dispatch("POST", "/v1/design",
                             {"spec": "<task>плита 200 Н", "max_new_tokens": 8})
    assert set(out) == {"generation", "analysis", "reward"}


def test_promote_and_rollback_via_api(api):
    code, mv = api.dispatch("POST", "/v1/registry/promote",
                            {"model": "core", "ref": "latest", "tag": "production"})
    assert mv["version"] == 2
    code, back = api.dispatch("POST", "/v1/registry/rollback",
                              {"model": "core", "tag": "production"})
    assert back["version"] == 1
    # сервис подхватил откат
    code, gen = api.dispatch("POST", "/v1/generate", {"prompt": "x", "max_new_tokens": 2})
    assert gen["version"]["version"] == 1


def test_eval_endpoint_reports_checks(api):
    code, report = api.dispatch("POST", "/v1/eval",
                                {"model": "core", "ref": "latest", "gate": False})
    assert "checks" in report and isinstance(report["passed"], bool)


def test_unknown_route_and_bad_params(api):
    code, err = api.dispatch("GET", "/nope", {})
    assert code == 404
    with pytest.raises(TypeError):
        api.dispatch("POST", "/v1/generate", {"unknown_param": 1})


def test_service_falls_back_to_untrained_model(tmp_path):
    service = InferenceService(str(tmp_path / "empty"), "missing", "production")
    out = service.generate("test", max_new_tokens=2)
    assert out["untrained"] is True and out["tokens_generated"] == 2


def test_http_server_end_to_end(tmp_path):
    reg = ModelRegistry(str(tmp_path / "registry"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict())
    server, _ = create_server("127.0.0.1", 0, registry_root=str(tmp_path / "registry"),
                              model="core", ref="latest")
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=10) as r:
            assert json.load(r)["status"] == "ok"
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/reward",
            data=json.dumps({"code": "cube([10,10,3],center=true);", "grid": 8}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            assert json.load(r)["compile"] == 1.0
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=10) as r:
            assert b"NEXUS-Engine API" in r.read()
    finally:
        server.shutdown()
        server.server_close()


def test_job_manager_runs_command(tmp_path):
    from nexus.serve.jobs import JobManager
    jm = JobManager(str(tmp_path / "jobs"))
    job = jm.submit(["vram", "--preset", "tiny"])
    for _ in range(300):
        if jm.get(job.id).status in ("done", "failed"):
            break
        import time
        time.sleep(0.1)
    finished = jm.get(job.id)
    assert finished.status == "done" and finished.returncode == 0
    assert "total_gb" in jm.logs(job.id)


# ─────────────────────────────────────────────── документация OpenAPI
def test_openapi_spec_is_complete_and_serializable():
    from nexus.serve.openapi import build_spec
    spec = build_spec(8000)
    assert spec["openapi"].startswith("3.1")
    json.dumps(spec)                                    # сериализуется без ошибок
    paths = spec["paths"]
    for required in ("/health", "/v1/generate", "/v1/analyze", "/v1/reward",
                     "/v1/design", "/v1/models", "/v1/jobs", "/metrics"):
        assert required in paths, required
    analyze = paths["/v1/analyze"]["post"]
    assert "requestBody" in analyze and "200" in analyze["responses"]
    props = analyze["requestBody"]["content"]["application/json"]["schema"]["properties"]
    assert {"code", "material", "force", "fixture", "grid"} <= set(props)


def test_docs_endpoints_served(tmp_path):
    import threading
    import urllib.request

    from nexus import NexusConfig, NexusEngine
    from nexus.registry import ModelRegistry
    from nexus.serve.app import create_server

    reg = ModelRegistry(str(tmp_path / "registry"))
    model = NexusEngine(NexusConfig.tiny())
    reg.save("core", model.state_dict(), model.cfg.to_dict())
    server, _ = create_server("127.0.0.1", 0, registry_root=str(tmp_path / "registry"),
                              model="core", ref="latest", api_key="secret")
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/openapi.json", timeout=10) as r:
            spec = json.load(r)                          # без ключа — документация открыта
        assert spec["info"]["title"] == "NEXUS-Engine API"
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/docs", timeout=10) as r:
            assert b"swagger" in r.read().lower()
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/favicon.ico", timeout=10) as r:
            assert r.status == 204
    finally:
        server.shutdown()
        server.server_close()
