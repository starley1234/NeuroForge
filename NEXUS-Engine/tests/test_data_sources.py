import json
import os
import subprocess
import sys

import pytest


# ───────────────────────────────────────────────────────────── каталог
def test_catalog_entries_are_wellformed():
    from nexus.data.catalog import CATALOG, MIXES, filter_catalog, mix_source
    assert len(CATALOG) >= 10
    for e in CATALOG:
        assert e.key and e.source and e.url and e.notes
        assert e.task in {"cad-code", "cad-brep", "sim", "math", "code", "text"}
        assert 1 <= e.priority <= 3
    assert any(e.commercial_ok for e in CATALOG)
    assert filter_catalog(task="math") and filter_catalog(commercial_only=True)
    for name in MIXES:
        spec = mix_source(name)
        assert spec.startswith("mix:")
        assert abs(sum(MIXES[name].values()) - 1.0) < 1e-6


# ───────────────────────────────────────────────── инженерная математика
def test_mathgen_answers_are_verified_by_formula():
    import math

    from nexus.data.mathgen import GENERATORS, generate
    samples = generate(60, seed=3)
    assert len({s.kind for s in samples}) >= 6
    for s in samples:
        assert s.answer == s.answer and abs(s.answer) < 1e12       # не NaN/inf
        assert s.unit and "Ответ:" in s.text and "Решение:" in s.text
        assert s.check(s.answer) and not s.check(s.answer * 3 + 1)


def test_mathgen_bending_matches_hand_calculation():
    """σ = 6FL/(bh²) — сверяем текст задачи с независимым расчётом."""
    import re

    from nexus.data.mathgen import cantilever_bending
    import random
    s = cantilever_bending(random.Random(0))
    f = float(re.search(r"F = ([\d.]+) Н", s.question).group(1))
    length = float(re.search(r"L = ([\d.]+) мм", s.question).group(1))
    b, h = (float(x) for x in re.search(r"b×h = ([\d.]+)×([\d.]+)", s.question).groups())
    expected = 6 * f * (length * 1e-3) / (b * 1e-3 * (h * 1e-3) ** 2) / 1e6
    assert s.check(expected)


def test_mathgen_jsonl_and_corpus_source(tmp_path):
    from nexus.data.corpora import iter_texts
    from nexus.data.mathgen import write_jsonl
    path = str(tmp_path / "math.jsonl")
    stats = write_jsonl(25, path, seed=1)
    assert stats["count"] == 25 and os.path.exists(path)
    assert len(list(iter_texts(f"jsonl:{path}#text"))) == 25
    assert len(list(iter_texts("mathgen:7"))) == 7


def test_mix_source_interleaves():
    from nexus.data.corpora import iter_texts
    texts = list(iter_texts("mix:mathgen:20=0.5,builtin:engineering=0.5", limit=25))
    assert len(texts) == 25
    assert any("Ответ:" in t for t in texts)          # из mathgen
    assert any("OpenSCAD" in t or "σ" in t or "module" in t for t in texts)


# ───────────────────────────────────────────────────────────── MCP-сервер
def _rpc(server, method, params=None, msg_id=1):
    return server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method,
                          "params": params or {}})


def _tool(server, name, args, msg_id=2):
    resp = _rpc(server, "tools/call", {"name": name, "arguments": args}, msg_id)
    payload = json.loads(resp["result"]["content"][0]["text"])
    return payload, resp["result"]["isError"]


@pytest.fixture()
def mcp(tmp_path):
    from nexus.mcp.server import MCPServer
    return MCPServer(str(tmp_path / "mcp"))


def test_mcp_initialize_and_tools_list(mcp):
    init = _rpc(mcp, "initialize", {"protocolVersion": "2024-11-05", "capabilities": {}})
    assert init["result"]["serverInfo"]["name"] == "nexus-engine"
    assert init["result"]["protocolVersion"] == "2024-11-05"
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None

    tools = _rpc(mcp, "tools/list")["result"]["tools"]
    names = {t["name"] for t in tools}
    assert {"scad_compile", "scad_analyze", "fem_analyze", "score_design",
            "propose_variation", "math_problem", "record_sample"} <= names
    for t in tools:
        assert t["description"] and t["inputSchema"]["type"] == "object"


def test_mcp_geometry_and_fem_tools(mcp):
    code = "difference(){cube([40,40,6],center=true); cylinder(h=20,r=5,center=true);}"
    ok, err = _tool(mcp, "scad_compile", {"code": code})
    assert ok["ok"] and not err

    bad, err = _tool(mcp, "scad_compile", {"code": "cube([1,1,1)"})
    assert bad["ok"] is False and err

    geo, _ = _tool(mcp, "scad_analyze", {"code": code, "grid": 12, "material": "alu6061"})
    assert geo["mass"]["mass_g"] > 0 and geo["audit"]["manifold"] is True

    fem, _ = _tool(mcp, "fem_analyze", {"code": code, "grid": 12, "material": "alu6061",
                                        "force_n": [0, 0, -300]})
    assert fem["fem"]["safety_factor"] > 0 and fem["fem"]["backend"] in ("hex", "loadpath")

    score, _ = _tool(mcp, "score_design", {"code": code, "grid": 10})
    assert score["compile"] == 1.0 and "total" in score


def test_mcp_record_sample_rejects_broken_design(mcp, tmp_path):
    good, _ = _tool(mcp, "record_sample", {"spec": "<task>плита",
                                           "code": "cube([20,20,4],center=true);"})
    assert good["ok"] and os.path.exists(good["path"])
    bad, err = _tool(mcp, "record_sample", {"spec": "<task>мусор", "code": "cube([1,1,1)"})
    assert bad["ok"] is False and err
    with open(good["path"], encoding="utf-8") as fh:
        assert len(fh.readlines()) == 1        # битый пример не записан


def test_mcp_unknown_tool_and_method(mcp):
    resp = _rpc(mcp, "tools/call", {"name": "nope", "arguments": {}})
    assert "error" in resp
    assert "error" in _rpc(mcp, "magic/method")


def test_mcp_stdio_roundtrip(tmp_path):
    """Полный цикл через подпроцесс — как это делает Claude Desktop."""
    messages = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "math_problem", "arguments": {"seed": 1}}},
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "nexus.mcp.server", "--workdir", str(tmp_path)],
        input="\n".join(json.dumps(m) for m in messages) + "\n",
        capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-500:]
    lines = [json.loads(x) for x in proc.stdout.strip().splitlines()]
    assert len(lines) == 3 and lines[0]["result"]["serverInfo"]["name"] == "nexus-engine"
    payload = json.loads(lines[2]["result"]["content"][0]["text"])
    assert payload["ok"] and "answer" in payload


# ──────────────────────────────────────────────── сбор данных дистилляцией
def test_collect_with_offline_teacher(tmp_path):
    from nexus.data.collect import TemplateDesigner, collect, default_tasks
    out = str(tmp_path / "collected.jsonl")
    stats = collect(TemplateDesigner(seed=1), default_tasks(6, seed=1), attempts=2,
                    threshold=2.0, out_path=out, workdir=str(tmp_path / "mcp"),
                    grid=12, verbose=False)
    assert stats.tasks == 6 and stats.accepted > 0
    with open(out, encoding="utf-8") as fh:
        records = [json.loads(line) for line in fh]
    assert records and all(r["reward"] >= 2.0 for r in records)
    assert all("<scad>" in r["text"] for r in records)
    assert os.path.exists(str(tmp_path / "collected_stats.json"))


def test_collect_rejects_bad_teacher_and_gives_feedback(tmp_path):
    from nexus.data.collect import Task, collect

    class BrokenDesigner:
        def __init__(self):
            self.feedbacks = []

        def propose(self, spec, feedback=None):
            self.feedbacks.append(feedback)
            return "cube([1,1,1)"          # всегда невалидно

    designer = BrokenDesigner()
    out = str(tmp_path / "bad.jsonl")
    stats = collect(designer, [Task("<task>тест")], attempts=3, threshold=3.5,
                    out_path=out, workdir=str(tmp_path / "mcp"), grid=8, verbose=False)
    assert stats.accepted == 0
    assert stats.failures.get("compile") == 1
    assert designer.feedbacks[1] and "не компилируется" in designer.feedbacks[1]
    assert os.path.getsize(out) == 0


def test_collected_corpus_is_trainable(tmp_path):
    from nexus.data.collect import TemplateDesigner, collect, default_tasks
    from nexus.data.corpora import PackedLMDataset
    out = str(tmp_path / "collected.jsonl")
    collect(TemplateDesigner(seed=2), default_tasks(4, seed=2), attempts=1,
            threshold=1.0, out_path=out, workdir=str(tmp_path / "mcp"), grid=10,
            verbose=False)
    ds = PackedLMDataset(f"jsonl:{out}#text", seq_len=64, min_blocks=2)
    assert len(ds) >= 2 and ds[0]["tokens"].shape == (64,)


def test_extract_code_handles_markdown():
    from nexus.data.collect import extract_code
    assert extract_code("бла\n```openscad\ncube([1,1,1]);\n```\nконец") == "cube([1,1,1]);"
    assert extract_code("cube([2,2,2]);") == "cube([2,2,2]);"
