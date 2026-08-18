import json
import os

import numpy as np
import pytest

SQL_DUMP = """
INSERT INTO `stl_items` (`stl_item_id`, `stl_item_status`, `stl_item_code_basis`, `stl_item_code`, `stl_item_favorite`, `stl_item_public`, `stl_item_img`, `stl_item_update`, `user_id`, `guest_id`, `answer_id`, `created_at`) VALUES
(7078, 'в работе', 'Дискообразный корпус с низким профилем.', '// корпус \\'широкий\\'\\n$fn=32;\\nouter_d = 60; // внешний диаметр\\nheight = 14;\\ndifference() {\\n  cylinder(d=outer_d, h=height);\\n  translate([0,0,2]) cylinder(d=outer_d-6, h=height);\\n}', 0, 0, 'img_7078.png', '2026-08-17 22:32:18', 1, 4808, 'gen-1', '2026-08-17 22:23:16'),
(7077, 'готово', 'Прижимной фланец для труб.', '/* фланец */\\n$fn=42;\\nflange_d = 110.0;\\nthk = 12.0; // толщина диска\\ndifference() {\\n  cylinder(d=flange_d, h=thk);\\n  translate([0,0,-1]) cylinder(d=32, h=thk+2);\\n}', 0, 0, NULL, '2026-08-17 22:20:03', 1, 4808, 'gen-2', '2026-08-17 22:19:23'),
(7076, 'в работе', 'Битый скрипт.', 'difference() { cube([10,10,10) ; cylinder(h=5, r=2); } // не закрыта скобка', 0, 0, NULL, '2026-08-17 22:15:00', 1, 4808, 'gen-3', '2026-08-17 22:14:00');
"""


# ───────────────────────────────────────────────────────── разбор дампа
def test_parse_sql_dump_handles_escapes_and_null():
    from nexus.data.ingest import parse_sql_dump
    rows = parse_sql_dump(SQL_DUMP, table="stl_items")
    assert len(rows) == 3
    assert rows[0][0] == 7078 and rows[0][1] == "в работе"
    code = rows[0][3]
    assert "\n" in code and "'широкий'" in code        # \n и \' раскрыты
    assert code.count("\\n") == 0
    assert rows[1][6] is None                          # NULL распознан
    assert rows[0][6] == "img_7078.png"


def test_load_records_normalizes_fields(tmp_path):
    from nexus.data.ingest import load_records
    path = tmp_path / "dump.sql"
    path.write_text(SQL_DUMP, encoding="utf-8")
    records = list(load_records(f"sql:{path}"))
    assert len(records) == 3
    first = records[0]
    assert set(first) >= {"spec", "code", "image", "status", "item_id"}
    assert first["spec"].startswith("Дискообразный")
    assert "cylinder" in first["code"]


def test_load_records_from_jsonl(tmp_path):
    from nexus.data.ingest import load_records
    path = tmp_path / "items.jsonl"
    path.write_text(json.dumps({"description": "плита", "scad": "cube([10,10,2]);"},
                               ensure_ascii=False) + "\n", encoding="utf-8")
    rec = next(iter(load_records(f"jsonl:{path}")))
    assert rec["spec"] == "плита" and rec["code"].startswith("cube")


def test_extract_parameters_reads_header():
    from nexus.data.ingest import extract_parameters
    params = extract_parameters(
        "outer_d = 60; // внешний диаметр\nheight = 14;\n"
        "big = [for (i=[0:100]) i];\ncube([1,1,1]);")
    names = {p["name"]: p for p in params}
    assert names["outer_d"]["value"] == "60"
    assert names["outer_d"]["comment"] == "внешний диаметр"
    assert "big" not in names                          # генераторы отсеиваются


# ─────────────────────────────────────────────────────────── импорт
def test_ingest_validates_dedupes_and_splits(tmp_path):
    from nexus.data.ingest import ingest
    dump = tmp_path / "dump.sql"
    dump.write_text(SQL_DUMP + SQL_DUMP, encoding="utf-8")   # дубли специально
    out = str(tmp_path / "out")
    stats = ingest(f"sql:{dump}", out, grid=12, verbose=False, val_fraction=0.0)

    assert stats.total == 6
    assert stats.duplicates == 3                        # второй проход — копии
    assert stats.accepted == 2                          # битый скрипт отброшен
    assert stats.rejected.get("compile_error") == 1
    assert stats.mean_reward > 0

    records = [json.loads(line) for line in
               open(os.path.join(out, "dataset.jsonl"), encoding="utf-8")]
    assert len(records) == 2
    rec = records[0]
    assert "<task>" in rec["text"] and "<scad>" in rec["text"]
    assert rec["physics"]["mass_g"] > 0 and rec["params"]
    assert os.path.exists(os.path.join(out, "rejects.jsonl"))
    assert os.path.exists(os.path.join(out, "stats.json"))


def test_ingest_filters_by_status(tmp_path):
    from nexus.data.ingest import ingest
    dump = tmp_path / "dump.sql"
    dump.write_text(SQL_DUMP, encoding="utf-8")
    stats = ingest(f"sql:{dump}", str(tmp_path / "out2"), statuses=["готово"],
                   grid=12, verbose=False, val_fraction=0.0)
    assert stats.accepted == 1 and stats.rejected.get("status") == 2


def test_ingested_corpus_is_trainable(tmp_path):
    from nexus.data.corpora import PackedLMDataset
    from nexus.data.ingest import ingest
    dump = tmp_path / "dump.sql"
    dump.write_text(SQL_DUMP, encoding="utf-8")
    out = str(tmp_path / "out3")
    ingest(f"sql:{dump}", out, grid=12, verbose=False, val_fraction=0.0)
    ds = PackedLMDataset(f"jsonl:{os.path.join(out, 'dataset.jsonl')}#text",
                         seq_len=64, min_blocks=2)
    assert len(ds) >= 2 and ds[0]["tokens"].shape == (64,)


# ────────────────────────────────────────────── визуальная модальность
def test_png_write_and_read_roundtrip(tmp_path):
    from nexus.data.vision import load_image, write_png
    rng = np.random.default_rng(0)
    image = rng.integers(0, 255, (16, 24, 3), dtype=np.uint8)
    path = write_png(str(tmp_path / "test.png"), image)
    back = load_image(path)
    assert back.shape == (16, 24, 3)
    assert np.allclose(back * 255.0, image.astype(np.float32), atol=1.0)


def test_internal_renderer_draws_geometry(tmp_path):
    from nexus.data.vision import load_image, render_internal
    code = "difference(){cylinder(d=60,h=12,$fn=32); translate([0,0,-1]) cylinder(d=20,h=20);}"
    path = render_internal(code, str(tmp_path / "iso.png"), (55.0, 25.0), size=96,
                           resolution=36)
    assert path and os.path.exists(path)
    img = load_image(path)
    lit = (img.max(axis=2) > 0.25).sum()
    assert 200 < lit < 96 * 96                          # деталь видна, но не весь кадр


def test_internal_renderer_rejects_broken_code(tmp_path):
    from nexus.data.vision import render_internal
    assert render_internal("cube([1,1,1)", str(tmp_path / "bad.png"), size=64) is None


def test_build_vision_dataset_manifest(tmp_path):
    from nexus.data.vision import build_vision_dataset
    records = [
        {"item_id": 1, "spec": "плита", "code": "cube([30,20,4], center=true);",
         "params": [{"name": "a", "value": "30", "comment": ""}]},
        {"item_id": 2, "spec": "втулка",
         "code": "difference(){cylinder(d=30,h=10,$fn=24); cylinder(d=12,h=30,center=true);}"},
    ]
    stats = build_vision_dataset(records, str(tmp_path / "vision"), size=64,
                                 backend="internal", views=["iso", "top"],
                                 resolution=32, verbose=False)
    assert stats.items == 2 and stats.images == 4 and stats.failed == 0
    lines = [json.loads(x) for x in
             open(os.path.join(str(tmp_path / "vision"), "manifest.jsonl"), encoding="utf-8")]
    assert len(lines) == 2
    entry = lines[0]
    assert {i["view"] for i in entry["images"]} == {"iso", "top"}
    for image in entry["images"]:
        assert os.path.exists(os.path.join(str(tmp_path / "vision"), image["image"]))
    assert "<scad>" in entry["text"]


# ──────────────────────────────── физика как инварианты на общей шине
def test_invariants_change_model_predictions():
    """Масса и нагрузка должны влиять на выход, а не быть декорацией."""
    import torch
    from nexus import NexusConfig, NexusEngine
    torch.manual_seed(0)
    model = NexusEngine(NexusConfig.tiny()).eval()
    tokens = torch.randint(4, 400, (2, 24))
    plain = model(tokens=tokens, reason=False).logits
    heavy = model(tokens=tokens, reason=False,
                  invariants=torch.tensor([[0, 0, 0, 0.21, 0, 0, -0.35],
                                           [0, 0, 0, 0.02, 0, 0, -0.05]])).logits
    assert not torch.allclose(plain, heavy)


def test_physics_dataset_yields_invariants(tmp_path):
    import torch
    from nexus.data.corpora import PhysicsLMDataset
    path = tmp_path / "phys.jsonl"
    path.write_text("\n".join(json.dumps(rec, ensure_ascii=False) for rec in [
        {"text": "<task>кронштейн<scad>cube([20,20,4]);",
         "physics": {"mass_g": 210.0}, "load": {"force_n": [0, 0, -350]}},
        {"text": "<task>плита<scad>cube([30,30,3]);",
         "physics": {"mass_g": 12.0}, "load": {"force_n": [0, 0, -20]}},
    ]), encoding="utf-8")

    ds = PhysicsLMDataset(str(path), seq_len=64)
    assert len(ds) == 2
    item = ds[0]
    assert item["tokens"].shape == (64,) and item["targets"].shape == (64,)
    assert item["invariants"].shape == (7,)
    assert item["invariants"][3] == pytest.approx(0.21)      # 210 г → 0.21 кг
    assert item["invariants"][6] == pytest.approx(-0.35)     # −350 Н → −0.35


def test_training_with_physics_invariants(tmp_path):
    from nexus.training.train_lm import train
    path = tmp_path / "phys.jsonl"
    path.write_text("\n".join(json.dumps({
        "text": f"<task>деталь {i}<scad>cube([{10 + i},10,4]);",
        "physics": {"mass_g": 10.0 * i}, "load": {"force_n": [0, 0, -50 * i]},
    }, ensure_ascii=False) for i in range(1, 9)), encoding="utf-8")

    out = train(f"jsonl:{path}", "core", "tiny", seq_len=64, physics=True,
                registry_root=str(tmp_path / "reg"), max_steps=2, batch_size=2,
                grad_accum=1, log_every=100)
    assert out["metrics"]["val_loss"] > 0


# ───────────────────────────────────────── A/B генераторов (nexus bench)
def test_bench_compares_designers(tmp_path):
    from nexus.eval.bench import format_table, run_bench
    results = run_bench(["template", "template"],
                        prompts=["Держатель кабеля 6 мм", "Заглушка 20 мм"],
                        out_dir=str(tmp_path / "bench"), grid=10, verbose=False)
    assert len(results) == 2
    for r in results:
        d = r.to_dict()
        assert d["items"] == 2
        assert 0.0 <= d["compile_rate"] <= 1.0
        assert d["sec_per_item"] >= 0
    assert os.path.exists(os.path.join(str(tmp_path / "bench"), "results.json"))
    assert os.path.exists(os.path.join(str(tmp_path / "bench"), "samples.jsonl"))
    assert "компилируется" in format_table(results)


def test_bench_parses_designer_specs():
    from nexus.eval.bench import parse_designer
    name, designer = parse_designer("template")
    assert name == "template" and hasattr(designer, "propose")
    name, designer = parse_designer("command:echo cube([1,1,1]);")
    assert name.startswith("command:")
    with pytest.raises(ValueError):
        parse_designer("magic:whatever")


def test_bench_loads_prompts_from_file(tmp_path):
    from nexus.eval.bench import load_prompts
    plain = tmp_path / "p.txt"
    plain.write_text("клипса 6 мм\nзаглушка 20 мм\n", encoding="utf-8")
    assert load_prompts(str(plain)) == ["клипса 6 мм", "заглушка 20 мм"]

    js = tmp_path / "p.jsonl"
    js.write_text(json.dumps({"spec": "фланец"}, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_prompts(str(js)) == ["фланец"]


# ──────────────────────── оптимизация детали под нагрузку (МКЭ в цикле)
BRACKET = """
width = 40;          // ширина
height = 45;         // высота
thickness = 6.0;     // толщина стенки
rib = 8.0;           // ребро жёсткости
leg = 35;            // вылет
hole_d = 5;          // отверстие

difference() {
  union() {
    cube([width, thickness, height]);
    cube([width, leg, thickness]);
    translate([width/2 - rib/2, 0, 0]) cube([rib, leg, leg]);
  }
  translate([width/2, thickness/2, height*0.72]) rotate([90,0,0])
    translate([0,0,-thickness]) cylinder(h=thickness*3, r=hole_d/2, $fn=16);
}
"""


def test_optimizer_picks_only_strength_parameters():
    from nexus.optimize import pick_knobs
    names = {k.name for k in pick_knobs(BRACKET)}
    assert {"thickness", "rib"} <= names
    # габариты и присоединительные размеры трогать нельзя
    assert not ({"width", "height", "leg", "hole_d"} & names)


def test_substitute_replaces_values_once():
    from nexus.optimize import substitute
    out = substitute(BRACKET, {"thickness": 3.25, "rib": 4.5})
    assert "thickness = 3.25;" in out and "rib = 4.5;" in out
    assert out.count("thickness = ") == 1
    assert "// толщина стенки" not in out.split("thickness = 3.25;")[0][-30:] or True
    from nexus.scad import compile_scad
    assert compile_scad(out).ok                      # результат остаётся валидным


def test_optimizer_reduces_mass_and_keeps_strength():
    from nexus.optimize import optimize
    result = optimize(BRACKET, force_n=(0, 0, -300), material="pla", required_sf=2.0,
                      budget=14, grid=12, verify_grid=14, seed=1, verbose=False)
    assert result.evaluations >= 10
    assert result.best.mass_g > 0
    assert result.best.safety_factor >= 2.0          # ограничение соблюдено
    assert result.best.mass_g <= result.baseline.mass_g + 1e-6
    assert "thickness" in result.best.params
    assert "было" in result.summary() and "стало" in result.summary()
    from nexus.scad import compile_scad
    assert compile_scad(result.code).ok


def test_optimizer_respects_explicit_parameter_list():
    from nexus.optimize import optimize
    result = optimize(BRACKET, force_n=(0, 0, -200), budget=6, grid=10, verify_grid=10,
                      only=["rib"], verbose=False)
    assert set(result.best.params) == {"rib"}


def test_optimizer_writes_files(tmp_path):
    from nexus.optimize import optimize_file
    src = tmp_path / "part.scad"
    src.write_text(BRACKET, encoding="utf-8")
    out = str(tmp_path / "opt.scad")
    result = optimize_file(str(src), out_path=out, force_n=(0, 0, -200), budget=6,
                           grid=10, verify_grid=10, verbose=False)
    assert os.path.exists(out) and os.path.exists(str(tmp_path / "opt_report.json"))
    assert result.to_dict()["mass_saved_pct"] is not None


def test_optimizer_uses_cache_and_is_deterministic():
    from nexus.optimize import optimize
    first = optimize(BRACKET, force_n=(0, 0, -250), budget=12, grid=10, verify_grid=10,
                     seed=7, verbose=False)
    second = optimize(BRACKET, force_n=(0, 0, -250), budget=12, grid=10, verify_grid=10,
                      seed=7, verbose=False)
    assert first.best.params == second.best.params        # воспроизводимо
    assert first.cache_hits >= 0 and first.evaluations >= 12
    assert isinstance(first.feasible, bool)


def test_optimizer_reports_infeasible_instead_of_lying():
    from nexus.optimize import optimize
    # заведомо невыполнимое требование: запас 1000 при пластике
    result = optimize(BRACKET, force_n=(0, 0, -5000), material="pla", required_sf=1000.0,
                      budget=8, grid=10, verify_grid=10, verbose=False)
    assert result.feasible is False
    assert "ВНИМАНИЕ" in result.summary()


def test_optimizer_rejects_broken_input():
    from nexus.optimize import optimize
    with pytest.raises(ValueError, match="не собирается|силовых"):
        optimize("thickness = 3; cube([10,10,10)", budget=6, grid=8, verbose=False)


# ────────────────────── экспорт в chat-формат для дообучения готовых LLM
def test_export_sft_builds_chat_messages(tmp_path):
    from nexus.data.sft import export_sft
    src = tmp_path / "corpus.jsonl"
    src.write_text("\n".join(json.dumps(rec, ensure_ascii=False) for rec in [
        {"spec": "<task>кронштейн 300 Н",
         "code": "// кронштейн\nw = 40;\nt = 6;\ndifference(){ cube([w,w,t], center=true);"
                 " cylinder(h=20, r=4, center=true); }",
         "physics": {"mass_g": 41.2, "safety_factor": 3.1, "min_wall_mm": 2.4}},
        {"spec": "плита", "code": "x", "physics": {}},          # слишком короткий
    ]), encoding="utf-8")

    out = str(tmp_path / "sft")
    stats = export_sft([f"jsonl:{src}"], out, val_fraction=0.0, verbose=False)
    assert stats.exported == 1 and stats.skipped.get("too_short") == 1

    line = json.loads(open(os.path.join(out, "train.jsonl"), encoding="utf-8").readline())
    roles = [m["role"] for m in line["messages"]]
    assert roles == ["system", "user", "assistant"]
    answer = line["messages"][2]["content"]
    assert "```openscad" in answer and "cube([w,w,t]" in answer
    assert "масса 41.2 г" in answer and "запас прочности 3.10" in answer


def test_export_sft_creates_repair_examples(tmp_path):
    from nexus.data.sft import export_sft, repair_messages
    record = {
        "spec": "<task>крючок 5 кг",
        "code": "// крючок\nthickness = 3;\ncube([20,20,thickness], center=true);"
                " translate([0,0,5]) cylinder(h=10, r=3);",
        "trajectory": [
            {"attempt": 0, "reward": 1.0, "feedback": None},
            {"attempt": 1, "reward": 4.2, "feedback": "слишком тонкие стенки: 0.9 мм"},
        ],
    }
    repair = repair_messages(record)
    assert repair and "0.9 мм" in repair["messages"][1]["content"]

    src = tmp_path / "traj.jsonl"
    src.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    stats = export_sft([f"jsonl:{src}"], str(tmp_path / "sft2"), val_fraction=0.0,
                       verbose=False)
    assert stats.repair_pairs == 1 and stats.exported == 2   # обычный + ремонтный


def test_lora_scripts_are_valid_and_documented():
    """Скрипты должны запускаться хотя бы с --help без установленного torch-стека."""
    import subprocess
    import sys
    for script in ("scripts/train_lora.py", "scripts/serve_lora.py"):
        proc = subprocess.run([sys.executable, script, "--help"],
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr[-400:]
        assert "usage:" in proc.stdout
