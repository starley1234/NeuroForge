import json
import tempfile
import unittest
from pathlib import Path

from neuroscad.openscad_runner import audit_source, extract_customizer_parameters
from training.annotate_teacher import parse_annotation
from training.build_distillation import build, replace_default
from training.evaluate_openscad import extract_code
from training.extract_sql_corpus import extract as extract_sql, iter_items
from training.ingest_openscad import ingest
from training.profile_openscad import profile
from training.retrieve_examples import retrieve

CODE = """// plate width
width = 40; // [20:1:80]
height = 4;
cube([width, 20, height], center=true);
"""

class OpenSCADDatasetTests(unittest.TestCase):
    def test_audit_and_customizer_parser(self):
        self.assertEqual(audit_source(CODE), [])
        self.assertIn("mesh_import", audit_source('import("other.stl");'))
        parameters = extract_customizer_parameters(CODE)
        self.assertEqual(parameters[0], {"name": "width", "default": 40.0, "minimum": 20.0, "step": 1.0, "maximum": 80.0})

    def test_dry_run_ingestion_deduplicates_before_split(self):
        with tempfile.TemporaryDirectory() as source_tmp, tempfile.TemporaryDirectory() as output_tmp:
            source = Path(source_tmp); (source / "a.scad").write_text(CODE); (source / "b.scad").write_text(CODE)
            (source / "a.json").write_text(json.dumps({"prompt": "Пластина 40 на 20", "license": "owned"}))
            manifest = ingest(source, Path(output_tmp), "owned", dry_run=True)
            self.assertEqual(manifest["unique_sources"], 1)
            self.assertEqual(manifest["counts"]["duplicate_source"], 1)
            record = json.loads((Path(output_tmp) / "records.jsonl").read_text())
            self.assertEqual(record["prompt"], "Пластина 40 на 20")
            self.assertEqual(profile(Path(output_tmp))["models"], 1)

    def test_teacher_json_and_distillation_build(self):
        annotation = {"description": "Пластина", "family": "plate", "requirements": {"width": 40},
                      "construction_plan": ["Create plate"], "paraphrases": ["plate", "пластина"], "quality_notes": []}
        fenced = "```json\n" + json.dumps(annotation) + "\n```"
        self.assertEqual(parse_annotation(fenced)["family"], "plate")
        with tempfile.TemporaryDirectory() as corpus_tmp, tempfile.TemporaryDirectory() as output_tmp:
            corpus = Path(corpus_tmp)
            record = {"id": "abc", "split": "test", "source_code": CODE, "parameters": extract_customizer_parameters(CODE), "license": "owned", "prompt": "Пластина"}
            (corpus / "records.jsonl").write_text(json.dumps(record) + "\n")
            labels = corpus / "annotations.jsonl"; labels.write_text(json.dumps({"id": "abc", "annotation": annotation}) + "\n")
            counts = build(corpus, labels, Path(output_tmp), edits_per_model=1)
            self.assertGreaterEqual(counts["test"], 3)
            rows = [json.loads(line) for line in (Path(output_tmp) / "test.jsonl").read_text().splitlines()]
            self.assertTrue(all(row["source_id"] == "abc" for row in rows))
            self.assertTrue(any(row["task"] == "edit" and "width = 20" in row["target"] for row in rows))

    def test_sql_extraction_decodes_code_and_omits_identifiers(self):
        sql = "INSERT INTO `stl_items` (`stl_item_id`,`stl_item_status`,`stl_item_code_basis`,`stl_item_code`,`user_id`) VALUES (7,'в работе','Кронштейн','a = 2;\\n// it\\\'s valid',99);"
        item = next(iter_items(sql)); self.assertIn("\n", item["stl_item_code"])
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as out:
            source = Path(tmp) / "items.sql"; source.write_text(sql)
            self.assertEqual(extract_sql(source, Path(out), "owned")["extracted"], 1)
            sidecar = json.loads((Path(out) / "stl_item_7.json").read_text())
            self.assertNotIn("user_id", json.dumps(sidecar))
            self.assertEqual(sidecar["prompt"], "Кронштейн")

    def test_retrieval_prefers_matching_owner_example(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [
                {"id": "plate", "prompt": "монтажная пластина с отверстиями", "source_code": "cube(1);", "parameters": []},
                {"id": "bushing", "prompt": "цилиндрическая втулка для вала", "source_code": "cylinder(1);", "parameters": []},
            ]
            (root / "records.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows))
            self.assertEqual(retrieve(root, None, "нужна втулка на вал", 1)[0]["id"], "bushing")

    def test_output_extractors(self):
        self.assertEqual(replace_default(CODE, "width", 55).splitlines()[1], "width = 55; // [20:1:80]")
        self.assertEqual(extract_code("<openscad>cube(2);</openscad>"), "cube(2);")

if __name__ == "__main__": unittest.main()
