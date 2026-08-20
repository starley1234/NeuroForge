import json
import tempfile
import unittest
from pathlib import Path
from training.prepare import prepare
from neuroscad import Program, validate

class TrainingDataTests(unittest.TestCase):
    def test_reproducible_valid_splits_without_group_leakage(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            counts_a = prepare(Path(a), 200, 7); counts_b = prepare(Path(b), 200, 7)
            self.assertEqual(counts_a, counts_b)
            groups = {}
            for split in ("train", "validation", "test"):
                self.assertEqual((Path(a) / f"{split}.jsonl").read_bytes(), (Path(b) / f"{split}.jsonl").read_bytes())
                for line in (Path(a) / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                    row = json.loads(line)
                    self.assertEqual(groups.get(row["group"], split), split)
                    groups[row["group"]] = split
                    validate(Program.from_dict(row["target"]))
            self.assertEqual(sum(counts_a.values()), 200)
