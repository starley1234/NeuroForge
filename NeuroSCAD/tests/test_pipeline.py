import unittest
from neuroscad import compile_openscad, generate, validate_program

class PipelineTests(unittest.TestCase):
    def test_russian_prompt_to_scad(self):
        program = generate("Кронштейн для камеры на трубу 25 мм с фиксацией винтом М4")
        defaults = {p.name: p.default for p in program.parameters}
        self.assertEqual(defaults["tube_d"], 25); self.assertEqual(defaults["screw_d"], 4.4)
        code = compile_openscad(program)
        self.assertIn("difference()", code); self.assertIn("tube_d = 25", code); self.assertIn("rotate([90, 0, 0])", code)
    def test_static_validation_is_honest(self):
        report = validate_program(generate("pipe clamp 32 mm M5"), render=False)
        self.assertTrue(report["valid"]); self.assertEqual(report["level"], "ir"); self.assertEqual(report["checks"]["geometry"]["status"], "not_run")
