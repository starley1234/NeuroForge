import unittest
from neuroscad import IRError, compile_openscad, generate, validate, validate_parameter_sweep, validate_program

class PipelineTests(unittest.TestCase):
    def test_russian_prompt_to_scad(self):
        program = generate("Кронштейн для камеры на трубу 25 мм с фиксацией винтом М4")
        defaults = {p.name: p.default for p in program.parameters}
        self.assertEqual(defaults["tube_d"], 25); self.assertEqual(defaults["screw_d"], 4.4)
        code = compile_openscad(program)
        self.assertIn("difference()", code); self.assertIn("tube_d = 25", code); self.assertIn("rotate([90, 0, 0])", code)
        self.assertIn("tube_d = 30", compile_openscad(program, {"tube_d": 30}))
    def test_dimension_word_is_not_m_fastener(self):
        program = generate("Кронштейн на трубу диаметром 80 мм с винтом М3")
        values = {p.name: p.default for p in program.parameters}
        self.assertEqual(values["tube_d"], 80)
        self.assertEqual(values["screw_d"], 3.4)

    def test_additional_product_families(self):
        plate = generate("Монтажная пластина 80x40x4 мм с четырьмя отверстиями М4")
        sleeve = generate("Втулка 20/8, высота 15 мм")
        self.assertEqual(plate.metadata["family"], "mounting_plate")
        self.assertEqual(sleeve.metadata["family"], "bushing")
        self.assertIn("plate_length = 80", compile_openscad(plate))
        self.assertIn("outer_d = 20", compile_openscad(sleeve))
        validate(plate); validate(sleeve)

    def test_constraints_and_slider_sweep(self):
        sleeve = generate("Втулка 20/8, высота 15 мм")
        with self.assertRaisesRegex(IRError, "minimum_wall"):
            validate(sleeve, {"outer_d": 10.4, "inner_d": 17.6})
        sweep = validate_parameter_sweep(sleeve)
        self.assertFalse(sweep["valid"])
        self.assertEqual(sweep["total"], 21)

    def test_static_validation_is_honest(self):
        report = validate_program(generate("pipe clamp 32 mm M5"), render=False)
        self.assertTrue(report["valid"]); self.assertEqual(report["level"], "ir"); self.assertEqual(report["checks"]["geometry"]["status"], "not_run")
