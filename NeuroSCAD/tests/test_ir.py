import unittest
from neuroscad import IRError, Node, Op, Program, calc, param, validate
from neuroscad.ir import Parameter, evaluate

class IRTests(unittest.TestCase):
    def test_expression(self):
        self.assertEqual(evaluate(calc("add", param("d"), calc("mul", 2, param("wall"))), {"d": 25, "wall": 4}), 33)
    def test_rejects_cycle(self):
        p = Program("a", (Node("a", Op.UNION, ("b", "b")), Node("b", Op.UNION, ("a", "a"))))
        with self.assertRaisesRegex(IRError, "cycle"): validate(p)
    def test_rejects_bad_dimension(self):
        with self.assertRaisesRegex(IRError, "positive"): validate(Program("box", (Node("box", Op.BOX, args={"size": [1, 2, -1]}),)))
    def test_round_trip(self):
        p = Program("box", (Node("box", Op.BOX, args={"size": [1, 2, 3]}),), (Parameter("x", 2, 1, 3),), {"name": "demo"})
        self.assertEqual(Program.from_dict(p.to_dict()).to_dict(), p.to_dict())
