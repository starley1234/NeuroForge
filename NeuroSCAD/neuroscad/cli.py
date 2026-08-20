from __future__ import annotations
import argparse, json
from pathlib import Path
from .compiler import compile_openscad
from .ir import Program
from .templates import generate
from .validator import validate_program

def main() -> None:
    parser = argparse.ArgumentParser(prog="neuroscad", description="Typed text-to-CSG prototype")
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate"); gen.add_argument("prompt"); gen.add_argument("--ir", type=Path); gen.add_argument("--scad", type=Path)
    val = sub.add_parser("validate"); val.add_argument("ir", type=Path); val.add_argument("--render", action="store_true")
    comp = sub.add_parser("compile"); comp.add_argument("ir", type=Path); comp.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        program = generate(args.prompt); code = compile_openscad(program)
        if args.ir: args.ir.write_text(json.dumps(program.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.scad: args.scad.write_text(code, encoding="utf-8")
        if not args.ir and not args.scad: print(code, end="")
    else:
        program = Program.from_dict(json.loads(args.ir.read_text(encoding="utf-8")))
        if args.command == "validate": print(json.dumps(validate_program(program, render=args.render), ensure_ascii=False, indent=2))
        else:
            code = compile_openscad(program)
            if args.output: args.output.write_text(code, encoding="utf-8")
            else: print(code, end="")
if __name__ == "__main__": main()
