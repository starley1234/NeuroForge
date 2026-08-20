# Report: supplied `stl_items.sql` sample

Analyzed: 2026-08-20. The SQL dump was parsed as inert text; it was not restored or executed.

## Extraction result

- 3 rows found and extracted successfully.
- 3/3 contain an original natural-language request in `stl_item_code_basis`.
- 3/3 contain non-empty OpenSCAD.
- SQL escaping (`\n`, escaped quotes) was decoded into normal `.scad` files.
- `user_id`, `guest_id` and `answer_id` are intentionally excluded from generated training sidecars.
- Exact-source duplicates in this sample: 0.

OpenSCAD is not installed in the current sandbox, so actual STL compilation and image rendering were not claimed. The new Docker ingestion command is the intended execution environment.

## Items

| ID | Description | Lines | Numeric parameters | Notable operations |
|---:|---|---:|---:|---|
| 7077 | Compression flange with pipe socket, O-ring groove and bolt recesses | 126 | 17 | cylinders, union/difference, modules, circular loop |
| 7076 | Spring clip for glasses on a car sun visor | 148 | 12 | 2D profile, hull, offset, polygon, linear extrude, loops |
| 7075 | Parametric ergonomic drawer handle | 185 | 9 | hull-based body, modules, conditions, scale, mounting-hole loop |

## Corpus quality signal

The examples are substantially more valuable than trivial primitive datasets:

- meaningful engineering prompts are already present;
- variables have semantic names and units;
- code is modular and heavily commented;
- all three scripts use reusable modules and loops;
- two scripts use hull-based shape construction;
- one includes 2D profile generation and extrusion;
- constructions include clearances, preload, fastening and print-orientation intent.

This supports the direct code-model distillation strategy. A pretrained code LLM can learn the house style and construction patterns without a custom CAD latent model.

## Issues to measure on the full dump

1. Only the drawer handle visibly uses explicit Customizer ranges for most parameters. Many numeric parameters in the flange and clip have defaults but no machine-readable min/max range.
2. Comments include engineering claims such as “optimized without supports”; these should be treated as labels to verify, not ground truth.
3. Different scripts may be variants from the same generation session. They need `group_id` assignment or similarity clustering before train/test splitting.
4. Compilation, watertightness and connected-component checks remain mandatory.
5. Rendered image paths in SQL are metadata only; standardized views should be regenerated from source for consistent teacher input.
6. The full corpus should be profiled for missing operations: `rotate_extrude`, `minkowski`, `intersection`, gears, threads, assemblies and external libraries.

## Next command for the full dump

```bash
python3 -m training.extract_sql_corpus \
  --input data/stl_items.sql \
  --output data/sql-extracted \
  --license owned

python3 -m training.ingest_openscad \
  --input data/sql-extracted \
  --output data/openscad-corpus \
  --license owned

python3 -m training.profile_openscad \
  --corpus data/openscad-corpus \
  --output data/openscad-corpus/profile.json
```

For only three items, train/validation/test proportions are not meaningful. The sample is for validating extraction and auditing; model training should wait for the full corpus.
