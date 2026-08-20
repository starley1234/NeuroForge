# Low-cost OpenSCAD distillation plan

## Revised decision

With 500 good owner-supplied OpenSCAD scripts, the cheapest credible path is **not** to train a diffusion CAD model first. OpenSCAD is already a compact programming language, and frontier code LLMs can generate it. We should distill that capability into a 3B code model and keep compilation, geometry checks and visual judging outside the model.

```text
500 trusted scripts + original prompts if available
                 │
       audit → compile → STL metrics
                 │
          standardized 8-view renders
                 │
 strong teacher: description + requirements + concise plan
                 │
       source-level split before augmentation
                 │
  3B code LLM LoRA: prompt/edit → plan + OpenSCAD
                 │
 sample 2–4 candidates → compile → rank → repair once
```

The hierarchical diffusion/flow architecture remains a longer-term research branch for broad shape exploration. It is unnecessary until direct code generation reaches a measured ceiling.

## Why this is likely to work

- The base model already knows programming, geometry vocabulary and natural language.
- The 500 scripts teach project-specific modeling style, parameterization and construction patterns.
- One teacher call can create several useful prompts and a concise construction plan for each validated model.
- Execution filtering provides cheap, objective supervision.
- The same student can learn generation, editing and repair because all are code tasks.

Research supports this direction: direct CadQuery generation with pretrained code LLMs improved with model scale; CAD-Coder found a small high-quality executable subset more useful than a much larger noisy set; execution and geometric feedback substantially improve single-pass generation.

## What 500 scripts are enough for

Enough for:

- a strong domain LoRA if the scripts cover meaningful diversity;
- retrieval examples for a teacher or student;
- learning preferred OpenSCAD style and parameter conventions;
- several thousand prompt/code pairs after honest paraphrase augmentation;
- generation and editing within represented construction patterns.

Not enough for:

- learning general mechanical engineering from scratch;
- proving performance on unseen part families;
- training a useful diffusion latent space from scratch;
- turning 500 near-duplicate brackets into a universal CAD model.

Diversity matters more than raw count. We must report family, operation and complexity coverage before training.

## Pipeline now implemented

### 1. Extract SQL dump when needed

For the supplied phpMyAdmin `stl_items` dump, do **not** restore the database. Parse it as inert text:

```bash
python3 -m training.extract_sql_corpus \
  --input data/stl_items.sql \
  --output data/sql-extracted \
  --license owned
```

The extractor writes one `.scad` and one sidecar `.json` per item. It intentionally omits `user_id`, `guest_id` and `answer_id` from training metadata.

### 2. Corpus audit and rendering

Put scripts in one directory, or use `data/sql-extracted` from the previous step. Optional sidecar with the same stem:

```json
{
  "prompt": "Кронштейн камеры на трубу 25 мм, винт М4",
  "license": "owned",
  "group_id": "camera-clamp-family-01",
  "tags": ["clamp", "camera"]
}
```

Dry audit, without execution:

```bash
python3 -m training.ingest_openscad \
  --input /path/to/500-scripts \
  --output data/openscad-corpus \
  --license owned \
  --dry-run
```

Compile, inspect STL and generate eight views:

```bash
python3 -m training.ingest_openscad \
  --input /path/to/500-scripts \
  --output data/openscad-corpus \
  --license owned
```

Recommended isolated container invocation (the host output directory must be writable by UID 10001):

```bash
docker build -t neuroscad:0.4.1 .
mkdir -p data/openscad-corpus
sudo chown 10001:10001 data/openscad-corpus
docker run --rm --network none --read-only --tmpfs /tmp:size=1g \
  -v /path/to/500-scripts:/input:ro \
  -v "$PWD/data/openscad-corpus:/output" \
  neuroscad:0.4.1 python -m training.ingest_openscad \
  --input /input --output /output --license owned
```

Strict mode rejects external `include`, `use`, `import` and `surface`. `--trusted` allows them but must only run in an isolated no-network container; dependency packaging is then the owner's responsibility.

Outputs:

```text
data/openscad-corpus/
├── records.jsonl
├── quarantine.jsonl
├── manifest.json
├── sources/
├── meshes/
└── renders/<source-id>/{iso_front,front,right,...}.png
```

Profile operation coverage before spending teacher tokens:

```bash
python3 -m training.profile_openscad \
  --corpus data/openscad-corpus \
  --output data/openscad-corpus/profile.json
```

The train/validation/test split is assigned **before** paraphrases and edits are created. Put variants of one construction into the same sidecar `group_id`; otherwise the source hash is used.

### 3. Cheap retrieval baseline

After adding original prompts or teacher annotations, retrieve similar trusted scripts without a vector database:

```bash
python3 -m training.retrieve_examples \
  --corpus data/openscad-corpus \
  --annotations data/openscad-corpus/annotations.jsonl \
  --prompt "корпус для платы 80 на 50 с вентиляцией" \
  --top-k 4
```

The dependency-free TF-IDF baseline is deliberately simple. Measure it before paying for an embedding service; Zero-to-CAD also found lightweight documentation retrieval sufficient in its synthesis loop.

### 4. Teacher annotation

Use any OpenAI-compatible multimodal teacher. Credentials stay in an environment variable and are never written to the dataset:

```bash
export TEACHER_API_KEY=...
python3 -m training.annotate_teacher \
  --corpus data/openscad-corpus \
  --output data/openscad-corpus/annotations.jsonl \
  --endpoint https://provider.example/v1/chat/completions \
  --model teacher-model
```

Each request contains code, parameters and up to four rendered views. The teacher returns:

- grounded description;
- part family;
- explicit requirements;
- short public construction plan;
- Russian and English paraphrases;
- visible quality concerns.

The script is resumable. Start with `--limit 10`, inspect annotations manually, then process the corpus.

### 5. Build distillation data

```bash
python3 -m training.build_distillation \
  --corpus data/openscad-corpus \
  --annotations data/openscad-corpus/annotations.jsonl \
  --output data/distilled \
  --edits-per-model 2
```

Targets use:

```text
<design_plan>...</design_plan>
<openscad>...</openscad>
```

This is a concise modeling plan, not hidden chain-of-thought. All paraphrases and edit examples inherit the source model's split, preventing leakage.

Expected first corpus size depends on annotation quality, but 500 models × roughly 5–8 grounded prompts plus edits gives approximately 3k–5k high-quality rows. That is suitable for LoRA adaptation, not foundation pretraining.

### 6. Train the student

```bash
pip install -e '.[training]'
python3 -m training.train --config training/openscad_config.example.json
```

Initial student: `Qwen2.5-Coder-3B-Instruct`, LoRA rank 32, context 3072. It is a cost/VRAM baseline, not a sacred choice. Compare 1.5B, 3B and 7B under the same held-out test.

### 7. Evaluate executable output

Run inference on held-out prompts and save JSONL rows with `prediction`. Then:

```bash
python3 -m training.evaluate_openscad predictions.jsonl \
  --details evaluation-details.jsonl
```

This measures strict source audit, compile rate, watertightness, positive volume and connected components. Semantic correctness still needs requirements checks and visual review.

## Cheap quality improvements in priority order

1. **RAG before training:** retrieve 2–4 similar owner scripts as examples. This can produce value immediately with the teacher and provides a baseline the student must beat.
2. **Best-of-4 execution filtering:** generate several candidates, reject compile failures, rank by requirements and mesh metrics.
3. **One repair pass:** feed OpenSCAD errors and exact metrics back to the model.
4. **Preference training:** retain accepted/rejected pairs and run DPO. This is cheaper and simpler than GRPO.
5. **Targeted teacher synthesis:** ask the teacher for variants that add one meaningful feature to an existing script, compile every result, and keep only valid, visually approved variants.
6. **External Apache data:** sample diverse Zero-to-CAD programs, translate a curated subset to OpenSCAD with the teacher, and retain only compiled equivalents. Do not bulk-convert a million examples before proving value on 1k–10k.
7. **GRPO only later:** use it when SFT+DPO plateaus and the reward reliably captures dimensions and requested features.

## Validation hierarchy

A script is accepted only when it passes all available layers:

1. static audit and source limits;
2. OpenSCAD execution with timeout;
3. non-empty STL;
4. positive volume, one component, watertightness;
5. exact requested dimensions and feature counts where testable;
6. multi-view teacher/VLM judgement for holistic shape;
7. human review for the small golden test set.

Visual judgement must not replace exact dimensions. Exact metrics must not replace visual feature checking. Use both.

## Golden benchmark from the 500 scripts

Reserve approximately:

- 400 source models for train;
- 50 for validation;
- 50 for untouched test.

Additionally create 50–100 human-written requests that are not teacher paraphrases. Track:

- compile rate;
- single-solid and watertight rates;
- parameter extraction accuracy;
- requested feature precision/recall;
- normalized Chamfer/IoU when a target mesh exists;
- human preference against the teacher and RAG baseline;
- latency and candidates consumed per accepted design.

Never tune prompts or hyperparameters on the final test set.

## When to revisit diffusion

Revisit hierarchical CAD flow when we have at least tens of thousands of structurally diverse, validated programs and direct code generation shows a specific deficiency such as poor design diversity or unstable global topology. Until then, pretrained code priors plus execution feedback provide a much better return per GPU-hour.
