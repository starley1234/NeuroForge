# NeuroSCAD Model Architecture v2 — Hierarchical CAD Flow

Status: design proposal, not trained. This document supersedes the original flat `class token + R9` CSG-Former idea as the target architecture. The existing Qwen LoRA script remains a baseline.

## Decision

Do not train a language model from scratch. Reuse a pretrained multilingual LLM/VLM for understanding and train the CAD representation, topology generator and geometric flow model for the domain.

The target is a hybrid architecture:

```text
Text / sketch / dimensions / loads
                 │
   pretrained language/vision encoders
       frozen first, then light LoRA
                 │
       requirement tokens (cross-attention)
                 ▼
┌──────────────────────────────────────────┐
│ Stage A: discrete topology diffusion     │
│ envelope → feature tree → operation tree │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ Stage B: rectified flow in CAD latent    │
│ proportions → dimensions → placements   │
└───────────────────┬──────────────────────┘
                    ▼
┌──────────────────────────────────────────┐
│ Stage C: residual detail refiner         │
│ holes → patterns → fillets → clearances  │
└───────────────────┬──────────────────────┘
                    ▼
       grammar-constrained IR decoder
                    ▼
     compiler → geometry kernel → validator
                    │
            repair / resample candidates
```

This is inspired by the useful principle behind FLUX-like systems—strong pretrained semantic conditioning plus generation in a learned latent space—but adapted to CAD's mixed discrete/continuous and constraint-heavy nature.

## Why not diffuse raw JSON or SCAD text

CAD has two different variable types:

- **Discrete:** operation type, feature-tree topology, references, boolean role.
- **Continuous:** dimensions, positions, directions, angles and tolerances.

Gaussian diffusion over raw token IDs has no meaningful geometry: token `DIFFERENCE + noise` is not a valid intermediate operation. A single continuous latent can also hide invalid topology. Therefore v2 factorizes the problem:

1. categorical masked diffusion for topology;
2. rectified flow for continuous geometry conditioned on topology;
3. deterministic constrained decoder and validator.

## Coarse-to-fine representation

The IR training representation must have explicit levels. Merely running more denoising steps does not guarantee that the model learns “outer shape first”.

### Level 0 — Engineering intent

- part family and function;
- units and coordinate frame;
- overall bounding envelope;
- symmetry and principal axes;
- interfaces: tube, bolt, PCB, shaft, mating face;
- manufacturing process and material when supplied.

### Level 1 — Structural feature graph

- base solids and primary sketches;
- union/difference/intersection structure;
- extrude/revolve/sweep operations;
- parent/attachment references;
- repeated patterns.

### Level 2 — Parameters

- exact dimensions;
- placements and orientations;
- constraint expressions;
- standard clearances and fits;
- parameter ranges.

### Level 3 — Details

- holes and counterbores;
- fillets/chamfers;
- ribs, bosses and local reinforcement;
- printability/manufacturing adjustments.

Training randomly hides or corrupts later levels while preserving earlier levels. The refiner learns to complete detail without changing approved interfaces and the global envelope.

## Components

### 1. Semantic conditioner

Recommended initial options:

- text: pretrained multilingual Qwen-class 0.6B–1.7B encoder/decoder backbone;
- image/sketch: SigLIP2 or DINO-family encoder;
- physical conditions: small MLP for load vectors, material and process tokens.

Use selected hidden states rather than generated prose. Project every modality to `d_model=512` and retain modality/type embeddings. Start frozen; train projectors; then unfreeze only LoRA adapters if validation shows benefit.

A 4B/8B text encoder similar in spirit to FLUX Klein is unnecessary for the first 16 GB experiment and would consume the memory needed by CAD generation. Scale it only after an ablation proves language conditioning is the bottleneck.

### 2. Hierarchical CAD autoencoder

Purpose: learn a compact, editable latent representation before generative training.

Proposed baseline:

- tree-aware Transformer encoder, 8 layers, `d=512`, 8 heads;
- 64 latent slots;
- residual vector quantization with separate codebooks for envelope, topology, features and details;
- grammar-constrained Transformer decoder, 10 layers;
- continuous parameter head with normalized values plus unit/type embeddings;
- approximate trainable parameters: 70–110M, depending on shared layers.

Loss:

```text
L_AE = L_command
     + λ_param L_dimension
     + λ_ref L_reference
     + λ_constraint L_constraint
     + λ_render L_geometry
     + λ_commit L_RVQ
```

The autoencoder is accepted only if reconstructed programs compile and preserve dimensions—not merely if token accuracy is high.

### 3. Topology generator

Use masked categorical diffusion over a bounded set of structural slots:

- mask/noise operation classes and parent references;
- denoise from envelope to primary features to detail features;
- apply grammar masks at every step;
- condition through cross-attention to semantic tokens;
- allow partial known trees for editing and completion.

A small Transformer denoiser, 8–12 layers at `d=512`, is sufficient for the first benchmark. Mamba/state-space blocks become attractive only when real sequences routinely exceed roughly 200–300 nodes; that decision must be benchmark-driven.

### 4. Geometry rectified-flow transformer

Conditioned on fixed topology, predict continuous latent values:

- dimensions and ratios;
- positions and rotations;
- sketch control points;
- clearances and feature parameters.

Use a multimodal diffusion transformer:

- separate semantic and CAD streams in early blocks;
- joint attention in later blocks;
- timestep embedding;
- tree-depth, parent and feature-level embeddings;
- classifier-free condition dropout for controllability;
- v-prediction / flow-matching objective.

Initial profile:

- 12 blocks;
- `d_model=512`;
- 8 heads;
- FFN 2048;
- 64–128 CAD latent tokens;
- approximately 70–100M trainable parameters.

Start with 20–32 flow steps. Distill to 4–8 steps only after the undistilled teacher is accurate. Training a short-step student directly from scratch usually trades away diversity and geometry quality.

### 5. Detail refiner

The same flow backbone can be reused with a level mask, or a smaller 6-layer model can refine only Level 3. Inputs include the validated coarse program and optional user edits. The global envelope and interface dimensions are locked.

### 6. Validator-guided search

Generate 4–8 cheap candidates, then rank by deterministic checks:

- IR/constraint validity;
- compile and non-empty solid;
- components and watertightness;
- requested dimension error;
- minimum wall and edge distance;
- text/feature alignment score.

Do not backpropagate through OpenSCAD initially. Use validator outcomes for rejection sampling and preference data. Differentiable SDF/point losses are auxiliary signals, not replacements for exact compilation.

## Training plan

### Phase 0 — Baselines

- deterministic family generators (already implemented);
- Qwen LoRA text-to-IR baseline (already implemented);
- retrieval + parameter fitting baseline.

No new architecture is accepted unless it beats these baselines on held-out real requests.

### Phase 1 — Representation

Train the CAD autoencoder on 100k–1M executable programs. Curriculum:

1. primitives and one feature;
2. two to five features;
3. patterns and constraints;
4. long feature trees and details.

Gate: `>=99.5%` IR validity and `>=98%` compile rate on reconstructions, plus dimension-error thresholds.

### Phase 2 — Semantic alignment

Contrastively align text, renders/sketches and CAD latents. Use hard negatives with similar appearance but different hole count, dimensions or topology.

### Phase 3 — Topology diffusion

Train categorical corruption/denoising from coarse structural conditions. Measure topology exact match, feature F1 and compile rate.

### Phase 4 — Geometry flow

Train continuous rectified flow conditioned on ground-truth topology first. Then gradually replace ground-truth topology with sampled topology.

### Phase 5 — Joint refinement

Jointly fine-tune projectors, topology and flow with validator-generated preference pairs. Keep exact compiler and safety checks outside the neural model.

### Phase 6 — Distillation

Distill generation steps and optionally the semantic encoder only after quality targets are reached.

## Data requirements

The current three-family synthetic v2 dataset is enough to test plumbing, not to train this architecture meaningfully.

Minimum useful research run:

- 100k–300k valid programs;
- at least 30–50 feature templates;
- at least 10 mechanically distinct families;
- multiple construction histories for equivalent geometry;
- text paraphrases plus precise expert descriptions;
- family-held-out and generator-held-out test sets;
- 2D sketches/multi-view renders for multimodal alignment.

A serious general model likely needs 1M+ diverse executable programs plus a smaller, carefully curated set of real engineering requests.

## RTX 5060 Ti 16 GB profile

Feasible locally:

- train CAD autoencoder components separately;
- train 70–120M topology/flow models in BF16 with gradient checkpointing;
- frozen 0.6B–1.7B language conditioner, optionally 4-bit with LoRA;
- batch size 1–8 plus accumulation;
- latent length 64–128.

Not a sensible single-GPU starting point:

- training a multilingual LLM from scratch;
- jointly training a 4B flow model and 8B text encoder;
- full-resolution B-Rep surface diffusion;
- claiming four-step high-quality sampling before teacher training/distillation.

## Required ablations

Compare:

1. Qwen autoregressive text-to-IR baseline;
2. topology AR + continuous regression;
3. categorical topology diffusion + regression;
4. categorical topology diffusion + rectified flow;
5. flat latent versus explicit coarse-to-fine levels;
6. frozen text encoder versus LoRA;
7. Transformer versus Mamba only on long sequences.

The winning architecture is the smallest one that improves compile rate, dimension accuracy, feature recall and held-out-family generalization—not the one with the most fashionable blocks.

## Final recommendation

Build v2 as **pretrained semantic conditioner + hierarchical CAD autoencoder + categorical topology diffusion + continuous rectified flow + constrained decoder + deterministic validator**.

This preserves the FLUX-like strengths the project needs—semantic conditioning, latent generation, iterative refinement and future multimodality—without pretending that CAD topology behaves like pixels.
