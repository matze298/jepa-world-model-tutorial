# JEPA-Style World Models

## From Representation Learning to Real-World Predictive Systems

This tutorial builds JEPA-style world models from first principles.

It starts with the mathematical and conceptual foundations of Joint-Embedding Predictive Architectures, then develops a minimal I-JEPA implementation, scales it into a research-grade codebase, extends it to temporal and action-conditioned world models, and finally applies the ideas to a real-world cycling intelligence problem.

The intended workflow is to write the code alongside the tutorial. The documentation explains each component, then the repository implementation in `src/`, `experiments/`, `notebooks/`, and `tests/` grows with it.

The goal is to bridge:

- research theory,
- concrete implementation,
- systems engineering,
- real-world sequential decision-making.

---

## What This Tutorial Is About

Modern AI systems often learn by reconstructing, classifying, or generating observations. JEPA-style models take a different route: they predict **representations** of unobserved or future parts of the world.

Instead of learning:

\[
x_{\mathrm{ctx}} \rightarrow x_{\mathrm{tgt}}
\]

they learn:

\[
x_{\mathrm{ctx}} \rightarrow z_{\mathrm{tgt}}
\]

where:

\[
z_{\mathrm{tgt}} = f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

This shifts the learning problem from pixel-level reconstruction to latent predictive abstraction.

The long-term goal is a world model:

\[
(z_{\leq t}, a_{t:t+k}) \rightarrow z_{t+k}
\]

that can support prediction, planning, and decision-making.

---

## Who This Is For

This tutorial assumes familiarity with:

- deep learning,
- PyTorch,
- transformers,
- self-supervised learning basics,
- representation learning,
- optimization,
- basic probability and linear algebra.

It is written for readers who want more than a high-level explanation. The aim is to understand the method well enough to rebuild it, debug it, modify it, and extend it.

---

## Tutorial Roadmap

```text
Chapter 1 — Foundations of JEPA
Chapter 2 — Minimal I-JEPA Implementation
Chapter 3 — Research-Grade Engineering
Chapter 4 — From Images to Video and Temporal JEPA
Chapter 5 — Action-Conditioned World Models
Chapter 6 — Real-World Application: Cycling Intelligence
Chapter 7 — Advanced Extensions and Open Problems
```

---

# Chapter 1 — Foundations of JEPA

Chapter 1 develops the conceptual and mathematical foundation.

It explains why JEPA predicts representations rather than pixels, how the objective is constructed, and what kind of representation geometry we want.

## Sections

### [1.0 Introduction](chapter-01/introduction.md)

Introduces the motivation for JEPA-style world models.

Topics:

- why raw observation prediction is often too detailed,
- why latent prediction is useful,
- how JEPA connects to world modeling,
- how the tutorial is structured,
- why cycling intelligence is a useful real-world application.

Status: **drafted**

---

### [1.1 The Self-Supervised Learning Landscape](chapter-01/ssl-landscape.md)

Places JEPA within the broader self-supervised learning landscape.

Topics:

- reconstruction-based methods,
- masked autoencoders,
- contrastive learning,
- BYOL-style methods,
- VICReg and redundancy reduction,
- masked predictive representation learning,
- temporal self-supervision,
- world models.

Status: **drafted**

---

### [1.2 Why Pixel Prediction Is Problematic](chapter-01/why-not-pixels.md)

Explains why predicting raw pixels or raw observations may be inefficient for semantic understanding and planning.

Topics:

- multimodal futures,
- nuisance detail,
- shortcut learning,
- pixel-space bottlenecks,
- observation prediction vs state prediction,
- why latent targets can be better for world models.

Status: **drafted**

---

### [1.3 The JEPA Objective](chapter-01/jepa-objective.md)

Derives the core JEPA objective.

Topics:

- online encoder,
- target encoder,
- predictor,
- stop-gradient,
- EMA target updates,
- latent-space losses,
- mask metadata,
- collapse risks,
- mask leakage,
- connection to temporal world models.

Status: **drafted**

---

### [1.4 Representation Geometry](chapter-01/representation-geometry.md)

Explains how JEPA shapes latent space.

Topics:

- similarity and distance,
- invariance,
- equivariance,
- latent manifolds,
- information bottlenecks,
- collapse,
- dimensional collapse,
- covariance and redundancy,
- nearest-neighbor structure,
- linear probing,
- trajectory geometry,
- planning-friendly latent spaces.

Status: **drafted**

---

# Chapter 2 — Minimal I-JEPA Implementation

Chapter 2 turns the mathematical objective into working code.

The goal is a minimal but correct PyTorch implementation of image JEPA. The implementation should be small enough to understand completely, but realistic enough to train and evaluate.

The current Chapter 2 draft covers the complete minimal image JEPA path, from patchification through the first end-to-end experiment and consolidation.

## Sections

### [2.0 Implementation Overview](chapter-02/implementation-overview.md)

Defines the full implementation target.

Topics:

- what the minimal system will include,
- what will be intentionally omitted,
- full data flow diagram,
- core tensor shapes,
- model components,
- training step overview.

Status: **drafted**

---

### [2.1 Project Setup](chapter-02/project-setup.md)

Sets up the local implementation environment.

Topics:

- repository structure,
- Python environment,
- PyTorch installation,
- dependencies,
- dataset folders,
- reproducibility utilities,
- typing and formatting conventions.

Status: **drafted**

---

### [2.2 Image Patchification](chapter-02/image-patchification.md)

Implements image-to-patch conversion.

Topics:

- patch embeddings,
- patch indexing,
- image grid layout,
- flattening and unflattening patches,
- patch visualization,
- shape discipline.

Status: **drafted**

---

### [2.3 Positional Embeddings](chapter-02/positional-embeddings.md)

Adds spatial information to patch tokens.

Topics:

- learned positional embeddings,
- sinusoidal positional embeddings,
- 2D grid positions,
- target-position conditioning,
- interpolation for different image sizes.

Status: **drafted**

---

### [2.4 Context and Target Mask Sampling](chapter-02/context-target-masks.md)

Implements the JEPA masking strategy.

Topics:

- random masks,
- block masks,
- target block sampling,
- context sampling,
- preventing overlap,
- mask visualization,
- mask leakage checks,
- batch padding or fixed target sizes.

Status: **drafted**

---

### [2.5 Minimal ViT Encoder](chapter-02/minimal-vit-encoder.md)

Builds the encoder backbone.

Topics:

- transformer blocks,
- patch token input,
- context token selection,
- LayerNorm,
- MLP blocks,
- attention heads,
- output representations.

Status: **drafted**

---

### [2.6 EMA Target Encoder](chapter-02/ema-target-encoder.md)

Implements the slowly moving target encoder.

Topics:

- copying online encoder weights,
- disabling gradients,
- EMA updates,
- EMA schedules,
- target encoder checkpointing,
- target representation stability.

Status: **drafted**

---

### [2.7 JEPA Predictor](chapter-02/jepa-predictor.md)

Builds the predictor network.

Topics:

- context projection,
- learned target queries,
- target positional embeddings,
- transformer predictor,
- output projection,
- predictor depth and width,
- predictor asymmetry.

Status: **drafted**

---

### [2.8 Losses and Diagnostics](chapter-02/losses-and-diagnostics.md)

Implements training losses and representation health checks.

Topics:

- MSE loss,
- Smooth L1 loss,
- cosine loss,
- collapse diagnostics,
- variance metrics,
- covariance metrics,
- effective rank,
- prediction-target similarity.

Status: **drafted**

---

### [2.9 Training Loop](chapter-02/training-loop.md)

Builds the full training loop.

Topics:

- dataloader,
- forward pass,
- loss computation,
- optimizer step,
- EMA update,
- logging,
- checkpointing,
- gradient clipping,
- mixed precision optional.

Status: **drafted**

---

### [2.10 Evaluation: k-NN and Linear Probe](chapter-02/evaluation-knn-minimal-probe.md)

Evaluates the learned representation.

Topics:

- frozen encoder evaluation,
- k-nearest-neighbor retrieval,
- linear classification probe,
- representation extraction,
- train/validation split,
- diagnostic plots.

Status: **drafted**

---

### [2.11 Running the First Experiment](chapter-02/first-experiment.md)

Runs the minimal implementation end-to-end.

Topics:

- dataset choice,
- minimal config,
- expected logs,
- sanity checks,
- common failure modes,
- first ablation ideas.

Status: **drafted**

---

### [2.12 Chapter 2 Consolidation](chapter-02/consolidation.md)

Summarizes the completed minimal image JEPA implementation and defines the handoff to Chapter 3.

Topics:

- final source layout,
- experiment scripts,
- test layout,
- debugging notebooks,
- end-to-end data flow,
- implementation invariants,
- completion criteria.

Status: **drafted**

---

# Chapter 3 — Research-Grade JEPA Training

Chapter 3 turns the minimal Chapter 2 implementation into a serious research harness.

The reader is assumed to already understand production-grade deep-learning infrastructure: structured configs, logging, checkpoints, mixed precision, distributed training, cloud runs, and experiment tracking.

Therefore this chapter focuses on what is **JEPA-specific**:

- preserving the online encoder, target encoder, predictor, and EMA state correctly,
- keeping mask sampling and target prediction explicit,
- logging collapse diagnostics as first-class metrics,
- making checkpoint/resume semantics correct for EMA-based training,
- scaling with Lightning Fabric without hiding the algorithm,
- supporting controlled ablations over masks, EMA, predictor capacity, and losses.

The core training logic remains:

```text
sample masks
encode context
encode target without gradients
predict target representation
compute latent loss
update online branch
EMA-update target branch
log diagnostics
```

The engineering around it becomes more robust.

## Sections

### [3.0 Research-Grade JEPA Training: Design Goals](chapter-03/research-grade-jepa-training.md)

Frames the transition from a minimal educational implementation to a research-grade JEPA training harness.

Topics:

- what stays unchanged from Chapter 2,
- what needs hardening,
- why JEPA has nonstandard training-state requirements,
- why the training loop should remain explicit,
- why Lightning Fabric is preferred before full PyTorch Lightning,
- what “research-grade” means specifically for JEPA.

Status: **drafted**

---

### [3.1 Configuration for JEPA Experiments](chapter-03/configuration-for-jepa-experiments.md)

Introduces structured experiment configs without re-explaining configuration systems from scratch.

Topics:

- JEPA-relevant config groups,
- model/mask/training/EMA/evaluation/runtime separation,
- validation rules that catch JEPA-specific mistakes,
- local, STL-10, and cloud config examples,
- config-driven model, mask, data, and optimizer construction.

Important validation rules:

```text
image_size % patch_size == 0
encoder_dim % encoder_heads == 0
predictor_dim % predictor_heads == 0
context_patches + target_patches <= total_patches
tau_base <= tau_final
target encoder excluded from optimizer
```

Planned files:

```text
src/jepa_world_model/configs/schema.py
src/jepa_world_model/configs/loader.py
configs/ijepa_cifar10_debug.yaml
configs/ijepa_stl10_base.yaml
configs/ijepa_stl10_cloud.yaml
```

Status: **drafted**

---

### [3.2 Run State and Experiment Manifests](chapter-03/run-state-and-experiment-manifests.md)

Defines the minimal auditable run state for JEPA experiments.

Topics:

- run directory layout,
- resolved config snapshots,
- manifest metadata,
- Git and environment capture,
- separating training checkpoints from representation artifacts,
- run resumption semantics.

JEPA-specific state that must be preserved:

```text
online encoder
target encoder
predictor
optimizer
scheduler
EMA schedule state / global step
mask config
loss config
representation diagnostics history
```

Planned files:

```text
src/jepa_world_model/engine/run_context.py
manifest.yaml
config.yaml
notes.md
standardized run directory layout
```

Status: **drafted**

---

### [3.3 Metrics, Diagnostics, and Logging Cadence](chapter-03/metrics-diagnostics-and-logging-cadence.md)

Defines what to log for JEPA and how often.

Topics:

- metric taxonomy,
- cheap vs expensive diagnostics,
- logging cadence,
- JSONL as canonical metric stream,
- TensorBoard/W&B as optional sinks,
- metric naming conventions,
- collapse early-warning signals.

Metric groups:

```text
optimization:
  loss
  lr
  grad_norm

prediction:
  pred_target/mse
  pred_target/cosine

target representation:
  target/std_mean
  target/dead_dim_fraction
  target/effective_rank
  target/norm_mean

prediction representation:
  pred/std_mean
  pred/effective_rank
  pred/norm_mean

mask:
  mask/overlap_fraction
  mask/context_ratio
  mask/target_ratio

EMA:
  ema_tau
  ema/relative_param_l2

evaluation:
  knn/accuracy
  linear_probe/val_acc
```

Planned files:

```text
src/jepa_world_model/engine/logging.py
MetricLogger
JSONL sink
TensorBoard sink
optional W&B sink
diagnostic cadence config
```

Status: **drafted**

---

### 3.4 Checkpointing and Resume Semantics

Makes checkpointing correct for JEPA, not just generic PyTorch.

Topics:

- what must be saved,
- what can be exported separately,
- resume behavior,
- strict vs non-strict checkpoint loading,
- online-only representation export,
- target encoder restoration,
- RNG state handling,
- scheduler and EMA consistency.

Critical distinction:

```text
Training checkpoint:
  online encoder
  target encoder
  predictor
  optimizer
  scheduler
  global step
  epoch
  config
  RNG state

Representation artifact:
  online encoder only
  preprocessing metadata
  model config
```

Important failure case:

```text
Resuming without target encoder state changes the JEPA target distribution.
```

Planned files:

```text
src/jepa_world_model/engine/checkpointing.py
save_training_checkpoint
load_training_checkpoint
export_encoder
resume validation
```

Status: **to generate**

---

### 3.5 Optimizer, Schedules, and Gradient Management

Consolidates training-control logic.

Topics:

- AdamW parameter groups,
- excluding the target encoder,
- learning-rate schedules,
- EMA tau schedule,
- gradient clipping,
- gradient accumulation,
- update ordering,
- sanity checks for trainable parameters.

JEPA-specific update order:

```text
forward
loss
backward
gradient clipping
optimizer step
scheduler step
EMA update
diagnostics/logging
```

Important invariants:

```text
target encoder is never in the optimizer
target encoder has no gradients
target encoder changes only through EMA
```

Planned files:

```text
src/jepa_world_model/engine/optim.py
src/jepa_world_model/engine/schedules.py
parameter group builders
gradient norm utilities
update-order test
```

Status: **to generate**

---

### 3.6 Mixed Precision and Numerical Stability

Adds precision control without making the JEPA loop opaque.

Topics:

- fp32 vs fp16 vs bf16,
- Fabric precision modes,
- where to keep diagnostics in fp32,
- covariance/effective-rank precision concerns,
- NaN/Inf detection,
- loss scaling implications,
- when to disable expensive diagnostics.

JEPA-specific issues:

```text
representation diagnostics should often cast to fp32
covariance/eig diagnostics can be numerically fragile
cosine metrics can hide norm collapse
bf16 is usually preferable on modern GPUs
```

Planned files:

```text
precision config
safe diagnostic casting
finite checks
grad-scaler/Fabric integration
```

Status: **to generate**

---

### 3.7 Lightning Fabric Training Harness

Replaces manual device/distributed boilerplate while keeping the JEPA loop explicit.

Topics:

- Fabric setup,
- model/optimizer/dataloader wrapping,
- Fabric backward,
- distributed-safe logging,
- checkpointing with Fabric,
- keeping EMA update explicit,
- avoiding full LightningModule for now.

The loop should still visibly contain:

```python
pred_repr, target_repr = model(...)
loss = latent_loss(...)
fabric.backward(loss)
optimizer.step()
update_ema(...)
```

Planned files:

```text
src/jepa_world_model/engine/fabric_trainer.py
experiments/train.py
FabricJEPAState
distributed-safe metric logging
```

Status: **to generate**

---

### 3.8 Evaluation During and After Training

Integrates representation evaluation into the research workflow.

Topics:

- evaluation cadence,
- k-NN and linear probe scheduling,
- online encoder vs target encoder evaluation,
- feature extraction consistency,
- checkpoint selection,
- random encoder baseline,
- evaluation artifact writing.

JEPA-specific evaluation questions:

```text
Does lower JEPA loss correlate with better probe accuracy?
Does target encoder outperform online encoder?
Does EMA smoothing help retrieval?
Does representation collapse appear before probe degradation?
```

Planned files:

```text
experiments/evaluate.py
evaluation hooks
optional periodic eval during training
retrieval artifact writing
probe result logging
```

Status: **to generate**

---

### 3.9 Cloud Execution and RunPod Workflow

Defines the cloud execution pattern without teaching cloud basics.

Topics:

- single training entry point,
- cloud config,
- persistent storage paths,
- environment reproduction with uv sync,
- artifact collection,
- interrupt/resume behavior,
- common RunPod footguns.

JEPA-specific cloud concerns:

```text
long runs must be resumable
diagnostics should be throttled
checkpoints must include target encoder
evaluation may be run separately
metrics should stream to persistent storage
```

Planned files:

```text
configs/ijepa_stl10_cloud.yaml
scripts/run_cloud_train.sh
scripts/resume_cloud_train.sh
cloud run checklist
```

Status: **to generate**

---

### 3.10 Ablations and Experiment Sweeps

Makes controlled JEPA ablations easy.

Topics:

- sweep spec format,
- config overrides,
- deterministic run naming,
- one-factor and small-grid ablations,
- result aggregation,
- avoiding confounded sweeps.

Recommended first ablations:

```text
mask.context_ratio
mask.target_block_height/width
mask.num_target_blocks
training.loss_type
ema.tau_base
model.predictor_depth
model.predictor_dim
model.encoder_depth
```

JEPA-specific analysis:

```text
loss vs linear probe
loss vs effective rank
context ratio vs collapse
target block size vs retrieval
predictor capacity vs encoder quality
```

Planned files:

```text
experiments/ablate.py
configs/ablations/*.yaml
sweep result directories
aggregation script
```

Status: **to generate**

---

### 3.11 Failure-Mode Playbook

Creates a practical debugging guide for JEPA-specific failures.

Topics:

- symptoms,
- likely causes,
- diagnostic metrics,
- first fixes,
- confirmation tests.

Failure modes:

```text
loss does not decrease
loss goes to zero too fast
target encoder gets gradients
EMA does not move
EMA diverges
mask overlap > 0
representation collapse
dimensional collapse
norm explosion
probe accuracy random
k-NN retrieval meaningless
cloud run differs from local
resume changes dynamics
mixed precision NaNs
```

Planned content:

```text
failure-mode table
diagnostic checklist
metric thresholds
debug commands
```

Status: **to generate**

---

### 3.12 Chapter 3 Consolidation

Summarizes the final research-grade training harness.

Topics:

- final file layout,
- final commands,
- run lifecycle,
- invariants,
- checklist before moving to temporal/video JEPA.

Final expected lifecycle:

```text
choose config
initialize run directory
train with Fabric
log JSONL + TensorBoard/W&B
checkpoint/resume
evaluate
aggregate ablations
inspect failure diagnostics
```

Status: **to generate**
---

# Chapter 4 — From Images to Video and Temporal JEPA

Chapter 4 extends the JEPA idea from static images to temporal prediction.

The key transition is:

\[
x_{\mathrm{ctx}} \rightarrow z_{\mathrm{tgt}}
\]

to:

\[
x_{\leq t} \rightarrow z_{t+k}
\]

## Sections

### 4.0 Temporal JEPA Overview

Topics:

- why time changes the problem,
- temporal context,
- future targets,
- spatiotemporal prediction,
- relation to V-JEPA.

Status: **to generate**

---

### 4.1 Video Tokenization and Tubelets

Topics:

- video tensors,
- frame patches,
- tubelet embeddings,
- temporal positional embeddings,
- memory costs.

Status: **to generate**

---

### 4.2 Spatiotemporal Masking

Topics:

- tube masks,
- future masks,
- block masks in space-time,
- context-target separation,
- mask visualization.

Status: **to generate**

---

### 4.3 V-JEPA-Lite Architecture

Topics:

- video encoder,
- target encoder,
- temporal predictor,
- target region conditioning,
- computational simplifications.

Status: **to generate**

---

### 4.4 Temporal Losses and Diagnostics

Topics:

- latent prediction loss across time,
- temporal smoothness,
- horizon-dependent error,
- representation drift,
- collapse over time.

Status: **to generate**

---

### 4.5 Evaluation on Video or Sequential Data

Topics:

- action recognition probes,
- future-state prediction,
- retrieval of similar clips,
- temporal nearest neighbors.

Status: **to generate**

---

# Chapter 5 — Action-Conditioned World Models

Chapter 5 turns passive temporal prediction into controllable latent dynamics.

The core objective becomes:

\[
\hat{z}_{t+k}
=
F_\phi(z_{\leq t}, a_{t:t+k})
\]

## Sections

### 5.0 From Prediction to Control

Topics:

- passive prediction vs action-conditioned prediction,
- state,
- action,
- transition models,
- planning in latent space.

Status: **to generate**

---

### 5.1 Latent Dynamics Models

Topics:

- deterministic dynamics,
- stochastic dynamics,
- transformer dynamics,
- recurrent dynamics,
- RSSM-style models.

Status: **to generate**

---

### 5.2 Action Encoders

Topics:

- continuous actions,
- discrete actions,
- action embeddings,
- action histories,
- planned action sequences.

Status: **to generate**

---

### 5.3 Multi-Step Rollouts

Topics:

- recursive prediction,
- open-loop rollout,
- closed-loop rollout,
- compounding error,
- rollout diagnostics.

Status: **to generate**

---

### 5.4 Uncertainty

Topics:

- aleatoric uncertainty,
- epistemic uncertainty,
- ensembles,
- stochastic latent variables,
- uncertainty-aware planning.

Status: **to generate**

---

### 5.5 Planning in Latent Space

Topics:

- random shooting,
- cross-entropy method,
- model predictive control,
- trajectory objectives,
- constraints.

Status: **to generate**

---

# Chapter 6 — Real-World Application: Cycling Intelligence

Chapter 6 applies JEPA-style world modeling to cycling telemetry.

The goal is to learn latent physiological and performance dynamics from real ride data.

## Sections

### 6.0 Application Overview

Topics:

- why cycling is a useful world-modeling domain,
- observations,
- hidden state,
- actions,
- prediction targets,
- planning tasks.

Status: **to generate**

---

### 6.1 Data Sources and Schema

Topics:

- FIT files,
- TCX/GPX files,
- Strava-like activity exports,
- telemetry channels,
- route data,
- weather/context features.

Status: **to generate**

---

### 6.2 Preprocessing Ride Data

Topics:

- resampling,
- missing values,
- smoothing,
- normalization,
- segment extraction,
- train/validation splits.

Status: **to generate**

---

### 6.3 State, Action, and Target Design

Topics:

- observation windows,
- future windows,
- pacing actions,
- effort zones,
- fatigue proxies,
- target representations.

Status: **to generate**

---

### 6.4 Temporal JEPA for Telemetry

Topics:

- context encoder,
- target encoder,
- dynamics predictor,
- action conditioning,
- latent loss,
- diagnostics.

Status: **to generate**

---

### 6.5 Probing Physiological State

Topics:

- heart-rate drift probe,
- effort sustainability probe,
- interval completion probe,
- recovery-cost proxy,
- fatigue classification.

Status: **to generate**

---

### 6.6 Pacing Rollouts

Topics:

- candidate pacing plans,
- latent rollout,
- fatigue cost,
- performance objective,
- constraints.

Status: **to generate**

---

### 6.7 Planning Demo

Topics:

- random-shooting planner,
- pacing recommendation,
- counterfactual comparisons,
- visualization,
- limitations.

Status: **to generate**

---

# Chapter 7 — Advanced Extensions and Open Problems

Chapter 7 explores research directions beyond the core tutorial.

## Sections

### 7.0 Research Directions

Topics:

- what JEPA-style world models still struggle with,
- open problems,
- evaluation challenges.

Status: **to generate**

---

### 7.1 Hierarchical World Models

Topics:

- multi-timescale representations,
- slow and fast latents,
- hierarchical prediction,
- planning at different horizons.

Status: **to generate**

---

### 7.2 Multimodal JEPA

Topics:

- video,
- text,
- audio,
- telemetry,
- actions,
- multimodal target prediction.

Status: **to generate**

---

### 7.3 JEPA with Generative Decoders

Topics:

- latent prediction plus reconstruction,
- optional decoders,
- hybrid objectives,
- visualization of latent predictions.

Status: **to generate**

---

### 7.4 Energy-Based JEPA

Topics:

- energy functions,
- compatible context-target pairs,
- negative sampling alternatives,
- uncertainty and multimodality.

Status: **to generate**

---

### 7.5 Active Learning and Curiosity

Topics:

- uncertainty-driven exploration,
- prediction error,
- information gain,
- active data collection.

Status: **to generate**

---

### 7.6 Limitations and Open Questions

Topics:

- representation collapse,
- semantic grounding,
- planning reliability,
- long-horizon uncertainty,
- evaluation benchmarks,
- causal confusion.

Status: **to generate**

---

# Implementation Track

The implementation will gradually evolve from a minimal file to a research codebase.

## Minimal Stage

```text
src/
└── jepa_world_model/
    ├── patchify.py
    ├── masks.py
    ├── vit.py
    ├── predictor.py
    ├── losses.py
    ├── diagnostics.py
    └── train_minimal.py
```

## Research Stage

```text
src/
└── jepa_world_model/
    ├── data/
    ├── models/
    ├── training/
    ├── evaluation/
    ├── diagnostics/
    ├── configs/
    └── utils/
```

## Application Stage

```text
src/
└── jepa_world_model/
    ├── cycling/
    │   ├── fit_parser.py
    │   ├── preprocessing.py
    │   ├── dataset.py
    │   ├── models.py
    │   ├── probes.py
    │   └── planning.py
    └── world_model/
        ├── dynamics.py
        ├── rollout.py
        └── planners.py
```

---

# Suggested Local Project Structure

```text
jepa-world-model-tutorial/
├── mkdocs.yml
├── docs/
│   ├── index.md
│   ├── chapter-01/
│   │   ├── introduction.md
│   │   ├── ssl-landscape.md
│   │   ├── why-not-pixels.md
│   │   ├── jepa-objective.md
│   │   └── representation-geometry.md
│   ├── chapter-02/
│   ├── chapter-03/
│   ├── chapter-04/
│   ├── chapter-05/
│   ├── chapter-06/
│   ├── chapter-07/
│   ├── references.md
│   └── assets/
│       ├── figures/
│       └── code/
├── src/
│   └── jepa_world_model/
├── notebooks/
├── configs/
├── experiments/
├── pyproject.toml
└── README.md
```

---

# Recommended Reading Path

Start with Chapter 1 if you want the conceptual foundation.

Start with Chapter 2 if you already understand self-supervised learning and want to build.

Start with Chapter 6 if your main interest is the cycling application, but return to Chapters 2 and 5 for implementation details.

---

# Core References

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.
  <https://openreview.net/forum?id=BZ5a1r-kVsf>

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**, 2020.
  <https://arxiv.org/abs/2006.07733>

- Adrien Bardes, Jean Ponce, Yann LeCun, **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**, 2021.
  <https://arxiv.org/abs/2105.04906>

- Jure Zbontar et al., **Barlow Twins: Self-Supervised Learning via Redundancy Reduction**, 2021.
  <https://arxiv.org/abs/2103.03230>

- Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, Nicolas Ballas, **Revisiting Feature Prediction for Learning Visual Representations from Video**, 2024.
  <https://arxiv.org/abs/2404.08471>

- Mahmoud Assran et al., **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**, 2025.
  <https://arxiv.org/abs/2506.09985>

- Danijar Hafner et al., **Learning Latent Dynamics for Planning from Pixels**, 2019.
  <https://arxiv.org/abs/1811.04551>

- Danijar Hafner et al., **Dream to Control: Learning Behaviors by Latent Imagination**, 2020.
  <https://arxiv.org/abs/1912.01603>
