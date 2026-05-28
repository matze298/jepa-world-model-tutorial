# 3.0 Research-Grade JEPA Training: Design Goals

Chapter 2 built a minimal I-JEPA implementation.

It was intentionally explicit: plain PyTorch, simple dataclass configs, direct training scripts, manual EMA updates, and local debugging with marimo. That was the right abstraction level for understanding the algorithm.

Chapter 3 has a different goal.

We now assume the reader already knows how to build production-grade deep-learning systems: structured configs, checkpointing, logging, mixed precision, distributed training, cloud execution, and experiment tracking.

So this chapter will not explain those concepts from scratch.

Instead, it focuses on what is **JEPA-specific**:

- preserving the online encoder, target encoder, predictor, and EMA state correctly,
- keeping mask sampling and target prediction explicit,
- ensuring the target branch never receives gradients,
- logging collapse diagnostics as first-class metrics,
- making checkpoint/resume semantics correct for EMA-based training,
- scaling the loop without hiding the algorithm,
- supporting controlled ablations over masking, predictor capacity, EMA, and losses.

The aim is to turn the Chapter 2 implementation into a serious research harness while keeping the JEPA mechanics visible.

---

## 3.0.1 What Stays the Same

The core algorithm does not change.

The training step is still:

```python
context_indices, target_indices = sample_mask_batch(...)

pred_repr, target_repr = model(
    images=images,
    context_indices=context_indices,
    target_indices=target_indices,
)

loss = latent_loss(
    pred=pred_repr,
    target=target_repr,
)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

update_ema(
    online=model.online_encoder,
    target=model.target_encoder,
    tau=ema_tau,
)
```

This is the center of the implementation.

Chapter 3 does not replace this with an opaque trainer abstraction. The JEPA step should remain inspectable because most serious bugs occur exactly here:

- target encoder accidentally receives gradients,
- target encoder is accidentally included in the optimizer,
- EMA update happens at the wrong time,
- checkpoint resume restores online encoder but not target encoder,
- context and target masks overlap,
- predictor returns the wrong token slice,
- loss decreases while representation variance collapses.

The infrastructure around the loop improves. The loop itself remains explicit.

---

## 3.0.2 What Changes

Chapter 2 optimized for clarity.

Chapter 3 optimizes for experimental reliability.

The shift is:

```text
Chapter 2:
    minimal implementation
    local scripts
    dataclass config
    console logs
    simple checkpoints
    manual local/cloud split

Chapter 3:
    structured experiment configs
    run manifests
    canonical metric streams
    resumable checkpoints
    optimizer/scheduler/EMA discipline
    precision control
    Lightning Fabric execution
    cloud-ready single entry point
    ablation support
    JEPA-specific failure-mode playbook
```

This is not a rewrite of the model.

It is a hardening of the training system.

The model components from Chapter 2 remain valid:

```text
patchify.py
position.py
masks.py
vit.py
ema.py
predictor.py
losses.py
diagnostics.py
model.py
```

Chapter 3 primarily adds or refactors:

```text
configs/
engine/
experiments/train.py
experiments/evaluate.py
experiments/ablate.py
```

The algorithmic code stays separate from the experiment engine.

---

## 3.0.3 The JEPA-Specific State Problem

A generic supervised checkpoint usually needs:

```text
model
optimizer
scheduler
epoch
global_step
config
```

A JEPA checkpoint needs more care.

The model contains at least three semantically distinct parts:

```text
online encoder
target encoder
predictor
```

The online encoder and predictor are optimized by gradients.

The target encoder is updated by EMA.

That means this is the training state:

```text
online encoder parameters
target encoder parameters
predictor parameters
optimizer state
scheduler state
global step
epoch
EMA schedule position
mask config
loss config
random state if exact resume is required
```

The critical point is:

> The target encoder is not a disposable copy of the online encoder during training.

It defines the target representation distribution.

If a run is resumed by restoring only the online encoder and reinitializing the target encoder from it, the training dynamics change. The model may still run, but it is not a faithful resume.

So Chapter 3 distinguishes between:

```text
training checkpoint:
    enough state to resume training faithfully

representation artifact:
    enough state to use the learned encoder downstream
```

A training checkpoint must include the target encoder.

An exported representation artifact usually does not need the predictor or optimizer.

---

## 3.0.4 The JEPA-Specific Logging Problem

For supervised learning, tracking loss and validation accuracy may be enough to catch many failures.

For JEPA, it is not.

A JEPA model can show a decreasing latent prediction loss while the representation becomes useless.

So diagnostics are not optional.

A serious JEPA run should log at least:

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

The most important early-warning metrics are:

```text
target/std_mean
target/dead_dim_fraction
target/effective_rank
mask/overlap_fraction
ema/relative_param_l2
```

A run with a nice loss curve but collapsing effective rank is not successful.

A run with nonzero mask overlap is invalid.

A run where EMA distance never changes probably has a broken update path.

A run where the target encoder receives gradients is not JEPA training as intended.

---

## 3.0.5 The JEPA-Specific Update Order

The update order should be explicit and tested.

The intended order is:

```text
sample masks
forward online branch
forward target branch without gradients
predict target representations
compute latent loss
backward
gradient clipping
optimizer step
scheduler step
EMA update
diagnostics/logging
```

In code, the structure should remain recognizable:

```python
pred_repr, target_repr = model(
    images=images,
    context_indices=context_indices,
    target_indices=target_indices,
)

loss = latent_loss(pred_repr, target_repr)

fabric.backward(loss)

if cfg.training.gradient_clip_norm is not None:
    fabric.clip_gradients(
        model,
        optimizer,
        max_norm=cfg.training.gradient_clip_norm,
    )

optimizer.step()
scheduler.step()
optimizer.zero_grad(set_to_none=True)

ema_tau = ema_schedule(global_step)
update_ema(model.online_encoder, model.target_encoder, ema_tau)
```

The exact Fabric API can vary depending on implementation, but the semantic order should not.

In particular:

```text
EMA update happens after optimizer step.
```

The target should move toward the newly updated online encoder.

---

## 3.0.6 Why Keep the Loop Explicit?

A full training framework can hide the details that matter most for JEPA.

For standard supervised training, that may be acceptable.

For JEPA, the following operations should remain visible:

```text
mask sampling
target no-grad forward
loss target detach
optimizer parameter selection
EMA update
collapse diagnostics
mask-overlap assertion
```

This is why Chapter 3 uses **Lightning Fabric** rather than immediately moving to a full `LightningModule`.

Fabric gives us:

```text
device management
mixed precision
distributed setup
Fabric-aware backward
Fabric-aware checkpoint utilities
```

without forcing the JEPA step into a high-level trainer abstraction.

The desired loop still looks like PyTorch:

```python
with fabric.autocast():
    pred_repr, target_repr = model(...)
    loss = latent_loss(pred_repr, target_repr)

fabric.backward(loss)
optimizer.step()
update_ema(...)
```

This is the right compromise for research code:

```text
less boilerplate
same algorithmic visibility
```

A full PyTorch Lightning version can be added later if the method stabilizes and the training flow becomes conventional enough.

---

## 3.0.7 Target Architecture for Chapter 3

Chapter 2 had a flat implementation layout.

Chapter 3 adds a research engine around it.

A plausible final source layout is:

```text
src/
└── jepa_world_model/
    ├── checkpointing.py
    ├── config.py
    ├── data.py
    ├── diagnostics.py
    ├── ema.py
    ├── evaluation.py
    ├── losses.py
    ├── masks.py
    ├── model.py
    ├── patchify.py
    ├── position.py
    ├── predictor.py
    ├── training.py
    ├── utils.py
    ├── vit.py
    ├── configs/
    │   ├── __init__.py
    │   ├── schema.py
    │   └── loader.py
    └── engine/
        ├── __init__.py
        ├── checkpointing.py
        ├── fabric_trainer.py
        ├── logging.py
        ├── optim.py
        ├── run_context.py
        └── schedules.py
```

The split is intentional.

Algorithmic components remain top-level:

```text
masks.py
vit.py
predictor.py
ema.py
losses.py
diagnostics.py
model.py
```

Experiment infrastructure lives under:

```text
engine/
configs/
```

This prevents training-system concerns from leaking into model definitions.

---

## 3.0.8 Experiment Entry Points

Chapter 2 used multiple scripts:

```text
train_minimal.py
train_cloud.py
overfit_tiny_batch.py
evaluate_minimal.py
```

Chapter 3 moves toward fewer, config-driven entry points:

```text
experiments/
├── train.py
├── evaluate.py
├── ablate.py
└── overfit_tiny_batch.py
```

The main local and cloud command should be the same shape:

```bash
uv run python experiments/train.py \
  --config configs/ijepa_stl10_base.yaml
```

For cloud:

```bash
uv run python experiments/train.py \
  --config configs/ijepa_stl10_cloud.yaml
```

The code path should be the same.

The config should change:

```text
data paths
run directory
batch size
precision
device
number of workers
training length
```

not the training algorithm.

---

## 3.0.9 Configuration Scope

Chapter 3 introduces structured configuration, but not for its own sake.

The config should capture experimental variables, not every internal implementation detail.

The first useful config groups are:

```text
experiment
model
mask
data
optimizer
training
ema
logging
checkpointing
runtime
evaluation
```

The config should validate JEPA-specific invariants:

```text
image_size % patch_size == 0
encoder_dim % encoder_heads == 0
predictor_dim % predictor_heads == 0
context_patches + target_patches <= total_patches
tau_base <= tau_final
gradient_accumulation_steps >= 1
target encoder excluded from optimizer
```

The last invariant is partly structural rather than purely config-based, but the training harness should test it.

The config should not silently allow impossible mask setups or mismatched model dimensions.

---

## 3.0.10 Run State and Manifest Scope

Every run should create a run directory:

```text
runs/
└── ijepa_stl10_base_2026-05-25_21-30-10/
    ├── config.yaml
    ├── manifest.yaml
    ├── metrics.jsonl
    ├── notes.md
    ├── checkpoints/
    ├── artifacts/
    └── logs/
```

The resolved config records intended settings.

The manifest records actual execution metadata:

```text
command
Git commit
Git dirty state
host
device
Python version
PyTorch version
CUDA availability
start time
```

The metrics file records the training trajectory.

The checkpoints preserve resumable state.

Artifacts contain plots and qualitative outputs such as retrieval examples.

This is standard research hygiene, but the JEPA-specific requirement is that the run state must let us verify:

```text
Was the target encoder restored?
Was EMA resumed at the correct global step?
Were masks valid?
Did representation geometry collapse?
Did evaluation use the online or target encoder?
```

---

## 3.0.11 Metrics and Diagnostic Cadence

Not every metric should be computed every step.

Some metrics are cheap:

```text
loss
pred_target/cosine
mask/overlap_fraction
ema_tau
lr
```

Some are moderately expensive:

```text
target/std_mean
pred/std_mean
norms
dead_dim_fraction
```

Some are expensive:

```text
covariance metrics
effective rank
linear probe
k-NN evaluation
retrieval artifacts
```

Chapter 3 should make diagnostic cadence configurable.

For example:

```yaml
logging:
  log_every_steps: 50
  representation_metrics_every_steps: 50
  covariance_metrics_every_steps: 500
  eval_every_epochs: 10
```

The key point is that expensive diagnostics should be throttled, not removed.

For JEPA, representation-health metrics are part of the training signal we use as researchers.

---

## 3.0.12 Checkpointing Scope

Chapter 3 checkpointing should support three operations:

```text
save training checkpoint
resume training checkpoint
export encoder artifact
```

These are different.

### Training checkpoint

Used to continue pretraining.

Must include:

```text
model.state_dict()
optimizer.state_dict()
scheduler.state_dict()
epoch
global_step
config
rng state if exact resume is required
```

Because `model.state_dict()` includes:

```text
online encoder
target encoder
predictor
```

this preserves the JEPA state.

### Resume

Should restore:

```text
model
optimizer
scheduler
global_step
epoch
EMA schedule position
```

Then training continues without silently resetting the target distribution.

### Encoder export

Used for downstream tasks.

Usually includes:

```text
online_encoder.state_dict()
model config
preprocessing metadata
patch size
image size
feature dimension
```

It does not need the predictor or optimizer.

This distinction avoids mixing training artifacts with representation artifacts.

---

## 3.0.13 Scaling Scope

Chapter 3 should make the implementation scalable without changing its semantics.

The scaling stack is:

```text
single GPU fp32
single GPU bf16
multi-GPU DDP through Fabric
cloud execution on RunPod
```

The same invariants must hold in every setting:

```text
target encoder excluded from optimizer
target encoder no gradients
EMA update applied consistently
metrics logged only from rank zero unless aggregated
mask sampling device-correct
checkpoints contain full training state
```

Distributed training introduces one additional concern:

```text
global batch size = per_device_batch_size × devices × accumulation_steps
```

This affects optimization and should be visible in logs or manifests.

---

## 3.0.14 First Ablations

The first serious ablations should target JEPA-specific design choices.

Good first candidates:

```text
mask.context_ratio
mask.target_block_height
mask.target_block_width
mask.num_target_blocks
training.loss_type
ema.tau_base
model.predictor_depth
model.predictor_dim
model.encoder_depth
```

The analysis should not only compare loss.

For each ablation, compare:

```text
final loss
pred_target/cosine
target/effective_rank
target/dead_dim_fraction
k-NN accuracy
linear probe validation accuracy
retrieval quality
```

The important research question is often:

> Which settings improve representation quality, not merely prediction loss?

A stronger predictor may reduce loss but weaken encoder pressure.

An easier mask may reduce loss but produce less semantic abstraction.

A slower EMA may stabilize training but slow representation adaptation.

These are JEPA-specific tradeoffs.

---

## 3.0.15 Failure-Mode Orientation

Chapter 3 should make failures easy to diagnose.

The failure-mode playbook should be built around symptoms:

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

Each failure should map to:

```text
likely causes
metrics to inspect
minimal reproduction
first fixes
confirmation test
```

Example:

```text
Symptom:
    loss decreases but linear probe remains random

Inspect:
    target/std_mean
    target/effective_rank
    mask/overlap_fraction
    pred_target/cosine
    retrieval examples

Likely causes:
    representation collapse
    task too easy
    predictor too strong
    masks not semantic enough
    training too short
    evaluation feature extraction bug
```

This is more useful than generic debugging advice.

---

## 3.0.16 Chapter 3 Deliverables

By the end of Chapter 3, we want:

```text
config-driven training
standardized run directories
manifest capture
JSONL metric stream
optional TensorBoard/W&B sinks
resumable training checkpoints
encoder export
optimizer/scheduler/EMA utilities
gradient clipping and accumulation
mixed precision support
Lightning Fabric harness
cloud-ready single entry point
evaluation integration
ablation runner
failure-mode playbook
```

The final lifecycle should be:

```text
choose config
initialize run directory
train with Fabric
log JSONL + optional TensorBoard/W&B
checkpoint
resume if needed
evaluate
export encoder
aggregate ablations
inspect failure diagnostics
```

---

## 3.0.17 What Chapter 3 Does Not Do

Chapter 3 does not change the modeling problem.

It does not yet implement:

```text
video JEPA
temporal JEPA
action-conditioned dynamics
cycling telemetry
planning
uncertainty-aware rollouts
multimodal JEPA
hierarchical world models
```

Those come later.

Chapter 3 keeps the image JEPA model and makes the training harness reliable enough to support later extensions.

---

## 3.0.18 Summary

Chapter 3 is about turning a working implementation into a research instrument.

The key design choice is to improve infrastructure without hiding the JEPA-specific loop.

We keep explicit:

```text
mask sampling
online context encoding
target no-grad encoding
latent prediction
loss computation
optimizer update
EMA update
diagnostics
```

We harden:

```text
configs
run state
logging
checkpoint/resume
precision
distributed execution
evaluation
ablations
failure diagnosis
```

The next section implements the configuration system for JEPA experiments.

---

## References and Further Reading

- Lightning Fabric documentation:
  <https://lightning.ai/docs/fabric/stable/>

- PyTorch Automatic Mixed Precision:
  <https://pytorch.org/docs/stable/amp.html>

- PyTorch Saving and Loading Models:
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>

- Hydra documentation:
  <https://hydra.cc/docs/intro/>

- OmegaConf documentation:
  <https://omegaconf.readthedocs.io/>

- TensorBoard with PyTorch:
  <https://pytorch.org/docs/stable/tensorboard.html>

- Weights & Biases documentation:
  <https://docs.wandb.ai/>
