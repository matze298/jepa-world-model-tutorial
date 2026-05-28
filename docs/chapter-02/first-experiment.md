# 2.11 Running the First Experiment

We now have the full minimal I-JEPA implementation path specified.

This section runs the first end-to-end experiment and defines what success should look like.

The purpose of the first experiment is not to achieve strong benchmark performance. The purpose is to verify that the implementation is correct enough to train:

- the forward pass works,
- the backward pass works,
- masks are valid,
- the target encoder is frozen,
- EMA updates happen,
- the loss decreases,
- representations do not collapse,
- evaluation runs end-to-end.

The first experiment is a debugging experiment.

---

## 2.11.1 Experiment Stages

We will run experiments in three stages:

```text
Stage 1:
    unit tests

Stage 2:
    tiny overfit run

Stage 3:
    short dataset run + evaluation
```

The stages answer different questions.

| Stage              | Question                                   |
| ------------------ | ------------------------------------------ |
| Unit tests         | Do individual modules behave correctly?    |
| Tiny overfit       | Can the model optimize at all?             |
| Short training run | Does the full system train on real images? |
| Evaluation         | Did the encoder learn anything useful?     |

Do not skip the tiny overfit run.

It is the fastest way to catch training-loop bugs.

---

## 2.11.2 Run the Test Suite

Start with:

```bash
pytest
```

At this point, the tests should cover:

```text
patchification
positional embeddings
mask sampling
ViT encoder
EMA target encoder
predictor
losses
diagnostics
model assembly
training step
evaluation utilities
```

A clean result should look like:

```text
passed
```

If tests fail, fix them before running training. Most training bugs are easier to diagnose in small unit tests than in a long experiment.

---

## 2.11.3 Run the Tiny Overfit Experiment

The tiny overfit run repeatedly trains on one batch.

Run:

```bash
python experiments/overfit_tiny_batch.py
```

Expected behavior:

```text
loss decreases
mask/overlap_fraction stays 0
target/std_mean stays nonzero
pred/std_mean stays nonzero
no NaNs or Infs appear
```

A typical log line might look like:

```text
step=0 | loss=0.2154 | pred_target/cosine=0.0312 | target/std_mean=0.6148 | pred/std_mean=0.2951 | target/effective_rank=31.7 | mask/overlap_fraction=0.0000 | ema_tau=0.9960
```

Later:

```text
step=100 | loss=0.1428 | pred_target/cosine=0.2685 | target/std_mean=0.6031 | pred/std_mean=0.3772 | target/effective_rank=30.9 | mask/overlap_fraction=0.0000 | ema_tau=0.9990
```

The exact values do not matter. The trend matters.

For a tiny overfit experiment, the model should usually be able to reduce the loss. If it cannot, there may be a bug in:

- mask indexing,
- encoder output shape,
- predictor target slicing,
- optimizer parameters,
- target detach behavior,
- EMA update timing.

---

## 2.11.4 What Tiny Overfit Does Not Prove

A successful tiny overfit run does not prove that the representation is good.

It only proves that the optimization path works.

A model can overfit one batch while still learning poor general representations.

The tiny overfit run answers:

```text
Can gradients flow through the online encoder and predictor?
Can the predictor match target representations for a fixed batch distribution?
Does EMA update without breaking?
Do diagnostics remain finite?
```

It does not answer:

```text
Does the encoder learn semantic features?
Does the model generalize?
Does k-NN retrieval improve?
Does linear probing improve?
```

Those require dataset-level evaluation.

---

## 2.11.5 Run a Local Debug Training Experiment

Next, run a short training experiment on CIFAR-10.

```bash
python experiments/train_minimal.py \
  --preset local_debug \
  --dataset cifar10 \
  --loss-type smooth_l1
```

The local debug preset is intentionally small:

```text
image_size = 32
patch_size = 4
encoder_dim = 64
encoder_depth = 2
predictor_dim = 64
predictor_depth = 1
batch_size = 16
epochs = 2
```

This run is not expected to learn a strong representation. It is a smoke test.

Expected behavior:

```text
training runs without crashing
loss decreases somewhat
mask overlap remains zero
representation statistics are finite
checkpoints are written
```

The checkpoint should appear under:

```text
checkpoints/
```

For example:

```text
checkpoints/minimal_ijepa_epoch_1.pt
checkpoints/minimal_ijepa_epoch_2.pt
```

---

## 2.11.6 Evaluate the Local Debug Checkpoint

After the local run, evaluate the checkpoint.

```bash
python experiments/evaluate_minimal.py \
  --checkpoint checkpoints/minimal_ijepa_epoch_2.pt \
  --dataset cifar10 \
  --max-batches 20
```

This extracts features from a subset of CIFAR-10 and runs:

- k-NN classification,
- linear probe.

For such a small run, results may be close to chance.

That is acceptable.

The purpose is to verify:

```text
checkpoint loads
encoder extracts full-image features
k-NN runs
linear probe runs
metrics are finite
```

A useful next comparison is a random encoder baseline. If the trained model performs no better than random after a very short run, that is not necessarily a problem. If it performs no better after a longer run, that is a signal to investigate.

---

## 2.11.7 Run a More Meaningful STL-10 Experiment

Once the CIFAR-10 debug run works, move to STL-10.

```bash
python experiments/train_minimal.py \
  --preset default \
  --dataset stl10 \
  --loss-type smooth_l1
```

STL-10 is more useful for this tutorial because:

- images are larger,
- there is an unlabeled split,
- patch-based masking is more meaningful,
- evaluation still has labeled train/test splits.

For the default config:

```text
image_size = 96
patch_size = 8
grid_size = 12
num_patches = 144
```

This gives a more realistic JEPA setting than \(32 \times 32\) CIFAR images.

---

## 2.11.8 Suggested First Cloud Run

For a larger experiment, use the cloud preset.

On a cloud GPU machine:

```bash
cd /workspace/jepa-world-model-tutorial

uv sync
source .venv/bin/activate

python experiments/train_cloud.py \
  --dataset stl10 \
  --loss-type smooth_l1
```

The cloud config might use:

```text
image_size = 96
patch_size = 8
encoder_dim = 384
encoder_depth = 8
predictor_dim = 256
predictor_depth = 4
batch_size = 256
epochs = 300
```

This is still not a full I-JEPA reproduction. It is a scaled version of the minimal implementation.

For cloud runs, checkpoints and logs should go to persistent paths such as:

```text
/workspace/checkpoints
/workspace/runs
```

Do not rely on ephemeral container storage for long experiments.

---

## 2.11.9 Minimal Metrics to Track

Every run should track at least:

```text
loss
pred_target/cosine
target/std_mean
pred/std_mean
target/dead_dim_fraction
pred/dead_dim_fraction
target/effective_rank
pred/effective_rank
mask/overlap_fraction
ema_tau
```

A minimal healthy run should show:

```text
loss:
    generally decreasing

pred_target/cosine:
    generally increasing

target/std_mean:
    nonzero and stable

pred/std_mean:
    nonzero and stable

target/dead_dim_fraction:
    near zero

mask/overlap_fraction:
    exactly zero

ema_tau:
    increasing toward 1.0
```

A suspicious run might show:

```text
loss decreases rapidly to almost zero
target/std_mean collapses to zero
effective_rank collapses
pred_target/cosine becomes high too quickly
mask/overlap_fraction is nonzero
```

The diagnostics are there to catch these failures early.

---

## 2.11.10 Expected Failure Modes

### Failure Mode 1: Loss does not decrease

Possible causes:

- learning rate too low,
- optimizer does not include online encoder or predictor,
- predictor output shape is wrong,
- loss is comparing wrong tensors,
- target representations are unstable,
- masks make the task too hard.

Checks:

```python
has_grad = any(
    p.grad is not None
    for p in model.online_encoder.parameters()
)
```

and:

```python
has_grad = any(
    p.grad is not None
    for p in model.predictor.parameters()
)
```

Both should be true after `loss.backward()`.

---

### Failure Mode 2: Target encoder receives gradients

This should not happen.

Check:

```python
target_has_grad = any(
    p.grad is not None
    for p in model.target_encoder.parameters()
)
```

This should be:

```python
False
```

If it is true, check:

- target encoder parameters are frozen,
- target forward happens under `torch.no_grad()`,
- target tensors are detached in the loss,
- optimizer excludes target parameters.

---

### Failure Mode 3: Mask overlap is nonzero

This is mask leakage.

If:

```text
mask/overlap_fraction > 0
```

then the model may see target patches in the context.

Stop the run and fix the mask sampler.

The target and context sets must satisfy:

\[
\mathcal{C} \cap \mathcal{T} = \emptyset
\]

---

### Failure Mode 4: Representation collapse

Symptoms:

```text
target/std_mean → 0
target/dead_dim_fraction → 1
target/effective_rank → 0 or very low
```

Possible causes:

- loss too easy,
- predictor too strong,
- target encoder update too fast,
- bad EMA schedule,
- insufficient masking difficulty,
- optimization instability,
- implementation bug.

First checks:

```text
Is the target encoder initialized from online encoder?
Is the target encoder updated by EMA?
Is target detached?
Are masks non-overlapping?
Are representations finite?
```

---

### Failure Mode 5: Norm explosion

Symptoms:

```text
pred/norm_mean grows rapidly
target/norm_mean grows rapidly
loss becomes NaN
```

Possible fixes:

- lower learning rate,
- add gradient clipping,
- check loss scale,
- use Smooth L1 instead of MSE,
- use AdamW with reasonable weight decay,
- inspect input normalization.

A simple gradient clip can be added:

```python
torch.nn.utils.clip_grad_norm_(
    trainable_jepa_parameters(model),
    max_norm=1.0,
)
```

Place it after:

```python
loss.backward()
```

and before:

```python
optimizer.step()
```

---

### Failure Mode 6: Evaluation is near random

For very short runs, this is expected.

If evaluation remains near random after longer runs, investigate:

- mask design,
- dataset size,
- model capacity,
- augmentation,
- training duration,
- representation collapse,
- linear probe settings,
- feature pooling strategy.

Also compare to a random encoder baseline.

---

## 2.11.11 Recommended First Ablations

After the first successful run, try small ablations.

### Loss type

Compare:

```bash
--loss-type mse
--loss-type smooth_l1
--loss-type cosine
```

Track:

```text
loss
pred_target/cosine
target/std_mean
linear_probe/val_acc
```

---

### Context ratio

Try:

```text
context_ratio = 0.4
context_ratio = 0.6
context_ratio = 0.8
```

Lower context ratio makes prediction harder.

Higher context ratio makes prediction easier.

---

### Target block size

Try:

```text
2x2 target blocks
3x3 target blocks
4x4 target blocks
```

Larger target blocks may encourage more semantic prediction but can become too difficult.

---

### Predictor depth

Try:

```text
predictor_depth = 1
predictor_depth = 2
predictor_depth = 4
```

A predictor that is too weak may fail to solve the task.

A predictor that is too strong may absorb too much of the learning pressure.

---

### EMA tau

Try:

```text
ema_tau_base = 0.99
ema_tau_base = 0.996
ema_tau_base = 0.999
```

A lower tau updates the target faster.

A higher tau makes the target more stable but slower to adapt.

---

## 2.11.12 What Counts as Success for Chapter 2?

Chapter 2 is successful if:

```text
the implementation trains end-to-end
the loss decreases
mask leakage is zero
target encoder receives no gradients
EMA updates target parameters
representations do not collapse
evaluation scripts run
trained encoder beats random baseline after sufficient training
```

It does not need to match official I-JEPA performance.

It does not need to scale perfectly.

It does not need distributed training.

It does not need an advanced config system.

Those belong in Chapter 3.

The Chapter 2 goal is:

> Build a minimal, inspectable, correct JEPA implementation.

---

## 2.11.13 Suggested Experiment Log

For every run, record:

```text
date
git commit
dataset
image size
patch size
model config
mask config
loss type
optimizer
learning rate
weight decay
EMA schedule
batch size
epochs
device
final loss
final diagnostics
k-NN accuracy
linear probe accuracy
notes
```

A simple Markdown log entry:

```markdown
## Run: minimal-stl10-smoothl1-001

- Dataset: STL-10 unlabeled
- Image size: 96
- Patch size: 8
- Encoder: dim 192, depth 6, heads 3
- Predictor: dim 128, depth 3, heads 4
- Mask: 4 target blocks, 3x3, context ratio 0.6
- Loss: Smooth L1
- Optimizer: AdamW
- LR: 5e-4
- Weight decay: 0.05
- EMA: cosine 0.996 → 1.0
- Batch size: 128
- Epochs: 100
- Device: local CUDA

### Results

- Final loss:
- Final pred-target cosine:
- Final target std:
- Final effective rank:
- k-NN:
- Linear probe:

### Notes

-
```

This is enough for early research hygiene.

Chapter 3 will introduce structured experiment tracking.

---

## 2.11.14 End of Chapter 2 Checklist

Before moving on, verify:

```text
[ ] pytest passes
[ ] marimo patch/mask visualization works
[ ] tiny overfit run reduces loss
[ ] local_debug training run completes
[ ] checkpoint is saved
[ ] evaluation script loads checkpoint
[ ] k-NN evaluation runs
[ ] linear probe runs
[ ] target encoder has no gradients
[ ] mask overlap is always zero
[ ] representation stats are non-collapsed
```

If all boxes are checked in the accompanying implementation, the minimal implementation is complete.

---

## 2.11.15 Summary

This section ran the first end-to-end JEPA experiment.

We covered:

- unit test execution,
- tiny overfit run,
- local debug training,
- STL-10 training,
- cloud run structure,
- evaluation,
- expected logs,
- failure modes,
- first ablations,
- experiment notes,
- Chapter 2 completion checklist.

At this point, the tutorial has specified a working minimal I-JEPA system.

The next chapter will turn this minimal implementation into a more research-grade training framework, adding stronger configuration, logging, checkpointing, mixed precision, and scalable cloud execution.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- PyTorch, **Saving and Loading Models**.
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>

- TorchVision, **CIFAR-10 Dataset**.
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html>

- TorchVision, **STL10 Dataset**.
  <https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html>

- marimo, **Documentation**.
  <https://docs.marimo.io/>
