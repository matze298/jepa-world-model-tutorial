# 2.12 Chapter 2 Consolidation

Chapter 2 specifies a complete minimal I-JEPA implementation.

The purpose of this consolidation section is to summarize the implementation target, how the pieces fit together, and what should be true before moving on to Chapter 3.

At this point, the accompanying implementation should contain:

- reusable model code,
- mask and patch utilities,
- EMA target encoder logic,
- losses and diagnostics,
- training scripts,
- evaluation scripts,
- tests,
- marimo debugging notebooks.

The implementation target is intentionally minimal, but complete enough to train, inspect, checkpoint, and evaluate.

---

## 2.12.1 Final Source Layout

The implementation source tree should look roughly like this:

```text
src/
└── jepa_world_model/
    ├── __init__.py
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
    ├── presets.py
    ├── training.py
    ├── utils.py
    └── vit.py
```

Each file has a focused role.

| File               | Purpose                                                            |
| ------------------ | ------------------------------------------------------------------ |
| `config.py`        | Minimal dataclass configuration                                    |
| `presets.py`       | Local-debug and cloud-run config presets                           |
| `utils.py`         | Device selection, seeding, parameter counting                      |
| `patchify.py`      | Patchification, unpatchification, patch gathering, patch embedding |
| `position.py`      | Learned and sinusoidal 2D positional embeddings                    |
| `masks.py`         | Context and target mask sampling                                   |
| `vit.py`           | Minimal ViT encoder                                                |
| `ema.py`           | Target encoder initialization and EMA updates                      |
| `predictor.py`     | JEPA predictor                                                     |
| `losses.py`        | Latent-space losses                                                |
| `diagnostics.py`   | Collapse, norm, covariance, rank, and mask diagnostics             |
| `model.py`         | Top-level `MinimalIJEPA` assembly                                  |
| `training.py`      | One-step training logic and log formatting                         |
| `data.py`          | Dataset and dataloader builders                                    |
| `checkpointing.py` | Save/load checkpoints                                              |
| `evaluation.py`    | k-NN and linear-probe evaluation utilities                         |

---

## 2.12.2 Final Experiment Scripts

The experiment directory should contain:

```text
experiments/
├── evaluate_minimal.py
├── overfit_tiny_batch.py
├── train_cloud.py
└── train_minimal.py
```

Their roles are:

| Script                  | Purpose                                              |
| ----------------------- | ---------------------------------------------------- |
| `overfit_tiny_batch.py` | Repeatedly train on one batch to verify optimization |
| `train_minimal.py`      | Local debug or small dataset training                |
| `train_cloud.py`        | Larger cloud-oriented training run                   |
| `evaluate_minimal.py`   | Load checkpoint and run k-NN / linear probe          |

The expected execution sequence is:

```bash
pytest
python experiments/overfit_tiny_batch.py
python experiments/train_minimal.py --preset local_debug --dataset cifar10
python experiments/evaluate_minimal.py \
  --checkpoint checkpoints/minimal_ijepa_epoch_2.pt \
  --dataset cifar10 \
  --max-batches 20
```

For a larger run:

```bash
python experiments/train_minimal.py \
  --preset default \
  --dataset stl10 \
  --loss-type smooth_l1
```

For cloud:

```bash
uv sync
source .venv/bin/activate

python experiments/train_cloud.py \
  --dataset stl10 \
  --loss-type smooth_l1
```

---

## 2.12.3 Final Test Layout

The test directory should contain:

```text
tests/
├── __init__.py
├── test_diagnostics.py
├── test_ema.py
├── test_evaluation.py
├── test_losses.py
├── test_masks.py
├── test_model.py
├── test_patchify.py
├── test_position.py
├── test_predictor.py
├── test_setup.py
├── test_training.py
└── test_vit.py
```

The tests cover:

| Test file             | Checks                                              |
| --------------------- | --------------------------------------------------- |
| `test_setup.py`       | Config, seeding, device selection                   |
| `test_patchify.py`    | Patchify/unpatchify, patch indices, patch gathering |
| `test_position.py`    | Learned and fixed positional embeddings             |
| `test_masks.py`       | Target blocks, context masks, overlap checks        |
| `test_vit.py`         | Transformer blocks and encoder outputs              |
| `test_ema.py`         | Target initialization, freezing, EMA updates        |
| `test_predictor.py`   | Predictor shapes and gradients                      |
| `test_losses.py`      | Latent loss functions                               |
| `test_diagnostics.py` | Representation and mask diagnostics                 |
| `test_model.py`       | Top-level model assembly                            |
| `test_training.py`    | One complete training step                          |
| `test_evaluation.py`  | k-NN and linear-probe utilities                     |

Run all tests with:

```bash
pytest
```

This should pass before starting serious experiments.

---

## 2.12.4 Final marimo Notebooks

The marimo notebooks are for debugging and visualization.

They should be stored as Python files:

```text
notebooks/
├── 00_debug_environment.py
├── 01_visualize_patches_and_masks.py
├── 02_debug_encoder.py
├── 03_debug_ema.py
├── 04_debug_predictor.py
├── 05_debug_losses_and_diagnostics.py
└── 06_retrieval_and_probe.py
```

Their roles are:

| Notebook                             | Purpose                                          |
| ------------------------------------ | ------------------------------------------------ |
| `00_debug_environment.py`            | Check device, imports, config, basic environment |
| `01_visualize_patches_and_masks.py`  | Inspect patchification and context/target masks  |
| `02_debug_encoder.py`                | Run encoder on sampled masks                     |
| `03_debug_ema.py`                    | Inspect EMA behavior and schedule                |
| `04_debug_predictor.py`              | Run predictor on encoder outputs                 |
| `05_debug_losses_and_diagnostics.py` | Inspect losses and representation metrics        |
| `06_retrieval_and_probe.py`          | Explore nearest neighbors and probe outputs      |

Open a notebook with:

```bash
marimo edit notebooks/01_visualize_patches_and_masks.py
```

Run one as a script with:

```bash
marimo run notebooks/02_debug_encoder.py
```

The notebooks should import from `src/jepa_world_model/`. They should not contain reusable model code.

---

## 2.12.5 Full Data Flow

The full minimal I-JEPA data flow is:

```text
images
  │
  ├── sample context_indices, target_indices
  │
  ▼
online encoder:
  patchify images
  embed patches
  gather context patches
  add context positional embeddings
  transformer encoder
  produce context_repr
  │
  ▼
predictor:
  project context_repr
  add context positional embeddings
  create target query tokens
  add target positional embeddings
  transformer predictor
  output pred_repr
  │
  ▼
loss:
  compare pred_repr to target_repr
```

Target branch:

```text
images
  │
  ▼
target encoder:
  patchify images
  embed patches
  gather target patches
  add target positional embeddings
  transformer encoder
  produce target_repr
  no gradients
  EMA-updated weights
```

Training update:

```text
loss.backward()
optimizer.step()
EMA target update
```

---

## 2.12.6 Core Shape Contract

The whole implementation depends on the following shape contract.

```python
images.shape
# [B, C, H, W]

context_indices.shape
# [B, N_ctx]

target_indices.shape
# [B, N_tgt]

context_repr.shape
# [B, N_ctx, D]

target_repr.shape
# [B, N_tgt, D]

pred_repr.shape
# [B, N_tgt, D]
```

The most important equality is:

```python
pred_repr.shape == target_repr.shape
```

If that is not true, the JEPA loss is invalid.

---

## 2.12.7 Core Training Step

The minimal training step is:

```python
context_indices, target_indices = sample_mask_batch(
    config=mask_config,
    batch_size=images.size(0),
    device=images.device,
)

pred_repr, target_repr = model(
    images=images,
    context_indices=context_indices,
    target_indices=target_indices,
)

loss = latent_loss(
    pred=pred_repr,
    target=target_repr,
    loss_type="smooth_l1",
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

This is the algorithmic heart of Chapter 2.

Everything else exists to make this step correct, inspectable, and evaluable.

---

## 2.12.8 Required Invariants

Before moving to Chapter 3, the implementation should satisfy these invariants.

### Mask invariant

```python
context ∩ target = ∅
```

In logs:

```text
mask/overlap_fraction = 0.0
```

### Target gradient invariant

The target encoder receives no gradients:

```python
target_has_grad = any(
    p.grad is not None
    for p in model.target_encoder.parameters()
)

assert not target_has_grad
```

### Optimizer invariant

The optimizer updates only:

```text
online encoder
predictor
```

not:

```text
target encoder
```

### EMA invariant

After optimizer steps, the target encoder moves toward the online encoder.

### Representation invariant

Target representations remain non-collapsed:

```text
target/std_mean > 0
target/dead_dim_fraction near 0
target/effective_rank nonzero
```

### Loss invariant

The loss is finite:

```python
torch.isfinite(loss)
```

### Evaluation invariant

The frozen online encoder can produce full-image features:

```python
features.shape
# [num_examples, D]
```

---

## 2.12.9 Minimal Command Checklist

Run this checklist before leaving Chapter 2.

```bash
pytest
```

```bash
python experiments/overfit_tiny_batch.py
```

```bash
python experiments/train_minimal.py \
  --preset local_debug \
  --dataset cifar10 \
  --loss-type smooth_l1
```

```bash
python experiments/evaluate_minimal.py \
  --checkpoint checkpoints/minimal_ijepa_epoch_2.pt \
  --dataset cifar10 \
  --max-batches 20
```

Optional STL-10 run:

```bash
python experiments/train_minimal.py \
  --preset default \
  --dataset stl10 \
  --loss-type smooth_l1
```

---

## 2.12.10 What Chapter 2 Does Not Yet Solve

Chapter 2 intentionally leaves several topics underdeveloped.

It does not yet provide:

- robust experiment configuration,
- distributed training,
- mixed precision,
- gradient accumulation,
- advanced checkpoint management,
- reproducible experiment manifests,
- TensorBoard or W&B integration as a default,
- Hydra/OmegaConf config hierarchy,
- Lightning Fabric,
- large-scale ablation infrastructure,
- optimized attention kernels,
- full official I-JEPA masking parity,
- serious benchmark results.

This is intentional.

Chapter 2 is about correctness and transparency.

Chapter 3 is about research-grade engineering.

---

## 2.12.11 Chapter 2 Completion Criteria

Chapter 2 is complete when:

```text
[ ] all unit tests pass
[ ] patch and mask visualization works in marimo
[ ] tiny overfit run reduces loss
[ ] local debug training completes
[ ] checkpoints are written
[ ] evaluation script loads a checkpoint
[ ] k-NN evaluation runs
[ ] linear probe runs
[ ] target encoder has no gradients
[ ] target encoder updates via EMA
[ ] mask overlap is always zero
[ ] representation diagnostics are finite
[ ] target representation does not collapse immediately
```

Once these are true, we have a minimal but complete I-JEPA implementation.

---

## 2.12.12 Transition to Chapter 3

Chapter 2 defined the working minimal implementation.

Chapter 3 will make it more usable for research.

The transition is:

```text
Chapter 2:
    plain PyTorch
    minimal scripts
    dataclass configs
    console logs
    simple checkpoints
    local debugging

Chapter 3:
    structured configs
    better logging
    checkpoint/resume discipline
    mixed precision
    Lightning Fabric
    cloud-run ergonomics
    ablation framework
    failure-mode playbook
```

The model itself should not change dramatically at the start of Chapter 3. The main change is the training and experimentation infrastructure around it.

This separation is important.

We first made the algorithm visible.

Now we can make it scalable.

---

## 2.12.13 Summary

Chapter 2 specified a complete minimal I-JEPA system.

The accompanying implementation should now have:

- patch-level image tokenization,
- position-aware patch encoders,
- context and target mask sampling,
- online and EMA target encoders,
- transformer predictor,
- latent-space losses,
- representation diagnostics,
- training loop,
- checkpointing,
- evaluation utilities,
- tests,
- marimo debugging notebooks.

The implementation target is intentionally small and inspectable.

Once implemented and verified, it is ready to be refactored into a more research-grade framework in Chapter 3.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- marimo, **Documentation**.
  <https://docs.marimo.io/>

- PyTorch, **Saving and Loading Models**.
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
