# 3.1 Configuration for JEPA Experiments

The Chapter 2 implementation used a compact dataclass because the goal was algorithmic clarity.

For serious experimentation, we need more structure. Not because configuration systems are interesting, but because JEPA experiments have several coupled choices that can silently invalidate a run:

```text
image size ↔ patch size ↔ number of patches
mask geometry ↔ available context patches
encoder dim ↔ attention heads
predictor dim ↔ predictor heads
EMA schedule ↔ global step
checkpoint resume ↔ target encoder state
evaluation features ↔ image preprocessing
```

This section defines a structured config system focused on those constraints.

We will not build a full Hydra stack yet. The first version uses:

```text
YAML files
typed dataclass schemas
explicit validation
small loader utilities
```

Hydra or OmegaConf can be layered on later. The important thing now is that every experiment is config-driven and JEPA-specific invariants are validated before training starts.

---

## 3.1.1 Config Groups

Use one top-level experiment config with focused subgroups:

```text
experiment
model
mask
data
optimizer
training
ema
diagnostics
logging
checkpointing
runtime
evaluation
```

The config should answer:

```text
What model is being trained?
What prediction task is the mask sampler creating?
How is the target encoder updated?
What is the optimization setup?
What diagnostics are computed?
How is the run executed and resumed?
How is representation quality evaluated?
```

A representative access pattern should look like:

```python
cfg.model.encoder_dim
cfg.mask.context_ratio
cfg.training.loss_type
cfg.ema.tau_base
cfg.diagnostics.effective_rank_every_steps
cfg.runtime.run_dir
```

This is much easier to reason about than a flat config once ablations begin.

---

## 3.1.2 Config Schema Layout

Add:

```text
src/jepa_world_model/configs/
├── __init__.py
├── schema.py
└── loader.py
```

The schema lives in:

```text
src/jepa_world_model/configs/schema.py
```

Start with imports:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
```

The final top-level schema will be:

```python
@dataclass(frozen=True)
class JEPAExperimentConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.model.validate()
        self.mask.validate(self.model)
        self.optimizer.validate()
        self.training.validate()
        self.ema.validate()
        self.diagnostics.validate()
        self.logging.validate()
        self.checkpointing.validate()
        self.evaluation.validate()
```

The schema itself should catch invalid experiments before a GPU is allocated.

---

## 3.1.3 Experiment Metadata

The experiment group is intentionally lightweight.

```python
@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "ijepa_experiment"
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
```

This is used for:

```text
run naming
manifest metadata
experiment grouping
sweep aggregation
```

Example:

```yaml
experiment:
  name: ijepa_stl10_base
  tags:
    - ijepa
    - stl10
    - baseline
  notes: "Baseline image JEPA run before mask ablations."
```

---

## 3.1.4 Model Config

The model config defines the image encoder and predictor architecture.

```python
@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 96
    patch_size: int = 8
    in_channels: int = 3

    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3

    predictor_dim: int = 128
    predictor_depth: int = 3
    predictor_heads: int = 4

    mlp_ratio: float = 4.0
    dropout: float = 0.0

    position_embedding: Literal["learned", "sincos"] = "learned"

    def validate(self) -> None:
        if self.image_size <= 0:
            raise ValueError("model.image_size must be positive.")

        if self.patch_size <= 0:
            raise ValueError("model.patch_size must be positive.")

        if self.image_size % self.patch_size != 0:
            raise ValueError(
                "model.image_size must be divisible by model.patch_size. "
                f"Got image_size={self.image_size}, patch_size={self.patch_size}."
            )

        if self.encoder_dim <= 0 or self.predictor_dim <= 0:
            raise ValueError("encoder_dim and predictor_dim must be positive.")

        if self.encoder_depth <= 0 or self.predictor_depth <= 0:
            raise ValueError("encoder_depth and predictor_depth must be positive.")

        if self.encoder_dim % self.encoder_heads != 0:
            raise ValueError(
                "model.encoder_dim must be divisible by model.encoder_heads."
            )

        if self.predictor_dim % self.predictor_heads != 0:
            raise ValueError(
                "model.predictor_dim must be divisible by model.predictor_heads."
            )

        if self.mlp_ratio <= 0:
            raise ValueError("model.mlp_ratio must be positive.")

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1).")

    @property
    def grid_size(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid_size * self.grid_size
```

The important JEPA-specific fields are:

```text
image_size
patch_size
grid_size
num_patches
encoder_dim
predictor_dim
```

These determine:

```text
mask feasibility
encoder token count
predictor input/output shapes
loss shape compatibility
evaluation full-patch indices
```

A config that breaks these constraints should fail immediately.

---

## 3.1.5 Mask Config

The mask config defines the self-supervised prediction task.

```python
@dataclass(frozen=True)
class MaskConfig:
    num_target_blocks: int = 4
    target_block_height: int = 3
    target_block_width: int = 3
    context_ratio: float = 0.6
    max_attempts: int = 100

    def num_target_patches(self) -> int:
        return (
            self.num_target_blocks
            * self.target_block_height
            * self.target_block_width
        )

    def num_context_patches(self, model: ModelConfig) -> int:
        return int(self.context_ratio * model.num_patches)

    def validate(self, model: ModelConfig) -> None:
        if self.num_target_blocks <= 0:
            raise ValueError("mask.num_target_blocks must be positive.")

        if self.target_block_height <= 0 or self.target_block_width <= 0:
            raise ValueError("mask target block dimensions must be positive.")

        if self.target_block_height > model.grid_size:
            raise ValueError(
                "mask.target_block_height cannot exceed model grid size."
            )

        if self.target_block_width > model.grid_size:
            raise ValueError(
                "mask.target_block_width cannot exceed model grid size."
            )

        if not 0.0 < self.context_ratio < 1.0:
            raise ValueError("mask.context_ratio must be in (0, 1).")

        if self.max_attempts <= 0:
            raise ValueError("mask.max_attempts must be positive.")

        n_tgt = self.num_target_patches()
        n_ctx = self.num_context_patches(model)

        if n_tgt >= model.num_patches:
            raise ValueError(
                "target patches must be fewer than total patches. "
                f"Got target={n_tgt}, total={model.num_patches}."
            )

        if n_ctx + n_tgt > model.num_patches:
            raise ValueError(
                "Infeasible mask config: context patches + target patches "
                "exceed total patches. "
                f"context={n_ctx}, target={n_tgt}, total={model.num_patches}."
            )
```

This validation is critical.

A bad mask config can produce:

```text
impossible sampling
unintended variable target length
context-target leakage
trivial prediction tasks
overly hard prediction tasks
```

The loader cannot determine semantic task quality, but it can catch impossible configurations.

---

## 3.1.6 Data Config

The data config controls dataset identity and input paths.

```python
@dataclass(frozen=True)
class DataConfig:
    dataset: Literal["cifar10", "stl10"] = "stl10"
    data_dir: str = "data"
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
```

Keep transforms minimal at first.

Augmentation policies are experimental variables, but they deserve their own group once introduced:

```text
augmentation:
    random_resized_crop
    color_jitter
    blur
    normalization
```

For now, the Chapter 2 transform path is enough:

```text
Resize(image_size)
ToTensor()
```

This keeps feature evaluation aligned with pretraining.

---

## 3.1.7 Optimizer Config

Optimizer config should be explicit about parameter groups.

```python
@dataclass(frozen=True)
class OptimizerConfig:
    name: Literal["adamw"] = "adamw"
    learning_rate: float = 5e-4
    weight_decay: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.95

    def validate(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be non-negative.")

        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("optimizer.beta1 must be in [0, 1).")

        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("optimizer.beta2 must be in [0, 1).")
```

Prefer `beta1` and `beta2` over a YAML tuple to avoid awkward type coercion.

The important invariant is not in the optimizer config itself. It is in the optimizer builder:

```text
target encoder parameters must be excluded
```

This will be enforced in the optimizer section.

---

## 3.1.8 Training Config

Training config contains loop-level choices.

```python
@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    batch_size: int = 128

    loss_type: Literal["mse", "smooth_l1", "cosine", "combined"] = "smooth_l1"

    gradient_clip_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1

    precision: Literal["32-true", "16-mixed", "bf16-mixed"] = "32-true"

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("training.epochs must be positive.")

        if self.batch_size <= 0:
            raise ValueError("training.batch_size must be positive.")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "training.gradient_accumulation_steps must be positive."
            )

        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                "training.gradient_clip_norm must be positive or null."
            )
```

The effective global batch size will later be computed from:

```text
batch_size × num_devices × gradient_accumulation_steps
```

That value should be logged in the run manifest once Fabric enters.

---

## 3.1.9 EMA Config

EMA is not an auxiliary detail. It defines the target representation dynamics.

```python
@dataclass(frozen=True)
class EMAConfig:
    tau_base: float = 0.996
    tau_final: float = 1.0

    def validate(self) -> None:
        if not 0.0 <= self.tau_base <= 1.0:
            raise ValueError("ema.tau_base must be in [0, 1].")

        if not 0.0 <= self.tau_final <= 1.0:
            raise ValueError("ema.tau_final must be in [0, 1].")

        if self.tau_base > self.tau_final:
            raise ValueError("ema.tau_base must be <= ema.tau_final.")
```

The EMA schedule depends on `global_step` and `total_steps`.

That means resume must restore `global_step`, not only epoch number.

---

## 3.1.10 Diagnostics Config

Chapter 2 computed diagnostics directly in the training step.

Chapter 3 should make diagnostic cadence configurable.

```python
@dataclass(frozen=True)
class DiagnosticsConfig:
    basic_every_steps: int = 20
    representation_every_steps: int = 20
    covariance_every_steps: int = 500
    parameter_distance_every_steps: int = 500
    mask_check_every_steps: int = 20

    compute_effective_rank: bool = True
    compute_covariance: bool = True
    assert_finite: bool = True
    assert_no_mask_overlap: bool = True

    def validate(self) -> None:
        positive_fields = [
            self.basic_every_steps,
            self.representation_every_steps,
            self.covariance_every_steps,
            self.parameter_distance_every_steps,
            self.mask_check_every_steps,
        ]

        if any(value <= 0 for value in positive_fields):
            raise ValueError("All diagnostics cadence fields must be positive.")
```

The main idea is:

```text
cheap metrics frequently
expensive metrics periodically
assertions enabled during debug
assertions optionally throttled during scale runs
```

This prevents covariance/effective-rank diagnostics from dominating large runs while preserving collapse visibility.

---

## 3.1.11 Logging Config

Logging config controls sinks and cadence.

```python
@dataclass(frozen=True)
class LoggingConfig:
    log_every_steps: int = 20

    jsonl: bool = True
    tensorboard: bool = True
    wandb: bool = False

    wandb_project: str | None = None
    wandb_entity: str | None = None

    def validate(self) -> None:
        if self.log_every_steps <= 0:
            raise ValueError("logging.log_every_steps must be positive.")

        if self.wandb and self.wandb_project is None:
            raise ValueError(
                "logging.wandb_project must be set when logging.wandb=true."
            )
```

The canonical stream should remain JSONL.

TensorBoard and W&B are views, not the only metric store.

---

## 3.1.12 Checkpointing Config

Checkpoint config controls cadence and retention.

```python
@dataclass(frozen=True)
class CheckpointingConfig:
    save_every_epochs: int = 10
    save_every_steps: int | None = None

    keep_last: bool = True
    keep_best: bool = False

    monitor_metric: str = "linear_probe/val_acc"
    mode: Literal["min", "max"] = "max"

    export_encoder_on_finish: bool = True

    def validate(self) -> None:
        if self.save_every_epochs <= 0:
            raise ValueError("checkpointing.save_every_epochs must be positive.")

        if self.save_every_steps is not None and self.save_every_steps <= 0:
            raise ValueError("checkpointing.save_every_steps must be positive.")

        if self.keep_best and not self.monitor_metric:
            raise ValueError(
                "checkpointing.monitor_metric must be set when keep_best=true."
            )
```

JEPA-specific checkpointing is handled later, but the config already distinguishes:

```text
training checkpoint cadence
best checkpoint selection
encoder export
```

---

## 3.1.13 Runtime Config

Runtime config separates machine-specific concerns from experiment design.

```python
@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 42
    run_dir: str = "runs"
    resume_from: str | None = None

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"

    fabric_accelerator: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    fabric_devices: str | int = "auto"
    fabric_strategy: str = "auto"
```

For local debug:

```yaml
runtime:
  run_dir: runs
  device: auto
```

For RunPod:

```yaml
runtime:
  run_dir: /workspace/runs
  device: cuda
  fabric_accelerator: cuda
  fabric_devices: auto
```

The training code should not contain cloud paths.

---

## 3.1.14 Evaluation Config

Evaluation config controls representation checks.

```python
@dataclass(frozen=True)
class EvaluationConfig:
    eval_every_epochs: int | None = 10

    run_knn: bool = True
    run_linear_probe: bool = True

    knn_k: int = 20
    probe_epochs: int = 100
    max_batches: int | None = None

    evaluate_online_encoder: bool = True
    evaluate_target_encoder: bool = False

    def validate(self) -> None:
        if self.eval_every_epochs is not None and self.eval_every_epochs <= 0:
            raise ValueError("evaluation.eval_every_epochs must be positive.")

        if self.knn_k <= 0:
            raise ValueError("evaluation.knn_k must be positive.")

        if self.probe_epochs <= 0:
            raise ValueError("evaluation.probe_epochs must be positive.")
```

Default evaluation should use the online encoder.

Target encoder evaluation can be useful for research questions:

```text
Does EMA smoothing improve retrieval?
Does target encoder probe better than online encoder?
```

but it should be opt-in.

---

## 3.1.15 Complete Schema File

Putting the pieces together:

```python
@dataclass(frozen=True)
class JEPAExperimentConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def validate(self) -> None:
        self.model.validate()
        self.mask.validate(self.model)
        self.optimizer.validate()
        self.training.validate()
        self.ema.validate()
        self.diagnostics.validate()
        self.logging.validate()
        self.checkpointing.validate()
        self.evaluation.validate()
```

This schema is intentionally explicit. It is easy to extend and easy to inspect.

---

## 3.1.16 YAML Loader

The loader lives in:

```text
src/jepa_world_model/configs/loader.py
```

Add dependency:

```bash
uv add pyyaml
```

Implementation:

```python
from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

from jepa_world_model.configs.schema import JEPAExperimentConfig


T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")

    return data


def dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """
    Recursively instantiate a dataclass from a dictionary.

    Unknown fields are rejected.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass type.")

    field_map = {field.name: field for field in fields(cls)}

    unknown = set(data.keys()) - set(field_map.keys())
    if unknown:
        raise ValueError(
            f"Unknown config fields for {cls.__name__}: {sorted(unknown)}"
        )

    defaults = cls()
    kwargs = {}

    for name, value in data.items():
        default_value = getattr(defaults, name)

        if is_dataclass(default_value):
            kwargs[name] = dataclass_from_dict(
                type(default_value),
                value,
            )
        else:
            kwargs[name] = value

    return cls(**kwargs)


def load_config(path: str | Path) -> JEPAExperimentConfig:
    data = load_yaml(path)

    cfg = dataclass_from_dict(
        JEPAExperimentConfig,
        data,
    )

    cfg.validate()

    return cfg


def save_config(
    cfg: JEPAExperimentConfig,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(cfg),
            f,
            sort_keys=False,
        )
```

This loader is deliberately small.

It does not yet implement:

```text
overrides
composition
config inheritance
sweeps
```

Those come later.

---

## 3.1.17 Example Debug Config

Create:

```text
configs/ijepa_cifar10_debug.yaml
```

```yaml
experiment:
  name: ijepa_cifar10_debug
  tags: [ijepa, cifar10, debug]

model:
  image_size: 32
  patch_size: 4
  in_channels: 3
  encoder_dim: 64
  encoder_depth: 2
  encoder_heads: 4
  predictor_dim: 64
  predictor_depth: 1
  predictor_heads: 4
  mlp_ratio: 4.0
  dropout: 0.0
  position_embedding: learned

mask:
  num_target_blocks: 2
  target_block_height: 2
  target_block_width: 2
  context_ratio: 0.5
  max_attempts: 100

data:
  dataset: cifar10
  data_dir: data
  num_workers: 0
  pin_memory: true
  drop_last: true

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.05
  beta1: 0.9
  beta2: 0.95

training:
  epochs: 2
  batch_size: 16
  loss_type: smooth_l1
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  precision: "32-true"

ema:
  tau_base: 0.996
  tau_final: 1.0

diagnostics:
  basic_every_steps: 10
  representation_every_steps: 10
  covariance_every_steps: 100
  parameter_distance_every_steps: 100
  mask_check_every_steps: 10
  compute_effective_rank: true
  compute_covariance: true
  assert_finite: true
  assert_no_mask_overlap: true

logging:
  log_every_steps: 10
  jsonl: true
  tensorboard: true
  wandb: false

checkpointing:
  save_every_epochs: 1
  save_every_steps: null
  keep_last: true
  keep_best: false
  monitor_metric: linear_probe/val_acc
  mode: max
  export_encoder_on_finish: true

runtime:
  seed: 42
  run_dir: runs
  resume_from: null
  device: auto
  fabric_accelerator: auto
  fabric_devices: auto
  fabric_strategy: auto

evaluation:
  eval_every_epochs: null
  run_knn: true
  run_linear_probe: true
  knn_k: 20
  probe_epochs: 10
  max_batches: 20
  evaluate_online_encoder: true
  evaluate_target_encoder: false
```

This config is for plumbing, not research conclusions.

---

## 3.1.18 Example STL-10 Base Config

Create:

```text
configs/ijepa_stl10_base.yaml
```

```yaml
experiment:
  name: ijepa_stl10_base
  tags: [ijepa, stl10, baseline]

model:
  image_size: 96
  patch_size: 8
  in_channels: 3
  encoder_dim: 192
  encoder_depth: 6
  encoder_heads: 3
  predictor_dim: 128
  predictor_depth: 3
  predictor_heads: 4
  mlp_ratio: 4.0
  dropout: 0.0
  position_embedding: learned

mask:
  num_target_blocks: 4
  target_block_height: 3
  target_block_width: 3
  context_ratio: 0.6
  max_attempts: 100

data:
  dataset: stl10
  data_dir: data
  num_workers: 4
  pin_memory: true
  drop_last: true

optimizer:
  name: adamw
  learning_rate: 0.0005
  weight_decay: 0.05
  beta1: 0.9
  beta2: 0.95

training:
  epochs: 100
  batch_size: 128
  loss_type: smooth_l1
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  precision: "32-true"

ema:
  tau_base: 0.996
  tau_final: 1.0

diagnostics:
  basic_every_steps: 20
  representation_every_steps: 20
  covariance_every_steps: 500
  parameter_distance_every_steps: 500
  mask_check_every_steps: 20
  compute_effective_rank: true
  compute_covariance: true
  assert_finite: true
  assert_no_mask_overlap: true

logging:
  log_every_steps: 20
  jsonl: true
  tensorboard: true
  wandb: false

checkpointing:
  save_every_epochs: 10
  save_every_steps: null
  keep_last: true
  keep_best: false
  monitor_metric: linear_probe/val_acc
  mode: max
  export_encoder_on_finish: true

runtime:
  seed: 42
  run_dir: runs
  resume_from: null
  device: auto
  fabric_accelerator: auto
  fabric_devices: auto
  fabric_strategy: auto

evaluation:
  eval_every_epochs: 10
  run_knn: true
  run_linear_probe: true
  knn_k: 20
  probe_epochs: 100
  max_batches: null
  evaluate_online_encoder: true
  evaluate_target_encoder: false
```

---

## 3.1.19 Example Cloud Config

Create:

```text
configs/ijepa_stl10_cloud.yaml
```

```yaml
experiment:
  name: ijepa_stl10_cloud
  tags: [ijepa, stl10, cloud, runpod]

model:
  image_size: 96
  patch_size: 8
  in_channels: 3
  encoder_dim: 384
  encoder_depth: 8
  encoder_heads: 6
  predictor_dim: 256
  predictor_depth: 4
  predictor_heads: 8
  mlp_ratio: 4.0
  dropout: 0.0
  position_embedding: learned

mask:
  num_target_blocks: 4
  target_block_height: 3
  target_block_width: 3
  context_ratio: 0.6
  max_attempts: 100

data:
  dataset: stl10
  data_dir: /workspace/data
  num_workers: 8
  pin_memory: true
  drop_last: true

optimizer:
  name: adamw
  learning_rate: 0.0005
  weight_decay: 0.05
  beta1: 0.9
  beta2: 0.95

training:
  epochs: 300
  batch_size: 256
  loss_type: smooth_l1
  gradient_clip_norm: 1.0
  gradient_accumulation_steps: 1
  precision: "bf16-mixed"

ema:
  tau_base: 0.996
  tau_final: 1.0

diagnostics:
  basic_every_steps: 50
  representation_every_steps: 50
  covariance_every_steps: 1000
  parameter_distance_every_steps: 1000
  mask_check_every_steps: 50
  compute_effective_rank: true
  compute_covariance: true
  assert_finite: true
  assert_no_mask_overlap: true

logging:
  log_every_steps: 50
  jsonl: true
  tensorboard: true
  wandb: false

checkpointing:
  save_every_epochs: 10
  save_every_steps: null
  keep_last: true
  keep_best: false
  monitor_metric: linear_probe/val_acc
  mode: max
  export_encoder_on_finish: true

runtime:
  seed: 42
  run_dir: /workspace/runs
  resume_from: null
  device: cuda
  fabric_accelerator: cuda
  fabric_devices: auto
  fabric_strategy: auto

evaluation:
  eval_every_epochs: 10
  run_knn: true
  run_linear_probe: true
  knn_k: 20
  probe_epochs: 100
  max_batches: null
  evaluate_online_encoder: true
  evaluate_target_encoder: false
```

The cloud config changes scale and paths, not the algorithm.

---

## 3.1.20 Config-Driven Builders

Chapter 2 builders can be kept for minimal scripts, but Chapter 3 should use config-driven builders.

Add to `model.py` or a new builder module:

```python
from jepa_world_model.configs.schema import JEPAExperimentConfig
from jepa_world_model.masks import BlockMaskConfig
from jepa_world_model.predictor import JEPAPredictor
from jepa_world_model.vit import MinimalViTEncoder
```

Model builder:

```python
def build_ijepa_from_config(
    cfg: JEPAExperimentConfig,
) -> MinimalIJEPA:
    m = cfg.model

    online_encoder = MinimalViTEncoder(
        image_size=m.image_size,
        patch_size=m.patch_size,
        in_channels=m.in_channels,
        embed_dim=m.encoder_dim,
        depth=m.encoder_depth,
        num_heads=m.encoder_heads,
        mlp_ratio=m.mlp_ratio,
        dropout=m.dropout,
    )

    target_encoder = MinimalViTEncoder(
        image_size=m.image_size,
        patch_size=m.patch_size,
        in_channels=m.in_channels,
        embed_dim=m.encoder_dim,
        depth=m.encoder_depth,
        num_heads=m.encoder_heads,
        mlp_ratio=m.mlp_ratio,
        dropout=m.dropout,
    )

    predictor = JEPAPredictor(
        grid_size=m.grid_size,
        encoder_dim=m.encoder_dim,
        predictor_dim=m.predictor_dim,
        depth=m.predictor_depth,
        num_heads=m.predictor_heads,
        mlp_ratio=m.mlp_ratio,
        dropout=m.dropout,
    )

    return MinimalIJEPA(
        online_encoder=online_encoder,
        target_encoder=target_encoder,
        predictor=predictor,
    )
```

Mask builder:

```python
def build_mask_config_from_config(
    cfg: JEPAExperimentConfig,
) -> BlockMaskConfig:
    return BlockMaskConfig(
        grid_height=cfg.model.grid_size,
        grid_width=cfg.model.grid_size,
        num_target_blocks=cfg.mask.num_target_blocks,
        target_block_height=cfg.mask.target_block_height,
        target_block_width=cfg.mask.target_block_width,
        context_ratio=cfg.mask.context_ratio,
        max_attempts=cfg.mask.max_attempts,
    )
```

Optimizer builder:

```python
import torch


def build_optimizer_from_config(
    cfg: JEPAExperimentConfig,
    model: MinimalIJEPA,
) -> torch.optim.Optimizer:
    if cfg.optimizer.name != "adamw":
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")

    return torch.optim.AdamW(
        trainable_jepa_parameters(model),
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
    )
```

This keeps experiment scripts thin.

---

## 3.1.21 Data Loader From Config

Update the Chapter 2 data builders or add a new one:

```python
def build_pretrain_dataloader_from_config(
    cfg: JEPAExperimentConfig,
) -> DataLoader:
    transform = build_image_transform(
        image_size=cfg.model.image_size,
    )

    if cfg.data.dataset == "stl10":
        dataset = STL10(
            root=cfg.data.data_dir,
            split="unlabeled",
            download=True,
            transform=transform,
        )

    elif cfg.data.dataset == "cifar10":
        dataset = CIFAR10(
            root=cfg.data.data_dir,
            train=True,
            download=True,
            transform=transform,
        )

    else:
        raise ValueError(f"Unsupported dataset: {cfg.data.dataset}")

    return DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory and torch.cuda.is_available(),
        drop_last=cfg.data.drop_last,
    )
```

Again, no cloud-specific logic appears here.

Paths and worker counts come from config.

---

## 3.1.22 First Config-Driven Training Entry Point

Create:

```text
experiments/train.py
```

At this stage, keep it plain PyTorch. Fabric enters later.

```python
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import tqdm

from jepa_world_model.checkpointing import save_checkpoint
from jepa_world_model.configs.loader import load_config, save_config
from jepa_world_model.data import build_pretrain_dataloader_from_config
from jepa_world_model.model import (
    build_ijepa_from_config,
    build_mask_config_from_config,
    build_optimizer_from_config,
)
from jepa_world_model.training import format_logs, train_step
from jepa_world_model.utils import get_device, seed_everything
```

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    return parser.parse_args()
```

```python
def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return get_device()

    return torch.device(device_name)
```

```python
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed_everything(cfg.runtime.seed)

    device = resolve_device(cfg.runtime.device)

    run_dir = Path(cfg.runtime.run_dir) / cfg.experiment.name
    run_dir.mkdir(parents=True, exist_ok=True)

    save_config(cfg, run_dir / "config.yaml")

    dataloader = build_pretrain_dataloader_from_config(cfg)

    model = build_ijepa_from_config(cfg).to(device)
    optimizer = build_optimizer_from_config(cfg, model)
    mask_config = build_mask_config_from_config(cfg)

    total_steps = cfg.training.epochs * len(dataloader)
    global_step = 0

    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.training.epochs):
        progress = tqdm(
            dataloader,
            desc=f"epoch {epoch + 1}/{cfg.training.epochs}",
        )

        for batch in progress:
            images = batch[0].to(device, non_blocking=True)

            logs = train_step(
                model=model,
                images=images,
                mask_config=mask_config,
                optimizer=optimizer,
                step=global_step,
                total_steps=total_steps,
                loss_type=cfg.training.loss_type,
                ema_tau_base=cfg.ema.tau_base,
                ema_tau_final=cfg.ema.tau_final,
                check_masks=cfg.diagnostics.assert_no_mask_overlap,
                log_parameter_distance=(
                    global_step % cfg.diagnostics.parameter_distance_every_steps == 0
                ),
            )

            if global_step % cfg.logging.log_every_steps == 0:
                progress.set_postfix_str(format_logs(logs))

            global_step += 1

        if (epoch + 1) % cfg.checkpointing.save_every_epochs == 0:
            save_checkpoint(
                path=checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                model=model,
                optimizer=optimizer,
                cfg=cfg,
                step=global_step,
                epoch=epoch + 1,
            )

    if cfg.checkpointing.keep_last:
        save_checkpoint(
            path=checkpoint_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            cfg=cfg,
            step=global_step,
            epoch=cfg.training.epochs,
        )


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python experiments/train.py \
  --config configs/ijepa_cifar10_debug.yaml
```

This is not the final Chapter 3 training harness yet. It is the bridge from Chapter 2 scripts to config-driven execution.

---

## 3.1.23 Config Tests

Create:

```text
tests/test_configs.py
```

Add:

```python
from pathlib import Path

import pytest

from jepa_world_model.configs.loader import load_config, save_config
from jepa_world_model.configs.schema import JEPAExperimentConfig


def test_default_config_validates():
    cfg = JEPAExperimentConfig()
    cfg.validate()


def test_save_load_roundtrip(tmp_path: Path):
    cfg = JEPAExperimentConfig()

    path = tmp_path / "config.yaml"
    save_config(cfg, path)

    loaded = load_config(path)

    assert loaded.model.image_size == cfg.model.image_size
    assert loaded.mask.context_ratio == cfg.mask.context_ratio
    assert loaded.ema.tau_base == cfg.ema.tau_base


def test_unknown_field_rejected(tmp_path: Path):
    path = tmp_path / "bad.yaml"

    path.write_text(
        """
model:
  image_size: 96
unknown_group:
  x: 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)


def test_invalid_patch_size_rejected(tmp_path: Path):
    path = tmp_path / "bad_patch.yaml"

    path.write_text(
        """
model:
  image_size: 95
  patch_size: 8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)


def test_infeasible_mask_rejected(tmp_path: Path):
    path = tmp_path / "bad_mask.yaml"

    path.write_text(
        """
model:
  image_size: 32
  patch_size: 4

mask:
  num_target_blocks: 10
  target_block_height: 4
  target_block_width: 4
  context_ratio: 0.8
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_config(path)


def test_debug_config_loads():
    cfg = load_config("configs/ijepa_cifar10_debug.yaml")

    assert cfg.data.dataset == "cifar10"
    assert cfg.model.image_size == 32


def test_stl10_base_config_loads():
    cfg = load_config("configs/ijepa_stl10_base.yaml")

    assert cfg.data.dataset == "stl10"
    assert cfg.model.image_size == 96


def test_cloud_config_loads():
    cfg = load_config("configs/ijepa_stl10_cloud.yaml")

    assert cfg.runtime.run_dir == "/workspace/runs"
    assert cfg.training.precision == "bf16-mixed"
```

Run:

```bash
uv run pytest tests/test_configs.py
```

These tests catch broken configs before training.

---

## 3.1.24 What We Are Not Adding Yet

Do not add all advanced config features immediately.

Specifically, defer:

```text
Hydra composition
command-line dotlist overrides
config inheritance
sweep expansion
environment variable interpolation
automatic run naming
schema migration
```

Those are useful, but they are not required for the next step.

The immediate goal is:

```text
YAML → typed config → validation → config-driven builders → train script
```

This is enough to stabilize the harness before adding Fabric and ablations.

---

## 3.1.25 Summary

This section introduced a structured config system for JEPA experiments.

The important pieces are:

```text
typed dataclass schema
explicit JEPA-specific validation
YAML config files
config-driven model construction
config-driven mask construction
config-driven dataloaders
config-driven optimizer construction
first config-driven training entry point
```

The most important validation checks are:

```text
image_size divisible by patch_size
attention dimensions divisible by heads
mask task feasible for patch grid
EMA schedule valid
diagnostics cadence valid
target encoder excluded structurally from optimizer
```

The next section defines run state and experiment manifests so that each config-driven run becomes auditable and resumable.

---

## References and Further Reading

- Python `dataclasses`:
  <https://docs.python.org/3/library/dataclasses.html>

- PyYAML documentation:
  <https://pyyaml.org/wiki/PyYAMLDocumentation>

- Hydra documentation:
  <https://hydra.cc/docs/intro/>

- OmegaConf documentation:
  <https://omegaconf.readthedocs.io/>

- PyTorch `AdamW`:
  <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>
