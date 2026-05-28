# 3.3 Metrics, Diagnostics, and Logging Cadence

Section 3.2 gave each run a durable home:

```text
config.yaml
manifest.yaml
metrics.jsonl
checkpoints/
artifacts/
logs/
```

This section defines what goes into the metric stream and how often.

For JEPA, logging is not just operational telemetry. It is part of the method’s safety system.

A decreasing latent prediction loss is not enough. The model can reduce loss while learning a weak representation, collapsing dimensions, leaking target information through masks, or accidentally updating the target encoder through gradients.

The logging layer should make those failures visible.

This section implements:

```text
metric taxonomy
diagnostic cadence
JSONL logging
TensorBoard sink
optional W&B sink
rank-zero-safe logging
metric aggregation helpers
```

The implementation remains lightweight, but the metric design is JEPA-specific.

---

## 3.3.1 Metric Groups

A serious JEPA run should log metrics in several groups.

```text
optimization
prediction
target representation
prediction representation
mask
EMA
runtime
evaluation
```

A typical metric row should eventually contain fields like:

```text
step
epoch
loss
lr
grad_norm
pred_target/cosine
target/std_mean
target/effective_rank
mask/overlap_fraction
ema_tau
ema/relative_param_l2
```

The purpose is to see not only whether the objective is improving, but whether the representation remains usable.

---

## 3.3.2 Metric Taxonomy

Use slash-separated metric names:

```text
group/name
```

Examples:

```text
pred_target/cosine
target/std_mean
mask/overlap_fraction
ema/relative_param_l2
linear_probe/val_acc
```

This convention works across:

```text
JSONL
TensorBoard
W&B
pandas
plotting scripts
```

Avoid names like:

```text
Target Std Mean
prediction cosine similarity
mask-overlap
```

Prefer stable machine-readable names.

Recommended metric groups:

```text
optimization:
    loss
    lr
    grad_norm
    global_batch_size

prediction:
    pred_target/mse
    pred_target/mae
    pred_target/cosine

target representation:
    target/norm_mean
    target/norm_std
    target/std_mean
    target/std_min
    target/dead_dim_fraction
    target/effective_rank
    target/cov_offdiag_abs_mean

prediction representation:
    pred/norm_mean
    pred/std_mean
    pred/dead_dim_fraction
    pred/effective_rank

mask:
    mask/context_count
    mask/target_count
    mask/context_ratio
    mask/target_ratio
    mask/overlap_fraction

EMA:
    ema_tau
    ema/param_l2
    ema/relative_param_l2

runtime:
    time/step_seconds
    time/samples_per_second
    memory/cuda_allocated_gb
    memory/cuda_reserved_gb

evaluation:
    knn/accuracy
    linear_probe/train_acc
    linear_probe/val_acc
```

Not every metric needs to be logged every step. The cadence matters.

---

## 3.3.3 Cheap vs Expensive Diagnostics

JEPA diagnostics have different costs.

Cheap metrics:

```text
loss
pred_target/cosine
pred_target/mse
mask/overlap_fraction
ema_tau
lr
```

Moderate metrics:

```text
norm statistics
feature std
dead-dimension fraction
gradient norm
EMA parameter distance
```

Expensive metrics:

```text
covariance diagnostics
effective rank
k-NN evaluation
linear probe
retrieval visualizations
```

The training harness should not compute everything every step.

A good default is:

```yaml
diagnostics:
  basic_every_steps: 20
  representation_every_steps: 20
  covariance_every_steps: 500
  parameter_distance_every_steps: 500
  mask_check_every_steps: 20
  compute_effective_rank: true
  compute_covariance: true
```

For a large cloud run, increase expensive cadences:

```yaml
diagnostics:
  basic_every_steps: 50
  representation_every_steps: 50
  covariance_every_steps: 1000
  parameter_distance_every_steps: 1000
```

The goal is not maximal logging. The goal is enough signal to detect failure without dominating training time.

---

## 3.3.4 JEPA Early-Warning Signals

The most important early-warning metrics are:

```text
mask/overlap_fraction
target/std_mean
target/dead_dim_fraction
target/effective_rank
ema/relative_param_l2
pred_target/cosine
```

Interpretation:

| Metric                     | Healthy behavior        | Warning sign                 |
| -------------------------- | ----------------------- | ---------------------------- |
| `mask/overlap_fraction`    | exactly `0.0`           | any nonzero value            |
| `target/std_mean`          | stable, nonzero         | trends toward zero           |
| `target/dead_dim_fraction` | near zero               | trends toward one            |
| `target/effective_rank`    | nonzero, not collapsing | sharp decline                |
| `ema/relative_param_l2`    | nonzero but bounded     | zero forever or exploding    |
| `pred_target/cosine`       | gradually improves      | high with collapsed variance |

A high `pred_target/cosine` is not sufficient. If representation variance collapses, cosine can look good while features become useless.

So the minimum useful training log is not:

```text
loss
```

It is closer to:

```text
loss
pred_target/cosine
target/std_mean
target/effective_rank
mask/overlap_fraction
ema_tau
```

---

## 3.3.5 Diagnostic Cadence Helper

Create:

```text
src/jepa_world_model/engine/logging.py
```

The file already contains `JSONLMetricWriter` from Section 3.2. Extend it.

Start with imports:

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
```

Define a cadence helper:

```python
def should_log(
    step: int,
    every_steps: int,
) -> bool:
    if every_steps <= 0:
        raise ValueError(f"every_steps must be positive, got {every_steps}.")

    return step % every_steps == 0
```

This keeps the training loop readable:

```python
if should_log(global_step, cfg.diagnostics.covariance_every_steps):
    ...
```

---

## 3.3.6 Metric Sink Protocol

Define a simple sink interface:

```python
class MetricSink(Protocol):
    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        ...
```

Each sink receives the same metrics.

This lets us fan out to:

```text
JSONL
TensorBoard
W&B
console
```

without coupling the training loop to a specific tool.

---

## 3.3.7 JSONL Sink

Update the JSONL writer to match the sink interface:

```python
class JSONLMetricWriter:
    """
    Append metrics to a JSONL file.

    This is the canonical metric stream for a run.
    """

    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        row = make_json_serializable(metrics)
        row["step"] = step

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
```

Serialization helper:

```python
def make_json_serializable(
    metrics: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {}

    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.item()

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, (int, float, str, bool)) or value is None:
            row[key] = value

    return row
```

This intentionally skips non-scalar tensors.

Metric rows should be scalar.

---

## 3.3.8 Reading JSONL Metrics

Keep the reader:

```python
def read_jsonl_metrics(
    path: str | Path,
) -> list[dict[str, Any]]:
    path = Path(path)

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rows.append(json.loads(line))

    return rows
```

This enables quick analysis:

```python
rows = read_jsonl_metrics("runs/.../metrics.jsonl")
```

In marimo:

```python
import pandas as pd

df = pd.DataFrame(rows)
df[["step", "loss", "target/effective_rank"]].plot(x="step")
```

---

## 3.3.9 TensorBoard Sink

TensorBoard is useful for fast visual inspection.

Add dependency if not already present:

```bash
uv add --dev tensorboard
```

Implement:

```python
class TensorBoardMetricWriter:
    """
    Write scalar metrics to TensorBoard.
    """

    def __init__(
        self,
        log_dir: str | Path,
    ):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir=str(log_dir),
        )

    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        row = make_json_serializable(metrics)

        for key, value in row.items():
            if key in {"step", "epoch"}:
                continue

            if isinstance(value, (int, float)):
                self.writer.add_scalar(
                    key,
                    value,
                    global_step=step,
                )

    def close(self) -> None:
        self.writer.close()
```

Write TensorBoard logs under:

```text
runs/.../logs/tensorboard/
```

Use:

```python
tensorboard_writer = TensorBoardMetricWriter(
    run_paths.logs / "tensorboard"
)
```

Then:

```bash
uv run tensorboard --logdir runs
```

For cloud:

```bash
uv run tensorboard \
  --logdir /workspace/runs \
  --host 0.0.0.0 \
  --port 6006
```

---

## 3.3.10 Optional W&B Sink

W&B should be optional.

Add:

```python
class WandBMetricWriter:
    """
    Optional W&B metric sink.

    Assumes wandb is installed and initialized here.
    """

    def __init__(
        self,
        project: str,
        entity: str | None,
        name: str,
        config: dict[str, Any],
    ):
        import wandb

        self.wandb = wandb

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
        )

    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        row = make_json_serializable(metrics)
        self.wandb.log(row, step=step)

    def close(self) -> None:
        self.run.finish()
```

This sink should only be constructed when:

```python
cfg.logging.wandb
```

is true.

Do not make W&B required for local or cloud training.

---

## 3.3.11 Multi-Sink Metric Logger

Define:

```python
class MetricLogger:
    """
    Fan-out metric logger.

    JSONL should usually be one of the sinks and remain the canonical record.
    """

    def __init__(
        self,
        sinks: list[MetricSink],
    ):
        self.sinks = sinks

    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        for sink in self.sinks:
            sink.write(
                metrics,
                step=step,
            )

    def close(self) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)

            if close is not None:
                close()
```

Builder:

```python
from dataclasses import asdict

from jepa_world_model.configs.schema import JEPAExperimentConfig
from jepa_world_model.engine.run_context import RunPaths
```

```python
def build_metric_logger(
    cfg: JEPAExperimentConfig,
    run_paths: RunPaths,
) -> MetricLogger:
    sinks: list[MetricSink] = []

    if cfg.logging.jsonl:
        sinks.append(
            JSONLMetricWriter(
                run_paths.metrics_path,
            )
        )

    if cfg.logging.tensorboard:
        sinks.append(
            TensorBoardMetricWriter(
                run_paths.logs / "tensorboard",
            )
        )

    if cfg.logging.wandb:
        if cfg.logging.wandb_project is None:
            raise ValueError(
                "cfg.logging.wandb_project must be set when W&B logging is enabled."
            )

        sinks.append(
            WandBMetricWriter(
                project=cfg.logging.wandb_project,
                entity=cfg.logging.wandb_entity,
                name=run_paths.root.name,
                config=asdict(cfg),
            )
        )

    return MetricLogger(sinks)
```

Now the training script only needs:

```python
metric_logger = build_metric_logger(cfg, run_paths)
```

and:

```python
metric_logger.write(row, step=global_step)
```

---

## 3.3.12 Console Formatting

Console output is not the metric store. It is a human-facing progress view.

Keep it compact.

```python
def format_console_metrics(
    metrics: dict[str, Any],
    keys: list[str] | None = None,
) -> str:
    if keys is None:
        keys = [
            "loss",
            "pred_target/cosine",
            "target/std_mean",
            "target/effective_rank",
            "mask/overlap_fraction",
            "ema_tau",
        ]

    parts = []

    for key in keys:
        if key not in metrics:
            continue

        value = metrics[key]

        if hasattr(value, "item"):
            value = value.item()

        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    return " | ".join(parts)
```

Example:

```text
loss=0.1842 | pred_target/cosine=0.2311 | target/std_mean=0.6123 | target/effective_rank=43.8210 | mask/overlap_fraction=0.0000 | ema_tau=0.9967
```

This is enough to monitor a run without overwhelming the terminal.

---

## 3.3.13 Runtime Throughput Metrics

Add a small timer:

```python
@dataclass
class StepTimer:
    last_time: float | None = None

    def tick(self) -> float | None:
        now = time.perf_counter()

        if self.last_time is None:
            self.last_time = now
            return None

        elapsed = now - self.last_time
        self.last_time = now

        return elapsed
```

Runtime metrics:

```python
def runtime_metrics(
    step_seconds: float | None,
    batch_size: int,
) -> dict[str, float]:
    if step_seconds is None:
        return {}

    samples_per_second = batch_size / max(step_seconds, 1e-12)

    return {
        "time/step_seconds": step_seconds,
        "time/samples_per_second": samples_per_second,
    }
```

CUDA memory metrics:

```python
def cuda_memory_metrics() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    return {
        "memory/cuda_allocated_gb": allocated,
        "memory/cuda_reserved_gb": reserved,
    }
```

These are operational metrics, but they matter for comparing local and cloud runs.

---

## 3.3.14 Gradient Norm Metrics

Gradient norm is useful for instability detection.

Add:

```python
def grad_global_norm(
    parameters,
) -> float:
    norms = []

    for param in parameters:
        if param.grad is None:
            continue

        norms.append(
            param.grad.detach().float().norm(2)
        )

    if not norms:
        return 0.0

    total = torch.stack(norms).norm(2)

    return total.item()
```

Usage before clipping or after backward:

```python
grad_norm = grad_global_norm(
    trainable_jepa_parameters(model)
)

logs["grad_norm"] = grad_norm
```

If using Fabric clipping, compute the norm before clipping if you want the raw signal, or after clipping if you want the applied norm. Name accordingly:

```text
grad_norm/raw
grad_norm/clipped
```

For now, log:

```text
grad_norm
```

before clipping.

---

## 3.3.15 Learning Rate Metrics

For a single optimizer group:

```python
def optimizer_lr_metrics(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    if not optimizer.param_groups:
        return {}

    return {
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
```

For multiple parameter groups:

```python
def optimizer_group_lr_metrics(
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    metrics = {}

    for idx, group in enumerate(optimizer.param_groups):
        metrics[f"lr/group_{idx}"] = float(group["lr"])

    if optimizer.param_groups:
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])

    return metrics
```

Chapter 3 may later use separate groups for encoder and predictor. If so, group-specific LR logging becomes useful.

---

## 3.3.16 Diagnostic Collection Function

The training step from Chapter 2 returned a fairly complete set of logs every step.

In Chapter 3, make diagnostics cadence-aware.

Create a helper in `training.py` or `engine/logging.py`.

```python
from jepa_world_model.diagnostics import (
    jepa_diagnostics,
    mask_stats,
    parameter_distance,
    prediction_target_metrics,
    representation_report,
)
```

The exact imports may differ depending on where `parameter_distance` lives. In Chapter 2 it was in `ema.py`.

A cadence-aware collector:

```python
def collect_jepa_metrics(
    *,
    pred_repr: torch.Tensor,
    target_repr: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    num_patches: int,
    model,
    step: int,
    cfg,
    loss_value: float,
    ema_tau: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "loss": loss_value,
        "ema_tau": ema_tau,
    }

    if should_log(step, cfg.diagnostics.basic_every_steps):
        metrics.update(
            prediction_target_metrics(
                pred_repr,
                target_repr,
            )
        )

    if should_log(step, cfg.diagnostics.representation_every_steps):
        metrics.update(
            representation_report(
                pred_repr,
                prefix="pred",
            )
        )
        metrics.update(
            representation_report(
                target_repr,
                prefix="target",
            )
        )

    if should_log(step, cfg.diagnostics.mask_check_every_steps):
        metrics.update(
            mask_stats(
                context_indices=context_indices,
                target_indices=target_indices,
                num_patches=num_patches,
            )
        )

    if should_log(step, cfg.diagnostics.parameter_distance_every_steps):
        from jepa_world_model.ema import parameter_distance

        metrics.update(
            parameter_distance(
                online=model.online_encoder,
                target=model.target_encoder,
            )
        )

    return metrics
```

This version assumes `representation_report` includes covariance and effective-rank metrics. If covariance is expensive, split `representation_report` into cheaper and expensive subreports.

A cleaner split is:

```text
basic representation metrics:
    norms
    std
    dead dims

expensive representation metrics:
    covariance
    effective rank
```

That refactor can happen as the implementation matures.

---

## 3.3.17 Splitting Cheap and Expensive Representation Reports

The Chapter 2 `representation_report` bundled several metrics together. For Chapter 3, split it.

In `diagnostics.py`, define:

```python
@torch.no_grad()
def cheap_representation_report(
    z: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    logs: dict[str, float] = {}

    logs.update(
        norm_metrics(
            z,
            prefix=prefix,
        )
    )

    logs.update(
        variance_metrics(
            z,
            prefix=prefix,
        )
    )

    return logs
```

Expensive report:

```python
@torch.no_grad()
def expensive_representation_report(
    z: torch.Tensor,
    prefix: str,
    compute_covariance: bool = True,
    compute_effective_rank: bool = True,
) -> dict[str, float]:
    logs: dict[str, float] = {}

    if compute_covariance:
        logs.update(
            covariance_metrics(
                z,
                prefix=prefix,
            )
        )

    if compute_effective_rank:
        logs.update(
            rank_metrics(
                z,
                prefix=prefix,
            )
        )

    return logs
```

Then the cadence logic becomes:

```python
if should_log(step, cfg.diagnostics.representation_every_steps):
    metrics.update(cheap_representation_report(pred_repr, "pred"))
    metrics.update(cheap_representation_report(target_repr, "target"))

if should_log(step, cfg.diagnostics.covariance_every_steps):
    metrics.update(
        expensive_representation_report(
            pred_repr,
            "pred",
            compute_covariance=cfg.diagnostics.compute_covariance,
            compute_effective_rank=cfg.diagnostics.compute_effective_rank,
        )
    )
    metrics.update(
        expensive_representation_report(
            target_repr,
            "target",
            compute_covariance=cfg.diagnostics.compute_covariance,
            compute_effective_rank=cfg.diagnostics.compute_effective_rank,
        )
    )
```

This avoids computing eigenvalues every logging step.

---

## 3.3.18 Rank-Zero Logging

Once Fabric or distributed training enters, only rank zero should write run-level logs unless metrics are explicitly reduced.

The logger should support an `enabled` flag.

```python
class MetricLogger:
    def __init__(
        self,
        sinks: list[MetricSink],
        enabled: bool = True,
    ):
        self.sinks = sinks
        self.enabled = enabled

    def write(
        self,
        metrics: dict[str, Any],
        step: int,
    ) -> None:
        if not self.enabled:
            return

        for sink in self.sinks:
            sink.write(metrics, step=step)

    def close(self) -> None:
        if not self.enabled:
            return

        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close is not None:
                close()
```

With Fabric later:

```python
metric_logger = build_metric_logger(
    cfg,
    run_paths,
    enabled=fabric.is_global_zero,
)
```

For now, plain PyTorch uses:

```python
enabled=True
```

Do not let every distributed worker write to the same JSONL file.

---

## 3.3.19 Updating `experiments/train.py`

Update the training script from Section 3.2.

Imports:

```python
from jepa_world_model.engine.logging import (
    StepTimer,
    build_metric_logger,
    cuda_memory_metrics,
    format_console_metrics,
    optimizer_group_lr_metrics,
    runtime_metrics,
)
```

Build logger:

```python
metric_logger = build_metric_logger(
    cfg=cfg,
    run_paths=run_paths,
)
```

Create timer:

```python
timer = StepTimer()
```

Inside loop:

```python
step_seconds = timer.tick()

logs = train_step(...)
```

Augment logs:

```python
logs.update(
    optimizer_group_lr_metrics(optimizer)
)

logs.update(
    runtime_metrics(
        step_seconds=step_seconds,
        batch_size=images.size(0),
    )
)

logs.update(
    cuda_memory_metrics()
)
```

Write:

```python
if global_step % cfg.logging.log_every_steps == 0:
    row = {
        "step": global_step,
        "epoch": epoch,
        **logs,
    }

    metric_logger.write(
        row,
        step=global_step,
    )

    progress.set_postfix_str(
        format_console_metrics(logs)
    )
```

Close logger at the end:

```python
metric_logger.close()
```

In a real script, wrap training in `try/finally`:

```python
try:
    train(...)
finally:
    metric_logger.close()
```

so TensorBoard and W&B sinks close cleanly.

---

## 3.3.20 Metric Rows

A metric row in `metrics.jsonl` should look like:

```json
{
  "step": 100,
  "epoch": 0,
  "loss": 0.1812,
  "lr": 0.0005,
  "grad_norm": 1.732,
  "ema_tau": 0.9964,
  "pred_target/cosine": 0.244,
  "target/std_mean": 0.612,
  "target/effective_rank": 42.8,
  "mask/overlap_fraction": 0.0,
  "time/step_seconds": 0.083,
  "time/samples_per_second": 1542.1
}
```

Not every row has every field. Expensive metrics appear only at their cadence.

That is fine. Downstream analysis should handle missing values.

---

## 3.3.21 Evaluation Metrics

Evaluation metrics should use the same metric stream.

Example after epoch 10:

```python
metric_logger.write(
    {
        "epoch": epoch,
        "eval/encoder": "online",
        "knn/accuracy": knn_acc,
        "linear_probe/train_acc": train_acc,
        "linear_probe/val_acc": val_acc,
    },
    step=global_step,
)
```

If evaluating both encoders, use separate names:

```text
online/knn/accuracy
target/knn/accuracy
online/linear_probe/val_acc
target/linear_probe/val_acc
```

or include:

```text
eval/encoder
```

as a categorical field.

For TensorBoard, scalar names are easier if encoder is in the metric name:

```text
online/linear_probe/val_acc
target/linear_probe/val_acc
```

---

## 3.3.22 Artifact Metadata

When writing artifacts, also write small metadata JSON files.

Example:

```text
artifacts/
├── retrieval_epoch_0010.png
└── retrieval_epoch_0010.json
```

The JSON might contain:

```json
{
  "epoch": 10,
  "step": 12500,
  "encoder": "online",
  "checkpoint": "checkpoints/epoch_0010.pt",
  "dataset": "stl10"
}
```

This prevents ambiguity later.

A helper:

```python
def write_artifact_metadata(
    path: str | Path,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(metadata),
            f,
            indent=2,
        )
```

Use this whenever writing retrieval plots, mask examples, or probe summaries.

---

## 3.3.23 Tests

Create:

```text
tests/test_logging.py
```

Add:

```python
from pathlib import Path

import torch

from jepa_world_model.configs.schema import (
    JEPAExperimentConfig,
    LoggingConfig,
    RuntimeConfig,
)
from jepa_world_model.engine.logging import (
    JSONLMetricWriter,
    MetricLogger,
    StepTimer,
    make_json_serializable,
    read_jsonl_metrics,
    should_log,
)
from jepa_world_model.engine.run_context import initialize_run


def test_should_log():
    assert should_log(0, 10)
    assert not should_log(1, 10)
    assert should_log(10, 10)


def test_make_json_serializable_scalar_tensor():
    row = make_json_serializable(
        {
            "loss": torch.tensor(1.25),
            "name": "test",
        }
    )

    assert row["loss"] == 1.25
    assert row["name"] == "test"


def test_jsonl_metric_writer_roundtrip(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"

    writer = JSONLMetricWriter(path)

    writer.write(
        {
            "loss": 1.0,
        },
        step=0,
    )

    writer.write(
        {
            "loss": 0.9,
        },
        step=1,
    )

    rows = read_jsonl_metrics(path)

    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert rows[1]["loss"] == 0.9


def test_metric_logger_disabled_does_not_write(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"

    sink = JSONLMetricWriter(path)
    logger = MetricLogger(
        sinks=[sink],
        enabled=False,
    )

    logger.write(
        {
            "loss": 1.0,
        },
        step=0,
    )

    assert not path.exists()


def test_metric_logger_enabled_writes(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"

    sink = JSONLMetricWriter(path)
    logger = MetricLogger(
        sinks=[sink],
        enabled=True,
    )

    logger.write(
        {
            "loss": 1.0,
        },
        step=0,
    )

    rows = read_jsonl_metrics(path)

    assert len(rows) == 1
    assert rows[0]["loss"] == 1.0


def test_step_timer_first_tick_none():
    timer = StepTimer()

    assert timer.tick() is None
```

If testing `build_metric_logger`, disable TensorBoard and W&B:

```python
def test_build_jsonl_metric_logger(tmp_path: Path):
    cfg = JEPAExperimentConfig(
        logging=LoggingConfig(
            jsonl=True,
            tensorboard=False,
            wandb=False,
        ),
        runtime=RuntimeConfig(
            run_dir=str(tmp_path),
        ),
    )

    run_paths = initialize_run(
        cfg,
        run_name="logging_test",
    )

    from jepa_world_model.engine.logging import build_metric_logger

    logger = build_metric_logger(
        cfg=cfg,
        run_paths=run_paths,
    )

    logger.write(
        {
            "loss": 1.0,
        },
        step=0,
    )

    rows = read_jsonl_metrics(run_paths.metrics_path)

    assert len(rows) == 1
```

Run:

```bash
uv run pytest tests/test_logging.py
```

---

## 3.3.24 Practical Invariants

After this section, logging should satisfy:

```text
JSONL metrics are always available unless explicitly disabled
TensorBoard/W&B are optional sinks
only scalar metrics are written
console logs are compact
rank-zero-only logging is supported
runtime metrics are recorded
diagnostic cadence is configurable
expensive diagnostics are throttled
```

For JEPA specifically:

```text
mask overlap is logged
target representation variance is logged
effective rank is logged periodically
EMA distance is logged periodically
prediction-target cosine is logged
evaluation metrics share the same metric stream
```

---

## 3.3.25 Summary

This section formalized metrics and logging cadence for JEPA training.

We added:

```text
metric taxonomy
diagnostic cadence
early-warning metrics
JSONL sink
TensorBoard sink
optional W&B sink
multi-sink MetricLogger
rank-zero logging support
console formatting
runtime throughput metrics
CUDA memory metrics
gradient norm metrics
learning-rate metrics
artifact metadata helper
tests
```

The central design decision is:

> JEPA diagnostics are not optional extras. They are part of the training harness.

The next section implements checkpointing and resume semantics, with special attention to target encoder state and EMA schedule consistency.

---

## References and Further Reading

- PyTorch TensorBoard utilities:
  <https://pytorch.org/docs/stable/tensorboard.html>

- Weights & Biases documentation:
  <https://docs.wandb.ai/>

- PyTorch CUDA memory management:
  <https://pytorch.org/docs/stable/notes/cuda.html#memory-management>

- PyTorch gradient clipping:
  <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>

- JSON Lines format:
  <https://jsonlines.org/>
