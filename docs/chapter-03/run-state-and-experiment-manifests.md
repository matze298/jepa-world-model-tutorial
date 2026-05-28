# 3.2 Run State and Experiment Manifests

Section 3.1 made experiments config-driven.

The next step is to make runs auditable.

For JEPA, a run directory is not just a place to dump checkpoints. It is the record that lets us answer questions such as:

```text
Was the target encoder restored correctly?
Which EMA schedule position was used after resume?
Were mask settings identical across ablations?
Did representation collapse begin before probe accuracy dropped?
Was evaluation run on the online encoder or the target encoder?
Was the cloud run using the intended precision and device count?
```

This section defines a run-state layer that records:

```text
resolved config
execution metadata
training metrics
checkpoints
representation artifacts
human notes
```

The emphasis is not on explaining why run directories exist. The emphasis is on what state a JEPA experiment must preserve and expose.

---

## 3.2.1 Target Run Layout

A run directory should look like:

```text
runs/
└── ijepa_stl10_base_2026-05-25_21-30-10_a1b2c3d/
    ├── config.yaml
    ├── manifest.yaml
    ├── metrics.jsonl
    ├── notes.md
    ├── checkpoints/
    │   ├── last.pt
    │   ├── epoch_0010.pt
    │   └── epoch_0020.pt
    ├── artifacts/
    │   ├── retrieval_epoch_0010.png
    │   ├── masks_epoch_0010.png
    │   └── probe_epoch_0010.json
    └── logs/
        └── tensorboard/
```

The important distinction is:

```text
config.yaml:
    intended experiment settings

manifest.yaml:
    actual execution metadata

metrics.jsonl:
    metric trajectory

checkpoints/:
    resumable training state

artifacts/:
    derived outputs and qualitative diagnostics

logs/:
    sink-specific logs, such as TensorBoard
```

For Chapter 3, `metrics.jsonl` is the canonical metric stream. TensorBoard and W&B can mirror it, but they should not be the only record.

---

## 3.2.2 JEPA Run State

A valid JEPA training run must preserve more than a generic supervised run.

The core state is:

```text
online encoder
target encoder
predictor
optimizer
scheduler
epoch
global step
EMA schedule position
config
```

Depending on reproducibility requirements, also preserve:

```text
torch RNG state
CUDA RNG state
Python RNG state
NumPy RNG state
dataloader sampler state
```

The non-negotiable JEPA-specific part is:

```text
target encoder state
global step
EMA schedule position
```

The target encoder defines the target representation distribution. Resuming without it is not a faithful resume.

The EMA schedule is typically a function of global step:

```python
ema_tau = cosine_ema_tau(
    step=global_step,
    total_steps=total_steps,
    tau_base=cfg.ema.tau_base,
    tau_final=cfg.ema.tau_final,
)
```

So `global_step` is part of the EMA state.

---

## 3.2.3 Training Checkpoint vs Representation Artifact

Do not conflate training checkpoints with exported encoders.

A **training checkpoint** is for resuming pretraining.

It should contain:

```text
model.state_dict()
optimizer.state_dict()
scheduler.state_dict()
epoch
global_step
config
RNG state, if exact resume is required
```

Since `model.state_dict()` includes:

```text
online_encoder
target_encoder
predictor
```

it preserves JEPA training state.

A **representation artifact** is for downstream use.

It should usually contain:

```text
online_encoder.state_dict()
model config
preprocessing metadata
image size
patch size
encoder dimension
feature pooling convention
```

It usually should not contain:

```text
target encoder
predictor
optimizer
scheduler
```

The distinction matters because a checkpoint optimized for resume and an artifact optimized for downstream use have different consumers.

---

## 3.2.4 Run Manifest

The manifest captures the actual execution context.

It should include:

```text
run name
created timestamp
experiment name
tags
command
Git commit
Git branch
Git dirty state
host
process ID
Python version
PyTorch version
CUDA availability
CUDA version
GPU count
GPU name
precision
device strategy
```

The manifest is not a replacement for the config. It records what environment executed the config.

A typical `manifest.yaml` might contain:

```yaml
run_name: ijepa_stl10_base_2026-05-25_21-30-10_a1b2c3d
created_at: "2026-05-25T21:30:10"
experiment_name: ijepa_stl10_base
tags:
  - ijepa
  - stl10
  - baseline

command:
  argv:
    - experiments/train.py
    - --config
    - configs/ijepa_stl10_base.yaml
  command: "experiments/train.py --config configs/ijepa_stl10_base.yaml"

git:
  commit: a1b2c3d4...
  short_commit: a1b2c3d
  branch: main
  dirty: false

environment:
  python: "3.14.x"
  torch_version: "2.12.x"
  cuda_available: true
  cuda_version: "..."
  cuda_device_count: 1
  cuda_device_name: "..."
  hostname: "..."
  platform: "..."

runtime:
  precision: bf16-mixed
  fabric_accelerator: cuda
  fabric_devices: auto
  fabric_strategy: auto
```

The exact fields can evolve, but the manifest should be machine-readable.

---

## 3.2.5 Implementing `run_context.py`

Create:

```text
src/jepa_world_model/engine/run_context.py
```

Start with:

```python
from __future__ import annotations

import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from jepa_world_model.configs.loader import save_config
from jepa_world_model.configs.schema import JEPAExperimentConfig
```

Define run paths:

```python
@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    artifacts: Path
    logs: Path

    config_path: Path
    manifest_path: Path
    metrics_path: Path
    notes_path: Path
```

The rest of the training code should receive `RunPaths` rather than constructing paths manually.

---

## 3.2.6 Run Naming

Use readable, sortable run names:

```python
def timestamp_string() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
```

Git helper:

```python
def run_command(
    command: list[str],
    cwd: str | Path | None = None,
) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None

    return result.stdout.strip()
```

Git metadata:

```python
def git_commit_hash() -> str | None:
    return run_command(["git", "rev-parse", "HEAD"])


def git_short_commit_hash() -> str | None:
    return run_command(["git", "rev-parse", "--short", "HEAD"])


def git_branch() -> str | None:
    return run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])


def git_is_dirty() -> bool | None:
    output = run_command(["git", "status", "--porcelain"])

    if output is None:
        return None

    return len(output) > 0
```

Run name:

```python
def create_run_name(
    cfg: JEPAExperimentConfig,
) -> str:
    short_commit = git_short_commit_hash()
    stamp = timestamp_string()

    if short_commit is None:
        return f"{cfg.experiment.name}_{stamp}"

    return f"{cfg.experiment.name}_{stamp}_{short_commit}"
```

This gives names such as:

```text
ijepa_stl10_base_2026-05-25_21-30-10_a1b2c3d
```

---

## 3.2.7 Environment Metadata

Add:

```python
def environment_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()

    metadata: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "mps_available": torch.backends.mps.is_available(),
    }

    if cuda_available:
        metadata.update(
            {
                "cuda_version": torch.version.cuda,
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
            }
        )

    return metadata
```

Command metadata:

```python
def command_metadata() -> dict[str, Any]:
    return {
        "argv": sys.argv,
        "command": " ".join(sys.argv),
    }
```

Runtime metadata derived from config:

```python
def runtime_metadata(
    cfg: JEPAExperimentConfig,
) -> dict[str, Any]:
    return {
        "precision": cfg.training.precision,
        "device": cfg.runtime.device,
        "fabric_accelerator": cfg.runtime.fabric_accelerator,
        "fabric_devices": cfg.runtime.fabric_devices,
        "fabric_strategy": cfg.runtime.fabric_strategy,
    }
```

---

## 3.2.8 Manifest Object

Define:

```python
@dataclass(frozen=True)
class RunManifest:
    run_name: str
    created_at: str
    experiment_name: str
    tags: list[str]
    notes: str | None
    command: dict[str, Any]
    git: dict[str, Any]
    environment: dict[str, Any]
    runtime: dict[str, Any]
```

Build it:

```python
def build_manifest(
    cfg: JEPAExperimentConfig,
    run_name: str,
) -> RunManifest:
    return RunManifest(
        run_name=run_name,
        created_at=datetime.now().isoformat(),
        experiment_name=cfg.experiment.name,
        tags=cfg.experiment.tags,
        notes=cfg.experiment.notes,
        command=command_metadata(),
        git={
            "commit": git_commit_hash(),
            "short_commit": git_short_commit_hash(),
            "branch": git_branch(),
            "dirty": git_is_dirty(),
        },
        environment=environment_metadata(),
        runtime=runtime_metadata(cfg),
    )
```

Save it:

```python
def save_manifest(
    manifest: RunManifest,
    path: str | Path,
) -> None:
    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            asdict(manifest),
            f,
            sort_keys=False,
        )
```

---

## 3.2.9 Creating Run Paths

Create directories explicitly:

```python
def create_run_paths(
    cfg: JEPAExperimentConfig,
    run_name: str | None = None,
    exist_ok: bool = False,
) -> RunPaths:
    if run_name is None:
        run_name = create_run_name(cfg)

    root = Path(cfg.runtime.run_dir) / run_name

    checkpoints = root / "checkpoints"
    artifacts = root / "artifacts"
    logs = root / "logs"

    for path in [root, checkpoints, artifacts, logs]:
        path.mkdir(
            parents=True,
            exist_ok=exist_ok,
        )

    return RunPaths(
        root=root,
        checkpoints=checkpoints,
        artifacts=artifacts,
        logs=logs,
        config_path=root / "config.yaml",
        manifest_path=root / "manifest.yaml",
        metrics_path=root / "metrics.jsonl",
        notes_path=root / "notes.md",
    )
```

For fresh runs, use:

```python
exist_ok=False
```

For true resume into an existing directory, use explicit resume logic rather than silently reusing a path.

---

## 3.2.10 Notes File

Create a human-editable notes file:

```python
def create_notes_file(
    cfg: JEPAExperimentConfig,
    path: str | Path,
) -> None:
    path = Path(path)

    notes = cfg.experiment.notes or ""

    content = f"""# Run Notes

## Experiment

{cfg.experiment.name}

## Tags

{", ".join(cfg.experiment.tags)}

## Initial Notes

{notes}

## Observations

-

## Issues

-

## Follow-ups

-
"""

    path.write_text(
        content,
        encoding="utf-8",
    )
```

This is not a replacement for metrics. It is where subjective observations belong:

```text
retrieval looked object-level by epoch 40
probe improved despite flat JEPA loss
cloud run resumed after preemption
mask artifacts look too easy
```

---

## 3.2.11 Run Initialization

Combine everything:

```python
def initialize_run(
    cfg: JEPAExperimentConfig,
    run_name: str | None = None,
) -> RunPaths:
    paths = create_run_paths(
        cfg=cfg,
        run_name=run_name,
        exist_ok=False,
    )

    save_config(
        cfg,
        paths.config_path,
    )

    manifest = build_manifest(
        cfg=cfg,
        run_name=paths.root.name,
    )

    save_manifest(
        manifest,
        paths.manifest_path,
    )

    create_notes_file(
        cfg=cfg,
        path=paths.notes_path,
    )

    return paths
```

Usage:

```python
run_paths = initialize_run(cfg)

print(f"Run directory: {run_paths.root}")
```

This should be called once at the beginning of a fresh run.

---

## 3.2.12 Resume Path Semantics

Resume should be explicit.

There are two valid patterns.

### Pattern A: Continue the same run

```text
runtime.resume_from: runs/old_run/checkpoints/last.pt
runtime.run_dir: runs
```

The code infers the run root from the checkpoint path and appends to:

```text
runs/old_run/metrics.jsonl
```

Use this when continuing an interrupted run.

### Pattern B: Branch from a checkpoint

```text
runtime.resume_from: runs/old_run/checkpoints/last.pt
experiment.name: ijepa_stl10_branch_lr_low
```

The code creates a new run directory and records the source checkpoint in the manifest.

Use this when intentionally changing config or continuing as a new experiment.

Do not silently mix these semantics.

A minimal helper can classify checkpoint paths later:

```python
def infer_run_root_from_checkpoint(
    checkpoint_path: str | Path,
) -> Path:
    path = Path(checkpoint_path).resolve()

    if path.parent.name != "checkpoints":
        raise ValueError(
            "Expected checkpoint path inside a checkpoints/ directory."
        )

    return path.parent.parent
```

Full resume logic will be implemented in the checkpointing section.

For now, the run context should support fresh run initialization cleanly.

---

## 3.2.13 JSONL Metric Stream

Metrics belong in:

```text
metrics.jsonl
```

Create:

```text
src/jepa_world_model/engine/logging.py
```

A minimal JSONL sink:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JSONLMetricWriter:
    def __init__(
        self,
        path: str | Path,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        metrics: dict[str, Any],
    ) -> None:
        row: dict[str, Any] = {}

        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()

            row[key] = value

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
```

Reader:

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

The next section will extend this into a multi-sink metric logger.

For now, JSONL gives us a canonical stream.

---

## 3.2.14 Integrating with `experiments/train.py`

Update the config-driven training script from Section 3.1.

Imports:

```python
from jepa_world_model.engine.logging import JSONLMetricWriter
from jepa_world_model.engine.run_context import initialize_run
```

Replace manual run directory creation:

```python
run_paths = initialize_run(cfg)
metric_writer = JSONLMetricWriter(run_paths.metrics_path)
```

Use run paths:

```python
checkpoint_dir = run_paths.checkpoints
```

When logging:

```python
if global_step % cfg.logging.log_every_steps == 0:
    row = {
        "step": global_step,
        "epoch": epoch,
        **logs,
    }

    metric_writer.write(row)

    progress.set_postfix_str(
        format_logs(logs)
    )
```

When saving checkpoints:

```python
save_checkpoint(
    path=run_paths.checkpoints / f"epoch_{epoch + 1:04d}.pt",
    model=model,
    optimizer=optimizer,
    cfg=cfg,
    step=global_step,
    epoch=epoch + 1,
)
```

Last checkpoint:

```python
save_checkpoint(
    path=run_paths.checkpoints / "last.pt",
    model=model,
    optimizer=optimizer,
    cfg=cfg,
    step=global_step,
    epoch=cfg.training.epochs,
)
```

Now every run produces:

```text
config.yaml
manifest.yaml
metrics.jsonl
notes.md
checkpoints/
artifacts/
logs/
```

---

## 3.2.15 Artifact Path Convention

Artifacts should be named by type and step or epoch.

Examples:

```text
artifacts/
├── masks_step_000000.png
├── retrieval_epoch_0010.png
├── probe_epoch_0010.json
├── rank_curve.png
└── final_summary.json
```

Add a helper:

```python
def artifact_path(
    run_paths: RunPaths,
    name: str,
) -> Path:
    return run_paths.artifacts / name
```

Usage:

```python
path = artifact_path(
    run_paths,
    f"retrieval_epoch_{epoch + 1:04d}.png",
)
```

Do not write artifacts directly into the run root.

Keep the root readable.

---

## 3.2.16 Cloud Path Requirements

For cloud runs, `runtime.run_dir` should point to persistent storage.

Example:

```yaml
runtime:
  run_dir: /workspace/runs
```

A cloud run should create:

```text
/workspace/runs/ijepa_stl10_cloud_.../
```

This directory should survive pod interruption.

A minimal cloud sanity check is:

```bash
touch /workspace/runs/test_write.txt
```

If that file would disappear when the pod stops, the run path is wrong.

The JEPA-specific reason this matters is that interrupted runs must resume with:

```text
target encoder state
optimizer state
global step
EMA schedule position
```

If the checkpoint is lost, the run is not recoverable.

---

## 3.2.17 Tests

Create:

```text
tests/test_run_context.py
```

Add:

```python
from pathlib import Path

import pytest

from jepa_world_model.configs.schema import (
    JEPAExperimentConfig,
    RuntimeConfig,
)
from jepa_world_model.engine.logging import (
    JSONLMetricWriter,
    read_jsonl_metrics,
)
from jepa_world_model.engine.run_context import (
    create_run_paths,
    infer_run_root_from_checkpoint,
    initialize_run,
)


def test_initialize_run_creates_expected_files(tmp_path: Path):
    cfg = JEPAExperimentConfig(
        runtime=RuntimeConfig(
            run_dir=str(tmp_path),
        )
    )

    paths = initialize_run(
        cfg,
        run_name="test_run",
    )

    assert paths.root.exists()
    assert paths.checkpoints.exists()
    assert paths.artifacts.exists()
    assert paths.logs.exists()
    assert paths.config_path.exists()
    assert paths.manifest_path.exists()
    assert paths.notes_path.exists()


def test_create_run_paths_refuses_existing_directory(tmp_path: Path):
    cfg = JEPAExperimentConfig(
        runtime=RuntimeConfig(
            run_dir=str(tmp_path),
        )
    )

    create_run_paths(
        cfg,
        run_name="duplicate",
        exist_ok=False,
    )

    with pytest.raises(FileExistsError):
        create_run_paths(
            cfg,
            run_name="duplicate",
            exist_ok=False,
        )


def test_metrics_writer_roundtrip(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"

    writer = JSONLMetricWriter(path)

    writer.write({"step": 0, "loss": 1.0})
    writer.write({"step": 1, "loss": 0.9})

    rows = read_jsonl_metrics(path)

    assert len(rows) == 2
    assert rows[0]["step"] == 0
    assert rows[1]["loss"] == 0.9


def test_infer_run_root_from_checkpoint(tmp_path: Path):
    run_root = tmp_path / "run"
    checkpoint_dir = run_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    checkpoint = checkpoint_dir / "last.pt"
    checkpoint.write_text("dummy", encoding="utf-8")

    inferred = infer_run_root_from_checkpoint(checkpoint)

    assert inferred == run_root.resolve()
```

The test imports `infer_run_root_from_checkpoint`, so include it in `run_context.py`:

```python
def infer_run_root_from_checkpoint(
    checkpoint_path: str | Path,
) -> Path:
    path = Path(checkpoint_path).resolve()

    if path.parent.name != "checkpoints":
        raise ValueError(
            "Expected checkpoint path inside a checkpoints/ directory."
        )

    return path.parent.parent
```

Run:

```bash
uv run pytest tests/test_run_context.py
```

---

## 3.2.18 Practical Invariants

After this section, a training run should satisfy:

```text
run directory is unique unless explicitly resuming
resolved config is saved before training
manifest is saved before training
metrics are appended to metrics.jsonl
checkpoints are written under checkpoints/
artifacts are written under artifacts/
cloud run_dir points to persistent storage
```

For JEPA specifically, the run structure should support verifying:

```text
checkpoint contains target encoder
checkpoint contains global_step
EMA schedule can resume at the correct step
diagnostics history is available
evaluation artifacts state which encoder was used
```

Some of these are implemented in later sections, but the directory structure already anticipates them.

---

## 3.2.19 Summary

This section defined run state and experiment manifests for JEPA training.

We added:

```text
RunPaths
run naming
Git metadata
environment metadata
runtime metadata
manifest.yaml
config snapshot
notes.md
metrics.jsonl
standard checkpoint/artifact/log paths
resume path semantics
tests
```

The key JEPA-specific design decision is that the run directory must support faithful reconstruction of:

```text
online encoder
target encoder
predictor
optimizer
scheduler
global step
EMA schedule position
diagnostic trajectory
```

The next section formalizes metric logging and diagnostic cadence.

---

## References and Further Reading

- Python `pathlib`:
  <https://docs.python.org/3/library/pathlib.html>

- Python `json`:
  <https://docs.python.org/3/library/json.html>

- Python `subprocess`:
  <https://docs.python.org/3/library/subprocess.html>

- PyYAML documentation:
  <https://pyyaml.org/wiki/PyYAMLDocumentation>

- PyTorch Saving and Loading Models:
  <https://pytorch.org/tutorials/beginner/saving_loading_models.html>
