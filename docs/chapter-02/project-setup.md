# 2.1 Project Setup

This section defines the development assumptions for the implementation chapters.

We assume the reader already knows how to:

- create a repository,
- create files and folders,
- use a virtual environment,
- manage Git,
- create `.gitignore`,
- run Python scripts,
- work with local development tools.

We will therefore not spend time on basic repository bootstrapping. Instead, this section establishes the project conventions used throughout the tutorial.

The setup has three guiding assumptions:

1. use **`uv`** for Python dependency management,
2. use **marimo notebooks** for local debugging and visualization,
3. use **cloud GPU runs**, for example RunPod, for larger experiments.

---

## 2.1.1 Runtime Assumptions

The code in this tutorial assumes a modern Python and PyTorch stack.

At the time of writing, the latest stable Python release is **Python 3.14.5**, released on May 10, 2026. The latest stable PyTorch release is **PyTorch 2.12.0**. Newer versions may also work, but the tutorial assumes this general generation of tooling.

Recommended baseline:

```text
Python: 3.14.x
PyTorch: 2.12.x
Package manager: uv
Notebook/debugging environment: marimo
CUDA: whatever is supported by your PyTorch install / cloud image
```

For cloud experiments, prefer a PyTorch/CUDA image that already has a working GPU stack. Then use `uv` to sync the project dependencies.

---

## 2.1.2 Dependency Management with `uv`

This tutorial uses `uv` exclusively.

The repository includes a bootstrap script that creates `.venv`, syncs the locked dependencies, and installs Git hooks:

```bash
./setup.py
```

After setup, activate the environment:

```bash
source .venv/bin/activate
```

Unless a command explicitly manages dependencies with `uv`, the rest of this chapter assumes that activated environment.

Use:

```bash
uv add package-name
```

not:

```bash
pip install package-name
```

Use:

```bash
python ...
```

inside the activated project environment.

Use:

```bash
uv sync
```

to recreate the project environment from `pyproject.toml` and `uv.lock`.

The project should commit:

```text
pyproject.toml
uv.lock
```

The project should not commit:

```text
.venv/
data/
runs/
checkpoints/
```

Those ignored files can be managed manually according to your normal Git workflow.

---

## 2.1.3 Minimal Dependency Set

The implementation chapters assume the following packages:

```bash
uv add \
  torch \
  torchvision \
  torchaudio \
  numpy \
  matplotlib \
  tqdm \
  einops \
  scikit-learn \
  pillow \
  rich
```

Development, documentation, and debugging tools:

```bash
uv add --dev \
  pytest \
  ruff \
  ty \
  marimo \
  tensorboard \
  mkdocs-material \
  mkdocs-glightbox \
  pymdown-extensions
```

Optional later:

```bash
uv add --dev \
  wandb \
  hydra-core \
  omegaconf
```

We will begin with plain dataclasses and simple scripts. Hydra, W&B, and more advanced experiment management belong in the research-engineering chapter.

---

## 2.1.4 Expected Project Shape

The exact setup is flexible, but the tutorial assumes this broad structure:

```text
jepa-world-model-tutorial/
├── docs/
│   ├── index.md
│   ├── chapter-01/
│   ├── chapter-02/
│   └── references.md
├── src/
│   └── jepa_world_model/
├── notebooks/
│   └── *.py
├── experiments/
├── configs/
├── tests/
├── pyproject.toml
└── uv.lock
```

The important convention is:

```text
docs/        explanation
src/         reusable implementation
notebooks/   marimo notebooks stored as .py files
experiments/ executable training scripts
tests/       unit and smoke tests
configs/     later experiment configs
```

Core model code should live in `src/`, not in marimo notebooks.

A good rule:

```text
If code is reused twice, move it into src/.
```

---

## 2.1.5 Why marimo Instead of IPython/Jupyter Notebooks

This tutorial uses **marimo** rather than traditional IPython/Jupyter notebooks.

The reasons are practical:

- marimo notebooks are stored as `.py` files,
- they are Git-friendly,
- they are reactive,
- dependent cells update automatically,
- notebooks can be run as scripts,
- notebooks can be served as interactive apps,
- they avoid much of the hidden-state problem common in traditional notebooks.

The official marimo site describes marimo as a reactive Python notebook stored as reproducible, Git-friendly Python, and the docs support installation and execution directly with `uv`.

For this project, marimo notebooks are for:

- visualizing patches,
- visualizing masks,
- checking tensor shapes,
- inspecting a single batch,
- overfitting a tiny batch,
- plotting representation diagnostics,
- debugging nearest-neighbor retrieval.

They are not the source of truth for reusable model code.

---

## 2.1.6 marimo Workflow

Create marimo notebooks as Python files, for example:

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

Start the marimo editor:

```bash
marimo edit notebooks/00_debug_environment.py
```

Run a marimo notebook as a script:

```bash
marimo run notebooks/02_debug_encoder.py
```

Launch the marimo tutorial:

```bash
marimo tutorial intro
```

A notebook should import from the package:

```python
from jepa_world_model.config import MinimalJEPAConfig
from jepa_world_model.utils import get_device, seed_everything
```

It should not define the core encoder, predictor, or training loop.

---

## 2.1.7 Local Debugging Workflow

The local environment is for fast iteration.

Use marimo notebooks for:

- visual inspection,
- interactive debugging,
- tensor-shape checks,
- one-batch experiments,
- plotting diagnostics,
- sanity-checking masks.

Use scripts for:

- deterministic smoke tests,
- short training runs,
- unit-testable functionality,
- reproducible experiments.

A typical local workflow is:

```text
marimo notebook:
    inspect one batch, patches, masks, shapes

unit tests:
    verify patchify, masks, positional embeddings

tiny script:
    overfit one or two batches

local GPU:
    run a short training experiment

cloud GPU:
    run the real experiment
```

The local command pattern should be:

```bash
pytest
marimo edit notebooks/01_visualize_patches_and_masks.py
python experiments/train_minimal.py --preset local_debug
```

---

## 2.1.8 Cloud Experiment Workflow

Large experiments should run on a cloud GPU provider such as RunPod.

The intended pattern is:

```text
local:
    implement and debug

git:
    push code and lockfile

cloud:
    clone repo
    uv sync
    run experiment script
    save logs/checkpoints to persistent storage
```

A typical cloud command sequence:

```bash
cd /workspace
git clone <repo-url>
cd jepa-world-model-tutorial

uv sync
source .venv/bin/activate
python experiments/train_cloud.py --dataset stl10
```

The training code should not care whether it runs locally or remotely. Only the config should change.

Local paths might be:

```text
data/
runs/
checkpoints/
```

Cloud paths might be:

```text
/workspace/data/
/workspace/runs/
/workspace/checkpoints/
```

This is why paths belong in config, not hard-coded inside training code.

---

## 2.1.9 Minimal Configuration

For Chapter 2, a dataclass is enough.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class MinimalJEPAConfig:
    image_size: int = 96
    patch_size: int = 8
    in_channels: int = 3

    encoder_dim: int = 192
    encoder_depth: int = 6
    encoder_heads: int = 3
    mlp_ratio: float = 4.0

    predictor_dim: int = 128
    predictor_depth: int = 3
    predictor_heads: int = 4

    num_target_blocks: int = 4
    target_block_height: int = 3
    target_block_width: int = 3
    context_ratio: float = 0.6

    batch_size: int = 128
    num_workers: int = 4

    epochs: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 0.05

    ema_tau_base: float = 0.996
    ema_tau_final: float = 1.0

    data_dir: str = "data"
    run_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"

    seed: int = 42
```

A tiny local-debug config can be defined separately:

```python
def local_debug_config() -> MinimalJEPAConfig:
    return MinimalJEPAConfig(
        image_size=32,
        patch_size=4,
        encoder_dim=64,
        encoder_depth=2,
        encoder_heads=4,
        predictor_dim=64,
        predictor_depth=1,
        predictor_heads=4,
        batch_size=16,
        epochs=2,
        num_workers=0,
    )
```

A cloud config can simply scale the same fields:

```python
def cloud_run_config() -> MinimalJEPAConfig:
    return MinimalJEPAConfig(
        image_size=96,
        patch_size=8,
        encoder_dim=384,
        encoder_depth=8,
        encoder_heads=6,
        predictor_dim=256,
        predictor_depth=4,
        predictor_heads=8,
        batch_size=256,
        epochs=300,
        num_workers=8,
        data_dir="/workspace/data",
        run_dir="/workspace/runs",
        checkpoint_dir="/workspace/checkpoints",
    )
```

Later, Chapter 3 will replace this with a proper configuration system.

---

## 2.1.10 Device Selection

Use a small helper for device selection:

```python
import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
```

For debugging, CPU is often useful even on a GPU machine:

```python
device = torch.device("cpu")
```

For actual training, prefer CUDA when available.

---

## 2.1.11 Reproducibility Helper

Use a single seeding function:

```python
import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
```

This does not guarantee perfect determinism across all GPU kernels, but it gives us a stable baseline for debugging.

---

## 2.1.12 Logging Convention

In Chapter 2, training steps return plain dictionaries:

```python
{
    "loss": 0.124,
    "ema_tau": 0.997,
    "pred_target_cosine": 0.41,
    "target/std_mean": 0.58,
    "pred/std_mean": 0.43,
}
```

This keeps the minimal implementation simple.

A console formatter is enough:

```python
def format_logs(logs: dict[str, float]) -> str:
    parts = []

    for key, value in logs.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")

    return " | ".join(parts)
```

Later, we will add TensorBoard or W&B.

---

## 2.1.13 Smoke Tests

The first tests should verify only the assumptions we rely on.

Example:

```python
from jepa_world_model.config import MinimalJEPAConfig
from jepa_world_model.utils import get_device, seed_everything


def test_config_is_valid():
    cfg = MinimalJEPAConfig()

    assert cfg.image_size > 0
    assert cfg.patch_size > 0
    assert cfg.image_size % cfg.patch_size == 0


def test_seed_everything_runs():
    seed_everything(42)


def test_device_selection_runs():
    device = get_device()
    assert device.type in {"cpu", "cuda", "mps"}
```

Run:

```bash
pytest
```

This is enough for setup. The meaningful tests begin with patchification and mask sampling.

---

## 2.1.14 Development Convention

For every implementation section, we will follow the same pattern:

```text
explain the concept
        ↓
define tensor shapes
        ↓
implement the module
        ↓
write a small test
        ↓
debug visually in marimo if useful
        ↓
use it in the training script
```

This keeps the tutorial synchronized with the code.

The implementation order is:

```text
1. patchify.py
2. position.py
3. masks.py
4. vit.py
5. ema.py
6. predictor.py
7. losses.py
8. diagnostics.py
9. model.py
10. data.py
11. checkpointing.py
12. evaluation.py
13. train_minimal.py
14. train_cloud.py
```

---

## 2.1.15 Summary

The project setup is intentionally simple.

We assume:

- `uv` for dependency and environment management,
- modern Python and PyTorch,
- marimo notebooks for local debugging and visualization,
- cloud GPU runs for larger experiments,
- plain dataclass configs for Chapter 2,
- simple log dictionaries before adding experiment infrastructure.

The next section begins the actual implementation: image patchification.

---

## References and Further Reading

- Astral, **uv — Working on projects**.
  <https://docs.astral.sh/uv/guides/projects/>

- marimo, **Documentation**.
  <https://docs.marimo.io/>

- marimo, **Installation**.
  <https://docs.marimo.io/getting_started/installation/>

- Python, **Python 3.14.5 Release**.
  <https://www.python.org/downloads/release/python-3145/>

- PyTorch, **Get Started Locally**.
  <https://pytorch.org/get-started/locally/>

- PyTorch, **Releases**.
  <https://github.com/pytorch/pytorch/releases>

- RunPod, **Documentation Overview**.
  <https://docs.runpod.io/overview>

- MkDocs Material, **Getting Started**.
  <https://squidfunk.github.io/mkdocs-material/getting-started/>
