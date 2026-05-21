# 2.10 Evaluation: k-NN and Linear Probe

Pretraining loss is not enough.

A JEPA model can reduce its latent prediction loss while still learning weak or collapsed representations. We therefore need evaluation methods that ask whether the learned encoder produces useful features.

In this section, we implement two basic evaluations:

1. **k-nearest-neighbor retrieval**
2. **linear probing**

Both use the encoder as a frozen feature extractor.

The goal is not to produce a full benchmark suite yet. The goal is to answer simple questions:

- Do similar images have nearby representations?
- Are labels linearly accessible from the learned features?
- Is the encoder doing better than random initialization?
- Did training collapse?
- Did the representation become useful beyond the JEPA loss?

---

## 2.10.1 What We Evaluate

After JEPA pretraining, the predictor is usually discarded.

The useful model is the encoder:

\[
z = f_\theta(x)
\]

For evaluation, we freeze the online encoder and use it to extract features.

Because our encoder normally accepts selected patch indices, we need a convention for evaluating a full image.

For an image with \(N\) patches, we pass all patch indices:

\[
\mathcal{I} = \{0, 1, \dots, N-1\}
\]

Then:

\[
z_{\mathcal{I}} = f_\theta(x, \mathcal{I})
\]

This gives patch-level representations:

```python
patch_repr.shape
# [B, N, D]
```

For image-level evaluation, we pool patch representations:

```python
image_repr = patch_repr.mean(dim=1)
```

So the image embedding is:

\[
z_{\mathrm{img}}
=
\frac{1}{N}
\sum_{i=1}^{N}
z_i
\]

This is simple and sufficient for Chapter 2.

Later, we can experiment with:

- attention pooling,
- class tokens,
- register tokens,
- multi-layer features,
- concatenated pooled features.

---

## 2.10.2 Evaluation Workflow

The evaluation workflow is:

```text
trained JEPA checkpoint
        ↓
load model
        ↓
freeze online encoder
        ↓
extract image representations
        ↓
run k-NN retrieval
        ↓
train linear probe
        ↓
compare to random encoder baseline
```

We will implement this in reusable utilities.

Create:

```text
src/jepa_world_model/evaluation.py
```

---

## 2.10.3 Full Patch Indices

The encoder needs patch indices.

For full-image evaluation, every image in the batch uses the same indices:

```python
[0, 1, 2, ..., N-1]
```

Add to `evaluation.py`:

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
```

Now add:

```python
def full_patch_indices(
    batch_size: int,
    num_patches: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Return all patch indices for each batch item.

    Returns:
        indices [B, N]
    """
    indices = torch.arange(
        num_patches,
        device=device,
        dtype=torch.long,
    )

    return indices.unsqueeze(0).expand(
        batch_size,
        -1,
    )
```

Example:

```python
indices = full_patch_indices(
    batch_size=4,
    num_patches=64,
    device=torch.device("cpu"),
)

indices.shape
# [4, 64]
```

---

## 2.10.4 Extracting Representations

We need a function that:

1. puts the encoder in eval mode,
2. disables gradients,
3. runs all patches through the encoder,
4. pools patch features,
5. returns features and labels.

Add:

```python
@torch.no_grad()
def extract_representations(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_patches: int,
    max_batches: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract pooled image representations from a frozen encoder.

    Args:
        encoder:
            Encoder with interface encoder(images, patch_indices).

        dataloader:
            DataLoader returning (images, labels).

        device:
            Device to run on.

        num_patches:
            Total number of image patches.

        max_batches:
            Optional limit for quick debugging.

    Returns:
        features:
            [num_examples, D]

        labels:
            [num_examples]
    """
    encoder.eval()

    features = []
    labels = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="extract")):
        if max_batches is not None and batch_idx >= max_batches:
            break

        images, y = batch
        images = images.to(device, non_blocking=True)

        indices = full_patch_indices(
            batch_size=images.size(0),
            num_patches=num_patches,
            device=device,
        )

        patch_repr = encoder(
            images=images,
            patch_indices=indices,
        )

        image_repr = patch_repr.mean(dim=1)

        features.append(image_repr.cpu())
        labels.append(y.cpu())

    features_tensor = torch.cat(features, dim=0)
    labels_tensor = torch.cat(labels, dim=0)

    return features_tensor, labels_tensor
```

This function is used by both k-NN and linear probing.

---

## 2.10.5 Feature Normalization

Many representation evaluations use normalized features.

For nearest-neighbor retrieval, cosine similarity is common:

\[
\mathrm{sim}(z_i, z_j)
=
\frac{z_i^\top z_j}{\|z_i\| \|z_j\|}
\]

So we normalize:

```python
features = F.normalize(features, dim=-1)
```

Add:

```python
def normalize_features(
    features: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    L2-normalize features along the last dimension.
    """
    return F.normalize(
        features.float(),
        dim=-1,
        eps=eps,
    )
```

---

## 2.10.6 k-Nearest-Neighbor Retrieval

k-NN retrieval asks:

> Which training examples are closest to this query representation?

For a query feature matrix:

```python
query_features.shape
# [Q, D]
```

and database features:

```python
database_features.shape
# [M, D]
```

we compute cosine similarity:

\[
S = QD^\top
\]

where:

```python
similarities.shape
# [Q, M]
```

Then we take the top \(k\) database examples for each query.

Add:

```python
@torch.no_grad()
def knn_indices(
    query_features: torch.Tensor,
    database_features: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Return indices of k nearest database features for each query.

    Uses cosine similarity.

    Args:
        query_features:
            [Q, D]

        database_features:
            [M, D]

        k:
            Number of neighbors.

    Returns:
        indices:
            [Q, k]
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}.")

    query = normalize_features(query_features)
    database = normalize_features(database_features)

    similarities = query @ database.T

    return similarities.topk(
        k=k,
        dim=1,
    ).indices
```

This function supports qualitative retrieval visualization.

---

## 2.10.7 k-NN Classification

For labeled datasets, we can also use k-NN as a classifier.

For each validation example:

1. find nearest training features,
2. collect their labels,
3. use majority vote.

Add:

```python
@torch.no_grad()
def knn_predict(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    query_features: torch.Tensor,
    k: int = 20,
) -> torch.Tensor:
    """
    k-NN classification with majority vote.

    Args:
        train_features:
            [N_train, D]

        train_labels:
            [N_train]

        query_features:
            [N_query, D]

        k:
            Number of neighbors.

    Returns:
        predictions:
            [N_query]
    """
    neighbor_idx = knn_indices(
        query_features=query_features,
        database_features=train_features,
        k=k,
    )

    neighbor_labels = train_labels[neighbor_idx]

    preds = []

    for labels in neighbor_labels:
        values, counts = labels.unique(
            return_counts=True,
        )

        pred = values[counts.argmax()]
        preds.append(pred)

    return torch.stack(preds)
```

Accuracy:

```python
def accuracy(
    predictions: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """
    Compute classification accuracy.
    """
    return (
        predictions.eq(labels).float().mean().item()
    )
```

Run:

```python
preds = knn_predict(
    train_features=train_features,
    train_labels=train_labels,
    query_features=val_features,
    k=20,
)

acc = accuracy(preds, val_labels)
```

This gives a simple nonparametric evaluation.

---

## 2.10.8 Linear Probe

A linear probe trains a linear classifier on frozen features.

The encoder remains frozen. Only the classifier learns.

Given features:

\[
z_i \in \mathbb{R}^D
\]

and class labels:

\[
y_i
\]

we train:

\[
\hat{y}_i = Wz_i + b
\]

The loss is cross-entropy:

\[
\mathcal{L}_{\mathrm{probe}}
=
\mathrm{CE}(Wz_i + b, y_i)
\]

A good representation should make labels linearly accessible.

Add:

```python
class LinearProbe(nn.Module):
    """
    Linear classifier on frozen features.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
    ):
        super().__init__()

        self.classifier = nn.Linear(
            feature_dim,
            num_classes,
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(features)
```

---

## 2.10.9 Training a Linear Probe on Extracted Features

Since features are already extracted, probe training is cheap.

Add:

```python
def train_linear_probe_on_features(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    val_features: torch.Tensor,
    val_labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
) -> dict[str, float]:
    """
    Train a linear classifier on frozen features.

    Returns:
        Final train and validation accuracy.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_features = train_features.float()
    val_features = val_features.float()

    feature_dim = train_features.size(1)

    probe = LinearProbe(
        feature_dim=feature_dim,
        num_classes=num_classes,
    ).to(device)

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    num_train = train_features.size(0)

    for _epoch in range(epochs):
        perm = torch.randperm(num_train)

        for start in range(0, num_train, batch_size):
            idx = perm[start : start + batch_size]

            x = train_features[idx].to(device)
            y = train_labels[idx].to(device)

            logits = probe(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        train_logits = probe(train_features.to(device))
        val_logits = probe(val_features.to(device))

        train_preds = train_logits.argmax(dim=1).cpu()
        val_preds = val_logits.argmax(dim=1).cpu()

    return {
        "linear_probe/train_acc": accuracy(train_preds, train_labels),
        "linear_probe/val_acc": accuracy(val_preds, val_labels),
    }
```

This trains a probe on precomputed features.

For larger datasets, use a `TensorDataset` and `DataLoader`. For Chapter 2, this direct implementation is enough.

---

## 2.10.10 Random Encoder Baseline

Evaluation is only meaningful with a baseline.

A trained encoder should outperform a random encoder.

The baseline workflow is:

```text
random encoder
        ↓
extract features
        ↓
k-NN / linear probe
        ↓
compare to trained encoder
```

If the trained encoder does not outperform random initialization, one of the following may be true:

- training did not run long enough,
- the loss is not useful,
- the model collapsed,
- the mask task is too easy or too hard,
- the evaluation is flawed,
- the architecture is too small,
- the dataset is too small or mismatched.

A random baseline is simple but essential.

---

## 2.10.11 Evaluation Dataloaders

Update `data.py` with evaluation dataloaders.

For CIFAR-10:

```python
def build_cifar10_eval_loaders(
    cfg: MinimalJEPAConfig,
) -> tuple[DataLoader, DataLoader]:
    transform = build_image_transform(
        image_size=cfg.image_size,
    )

    train_dataset = CIFAR10(
        root=cfg.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    val_dataset = CIFAR10(
        root=cfg.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
```

For STL-10:

```python
def build_stl10_eval_loaders(
    cfg: MinimalJEPAConfig,
) -> tuple[DataLoader, DataLoader]:
    transform = build_image_transform(
        image_size=cfg.image_size,
    )

    train_dataset = STL10(
        root=cfg.data_dir,
        split="train",
        download=True,
        transform=transform,
    )

    val_dataset = STL10(
        root=cfg.data_dir,
        split="test",
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return train_loader, val_loader
```

Generic builder:

```python
def build_eval_loaders(
    cfg: MinimalJEPAConfig,
    dataset_name: str,
) -> tuple[DataLoader, DataLoader]:
    if dataset_name == "cifar10":
        return build_cifar10_eval_loaders(cfg)

    if dataset_name == "stl10":
        return build_stl10_eval_loaders(cfg)

    raise ValueError(f"Unknown dataset_name: {dataset_name}")
```

---

## 2.10.12 Loading a Checkpoint for Evaluation

Create:

```text
experiments/evaluate_minimal.py
```

Add:

```python
from __future__ import annotations

import argparse

import torch

from jepa_world_model.checkpointing import load_checkpoint
from jepa_world_model.data import build_eval_loaders
from jepa_world_model.evaluation import (
    accuracy,
    extract_representations,
    knn_predict,
    train_linear_probe_on_features,
)
from jepa_world_model.model import build_minimal_ijepa
from jepa_world_model.presets import local_debug_config
from jepa_world_model.utils import get_device, seed_everything
```

Argument parsing:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "stl10"],
    )

    parser.add_argument(
        "--preset",
        type=str,
        default="local_debug",
        choices=["local_debug"],
    )

    parser.add_argument(
        "--knn-k",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
    )

    return parser.parse_args()
```

Main:

```python
def main() -> None:
    args = parse_args()

    cfg = local_debug_config()

    seed_everything(cfg.seed)

    device = get_device()

    model = build_minimal_ijepa(cfg).to(device)

    load_checkpoint(
        path=args.checkpoint,
        model=model,
        optimizer=None,
        map_location=device,
    )

    encoder = model.online_encoder

    train_loader, val_loader = build_eval_loaders(
        cfg=cfg,
        dataset_name=args.dataset,
    )

    num_patches = (
        cfg.image_size // cfg.patch_size
    ) ** 2

    train_features, train_labels = extract_representations(
        encoder=encoder,
        dataloader=train_loader,
        device=device,
        num_patches=num_patches,
        max_batches=args.max_batches,
    )

    val_features, val_labels = extract_representations(
        encoder=encoder,
        dataloader=val_loader,
        device=device,
        num_patches=num_patches,
        max_batches=args.max_batches,
    )

    knn_preds = knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        query_features=val_features,
        k=args.knn_k,
    )

    knn_acc = accuracy(
        knn_preds,
        val_labels,
    )

    print(f"k-NN accuracy: {knn_acc:.4f}")

    num_classes = 10

    probe_logs = train_linear_probe_on_features(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_classes=num_classes,
        epochs=args.probe_epochs,
        device=device,
    )

    print(probe_logs)


if __name__ == "__main__":
    main()
```

Run:

```bash
uv run python experiments/evaluate_minimal.py \
  --checkpoint checkpoints/minimal_ijepa_epoch_2.pt \
  --dataset cifar10 \
  --max-batches 20
```

For a tiny local-debug checkpoint, do not expect strong performance. The goal is to verify that evaluation works.

---

## 2.10.13 Qualitative Retrieval Visualization

k-NN retrieval is especially useful visually.

Given a query image, show the nearest training images according to representation similarity.

Add utility:

```python
@torch.no_grad()
def retrieve_neighbors(
    query_features: torch.Tensor,
    database_features: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """
    Return nearest-neighbor indices for visualization.
    """
    return knn_indices(
        query_features=query_features,
        database_features=database_features,
        k=k,
    )
```

In marimo, after extracting features:

```python
neighbors = retrieve_neighbors(
    query_features=val_features[:8],
    database_features=train_features,
    k=5,
)
```

Then visualize the corresponding images from the dataset.

This is intentionally qualitative. It can reveal whether nearest neighbors are semantically meaningful or random.

---

## 2.10.14 marimo Retrieval Notebook

Create:

```text
notebooks/06_retrieval_and_probe.py
```

Open:

```bash
uv run marimo edit notebooks/06_retrieval_and_probe.py
```

Suggested workflow:

```python
import matplotlib.pyplot as plt
import torch

from jepa_world_model.checkpointing import load_checkpoint
from jepa_world_model.data import build_eval_loaders
from jepa_world_model.evaluation import (
    extract_representations,
    retrieve_neighbors,
)
from jepa_world_model.model import build_minimal_ijepa
from jepa_world_model.presets import local_debug_config
from jepa_world_model.utils import get_device
```

Load model:

```python
cfg = local_debug_config()
device = get_device()

model = build_minimal_ijepa(cfg).to(device)

load_checkpoint(
    "checkpoints/minimal_ijepa_epoch_2.pt",
    model=model,
    map_location=device,
)

encoder = model.online_encoder
```

Load data:

```python
train_loader, val_loader = build_eval_loaders(
    cfg=cfg,
    dataset_name="cifar10",
)
```

Extract features:

```python
num_patches = (cfg.image_size // cfg.patch_size) ** 2

train_features, train_labels = extract_representations(
    encoder=encoder,
    dataloader=train_loader,
    device=device,
    num_patches=num_patches,
    max_batches=20,
)

val_features, val_labels = extract_representations(
    encoder=encoder,
    dataloader=val_loader,
    device=device,
    num_patches=num_patches,
    max_batches=5,
)
```

Retrieve:

```python
neighbors = retrieve_neighbors(
    query_features=val_features[:8],
    database_features=train_features,
    k=5,
)

neighbors.shape
```

Visualization requires access to original images. For a simple first version, keep a dataset object available and index into it.

---

## 2.10.15 Linear Probe on the Fly

The feature-extraction approach is simple and fast for small datasets.

For larger datasets, one might train the probe directly on batches:

```text
image batch
  ↓
frozen encoder
  ↓
pooled features
  ↓
linear classifier
```

This avoids storing all features in memory.

For Chapter 2, precomputing features is clearer.

Chapter 3 can add a more scalable probe trainer.

---

## 2.10.16 Unit Tests

Create:

```text
tests/test_evaluation.py
```

Add:

```python
import torch

from jepa_world_model.evaluation import (
    LinearProbe,
    accuracy,
    full_patch_indices,
    knn_indices,
    knn_predict,
    normalize_features,
    train_linear_probe_on_features,
)


def test_full_patch_indices_shape():
    indices = full_patch_indices(
        batch_size=3,
        num_patches=8,
        device=torch.device("cpu"),
    )

    assert indices.shape == (3, 8)

    expected = torch.arange(8)

    torch.testing.assert_close(
        indices[0],
        expected,
    )


def test_normalize_features_unit_norm():
    x = torch.randn(4, 8)

    y = normalize_features(x)

    norms = y.norm(dim=-1)

    torch.testing.assert_close(
        norms,
        torch.ones_like(norms),
        atol=1e-5,
        rtol=1e-5,
    )


def test_knn_indices_shape():
    query = torch.randn(3, 8)
    database = torch.randn(10, 8)

    idx = knn_indices(
        query_features=query,
        database_features=database,
        k=4,
    )

    assert idx.shape == (3, 4)


def test_knn_predict_shape():
    train_features = torch.randn(10, 8)
    train_labels = torch.arange(10) % 2
    query_features = torch.randn(3, 8)

    preds = knn_predict(
        train_features=train_features,
        train_labels=train_labels,
        query_features=query_features,
        k=3,
    )

    assert preds.shape == (3,)


def test_accuracy():
    preds = torch.tensor([0, 1, 1, 0])
    labels = torch.tensor([0, 1, 0, 0])

    acc = accuracy(preds, labels)

    assert acc == 0.75


def test_linear_probe_shape():
    probe = LinearProbe(
        feature_dim=8,
        num_classes=3,
    )

    x = torch.randn(4, 8)

    logits = probe(x)

    assert logits.shape == (4, 3)


def test_train_linear_probe_on_tiny_features():
    train_features = torch.randn(20, 8)
    train_labels = torch.randint(0, 2, (20,))

    val_features = torch.randn(10, 8)
    val_labels = torch.randint(0, 2, (10,))

    logs = train_linear_probe_on_features(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_classes=2,
        epochs=2,
        batch_size=5,
        device=torch.device("cpu"),
    )

    assert "linear_probe/train_acc" in logs
    assert "linear_probe/val_acc" in logs
```

Run:

```bash
uv run pytest tests/test_evaluation.py
```

---

## 2.10.17 Interpreting Results

For a tiny local-debug run, performance may be weak.

That is fine.

The first expected milestones are:

```text
random encoder:
    near chance linear probe
    poor nearest-neighbor retrieval

short JEPA run:
    maybe slightly better than random
    loss decreases
    diagnostics remain healthy

longer JEPA run:
    k-NN improves
    linear probe improves
    retrieval becomes visually meaningful
```

For CIFAR-10:

```text
chance accuracy = 10%
```

For STL-10:

```text
chance accuracy = 10%
```

A minimal model trained briefly may not dramatically outperform chance. This does not automatically mean the implementation is wrong. Representation learning may need:

- more epochs,
- larger model,
- better augmentation,
- better masks,
- better dataset,
- tuned learning rate,
- longer training.

The key comparison is trained encoder vs random encoder under the same evaluation.

---

## 2.10.18 Common Bugs

### Bug 1: Evaluating the target encoder instead of the online encoder

Use:

```python
encoder = model.online_encoder
```

The target encoder is a moving average training target. The online encoder is the representation model we normally evaluate.

That said, evaluating the target encoder can also be interesting. It may sometimes be smoother. But use the online encoder as the default.

---

### Bug 2: Forgetting full patch indices

The encoder requires patch indices.

For full image evaluation:

```python
indices = full_patch_indices(batch_size, num_patches, device)
```

Do not pass context masks during evaluation unless you intentionally want partial-image representations.

---

### Bug 3: Probe accidentally trains the encoder

When extracting features first, this cannot happen because features are detached CPU tensors.

If training the probe on the fly, wrap encoder forward in:

```python
with torch.no_grad():
    features = encoder(...)
```

---

### Bug 4: Different transforms for train and validation

For basic probing, keep transforms simple and consistent.

Start with:

```python
Resize
ToTensor
```

Add augmentation later only if intentionally evaluating robustness.

---

### Bug 5: Linear probe overfits tiny feature subsets

If using `--max-batches`, the probe may overfit. This is fine for a smoke test but not a serious evaluation.

For real results, extract features over the full train and validation splits.

---

## 2.10.19 Summary

This section implemented basic representation evaluation.

We added:

- full-patch evaluation indices,
- frozen feature extraction,
- feature normalization,
- k-NN retrieval,
- k-NN classification,
- linear probe training,
- evaluation dataloaders,
- checkpoint evaluation script,
- marimo retrieval workflow,
- unit tests.

These tools tell us whether the encoder learned useful representations beyond simply reducing JEPA loss.

The next section runs the first end-to-end experiment and defines what a successful minimal run should look like.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words**, 2020.
  <https://arxiv.org/abs/2010.11929>

- Kaiming He et al., **Masked Autoencoders Are Scalable Vision Learners**, 2021.
  <https://arxiv.org/abs/2111.06377>

- TorchVision, **Datasets**.
  <https://pytorch.org/vision/stable/datasets.html>

- PyTorch, **torch.nn.Linear**.
  <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html>
