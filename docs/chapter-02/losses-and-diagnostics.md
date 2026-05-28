# 2.8 Losses and Diagnostics

We now have the main JEPA components:

- patchification,
- positional embeddings,
- context and target masks,
- online and target encoders,
- EMA target updates,
- predictor.

The next step is to define:

1. the loss that trains the predictor and online encoder,
2. diagnostics that tell us whether the representation is healthy.

This section implements both.

The loss answers:

> Is the predicted target representation close to the target encoder representation?

The diagnostics answer:

> Are the representations non-collapsed, well-scaled, and numerically stable?

Both are necessary.

A JEPA model can have a decreasing loss while still learning a weak or collapsed representation. We therefore treat diagnostics as part of the implementation, not as optional logging decoration.

---

## 2.8.1 The JEPA Loss

The JEPA objective is:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
d(
\hat{z}_{\mathcal{T}},
\mathrm{sg}(z_{\mathcal{T}})
)
\]

where:

- \(\hat{z}_{\mathcal{T}}\) is the predictor output,
- \(z_{\mathcal{T}}\) is the target encoder output,
- \(\mathrm{sg}(\cdot)\) denotes stop-gradient,
- \(d\) is a distance function.

In code:

```python
loss = loss_fn(
    pred_repr,
    target_repr.detach(),
)
```

The prediction and target tensors must have the same shape:

```python
pred_repr.shape
# [B, N_tgt, D]

target_repr.shape
# [B, N_tgt, D]
```

The loss should return a scalar.

---

## 2.8.2 Candidate Loss Functions

We will implement three useful latent-space losses:

1. mean squared error,
2. Smooth L1 loss,
3. cosine loss.

A combined loss is also useful for experiments.

---

### Mean Squared Error

MSE is the simplest choice:

\[
\mathcal{L}_{\mathrm{MSE}}
=
\frac{1}{B N_{\mathrm{tgt}} D}
\sum_{b,i,j}
(\hat{z}_{b,i,j} - z_{b,i,j})^2
\]

Here \(B\) is batch size, \(N_{\mathrm{tgt}}\) is the number of target tokens, \(D\) is representation dimension, and \(\hat{z}_{b,i,j}\) and \(z_{b,i,j}\) are matching predicted and target representation elements.

It preserves magnitude information.

```python
loss = F.mse_loss(pred_repr, target_repr.detach())
```

MSE is sensitive to scale. If representation norms grow, MSE can become large. If representations shrink, MSE can become small even if the geometry is poor.

---

### Smooth L1 Loss

Smooth L1 is less sensitive to outliers.

It behaves like squared error near zero and like absolute error for larger deviations.

```python
loss = F.smooth_l1_loss(pred_repr, target_repr.detach())
```

This is a reasonable default for the minimal implementation.

---

### Cosine Loss

Cosine loss compares directions rather than magnitudes.

\[
\mathcal{L}_{\mathrm{cos}}
=
1
-
\frac{
\hat{z}^\top z
}{
\|\hat{z}\| \|z\|
}
\]

In this expression, \(\hat{z}\) and \(z\) denote a matching predicted-target representation pair after selecting one batch item and one target token; the implementation averages this quantity over the batch and target-token dimensions.

In code:

```python
pred = F.normalize(pred_repr, dim=-1)
target = F.normalize(target_repr.detach(), dim=-1)

loss = 1.0 - (pred * target).sum(dim=-1).mean()
```

Cosine loss is scale-invariant, but it discards norm information. This may or may not be desirable.

---

## 2.8.3 Implementing `losses.py`

Create:

```text
src/jepa_world_model/losses.py
```

Add:

```python
from __future__ import annotations

import torch
import torch.nn.functional as F
```

First, implement a shape check:

```python
def assert_prediction_target_shape(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    """
    Ensure prediction and target representations have the same shape.

    Expected:
        pred:
            [B, N_tgt, D]

        target:
            [B, N_tgt, D]
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction and target shapes must match. "
            f"Got pred={pred.shape}, target={target.shape}."
        )

    if pred.dim() != 3:
        raise ValueError(
            f"Expected tensors with shape [B, N_tgt, D], got {pred.shape}."
        )
```

Now implement MSE:

```python
def mse_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Mean squared error loss in latent space.
    """
    assert_prediction_target_shape(pred, target)

    return F.mse_loss(
        pred,
        target.detach(),
    )
```

Smooth L1:

```python
def smooth_l1_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    Smooth L1 loss in latent space.
    """
    assert_prediction_target_shape(pred, target)

    return F.smooth_l1_loss(
        pred,
        target.detach(),
        beta=beta,
    )
```

Cosine:

```python
def cosine_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Cosine distance loss in latent space.

    Returns:
        1 - average cosine similarity.
    """
    assert_prediction_target_shape(pred, target)

    pred_norm = F.normalize(
        pred,
        dim=-1,
        eps=eps,
    )

    target_norm = F.normalize(
        target.detach(),
        dim=-1,
        eps=eps,
    )

    cosine = (pred_norm * target_norm).sum(dim=-1)

    return 1.0 - cosine.mean()
```

Combined loss:

```python
def combined_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mse_weight: float = 1.0,
    cosine_weight: float = 0.1,
) -> torch.Tensor:
    """
    Combined MSE and cosine latent loss.
    """
    assert_prediction_target_shape(pred, target)

    mse = mse_latent_loss(
        pred,
        target,
    )

    cosine = cosine_latent_loss(
        pred,
        target,
    )

    return mse_weight * mse + cosine_weight * cosine
```

Default loss selector:

```python
def latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = "smooth_l1",
) -> torch.Tensor:
    """
    Dispatch latent loss by name.

    Supported:
        "mse"
        "smooth_l1"
        "cosine"
        "combined"
    """
    if loss_type == "mse":
        return mse_latent_loss(pred, target)

    if loss_type == "smooth_l1":
        return smooth_l1_latent_loss(pred, target)

    if loss_type == "cosine":
        return cosine_latent_loss(pred, target)

    if loss_type == "combined":
        return combined_latent_loss(pred, target)

    raise ValueError(f"Unknown loss_type: {loss_type}")
```

For the first experiments, use:

```python
loss = latent_loss(
    pred_repr,
    target_repr,
    loss_type="smooth_l1",
)
```

---

## 2.8.4 Why `.detach()` Belongs Inside the Loss

The target encoder forward pass should already use:

```python
with torch.no_grad():
    target_repr = target_encoder(...)
```

Still, the loss function also detaches the target:

```python
target.detach()
```

This is intentional redundancy.

It ensures that even if a future caller forgets `torch.no_grad()`, the target branch is not optimized through the latent loss.

The target encoder should be updated only by EMA.

---

## 2.8.5 Basic Prediction Diagnostics

The first diagnostics compare prediction and target.

Useful metrics:

- loss,
- cosine similarity,
- MSE,
- mean absolute error,
- prediction norm,
- target norm.

Create:

```text
src/jepa_world_model/diagnostics.py
```

Start with:

```python
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
```

Add:

```python
@torch.no_grad()
def prediction_target_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Compute basic prediction-target metrics.

    Args:
        pred:
            [B, N_tgt, D]

        target:
            [B, N_tgt, D]

    Returns:
        Dictionary of scalar metrics.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, target={target.shape}."
        )

    pred_flat = pred.float().flatten(0, 1)
    target_flat = target.float().flatten(0, 1)

    mse = F.mse_loss(
        pred_flat,
        target_flat,
    )

    mae = F.l1_loss(
        pred_flat,
        target_flat,
    )

    cosine = F.cosine_similarity(
        pred_flat,
        target_flat,
        dim=-1,
    ).mean()

    return {
        "pred_target/mse": mse.item(),
        "pred_target/mae": mae.item(),
        "pred_target/cosine": cosine.item(),
    }
```

This gives us immediate feedback about prediction quality.

During early training, cosine may be low. It should improve if the model learns.

---

## 2.8.6 Representation Norm Diagnostics

Norms tell us whether representations are exploding, shrinking, or stable.

For tensor:

```python
z.shape
# [B, N, D]
```

we flatten batch and token dimensions:

```python
z_flat.shape
# [B * N, D]
```

Then compute norms:

\[
\|z_i\|_2
\]

Add:

```python
@torch.no_grad()
def norm_metrics(
    z: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """
    Compute representation norm statistics.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    if z.dim() != 2:
        raise ValueError(
            f"Expected [B, D] or [B, N, D], got {z.shape}."
        )

    norms = z.float().norm(dim=-1)

    return {
        f"{prefix}/norm_mean": norms.mean().item(),
        f"{prefix}/norm_std": norms.std().item(),
        f"{prefix}/norm_min": norms.min().item(),
        f"{prefix}/norm_max": norms.max().item(),
    }
```

Example:

```python
logs.update(norm_metrics(target_repr, prefix="target"))
logs.update(norm_metrics(pred_repr, prefix="pred"))
```

Warning signs:

```text
norm_mean → 0
norm_mean → very large
norm_std → 0
```

---

## 2.8.7 Feature Variance Diagnostics

Collapse often appears as low feature variance.

For representations:

\[
Z \in \mathbb{R}^{M \times D}
\]

where:

\[
M = B \cdot N
\]

compute standard deviation per feature dimension:

\[
\sigma_j = \mathrm{std}(Z_{:,j})
\]

A collapsed representation has:

\[
\sigma_j \approx 0
\]

for many or all dimensions.

Add:

```python
@torch.no_grad()
def variance_metrics(
    z: torch.Tensor,
    prefix: str,
    eps: float = 1e-4,
) -> dict[str, float]:
    """
    Compute feature variance / std diagnostics.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    if z.dim() != 2:
        raise ValueError(
            f"Expected [B, D] or [B, N, D], got {z.shape}."
        )

    z = z.float()

    std = z.std(dim=0)
    var = z.var(dim=0)

    dead_fraction = (std < eps).float().mean()

    return {
        f"{prefix}/std_mean": std.mean().item(),
        f"{prefix}/std_min": std.min().item(),
        f"{prefix}/std_max": std.max().item(),
        f"{prefix}/var_mean": var.mean().item(),
        f"{prefix}/dead_dim_fraction": dead_fraction.item(),
    }
```

A healthy representation should have nonzero standard deviation across many dimensions.

This does not guarantee semantic quality, but it catches obvious collapse.

---

## 2.8.8 Covariance Diagnostics

Variance checks each feature independently.

Covariance checks redundancy between features.

Given centered features:

\[
Z_c = Z - \mu
\]

the covariance matrix is:

\[
C =
\frac{1}{M - 1} Z_c^\top Z_c
\]

The diagonal contains feature variances. The off-diagonal terms measure correlation or redundancy between different dimensions.

Add:

```python
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """
    Return flattened off-diagonal elements of a square matrix.
    """
    if x.dim() != 2 or x.size(0) != x.size(1):
        raise ValueError(f"Expected square matrix, got {x.shape}.")

    n = x.size(0)

    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
```

Now covariance metrics:

```python
@torch.no_grad()
def covariance_metrics(
    z: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """
    Compute covariance diagnostics for representations.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    if z.dim() != 2:
        raise ValueError(
            f"Expected [B, D] or [B, N, D], got {z.shape}."
        )

    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)

    num_samples = z.size(0)

    if num_samples < 2:
        raise ValueError("Need at least two samples for covariance metrics.")

    cov = (z.T @ z) / (num_samples - 1)

    diag = torch.diagonal(cov)
    off = off_diagonal(cov)

    return {
        f"{prefix}/cov_diag_mean": diag.mean().item(),
        f"{prefix}/cov_diag_min": diag.min().item(),
        f"{prefix}/cov_diag_max": diag.max().item(),
        f"{prefix}/cov_offdiag_abs_mean": off.abs().mean().item(),
        f"{prefix}/cov_offdiag_sq_mean": off.pow(2).mean().item(),
    }
```

High off-diagonal values may indicate redundant dimensions.

This is not necessarily a failure, but it is useful to monitor.

---

## 2.8.9 Effective Rank

Feature variance can be nonzero while most information lies in only a few directions.

Effective rank estimates how many dimensions are meaningfully used.

Given covariance eigenvalues:

\[
\lambda_1,\dots,\lambda_D
\]

normalize them:

\[
p_i = \frac{\lambda_i}{\sum_j \lambda_j}
\]

Compute entropy:

\[
H(p) = -\sum_i p_i \log p_i
\]

Then:

\[
\mathrm{effective\ rank} = \exp(H(p))
\]

Add:

```python
@torch.no_grad()
def effective_rank(
    z: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """
    Estimate effective rank of the representation covariance.
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    if z.dim() != 2:
        raise ValueError(
            f"Expected [B, D] or [B, N, D], got {z.shape}."
        )

    z = z.float()
    z = z - z.mean(dim=0, keepdim=True)

    num_samples = z.size(0)

    if num_samples < 2:
        return 0.0

    cov = (z.T @ z) / (num_samples - 1)

    eigvals = torch.linalg.eigvalsh(cov).clamp_min(0.0)

    total = eigvals.sum()

    if total <= eps:
        return 0.0

    probs = eigvals / total
    entropy = -(probs * torch.log(probs + eps)).sum()

    return torch.exp(entropy).item()
```

Add a wrapper:

```python
@torch.no_grad()
def rank_metrics(
    z: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """
    Compute effective-rank diagnostics.
    """
    return {
        f"{prefix}/effective_rank": effective_rank(z),
    }
```

Effective rank is useful for detecting dimensional collapse.

---

## 2.8.10 Unified Representation Report

We want one function that combines:

- norms,
- variance,
- covariance,
- effective rank.

Add:

```python
@torch.no_grad()
def representation_report(
    z: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    """
    Compute a combined representation diagnostic report.
    """
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

    logs.update(
        covariance_metrics(
            z,
            prefix=prefix,
        )
    )

    logs.update(
        rank_metrics(
            z,
            prefix=prefix,
        )
    )

    return logs
```

Usage:

```python
logs.update(
    representation_report(
        target_repr,
        prefix="target",
    )
)

logs.update(
    representation_report(
        pred_repr,
        prefix="pred",
    )
)
```

This gives a compact health report for both target and prediction representations.

---

## 2.8.11 Full JEPA Diagnostics

Now combine prediction-target metrics and representation reports.

```python
@torch.no_grad()
def jepa_diagnostics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """
    Compute diagnostics for JEPA prediction and representation health.
    """
    logs: dict[str, float] = {}

    logs.update(
        prediction_target_metrics(
            pred,
            target,
        )
    )

    logs.update(
        representation_report(
            pred,
            prefix="pred",
        )
    )

    logs.update(
        representation_report(
            target,
            prefix="target",
        )
    )

    return logs
```

During training:

```python
with torch.no_grad():
    logs = {
        "loss": loss.item(),
        "ema_tau": ema_tau,
    }

    logs.update(
        jepa_diagnostics(
            pred_repr,
            target_repr,
        )
    )
```

---

## 2.8.12 Mask Diagnostics

The diagnostic module can also include mask checks.

Add:

```python
@torch.no_grad()
def mask_stats(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    num_patches: int,
) -> dict[str, float]:
    """
    Compute basic mask statistics.
    """
    if context_indices.dim() != 2:
        raise ValueError(
            f"Expected context_indices [B, N_ctx], got {context_indices.shape}."
        )

    if target_indices.dim() != 2:
        raise ValueError(
            f"Expected target_indices [B, N_tgt], got {target_indices.shape}."
        )

    batch_size = context_indices.size(0)

    overlaps = []

    for batch_idx in range(batch_size):
        overlap = torch.isin(
            target_indices[batch_idx],
            context_indices[batch_idx],
        )

        overlaps.append(overlap.float().mean())

    overlap_fraction = torch.stack(overlaps).mean().item()

    return {
        "mask/context_count": float(context_indices.size(1)),
        "mask/target_count": float(target_indices.size(1)),
        "mask/context_ratio": float(context_indices.size(1) / num_patches),
        "mask/target_ratio": float(target_indices.size(1) / num_patches),
        "mask/overlap_fraction": overlap_fraction,
    }
```

This should report:

```text
mask/overlap_fraction = 0.0
```

If not, the training run is invalid.

---

## 2.8.13 NaN and Inf Checks

Numerical failures should be caught immediately.

Add:

```python
def assert_finite_tensor(
    x: torch.Tensor,
    name: str,
) -> None:
    """
    Raise an error if tensor contains NaN or Inf.
    """
    if not torch.isfinite(x).all():
        raise FloatingPointError(
            f"Tensor {name} contains NaN or Inf."
        )
```

Use during development:

```python
assert_finite_tensor(pred_repr, "pred_repr")
assert_finite_tensor(target_repr, "target_repr")
assert_finite_tensor(loss, "loss")
```

This is especially useful once mixed precision enters the picture.

---

## 2.8.14 Logging Interpretation

A healthy early training run might show:

```text
loss                         decreasing
pred_target/cosine            increasing
target/std_mean               stable and nonzero
pred/std_mean                 stable and nonzero
target/dead_dim_fraction      near 0
pred/dead_dim_fraction        near 0
mask/overlap_fraction         exactly 0
```

Warning signs:

```text
loss goes to zero very quickly
target/std_mean goes to zero
pred/std_mean goes to zero
dead_dim_fraction approaches 1
effective_rank collapses
norm_mean explodes
mask/overlap_fraction > 0
pred_target/cosine is high but representation variance is tiny
```

The last case is especially important. High cosine similarity is meaningless if all vectors are nearly constant.

---

## 2.8.15 Unit Tests for Losses

Create:

```text
tests/test_losses.py
```

Add:

```python
import pytest
import torch

from jepa_world_model.losses import (
    assert_prediction_target_shape,
    combined_latent_loss,
    cosine_latent_loss,
    latent_loss,
    mse_latent_loss,
    smooth_l1_latent_loss,
)


def test_shape_check_accepts_matching_shapes():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    assert_prediction_target_shape(pred, target)


def test_shape_check_rejects_mismatched_shapes():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 5, 8)

    with pytest.raises(ValueError):
        assert_prediction_target_shape(pred, target)


def test_mse_latent_loss_scalar():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    loss = mse_latent_loss(pred, target)

    assert loss.dim() == 0


def test_smooth_l1_latent_loss_scalar():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    loss = smooth_l1_latent_loss(pred, target)

    assert loss.dim() == 0


def test_cosine_latent_loss_scalar():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    loss = cosine_latent_loss(pred, target)

    assert loss.dim() == 0


def test_combined_latent_loss_scalar():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    loss = combined_latent_loss(pred, target)

    assert loss.dim() == 0


def test_latent_loss_dispatch():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    for loss_type in ["mse", "smooth_l1", "cosine", "combined"]:
        loss = latent_loss(
            pred,
            target,
            loss_type=loss_type,
        )

        assert loss.dim() == 0


def test_latent_loss_rejects_unknown_type():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    with pytest.raises(ValueError):
        latent_loss(
            pred,
            target,
            loss_type="unknown",
        )
```

Run:

```bash
pytest tests/test_losses.py
```

---

## 2.8.16 Unit Tests for Diagnostics

Create:

```text
tests/test_diagnostics.py
```

Add:

```python
import pytest
import torch

from jepa_world_model.diagnostics import (
    assert_finite_tensor,
    covariance_metrics,
    effective_rank,
    jepa_diagnostics,
    mask_stats,
    norm_metrics,
    prediction_target_metrics,
    representation_report,
    variance_metrics,
)


def test_prediction_target_metrics_keys():
    pred = torch.randn(2, 4, 8)
    target = torch.randn(2, 4, 8)

    logs = prediction_target_metrics(pred, target)

    assert "pred_target/mse" in logs
    assert "pred_target/mae" in logs
    assert "pred_target/cosine" in logs


def test_norm_metrics_keys():
    z = torch.randn(2, 4, 8)

    logs = norm_metrics(z, prefix="target")

    assert "target/norm_mean" in logs
    assert "target/norm_std" in logs


def test_variance_metrics_keys():
    z = torch.randn(2, 4, 8)

    logs = variance_metrics(z, prefix="target")

    assert "target/std_mean" in logs
    assert "target/dead_dim_fraction" in logs


def test_covariance_metrics_keys():
    z = torch.randn(8, 4, 16)

    logs = covariance_metrics(z, prefix="target")

    assert "target/cov_diag_mean" in logs
    assert "target/cov_offdiag_abs_mean" in logs


def test_effective_rank_positive_for_random_data():
    z = torch.randn(32, 16)

    rank = effective_rank(z)

    assert rank > 0.0


def test_representation_report_keys():
    z = torch.randn(8, 4, 16)

    logs = representation_report(z, prefix="target")

    assert "target/norm_mean" in logs
    assert "target/std_mean" in logs
    assert "target/effective_rank" in logs


def test_jepa_diagnostics_keys():
    pred = torch.randn(8, 4, 16)
    target = torch.randn(8, 4, 16)

    logs = jepa_diagnostics(pred, target)

    assert "pred_target/mse" in logs
    assert "pred/norm_mean" in logs
    assert "target/norm_mean" in logs


def test_mask_stats_overlap_zero():
    context = torch.tensor([
        [0, 1, 2],
        [3, 4, 5],
    ])

    target = torch.tensor([
        [3, 4],
        [0, 1],
    ])

    logs = mask_stats(
        context_indices=context,
        target_indices=target,
        num_patches=8,
    )

    assert logs["mask/overlap_fraction"] == 0.0


def test_mask_stats_detects_overlap():
    context = torch.tensor([
        [0, 1, 2],
    ])

    target = torch.tensor([
        [2, 3],
    ])

    logs = mask_stats(
        context_indices=context,
        target_indices=target,
        num_patches=8,
    )

    assert logs["mask/overlap_fraction"] > 0.0


def test_assert_finite_tensor_accepts_finite():
    x = torch.ones(3)

    assert_finite_tensor(x, "x")


def test_assert_finite_tensor_rejects_nan():
    x = torch.tensor([1.0, float("nan")])

    with pytest.raises(FloatingPointError):
        assert_finite_tensor(x, "x")
```

Run:

```bash
pytest tests/test_diagnostics.py
```

---

## 2.8.17 Integration with Encoder and Predictor

At this point, we can run the full forward path up to the loss.

```python
import torch

from jepa_world_model.diagnostics import jepa_diagnostics
from jepa_world_model.losses import latent_loss
from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch
from jepa_world_model.predictor import JEPAPredictor
from jepa_world_model.vit import MinimalViTEncoder
```

```python
images = torch.randn(2, 3, 32, 32)

online_encoder = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
)

target_encoder = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
)

predictor = JEPAPredictor(
    grid_size=8,
    encoder_dim=64,
    predictor_dim=32,
    depth=2,
    num_heads=4,
)

mask_config = BlockMaskConfig(
    grid_height=8,
    grid_width=8,
    num_target_blocks=2,
    target_block_height=2,
    target_block_width=2,
    context_ratio=0.5,
)

context_indices, target_indices = sample_mask_batch(
    config=mask_config,
    batch_size=images.size(0),
    device=images.device,
)
```

Forward:

```python
context_repr = online_encoder(
    images=images,
    patch_indices=context_indices,
)

with torch.no_grad():
    target_repr = target_encoder(
        images=images,
        patch_indices=target_indices,
    )

pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)
```

Loss:

```python
loss = latent_loss(
    pred=pred_repr,
    target=target_repr,
    loss_type="smooth_l1",
)
```

Diagnostics:

```python
logs = jepa_diagnostics(
    pred=pred_repr,
    target=target_repr,
)

logs["loss"] = loss.item()
```

This is almost the full training step. The next sections will wrap this into the top-level model and full training loop.

---

## 2.8.18 marimo Debug Notebook

Create:

```text
notebooks/05_debug_losses_and_diagnostics.py
```

Open:

```bash
marimo edit notebooks/05_debug_losses_and_diagnostics.py
```

Suggested cells:

```python
import torch

from jepa_world_model.diagnostics import jepa_diagnostics
from jepa_world_model.losses import latent_loss
from jepa_world_model.masks import BlockMaskConfig, sample_mask_batch
from jepa_world_model.predictor import JEPAPredictor
from jepa_world_model.vit import MinimalViTEncoder
```

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

images = torch.randn(4, 3, 32, 32, device=device)
```

```python
online_encoder = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
).to(device)

target_encoder = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
).to(device)

predictor = JEPAPredictor(
    grid_size=8,
    encoder_dim=64,
    predictor_dim=32,
    depth=2,
    num_heads=4,
).to(device)
```

```python
mask_config = BlockMaskConfig(
    grid_height=8,
    grid_width=8,
    num_target_blocks=2,
    target_block_height=2,
    target_block_width=2,
    context_ratio=0.5,
)

context_indices, target_indices = sample_mask_batch(
    config=mask_config,
    batch_size=images.size(0),
    device=device,
)
```

```python
context_repr = online_encoder(images, context_indices)

with torch.no_grad():
    target_repr = target_encoder(images, target_indices)

pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)
```

```python
loss = latent_loss(
    pred=pred_repr,
    target=target_repr,
    loss_type="smooth_l1",
)

loss.item()
```

```python
logs = jepa_diagnostics(pred_repr, target_repr)
logs
```

The purpose is to check that metrics are finite and sensible before building the training loop.

---

## 2.8.19 Common Bugs

### Bug 1: Loss compares wrong shapes

The prediction and target must both be:

```python
[B, N_tgt, D]
```

If the predictor returns all tokens, the shape may be:

```python
[B, N_ctx + N_tgt, D]
```

That is wrong.

Only compare target predictions.

---

### Bug 2: Target not detached

The loss should detach target representations:

```python
target.detach()
```

The target encoder should not receive gradients.

---

### Bug 3: Loss decreases but representation collapses

Always monitor:

```text
target/std_mean
target/dead_dim_fraction
target/effective_rank
```

A low loss is not enough.

---

### Bug 4: Cosine looks good but norms collapse

Cosine similarity ignores magnitude.

If using cosine loss, also monitor norms and feature variance.

---

### Bug 5: Covariance metrics are expensive

Covariance requires a \(D \times D\) matrix.

For small Chapter 2 models, this is fine. For larger models, log covariance metrics less frequently.

---

## 2.8.20 Summary

This section implemented the loss and diagnostic layer for minimal I-JEPA.

We implemented:

- MSE latent loss,
- Smooth L1 latent loss,
- cosine latent loss,
- combined latent loss,
- prediction-target metrics,
- norm diagnostics,
- variance diagnostics,
- covariance diagnostics,
- effective rank,
- mask diagnostics,
- NaN/Inf checks,
- unit tests,
- marimo debugging workflow.

The next section assembles the full `MinimalIJEPA` model and training step.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Adrien Bardes, Jean Ponce, Yann LeCun, **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**, 2021.
  <https://arxiv.org/abs/2105.04906>

- Jure Zbontar et al., **Barlow Twins: Self-Supervised Learning via Redundancy Reduction**, 2021.
  <https://arxiv.org/abs/2103.03230>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent**, 2020.
  <https://arxiv.org/abs/2006.07733>

- PyTorch, **torch.nn.functional.mse_loss**.
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.mse_loss.html>

- PyTorch, **torch.nn.functional.smooth_l1_loss**.
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.smooth_l1_loss.html>

- PyTorch, **torch.linalg.eigvalsh**.
  <https://pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html>
