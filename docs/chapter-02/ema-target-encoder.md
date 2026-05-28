# 2.6 EMA Target Encoder

JEPA uses two encoder branches:

1. an **online encoder**,
2. a **target encoder**.

The online encoder processes context patches and is updated by gradient descent.

The target encoder processes target patches and provides the representation target. It is not updated directly by backpropagation. Instead, it is updated as an exponential moving average of the online encoder.

This gives the model a slowly moving target representation.

The core update is:

\[
\bar{\theta}
\leftarrow
\tau \bar{\theta}
+
(1-\tau)\theta
\]

where:

- \(\theta\) are online encoder parameters,
- \(\bar{\theta}\) are target encoder parameters,
- \(\tau\) is the EMA momentum.

This section implements the target encoder mechanics.

---

## 2.6.1 Why We Need a Target Encoder

The JEPA loss is:

\[
\mathcal{L}
=
\left\|
g_\theta(f_\theta(x_{\mathcal{C}}), \mathcal{T})
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathcal{T}}))
\right\|^2
\]

The target representation is:

\[
z_{\mathcal{T}}
=
f_{\bar{\theta}}(x_{\mathcal{T}})
\]

The predicted target representation is:

\[
\hat{z}_{\mathcal{T}}
=
g_\theta(f_\theta(x_{\mathcal{C}}), \mathcal{T})
\]

The prediction branch learns by gradient descent.

The target branch does not.

This asymmetry matters. If both branches were updated directly by the same loss, the system would be more prone to unstable moving targets or trivial solutions.

The target encoder provides a slowly evolving representation space that the online branch learns to predict.

---

## 2.6.2 Online vs Target Branches

The online branch consists of:

```text
online encoder + predictor
```

It receives gradients.

The target branch consists of:

```text
target encoder
```

It does not receive gradients.

During a forward pass:

```python
context_repr = online_encoder(images, context_indices)

with torch.no_grad():
    target_repr = target_encoder(images, target_indices)

pred_repr = predictor(context_repr, context_indices, target_indices)

loss = loss_fn(pred_repr, target_repr)
```

During an optimizer step:

```python
loss.backward()
optimizer.step()
```

Only the online encoder and predictor should update.

Then the target encoder is updated manually:

```python
update_ema(
    online=online_encoder,
    target=target_encoder,
    tau=ema_tau,
)
```

This happens after the optimizer step.

---

## 2.6.3 EMA Intuition

The EMA update is:

\[
\bar{\theta}_{t+1}
=
\tau \bar{\theta}_{t}
+
(1-\tau)\theta_t
\]

If:

\[
\tau = 0.996
\]

then the target encoder keeps 99.6% of its previous parameters and receives 0.4% of the current online parameters at each update.

A larger \(\tau\) means a slower target.

A smaller \(\tau\) means a faster target.

Typical behavior:

```text
early training:
    target tracks online encoder moderately quickly

late training:
    target becomes very stable
```

A common schedule increases \(\tau\) toward 1.0 during training.

---

## 2.6.4 Implementing `ema.py`

Create:

```text
src/jepa_world_model/ema.py
```

Add:

```python
from __future__ import annotations

import math

import torch
import torch.nn as nn
```

Now implement target initialization.

```python
@torch.no_grad()
def initialize_target_encoder(
    online: nn.Module,
    target: nn.Module,
) -> None:
    """
    Initialize target encoder from online encoder and freeze it.

    Args:
        online:
            Online encoder.

        target:
            Target encoder.
    """
    target.load_state_dict(online.state_dict())

    for param in target.parameters():
        param.requires_grad = False
```

This function does two things:

1. copies parameters,
2. disables gradients.

It should be called once when constructing the JEPA model.

---

## 2.6.5 EMA Update

Add:

```python
@torch.no_grad()
def update_ema(
    online: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    """
    Update target parameters toward online parameters using EMA.

    target = tau * target + (1 - tau) * online

    Args:
        online:
            Online encoder.

        target:
            Target encoder.

        tau:
            EMA momentum in [0, 1].
    """
    if not 0.0 <= tau <= 1.0:
        raise ValueError(f"tau must be in [0, 1], got {tau}.")

    online_params = dict(online.named_parameters())
    target_params = dict(target.named_parameters())

    if online_params.keys() != target_params.keys():
        raise ValueError(
            "Online and target modules have different parameter names."
        )

    for name in online_params:
        online_param = online_params[name]
        target_param = target_params[name]

        if online_param.shape != target_param.shape:
            raise ValueError(
                f"Shape mismatch for parameter {name}: "
                f"online={online_param.shape}, target={target_param.shape}."
            )

        target_param.data.mul_(tau).add_(
            online_param.data,
            alpha=1.0 - tau,
        )
```

This version checks parameter names and shapes. It is slightly more verbose than zipping parameters, but it is safer.

A shorter version would be:

```python
for online_param, target_param in zip(online.parameters(), target.parameters()):
    target_param.data.mul_(tau).add_(online_param.data, alpha=1.0 - tau)
```

The explicit version is better for tutorial code because it fails loudly if the modules do not match.

---

## 2.6.6 Handling Buffers

Modules may contain buffers, for example BatchNorm running statistics.

Our minimal ViT encoder uses LayerNorm and does not have running-stat buffers that need EMA updates. But for completeness, we can also copy buffers.

Add:

```python
@torch.no_grad()
def copy_buffers(
    online: nn.Module,
    target: nn.Module,
) -> None:
    """
    Copy buffers from online module to target module.

    For the minimal ViT encoder this is usually not important, but it
    makes the utility safer if future modules add buffers.
    """
    online_buffers = dict(online.named_buffers())
    target_buffers = dict(target.named_buffers())

    if online_buffers.keys() != target_buffers.keys():
        return

    for name in online_buffers:
        target_buffers[name].copy_(online_buffers[name])
```

Then we can optionally call it after EMA:

```python
copy_buffers(online, target)
```

For this minimal implementation, parameter EMA is the main mechanism.

---

## 2.6.7 EMA Momentum Schedule

A constant EMA momentum is fine for initial tests:

```python
tau = 0.996
```

But a schedule is useful for longer training.

A common choice is a cosine schedule from `tau_base` to `tau_final`.

\[
\tau_t
=
\tau_{\mathrm{final}}
-
(\tau_{\mathrm{final}} - \tau_{\mathrm{base}})
\cdot
\frac{1 + \cos(\pi t / T)}{2}
\]

At \(t = 0\):

\[
\tau_t = \tau_{\mathrm{base}}
\]

At \(t = T\):

\[
\tau_t = \tau_{\mathrm{final}}
\]

Add:

```python
def cosine_ema_tau(
    step: int,
    total_steps: int,
    tau_base: float = 0.996,
    tau_final: float = 1.0,
) -> float:
    """
    Cosine schedule for EMA momentum.

    Args:
        step:
            Current training step.

        total_steps:
            Total number of training steps.

        tau_base:
            Initial EMA momentum.

        tau_final:
            Final EMA momentum.

    Returns:
        EMA momentum for current step.
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be positive, got {total_steps}.")

    if step < 0:
        raise ValueError(f"step must be non-negative, got {step}.")

    if not 0.0 <= tau_base <= 1.0:
        raise ValueError(f"tau_base must be in [0, 1], got {tau_base}.")

    if not 0.0 <= tau_final <= 1.0:
        raise ValueError(f"tau_final must be in [0, 1], got {tau_final}.")

    if tau_base > tau_final:
        raise ValueError(
            f"tau_base should be <= tau_final. "
            f"Got tau_base={tau_base}, tau_final={tau_final}."
        )

    progress = min(step / total_steps, 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return tau_final - (tau_final - tau_base) * cosine
```

Example:

```python
tau = cosine_ema_tau(
    step=global_step,
    total_steps=total_steps,
    tau_base=0.996,
    tau_final=1.0,
)
```

---

## 2.6.8 EMA Update Timing

The target encoder should usually be updated after the optimizer step.

The training sequence is:

```python
pred_repr, target_repr = model(
    images,
    context_indices,
    target_indices,
)

loss = loss_fn(pred_repr, target_repr)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()

update_ema(
    online=model.online_encoder,
    target=model.target_encoder,
    tau=ema_tau,
)
```

Why after `optimizer.step()`?

Because the target should move toward the newly updated online encoder.

If we update EMA before the optimizer step, the target lags by one update. That is not catastrophic, but updating after the optimizer step is the convention we will use.

---

## 2.6.9 Target Encoder Must Not Receive Gradients

The target encoder should satisfy:

```python
all(not p.requires_grad for p in target_encoder.parameters())
```

During forward pass, target encoding should happen inside:

```python
with torch.no_grad():
    target_repr = target_encoder(images, target_indices)
```

The target representation can also be detached before the loss:

```python
loss = loss_fn(pred_repr, target_repr.detach())
```

Using both `torch.no_grad()` and `.detach()` is redundant but safe. It makes the intended behavior clear.

The target branch is a target, not a trainable path.

---

## 2.6.10 Verifying EMA Behavior

A simple scalar example helps.

Suppose one online parameter is:

\[
\theta = 10
\]

and the corresponding target parameter is:

\[
\bar{\theta} = 0
\]

With:

\[
\tau = 0.9
\]

the update gives:

\[
\bar{\theta}
\leftarrow
0.9 \cdot 0 + 0.1 \cdot 10
=
1
\]

Another update with the same online value gives:

\[
\bar{\theta}
\leftarrow
0.9 \cdot 1 + 0.1 \cdot 10
=
1.9
\]

The target approaches the online value gradually.

In code:

```python
target = tau * target + (1.0 - tau) * online
```

This is the same rule applied to every parameter tensor.

---

## 2.6.11 EMA Distance Diagnostic

It is useful to monitor how far the target encoder is from the online encoder.

Add:

```python
@torch.no_grad()
def parameter_distance(
    online: nn.Module,
    target: nn.Module,
) -> dict[str, float]:
    """
    Compute average parameter distance between online and target modules.
    """
    online_params = dict(online.named_parameters())
    target_params = dict(target.named_parameters())

    if online_params.keys() != target_params.keys():
        raise ValueError(
            "Online and target modules have different parameter names."
        )

    sq_dist = 0.0
    sq_norm = 0.0
    num_tensors = 0

    for name in online_params:
        online_param = online_params[name].detach().float()
        target_param = target_params[name].detach().float()

        diff = online_param - target_param

        sq_dist += diff.pow(2).sum().item()
        sq_norm += online_param.pow(2).sum().item()
        num_tensors += 1

    l2_dist = math.sqrt(sq_dist)
    online_norm = math.sqrt(sq_norm)

    relative = l2_dist / max(online_norm, 1e-12)

    return {
        "ema/param_l2": l2_dist,
        "ema/relative_param_l2": relative,
        "ema/num_tensors": float(num_tensors),
    }
```

This can be logged occasionally:

```python
logs.update(
    parameter_distance(
        online=model.online_encoder,
        target=model.target_encoder,
    )
)
```

If the distance is exactly zero for all training, EMA may not be updating or the online encoder may not be learning.

If the distance explodes, training may be unstable.

---

## 2.6.12 Integrating EMA into the JEPA Model

The top-level JEPA model will own both encoders.

A minimal version:

```python
import torch
import torch.nn as nn

from jepa_world_model.ema import initialize_target_encoder


class MinimalIJEPA(nn.Module):
    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        predictor: nn.Module,
    ):
        super().__init__()

        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.predictor = predictor

        initialize_target_encoder(
            online=self.online_encoder,
            target=self.target_encoder,
        )

    def forward(
        self,
        images: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_repr = self.online_encoder(
            images=images,
            patch_indices=context_indices,
        )

        with torch.no_grad():
            target_repr = self.target_encoder(
                images=images,
                patch_indices=target_indices,
            )

        pred_repr = self.predictor(
            context_repr=context_repr,
            context_indices=context_indices,
            target_indices=target_indices,
        )

        return pred_repr, target_repr
```

The model initializes and freezes the target encoder, but it does not update EMA internally.

EMA update belongs in the training loop because it depends on the global step and schedule.

---

## 2.6.13 Unit Tests

Create:

```text
tests/test_ema.py
```

Add:

```python
import torch
import torch.nn as nn

from jepa_world_model.ema import (
    cosine_ema_tau,
    initialize_target_encoder,
    parameter_distance,
    update_ema,
)


def test_initialize_target_encoder_copies_weights():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    initialize_target_encoder(
        online=online,
        target=target,
    )

    for p_online, p_target in zip(
        online.parameters(),
        target.parameters(),
    ):
        torch.testing.assert_close(p_online, p_target)


def test_initialize_target_encoder_freezes_target():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    initialize_target_encoder(
        online=online,
        target=target,
    )

    assert all(
        not p.requires_grad
        for p in target.parameters()
    )


def test_update_ema_tau_zero_copies_online():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    with torch.no_grad():
        for p in online.parameters():
            p.fill_(2.0)

        for p in target.parameters():
            p.fill_(0.0)

    update_ema(
        online=online,
        target=target,
        tau=0.0,
    )

    for p_target in target.parameters():
        torch.testing.assert_close(
            p_target,
            torch.full_like(p_target, 2.0),
        )


def test_update_ema_tau_one_keeps_target():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    with torch.no_grad():
        for p in online.parameters():
            p.fill_(2.0)

        for p in target.parameters():
            p.fill_(0.0)

    update_ema(
        online=online,
        target=target,
        tau=1.0,
    )

    for p_target in target.parameters():
        torch.testing.assert_close(
            p_target,
            torch.zeros_like(p_target),
        )


def test_update_ema_midpoint():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    with torch.no_grad():
        for p in online.parameters():
            p.fill_(10.0)

        for p in target.parameters():
            p.fill_(0.0)

    update_ema(
        online=online,
        target=target,
        tau=0.9,
    )

    for p_target in target.parameters():
        torch.testing.assert_close(
            p_target,
            torch.full_like(p_target, 1.0),
        )


def test_cosine_ema_tau_start_and_end():
    tau_start = cosine_ema_tau(
        step=0,
        total_steps=100,
        tau_base=0.996,
        tau_final=1.0,
    )

    tau_end = cosine_ema_tau(
        step=100,
        total_steps=100,
        tau_base=0.996,
        tau_final=1.0,
    )

    assert abs(tau_start - 0.996) < 1e-12
    assert abs(tau_end - 1.0) < 1e-12


def test_parameter_distance_zero_after_initialization():
    online = nn.Linear(4, 8)
    target = nn.Linear(4, 8)

    initialize_target_encoder(
        online=online,
        target=target,
    )

    stats = parameter_distance(
        online=online,
        target=target,
    )

    assert stats["ema/param_l2"] == 0.0
```

Run:

```bash
pytest tests/test_ema.py
```

---

## 2.6.14 Integration Test with ViT Encoder

Add an integration test:

```python
from jepa_world_model.ema import initialize_target_encoder, update_ema
from jepa_world_model.vit import MinimalViTEncoder


def test_ema_with_vit_encoder():
    online = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    target = MinimalViTEncoder(
        image_size=32,
        patch_size=4,
        in_channels=3,
        embed_dim=64,
        depth=2,
        num_heads=4,
    )

    initialize_target_encoder(
        online=online,
        target=target,
    )

    # Modify online parameters.
    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.01 * torch.randn_like(p))

    before = parameter_distance(
        online=online,
        target=target,
    )

    update_ema(
        online=online,
        target=target,
        tau=0.9,
    )

    after = parameter_distance(
        online=online,
        target=target,
    )

    assert after["ema/param_l2"] < before["ema/param_l2"]
```

This confirms that EMA moves the target encoder toward the online encoder.

---

## 2.6.15 marimo Debug Notebook

Create:

```text
notebooks/03_debug_ema.py
```

Open:

```bash
marimo edit notebooks/03_debug_ema.py
```

Useful cells:

```python
import torch

from jepa_world_model.ema import (
    cosine_ema_tau,
    initialize_target_encoder,
    parameter_distance,
    update_ema,
)
from jepa_world_model.vit import MinimalViTEncoder
```

```python
online = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
)

target = MinimalViTEncoder(
    image_size=32,
    patch_size=4,
    in_channels=3,
    embed_dim=64,
    depth=2,
    num_heads=4,
)

initialize_target_encoder(online, target)

parameter_distance(online, target)
```

Perturb online encoder:

```python
with torch.no_grad():
    for p in online.parameters():
        p.add_(0.01 * torch.randn_like(p))

parameter_distance(online, target)
```

Apply EMA:

```python
update_ema(
    online=online,
    target=target,
    tau=0.9,
)

parameter_distance(online, target)
```

Plot the EMA schedule:

```python
import matplotlib.pyplot as plt

total_steps = 10_000

taus = [
    cosine_ema_tau(
        step=s,
        total_steps=total_steps,
        tau_base=0.996,
        tau_final=1.0,
    )
    for s in range(total_steps + 1)
]

plt.figure(figsize=(6, 4))
plt.plot(taus)
plt.xlabel("step")
plt.ylabel("EMA tau")
plt.title("Cosine EMA schedule")
plt.show()
```

---

## 2.6.16 Common Bugs

### Bug 1: Target encoder receives gradients

Check:

```python
assert all(
    not p.requires_grad
    for p in model.target_encoder.parameters()
)
```

Also check after backward:

```python
target_has_grad = any(
    p.grad is not None
    for p in model.target_encoder.parameters()
)

assert not target_has_grad
```

---

### Bug 2: Forgetting `torch.no_grad()` around target forward

Use:

```python
with torch.no_grad():
    target_repr = target_encoder(images, target_indices)
```

This saves memory and prevents accidental graph construction.

---

### Bug 3: Updating EMA before optimizer step

Use:

```python
optimizer.step()
update_ema(...)
```

not:

```python
update_ema(...)
optimizer.step()
```

---

### Bug 4: Online and target architectures differ

EMA requires matching parameter names and shapes.

This should fail:

```python
online = MinimalViTEncoder(embed_dim=64, ...)
target = MinimalViTEncoder(embed_dim=128, ...)
```

The explicit `update_ema` implementation will catch this.

---

### Bug 5: Including target encoder in optimizer

The optimizer should only receive parameters from:

```text
online encoder
predictor
```

not the target encoder.

Correct:

```python
optimizer = torch.optim.AdamW(
    list(model.online_encoder.parameters())
    + list(model.predictor.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay,
)
```

Avoid:

```python
optimizer = torch.optim.AdamW(model.parameters(), ...)
```

because that includes the target encoder. Even if target parameters have `requires_grad=False`, being explicit is safer.

---

## 2.6.17 Summary

The EMA target encoder is a central part of JEPA training.

It provides a slowly moving representation target for the prediction branch.

In this section, we implemented:

- target encoder initialization,
- target encoder freezing,
- EMA parameter updates,
- optional buffer copying,
- cosine EMA momentum schedule,
- parameter-distance diagnostics,
- unit tests,
- integration tests,
- marimo debugging workflow.

The next section implements the JEPA predictor: the network that maps context representations and target positions into predicted target representations.

---

## References and Further Reading

- Mahmoud Assran et al., **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.
  <https://arxiv.org/abs/2301.08243>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**, 2020.
  <https://arxiv.org/abs/2006.07733>

- Xinlei Chen and Kaiming He, **Exploring Simple Siamese Representation Learning**, 2020.
  <https://arxiv.org/abs/2011.10566>

- Facebook Research, **Official I-JEPA Codebase**.
  <https://github.com/facebookresearch/ijepa>
