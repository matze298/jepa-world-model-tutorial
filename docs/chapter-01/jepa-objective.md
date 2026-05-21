# 1.3 The JEPA Objective

The core idea of JEPA is simple:

> Predict the representation of an unobserved target from the representation of an observed context.

This differs from reconstruction-based self-supervised learning. A masked autoencoder predicts missing pixels. A JEPA-style model predicts missing **representations**.

For an image, this means the model may observe one set of patches and predict the representation of another set of patches. For video, it may observe some spatiotemporal regions and predict the representation of hidden or future regions. For time-series world modeling, it may observe a past window and predict the representation of a future window.

The generic structure is:

\[
\text{context}
\longrightarrow
\text{predicted target representation}
\]

More explicitly:

\[
x_{\mathrm{ctx}}
\longrightarrow
\hat{z}_{\mathrm{tgt}}
\]

where:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The target is not the raw observation \(x_{\mathrm{tgt}}\), but an embedding produced by a target encoder.

This section develops the JEPA objective from first principles and translates every mathematical component into an implementation concept.

---

## 1.3.1 The Core Setup

Let \(x\) be an observation.

For an image:

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

For a video:

\[
x \in \mathbb{R}^{3 \times T \times H \times W}
\]

For a multivariate time series:

\[
x \in \mathbb{R}^{T \times D}
\]

JEPA begins by splitting the observation into two parts:

\[
x_{\mathrm{ctx}}
\]

and:

\[
x_{\mathrm{tgt}}
\]

where:

- \(x_{\mathrm{ctx}}\) is the context available to the predictor,
- \(x_{\mathrm{tgt}}\) is the target that must be predicted in representation space.

In an image, \(x_{\mathrm{ctx}}\) may correspond to visible patch blocks, while \(x_{\mathrm{tgt}}\) may correspond to hidden target blocks.

In a temporal model, \(x_{\mathrm{ctx}}\) may be a past window, while \(x_{\mathrm{tgt}}\) may be a future window.

The online encoder maps the context to a representation:

\[
z_{\mathrm{ctx}}
=
f_\theta(x_{\mathrm{ctx}})
\]

The target encoder maps the target to a representation:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The predictor maps the context representation to a predicted target representation:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}}, m_{\mathrm{tgt}})
\]

where \(m_{\mathrm{tgt}}\) contains information about the target. For an image, this might be the target patch positions. For a temporal model, this might be the prediction horizon.

The JEPA loss compares the prediction to the target representation:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
d(
\hat{z}_{\mathrm{tgt}},
\mathrm{sg}(z_{\mathrm{tgt}})
)
\]

where:

- \(d\) is a distance function,
- \(\mathrm{sg}(\cdot)\) denotes stop-gradient.

A common version is:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
\left\|
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathrm{tgt}}))
\right\|_2^2
\]

This is the central equation.

Everything else in the method exists to make this equation work.

---

## 1.3.2 What Each Component Does

The JEPA objective contains four main components:

1. online encoder,
2. target encoder,
3. predictor,
4. mask or target metadata.

We will use the following notation:

\[
f_\theta
\]

for the online encoder,

\[
f_{\bar{\theta}}
\]

for the target encoder,

\[
g_\theta
\]

for the predictor,

and:

\[
m
\]

for mask or target-position information.

---

### Online Encoder

The online encoder processes the context.

\[
z_{\mathrm{ctx}}
=
f_\theta(x_{\mathrm{ctx}})
\]

It is updated by backpropagation.

For images, the online encoder will usually be a Vision Transformer applied to context patch tokens. For temporal telemetry, it might be a transformer, temporal convolution network, recurrent model, or state-space model.

A minimal abstract interface is:

```python
class ContextEncoder(torch.nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:
            Input observation.

        context_indices:
            Indices specifying which parts of x are visible context.

        returns:
            Context representation.
        """
        raise NotImplementedError
```

For images, we will eventually use:

```python
context_repr = online_encoder(
    images=images,
    patch_indices=context_indices,
)
```

For time series, the interface might become:

```python
context_repr = context_encoder(
    past_window=past_observations,
)
```

The online encoder is the representation learner we will usually keep after pretraining.

---

### Target Encoder

The target encoder processes the target.

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

Unlike the online encoder, the target encoder is not directly updated by gradient descent from the JEPA loss. Its output is stop-gradient-ed:

\[
\mathrm{sg}(z_{\mathrm{tgt}})
\]

The target encoder usually tracks the online encoder using exponential moving average:

\[
\bar{\theta}
\leftarrow
\tau \bar{\theta}
+
(1-\tau)\theta
\]

where \(\tau\) is close to one.

In code:

```python
@torch.no_grad()
def ema_update(
    online: torch.nn.Module,
    target: torch.nn.Module,
    tau: float,
) -> None:
    for online_param, target_param in zip(
        online.parameters(),
        target.parameters(),
    ):
        target_param.data.mul_(tau).add_(
            online_param.data,
            alpha=1.0 - tau,
        )
```

The target encoder gives the model a slowly moving prediction target.

This is important because if both branches were updated freely by the same loss, the system could drift toward trivial solutions more easily.

---

### Predictor

The predictor maps context representations to target representations.

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}}, m_{\mathrm{tgt}})
\]

The predictor is not just a small head. In JEPA-style models, it often has a meaningful architecture, such as a lightweight transformer.

The predictor must answer:

> Given the context representation, what should the target representation be?

For images, it also needs to know **where** the target is. Predicting the representation of an arbitrary missing patch is under-specified. Predicting the representation of a specific target block is meaningful.

A useful predictor interface is:

```python
class JEPAPredictor(torch.nn.Module):
    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        context_repr:
            [batch, num_context_tokens, dim]

        context_indices:
            [batch, num_context_tokens]

        target_indices:
            [batch, num_target_tokens]

        returns:
            Predicted target representations,
            [batch, num_target_tokens, dim]
        """
        raise NotImplementedError
```

The predictor is trained by gradient descent together with the online encoder.

After pretraining, the predictor is often discarded. The encoder is the main learned representation model.

---

### Mask or Target Metadata

The variable \(m_{\mathrm{tgt}}\) contains information about the target.

For image JEPA, this includes target patch positions:

\[
m_{\mathrm{tgt}} = \mathcal{T}
\]

For temporal JEPA, it may include prediction horizon:

\[
m_{\mathrm{tgt}} = k
\]

For action-conditioned world models, it may include future actions:

\[
m_{\mathrm{tgt}} = a_{t:t+k}
\]

This is why the generic JEPA objective is better written as:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}}, m_{\mathrm{tgt}})
\]

rather than:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}})
\]

The target metadata defines what the prediction is about.

---

## 1.3.3 A First Complete PyTorch Skeleton

Before going deeper into the math, it is useful to see the entire skeleton.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class JEPAModel(nn.Module):
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

        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(
            self.online_encoder.state_dict()
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        x: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_repr = self.online_encoder(
            x=x,
            indices=context_indices,
        )

        with torch.no_grad():
            target_repr = self.target_encoder(
                x=x,
                indices=target_indices,
            )

        pred_repr = self.predictor(
            context_repr=context_repr,
            context_indices=context_indices,
            target_indices=target_indices,
        )

        return pred_repr, target_repr
```

The loss:

```python
def jepa_loss(
    pred_repr: torch.Tensor,
    target_repr: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(
        pred_repr,
        target_repr.detach(),
    )
```

The EMA update:

```python
@torch.no_grad()
def update_target_encoder(
    online_encoder: nn.Module,
    target_encoder: nn.Module,
    tau: float,
) -> None:
    for p_online, p_target in zip(
        online_encoder.parameters(),
        target_encoder.parameters(),
    ):
        p_target.data.mul_(tau).add_(
            p_online.data,
            alpha=1.0 - tau,
        )
```

The training step:

```python
def train_jepa_step(
    model: JEPAModel,
    x: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    tau: float,
) -> torch.Tensor:
    pred_repr, target_repr = model(
        x=x,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    loss = jepa_loss(pred_repr, target_repr)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    update_target_encoder(
        online_encoder=model.online_encoder,
        target_encoder=model.target_encoder,
        tau=tau,
    )

    return loss.detach()
```

This skeleton already contains the entire algorithmic structure.

What it does not yet contain are the details that make the method work:

- patch embeddings,
- positional encodings,
- context and target masks,
- predictor architecture,
- normalization,
- EMA schedule,
- diagnostics.

Those will come later.

---

## 1.3.4 Stop-Gradient

The stop-gradient operator is essential.

The target representation is:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The loss compares:

\[
\hat{z}_{\mathrm{tgt}}
\]

to:

\[
\mathrm{sg}(z_{\mathrm{tgt}})
\]

The stop-gradient means:

\[
\frac{\partial \mathrm{sg}(z_{\mathrm{tgt}})}
{\partial z_{\mathrm{tgt}}}
=
0
\]

So gradients do not flow into the target encoder through the loss.

In code, this is achieved either with `torch.no_grad()`:

```python
with torch.no_grad():
    target_repr = target_encoder(x, target_indices)
```

or by detaching the target:

```python
target_repr = target_encoder(x, target_indices).detach()
```

Using both is common and safe:

```python
with torch.no_grad():
    target_repr = target_encoder(x, target_indices)

loss = F.smooth_l1_loss(pred_repr, target_repr.detach())
```

The stop-gradient makes the loss asymmetric.

The prediction branch changes to match the target branch. The target branch does not directly change to reduce the loss.

This asymmetry is shared with methods such as BYOL and SimSiam, where stop-gradient and predictor asymmetry play an important role in avoiding collapse. See [Grill et al., 2020 — BYOL](https://arxiv.org/abs/2006.07733) and [Chen and He, 2020 — SimSiam](https://arxiv.org/abs/2011.10566).

---

## 1.3.5 Exponential Moving Average Target Encoder

The target encoder is usually updated as an exponential moving average of the online encoder.

The update is:

\[
\bar{\theta}_{t+1}
=
\tau_t \bar{\theta}_t
+
(1-\tau_t)\theta_t
\]

where:

- \(\theta_t\) are the online encoder parameters,
- \(\bar{\theta}_t\) are the target encoder parameters,
- \(\tau_t\) is the EMA momentum.

If \(\tau_t\) is close to one, the target encoder changes slowly.

This gives the online encoder a stable target. The target representations are not fixed forever, but they do not move as quickly as the online parameters.

A constant EMA momentum is simple:

```python
tau = 0.996
```

A scheduled EMA momentum is often better. For example, \(\tau\) can increase from a lower value to a value close to one over training:

```python
import math


def cosine_ema_schedule(
    step: int,
    total_steps: int,
    tau_base: float = 0.996,
    tau_final: float = 1.0,
) -> float:
    progress = step / total_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return tau_final - (tau_final - tau_base) * cosine
```

This starts at `tau_base` and gradually approaches `tau_final`.

In training:

```python
tau = cosine_ema_schedule(
    step=global_step,
    total_steps=total_steps,
    tau_base=0.996,
    tau_final=1.0,
)

update_target_encoder(
    online_encoder=model.online_encoder,
    target_encoder=model.target_encoder,
    tau=tau,
)
```

The intuition is:

- early in training, the target can track the online encoder more quickly,
- later in training, the target becomes more stable.

This is not merely an optimization trick. It changes the dynamics of the learning system.

---

## 1.3.6 The Predictor as an Asymmetric Bottleneck

The predictor is a crucial part of the architecture.

Without a predictor, we might train:

\[
f_\theta(x_{\mathrm{ctx}})
\approx
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

But context and target may have different numbers of tokens, different positions, and different information content. The predictor maps from one structure to another.

For image JEPA, suppose:

\[
z_{\mathrm{ctx}}
\in
\mathbb{R}^{B \times N_c \times D}
\]

and:

\[
z_{\mathrm{tgt}}
\in
\mathbb{R}^{B \times N_t \times D}
\]

where:

- \(B\) is batch size,
- \(N_c\) is the number of context tokens,
- \(N_t\) is the number of target tokens,
- \(D\) is the embedding dimension.

The predictor must output:

\[
\hat{z}_{\mathrm{tgt}}
\in
\mathbb{R}^{B \times N_t \times D}
\]

One simple predictor design is:

1. project context tokens to predictor dimension,
2. create learned mask/query tokens for targets,
3. add target positional embeddings,
4. run a transformer over context and target queries,
5. return the output corresponding to target queries.

A simplified predictor skeleton:

```python
class TransformerPredictor(nn.Module):
    def __init__(
        self,
        encoder_dim: int,
        predictor_dim: int,
        depth: int,
        num_heads: int,
        num_patches: int,
    ):
        super().__init__()

        self.context_proj = nn.Linear(encoder_dim, predictor_dim)
        self.target_query = nn.Parameter(
            torch.zeros(1, 1, predictor_dim)
        )

        self.pos_embed = nn.Embedding(
            num_patches,
            predictor_dim,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=predictor_dim,
            nhead=num_heads,
            batch_first=True,
            dim_feedforward=4 * predictor_dim,
            activation="gelu",
            norm_first=True,
        )

        self.blocks = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=depth,
        )

        self.out_proj = nn.Linear(predictor_dim, encoder_dim)

    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_repr.size(0)

        context_tokens = self.context_proj(context_repr)
        context_tokens = context_tokens + self.pos_embed(context_indices)

        num_targets = target_indices.size(1)

        target_queries = self.target_query.expand(
            batch_size,
            num_targets,
            -1,
        )

        target_queries = target_queries + self.pos_embed(target_indices)

        tokens = torch.cat(
            [context_tokens, target_queries],
            dim=1,
        )

        tokens = self.blocks(tokens)

        target_out = tokens[:, -num_targets:]
        pred = self.out_proj(target_out)

        return pred
```

This is not yet optimized. It is intentionally readable.

The main point is that target locations enter through positional embeddings:

```python
target_queries = target_queries + self.pos_embed(target_indices)
```

Without this information, the predictor would not know which target representation to produce.

---

## 1.3.7 Choice of Distance Function

The JEPA objective needs a distance function:

\[
d(\hat{z}, z)
\]

Several choices are possible.

### Mean Squared Error

\[
d_{\mathrm{MSE}}(\hat{z}, z)
=
\|\hat{z} - z\|_2^2
\]

Code:

```python
def mse_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred, target.detach())
```

MSE is simple and stable, but it is sensitive to scale.

---

### Smooth L1 Loss

Smooth L1 is less sensitive to outliers.

```python
def smooth_l1_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target.detach())
```

This is often a reasonable default.

---

### Cosine Distance

Cosine distance focuses on representation direction:

\[
d_{\mathrm{cos}}(\hat{z}, z)
=
1
-
\frac{
\hat{z}^\top z
}{
\|\hat{z}\|\|z\|
}
\]

Code:

```python
def cosine_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)

    return 1.0 - (pred * target).sum(dim=-1).mean()
```

Cosine loss is scale-invariant, but the representation norm may still carry useful information. If everything is normalized, the model cannot use magnitude as part of the representation.

---

### Combined Loss

A combined loss can use both magnitude and direction:

```python
def combined_latent_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 1.0,
    beta: float = 0.1,
) -> torch.Tensor:
    mse = F.mse_loss(pred, target.detach())

    pred_norm = F.normalize(pred, dim=-1)
    target_norm = F.normalize(target.detach(), dim=-1)
    cosine = 1.0 - (pred_norm * target_norm).sum(dim=-1).mean()

    return alpha * mse + beta * cosine
```

The best choice is empirical. In this tutorial, we will begin with Smooth L1 or MSE because they are simple and align well with the basic JEPA formulation.

---

## 1.3.8 Collapse and Trivial Solutions

A central concern in joint-embedding methods is collapse.

A collapsed representation maps many or all inputs to the same vector:

\[
f(x) = c
\]

for all \(x\).

If this happens, the prediction problem becomes trivial:

\[
\hat{z}_{\mathrm{tgt}} = c
\]

\[
z_{\mathrm{tgt}} = c
\]

The loss can be low, but the representation is useless.

JEPA avoids trivial solutions through a combination of design choices:

- target encoder stop-gradient,
- EMA target encoder,
- predictor asymmetry,
- structured masking,
- sufficient target difficulty,
- normalization and architecture choices,
- monitoring representation statistics.

There is no single magic component. The system works as a training dynamic.

This means diagnostics are essential.

A simple collapse diagnostic checks variance across the batch and token dimensions:

```python
@torch.no_grad()
def collapse_diagnostics(
    z: torch.Tensor,
    eps: float = 1e-4,
) -> dict[str, float]:
    """
    z:
        [batch, tokens, dim] or [batch, dim]
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    z = z.float()

    std_per_dim = z.std(dim=0)
    dead_dims = (std_per_dim < eps).float().mean()

    return {
        "repr/std_mean": std_per_dim.mean().item(),
        "repr/std_min": std_per_dim.min().item(),
        "repr/std_max": std_per_dim.max().item(),
        "repr/dead_dim_fraction": dead_dims.item(),
        "repr/norm_mean": z.norm(dim=-1).mean().item(),
    }
```

During training, we should log diagnostics for both prediction and target:

```python
with torch.no_grad():
    pred_stats = collapse_diagnostics(pred_repr)
    target_stats = collapse_diagnostics(target_repr)
```

A falling loss is not enough.

If the target representation variance collapses, the model may be solving nothing.

---

## 1.3.9 Mask Leakage

Mask leakage occurs when the model has access to information that should have been hidden.

For image JEPA, this can happen if:

- context and target patch indices overlap,
- positional information reveals too much,
- augmentation or preprocessing accidentally includes target content,
- target tokens are accidentally passed into the online branch.

The most obvious leakage is index overlap.

We should assert:

\[
\mathcal{C} \cap \mathcal{T} = \emptyset
\]

In code:

```python
def assert_no_mask_overlap(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> None:
    """
    context_indices:
        [batch, num_context]

    target_indices:
        [batch, num_target]
    """
    batch_size = context_indices.size(0)

    for b in range(batch_size):
        overlap = torch.isin(
            target_indices[b],
            context_indices[b],
        )

        if overlap.any():
            raise ValueError(
                f"Mask leakage detected in batch item {b}."
            )
```

For large-scale training, a loop over batch items may be slow. But during development, explicit assertions are valuable.

A vectorized diagnostic can estimate overlap fraction:

```python
@torch.no_grad()
def mask_overlap_fraction(
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> float:
    overlaps = []

    for b in range(context_indices.size(0)):
        overlap = torch.isin(
            target_indices[b],
            context_indices[b],
        )
        overlaps.append(overlap.float().mean())

    return torch.stack(overlaps).mean().item()
```

This should be exactly zero.

Mask leakage can produce deceptively good losses. The predictor may appear strong because the target was never truly hidden.

---

## 1.3.10 Multiple Targets

I-JEPA often predicts multiple target blocks from a single context.

Let there be \(K\) target regions:

\[
x_{\mathrm{tgt}}^{(1)}, \dots, x_{\mathrm{tgt}}^{(K)}
\]

The target encoder produces:

\[
z_{\mathrm{tgt}}^{(k)}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}}^{(k)})
\]

The predictor outputs:

\[
\hat{z}_{\mathrm{tgt}}^{(k)}
=
g_\theta(z_{\mathrm{ctx}}, m_{\mathrm{tgt}}^{(k)})
\]

The total loss is:

\[
\mathcal{L}
=
\frac{1}{K}
\sum_{k=1}^{K}
d(
\hat{z}_{\mathrm{tgt}}^{(k)},
\mathrm{sg}(z_{\mathrm{tgt}}^{(k)})
)
\]

In implementation, we can either concatenate all target tokens or keep a target-block dimension.

Concatenated representation:

```python
target_indices.shape
# [batch, total_target_tokens]

pred_repr.shape
# [batch, total_target_tokens, dim]

target_repr.shape
# [batch, total_target_tokens, dim]
```

Block-structured representation:

```python
target_indices.shape
# [batch, num_blocks, tokens_per_block]

pred_repr.shape
# [batch, num_blocks, tokens_per_block, dim]

target_repr.shape
# [batch, num_blocks, tokens_per_block, dim]
```

The block-structured version is more explicit. The concatenated version is simpler.

A loss function can support both by flattening:

```python
def multi_target_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Supports:
        [B, T, D]
        [B, K, T, D]
    """
    if pred.dim() == 4:
        pred = pred.flatten(1, 2)
        target = target.flatten(1, 2)

    return F.smooth_l1_loss(pred, target.detach())
```

Multiple targets improve the training signal because one context produces several prediction tasks.

---

## 1.3.11 Batch and Shape Discipline

JEPA implementations are shape-sensitive.

A large fraction of bugs come from silently mixing:

- batch dimension,
- token dimension,
- block dimension,
- feature dimension.

We will use the following conventions:

```text
B = batch size
N = total number of image patches
C = number of context tokens
T = number of target tokens
K = number of target blocks
D = encoder embedding dimension
P = predictor embedding dimension
```

Common shapes:

```python
images.shape
# [B, 3, H, W]

patch_tokens.shape
# [B, N, D]

context_indices.shape
# [B, C]

target_indices.shape
# [B, T]

context_repr.shape
# [B, C, D]

target_repr.shape
# [B, T, D]

pred_repr.shape
# [B, T, D]
```

A useful habit is to assert shapes during development:

```python
def assert_jepa_shapes(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape}, target {target.shape}"
        )

    if pred.dim() != 3:
        raise ValueError(
            f"Expected [B, T, D], got {pred.shape}"
        )
```

Then:

```python
assert_jepa_shapes(pred_repr, target_repr)
loss = F.smooth_l1_loss(pred_repr, target_repr.detach())
```

This may look mundane, but strict shape discipline makes research code dramatically easier to debug.

---

## 1.3.12 JEPA as Conditional Representation Prediction

Mathematically, JEPA can be viewed as conditional prediction in latent space.

Instead of modeling:

\[
p(x_{\mathrm{tgt}} \mid x_{\mathrm{ctx}})
\]

JEPA learns a predictor for:

\[
p(z_{\mathrm{tgt}} \mid z_{\mathrm{ctx}}, m_{\mathrm{tgt}})
\]

In deterministic form:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(z_{\mathrm{ctx}}, m_{\mathrm{tgt}})
\]

This is a point prediction. It predicts one latent target.

But the target may be uncertain. There may be multiple plausible target representations. A deterministic predictor may therefore learn a conditional average in latent space.

Later, when we move toward world models, we may consider stochastic or energy-based variants:

\[
E_\theta(z_{\mathrm{ctx}}, z_{\mathrm{tgt}}, m)
\]

where the model assigns low energy to compatible context-target pairs and high energy to incompatible ones.

The original JEPA vision is broader than a single MSE objective. It is a framework for predictive learning in representation space.

For this tutorial, we begin with the deterministic version because it is the simplest to implement and analyze.

---

## 1.3.13 Connection to World Models

The image JEPA objective is:

\[
x_{\mathrm{ctx}}
\rightarrow
z_{\mathrm{tgt}}
\]

The temporal JEPA objective is:

\[
x_{\leq t}
\rightarrow
z_{t+k}
\]

The action-conditioned world model objective is:

\[
(x_{\leq t}, a_{t:t+k})
\rightarrow
z_{t+k}
\]

The same structure appears in all three cases.

For a temporal world model:

\[
z_{\mathrm{past}}
=
f_\theta(x_{t-L:t})
\]

\[
z_{\mathrm{future}}
=
f_{\bar{\theta}}(x_{t+1:t+H})
\]

\[
\hat{z}_{\mathrm{future}}
=
F_\phi(z_{\mathrm{past}}, a_{t:t+H})
\]

\[
\mathcal{L}
=
\left\|
\hat{z}_{\mathrm{future}}
-
\mathrm{sg}(z_{\mathrm{future}})
\right\|^2
\]

This is JEPA with a temporal mask.

A skeleton implementation:

```python
class TemporalJEPA(nn.Module):
    def __init__(
        self,
        context_encoder: nn.Module,
        target_encoder: nn.Module,
        dynamics: nn.Module,
    ):
        super().__init__()
        self.context_encoder = context_encoder
        self.target_encoder = target_encoder
        self.dynamics = dynamics

        self.target_encoder.load_state_dict(
            self.context_encoder.state_dict()
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        past_obs: torch.Tensor,
        future_obs: torch.Tensor,
        future_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        past_z = self.context_encoder(past_obs)

        with torch.no_grad():
            future_z = self.target_encoder(future_obs)

        pred_future_z = self.dynamics(
            past_z=past_z,
            future_actions=future_actions,
        )

        return pred_future_z, future_z
```

The loss is the same:

```python
pred_z, target_z = model(
    past_obs=past_obs,
    future_obs=future_obs,
    future_actions=future_actions,
)

loss = F.mse_loss(pred_z, target_z.detach())
```

This is why JEPA is a natural bridge from self-supervised representation learning to world modeling.

---

## 1.3.14 A Minimal Training Loop with Logging

A practical training step should return more than the loss.

It should also return diagnostics.

```python
def train_step_with_logs(
    model: JEPAModel,
    x: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    step: int,
    total_steps: int,
) -> dict[str, float]:
    tau = cosine_ema_schedule(
        step=step,
        total_steps=total_steps,
        tau_base=0.996,
        tau_final=1.0,
    )

    pred_repr, target_repr = model(
        x=x,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    assert_jepa_shapes(pred_repr, target_repr)

    loss = smooth_l1_latent_loss(pred_repr, target_repr)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    update_target_encoder(
        online_encoder=model.online_encoder,
        target_encoder=model.target_encoder,
        tau=tau,
    )

    with torch.no_grad():
        pred_stats = collapse_diagnostics(pred_repr)
        target_stats = collapse_diagnostics(target_repr)

        cosine = F.cosine_similarity(
            pred_repr.flatten(0, 1).float(),
            target_repr.flatten(0, 1).float(),
            dim=-1,
        ).mean()

    logs = {
        "loss": loss.item(),
        "ema_tau": tau,
        "pred_target/cosine": cosine.item(),
    }

    logs.update({f"pred/{k}": v for k, v in pred_stats.items()})
    logs.update({f"target/{k}": v for k, v in target_stats.items()})

    return logs
```

This is closer to what we will use in real training.

The loss tells us whether the predictor is matching the target. The diagnostics tell us whether the representations remain alive.

---

## 1.3.15 What the Objective Does Not Guarantee

The JEPA objective is powerful, but it does not automatically guarantee semantic representations.

Several failure modes are possible.

### Collapse

The encoder may produce constant or near-constant representations.

### Shortcut learning

The predictor may exploit local cues or mask leakage.

### Weak target representations

If the target encoder learns poor features, predicting those features is not useful.

### Overly easy prediction

If targets are too predictable from low-level texture, the model may not learn high-level structure.

### Overly hard prediction

If targets are impossible to infer from context, the model may learn weak averages.

### Poor transfer

A low JEPA loss does not guarantee strong downstream performance.

Therefore the objective must be paired with:

- careful mask design,
- representation diagnostics,
- downstream evaluation,
- ablations,
- visualization,
- sanity checks.

The objective is the center of the method, but not the entire method.

---

## 1.3.16 Summary

The JEPA objective trains a model to predict target representations from context representations.

The central equation is:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
\left\|
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathrm{tgt}}))
\right\|^2
\]

Its main components are:

- an online encoder,
- a target encoder,
- a predictor,
- mask or target metadata,
- a latent-space loss,
- stop-gradient,
- EMA target updates.

This objective changes the self-supervised learning problem from observation reconstruction to latent prediction.

That is why it is useful as a stepping stone toward world models.

The next section studies the representation geometry induced by this objective: invariances, bottlenecks, collapse, and why predicting in latent space can encourage abstraction.

---

## References and Further Reading

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**, 2020.  
  <https://arxiv.org/abs/2006.07733>

- Xinlei Chen, Kaiming He, **Exploring Simple Siamese Representation Learning**, 2020.  
  <https://arxiv.org/abs/2011.10566>

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.  
  <https://openreview.net/forum?id=BZ5a1r-kVsf>

- Facebook Research, **Official I-JEPA Codebase**.  
  <https://github.com/facebookresearch/ijepa>

- Meta AI, **V-JEPA: The next step toward advanced machine intelligence**, 2024.  
  <https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/>
