# 2.0 Minimal I-JEPA Implementation Overview

Chapter 1 developed the conceptual foundation of JEPA-style learning. We introduced the central objective:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
\left\|
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
-
\mathrm{sg}(f_{\bar{\theta}}(x_{\mathrm{tgt}}))
\right\|^2
\]

This chapter turns that equation into working PyTorch code.

We assume the reader already knows standard PyTorch concepts such as tensors, `nn.Module`, autograd, optimizers, and training loops. The focus here is on the JEPA-specific architecture and data flow, not on basic PyTorch usage.

The goal is to build a minimal but correct implementation of **Image JEPA**. It should be small enough to understand completely, but faithful enough to expose the core algorithmic ideas:

- patch embeddings,
- context and target masks,
- online encoder,
- EMA target encoder,
- predictor,
- latent-space loss,
- collapse diagnostics,
- training loop,
- basic evaluation.

We are not trying to reproduce the full official I-JEPA training setup yet. That comes later. This chapter focuses on building a transparent implementation that makes every component visible.

---

## 2.0.1 What We Are Building

We will implement a small image-based JEPA model.

Given an image:

\[
x \in \mathbb{R}^{3 \times H \times W}
\]

we divide it into patches:

\[
x \rightarrow [p_1, p_2, \dots, p_N]
\]

where \(N\) is the number of patches.

We sample two sets of patch indices:

\[
\mathcal{C}
\]

for context patches, and:

\[
\mathcal{T}
\]

for target patches.

The online encoder processes the context:

\[
z_{\mathcal{C}}
=
f_\theta(x_{\mathcal{C}})
\]

The target encoder processes the target:

\[
z_{\mathcal{T}}
=
f_{\bar{\theta}}(x_{\mathcal{T}})
\]

The predictor receives the context representation and the target positions:

\[
\hat{z}_{\mathcal{T}}
=
g_\theta(z_{\mathcal{C}}, \mathcal{T})
\]

The loss compares predicted target representations to target encoder representations:

\[
\mathcal{L}
=
\left\|
\hat{z}_{\mathcal{T}}
-
\mathrm{sg}(z_{\mathcal{T}})
\right\|^2
\]

The target encoder is updated using exponential moving average:

\[
\bar{\theta}
\leftarrow
\tau \bar{\theta}
+
(1-\tau)\theta
\]

This is the complete algorithm.

Everything in this chapter is an implementation of one piece of this pipeline.

---

## 2.0.2 Minimal System Diagram

The data flow is:

```text
image
  │
  ▼
patch embedding
  │
  ├── context indices ───────► online encoder ───────┐
  │                                                   │
  └── target indices ────────► target encoder ──┐     │
                                                │     │
                                                ▼     ▼
                                      target representation
                                                ▲
                                                │
context representation + target positions ─► predictor
                                                │
                                                ▼
                                  predicted target representation
                                                │
                                                ▼
                                      latent prediction loss
```

In code, the core forward pass will eventually look like:

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

loss = loss_fn(pred_repr, target_repr.detach())
```

This is the heart of the implementation.

---

## 2.0.3 What This Minimal Implementation Includes

The minimal implementation will include the following modules.

```text
src/jepa_world_model/
├── patchify.py
├── masks.py
├── position.py
├── vit.py
├── predictor.py
├── losses.py
├── diagnostics.py
├── ema.py
├── data.py
└── train_minimal.py
```

Each file has a focused purpose.

---

### `patchify.py`

Responsible for turning images into patches and patch tokens.

Includes:

- image patchification,
- patch grid indexing,
- patch reconstruction utilities,
- patch visualization helpers.

Core concepts:

```python
patches = patchify(images, patch_size)
tokens = patch_embed(images)
selected_tokens = gather_tokens(tokens, patch_indices)
```

---

### `masks.py`

Responsible for context and target mask sampling.

Includes:

- random patch masks,
- rectangular block masks,
- context-target overlap checks,
- batch mask sampling,
- mask visualization utilities.

Core concepts:

```python
context_indices, target_indices = mask_sampler(
    batch_size=batch_size,
    grid_size=grid_size,
    device=device,
)
```

---

### `position.py`

Responsible for positional embeddings.

Includes:

- learned positional embeddings,
- optional sinusoidal 2D embeddings,
- position gathering by patch index.

Core concepts:

```python
pos = pos_embed(indices)
tokens = tokens + pos
```

---

### `vit.py`

Responsible for the minimal Vision Transformer encoder.

Includes:

- patch embedding,
- transformer blocks,
- context token selection,
- output token representations.

Core concepts:

```python
context_repr = encoder(images, context_indices)
target_repr = encoder(images, target_indices)
```

---

### `predictor.py`

Responsible for predicting target representations from context representations.

Includes:

- context projection,
- learned target query tokens,
- target positional conditioning,
- transformer predictor blocks,
- output projection.

Core concepts:

```python
pred_repr = predictor(
    context_repr=context_repr,
    context_indices=context_indices,
    target_indices=target_indices,
)
```

---

### `losses.py`

Responsible for latent prediction losses.

Includes:

- MSE loss,
- Smooth L1 loss,
- cosine loss,
- optional combined loss.

Core concepts:

```python
loss = smooth_l1_latent_loss(pred_repr, target_repr)
```

---

### `diagnostics.py`

Responsible for representation health checks.

Includes:

- feature variance,
- representation norms,
- effective rank,
- cosine similarity,
- collapse diagnostics,
- mask overlap diagnostics.

Core concepts:

```python
logs = representation_geometry_report(target_repr, prefix="target")
```

---

### `ema.py`

Responsible for target encoder updates.

Includes:

- target initialization,
- EMA update,
- EMA schedule.

Core concepts:

```python
update_ema(
    online=model.online_encoder,
    target=model.target_encoder,
    tau=tau,
)
```

---

### `data.py`

Responsible for loading image datasets.

Includes:

- CIFAR-10 or STL-10 loaders,
- transforms,
- train/validation split,
- batch collation.

Core concepts:

```python
train_loader, val_loader = build_dataloaders(config)
```

---

### `train_minimal.py`

Responsible for the first end-to-end training run.

Includes:

- model creation,
- optimizer,
- training loop,
- logging,
- checkpointing,
- evaluation hooks.

Core concepts:

```python
for step, batch in enumerate(train_loader):
    logs = train_step(...)
```

---

## 2.0.4 What We Intentionally Omit at First

A minimal implementation should not try to solve every engineering problem immediately.

We will intentionally omit or simplify:

- distributed training,
- large-scale datasets,
- multi-node training,
- complex augmentation pipelines,
- sophisticated config systems,
- highly optimized attention kernels,
- full parity with official I-JEPA,
- extensive hyperparameter sweeps,
- production checkpoint management.

These topics matter, but adding them too early obscures the core method.

The purpose of Chapter 2 is clarity.

Chapter 3 will turn the minimal implementation into a more serious research framework.

---

## 2.0.5 Dataset Choice

For the minimal implementation, we want a dataset that is:

- easy to download,
- small enough to train quickly,
- image-based,
- compatible with simple ViT experiments,
- useful for representation probing.

Good first choices are:

| Dataset         | Why it is useful               | Limitation           |
| --------------- | ------------------------------ | -------------------- |
| CIFAR-10        | very easy, fast iteration      | low resolution       |
| STL-10          | unlabeled split, larger images | slightly heavier     |
| Tiny ImageNet   | more diverse                   | more setup           |
| ImageNet subset | closer to real setup           | requires data access |

For the first implementation, **STL-10** is a good default because it includes an unlabeled split and \(96 \times 96\) images.

CIFAR-10 is also acceptable for fast smoke tests, but the images are small. With \(32 \times 32\) images, patch-based masking is less interesting unless patch size is very small.

A practical progression:

```text
CIFAR-10 smoke test
        ↓
STL-10 minimal training
        ↓
Tiny ImageNet or ImageNet subset
```

For the first working implementation, we can design the code so that switching datasets is easy.

---

## 2.0.6 Core Tensor Shapes

Strict shape discipline is essential.

We will use the following notation:

```text
B = batch size
C = image channels
H = image height
W = image width
P = patch size
G = grid size
N = number of patches
D = encoder dimension
D_p = predictor dimension
N_ctx = number of context patches
N_tgt = number of target patches
```

For square images:

\[
G = H / P = W / P
\]

and:

\[
N = G^2
\]

Example:

```text
image size = 96
patch size = 8
grid size  = 12
num patches = 144
```

Common shapes:

```python
images.shape
# [B, 3, H, W]

patches.shape
# [B, N, 3 * P * P]

patch_tokens.shape
# [B, N, D]

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

The loss requires:

```python
pred_repr.shape == target_repr.shape
```

During development, we will assert this aggressively.

```python
def assert_same_shape(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred={pred.shape}, target={target.shape}"
        )
```

This kind of defensive programming saves hours of debugging.

---

## 2.0.7 Patch Indexing Convention

We will flatten a 2D patch grid into a 1D sequence.

For a grid of size:

\[
G \times G
\]

the patch at row \(r\) and column \(c\) has index:

\[
i = rG + c
\]

So:

```text
0      1      2      ... G-1
G      G+1    G+2    ... 2G-1
2G     2G+1   2G+2   ... 3G-1
...
```

In code:

```python
def patch_index(row: int, col: int, grid_size: int) -> int:
    return row * grid_size + col
```

Vectorized:

```python
rows = torch.arange(grid_size)
cols = torch.arange(grid_size)
rr, cc = torch.meshgrid(rows, cols, indexing="ij")
indices = rr * grid_size + cc
```

This convention will be used by:

- patchification,
- positional embeddings,
- context masks,
- target masks,
- predictor target queries,
- visualization.

Keeping this convention consistent is crucial.

---

## 2.0.8 Minimal Model Configuration

A first model should be small.

The goal is not state-of-the-art performance. The goal is a model that trains quickly and exposes failure modes.

A reasonable starting config:

```yaml
image_size: 96
patch_size: 8
in_channels: 3

encoder:
  dim: 192
  depth: 6
  num_heads: 3
  mlp_ratio: 4.0
  dropout: 0.0

predictor:
  dim: 128
  depth: 3
  num_heads: 4
  mlp_ratio: 4.0

mask:
  num_target_blocks: 4
  target_block_size: [3, 3]
  context_ratio: 0.6

training:
  batch_size: 128
  epochs: 100
  learning_rate: 0.0005
  weight_decay: 0.05
  ema_tau_base: 0.996
  ema_tau_final: 1.0
```

For CIFAR-10, a smaller config is better:

```yaml
image_size: 32
patch_size: 4

encoder:
  dim: 128
  depth: 4
  num_heads: 4

predictor:
  dim: 96
  depth: 2
  num_heads: 4
```

The minimal implementation can begin without YAML. We can hard-code a dataclass first:

```python
from dataclasses import dataclass


@dataclass
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

    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 0.05

    ema_tau_base: float = 0.996
    ema_tau_final: float = 1.0
```

Later, Chapter 3 can replace this with Hydra or OmegaConf.

---

## 2.0.9 The Minimal I-JEPA Class

At the top level, our model will combine:

- online encoder,
- target encoder,
- predictor.

```python
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

        self._init_target_encoder()

    def _init_target_encoder(self) -> None:
        self.target_encoder.load_state_dict(
            self.online_encoder.state_dict()
        )

        for p in self.target_encoder.parameters():
            p.requires_grad = False

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

This is the same structure introduced in Chapter 1, but now it becomes the organizing class for the implementation.

---

## 2.0.10 The Minimal Training Step

The training step is the smallest complete unit of learning.

```python
def train_step(
    model: MinimalIJEPA,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    ema_tau: float,
) -> dict[str, float]:
    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    if pred_repr.shape != target_repr.shape:
        raise ValueError(
            f"pred_repr shape {pred_repr.shape} does not match "
            f"target_repr shape {target_repr.shape}"
        )

    loss = torch.nn.functional.smooth_l1_loss(
        pred_repr,
        target_repr.detach(),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    update_ema(
        online=model.online_encoder,
        target=model.target_encoder,
        tau=ema_tau,
    )

    with torch.no_grad():
        cosine = torch.nn.functional.cosine_similarity(
            pred_repr.flatten(0, 1).float(),
            target_repr.flatten(0, 1).float(),
            dim=-1,
        ).mean()

    return {
        "loss": loss.item(),
        "pred_target_cosine": cosine.item(),
        "ema_tau": ema_tau,
    }
```

This step does five things:

1. computes predicted and target representations,
2. computes the latent prediction loss,
3. updates the online encoder and predictor,
4. updates the target encoder with EMA,
5. returns basic logs.

Later, we will add:

- mixed precision,
- gradient clipping,
- representation diagnostics,
- checkpointing,
- learning-rate scheduling,
- validation.

But the algorithmic core is already here.

---

## 2.0.11 Target Encoder Update

The target encoder is not optimized by gradient descent.

It is updated after each optimizer step:

```python
@torch.no_grad()
def update_ema(
    online: nn.Module,
    target: nn.Module,
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

The EMA schedule can be:

```python
import math


def cosine_ema_tau(
    step: int,
    total_steps: int,
    tau_base: float,
    tau_final: float = 1.0,
) -> float:
    progress = step / max(total_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

    return tau_final - (tau_final - tau_base) * cosine
```

At step zero:

```python
tau ≈ tau_base
```

Near the end:

```python
tau ≈ tau_final
```

This means the target encoder becomes increasingly stable during training.

---

## 2.0.12 First Diagnostics

A JEPA implementation should log representation diagnostics from the beginning.

The minimal logs should include:

- loss,
- prediction-target cosine similarity,
- target representation norm,
- prediction representation norm,
- target feature standard deviation,
- prediction feature standard deviation,
- dead dimension fraction.

A minimal diagnostic function:

```python
@torch.no_grad()
def representation_stats(
    z: torch.Tensor,
    prefix: str,
    eps: float = 1e-4,
) -> dict[str, float]:
    if z.dim() == 3:
        z = z.flatten(0, 1)

    z = z.float()

    norms = z.norm(dim=-1)
    std = z.std(dim=0)

    return {
        f"{prefix}/norm_mean": norms.mean().item(),
        f"{prefix}/norm_std": norms.std().item(),
        f"{prefix}/std_mean": std.mean().item(),
        f"{prefix}/std_min": std.min().item(),
        f"{prefix}/dead_dim_fraction": (std < eps).float().mean().item(),
    }
```

In the training step:

```python
with torch.no_grad():
    logs = {
        "loss": loss.item(),
        "ema_tau": ema_tau,
    }

    logs.update(representation_stats(pred_repr, prefix="pred"))
    logs.update(representation_stats(target_repr, prefix="target"))
```

A decreasing loss is not sufficient. If representation variance collapses, the model is not learning useful features.

---

## 2.0.13 First Sanity Checks

Before training seriously, we need smoke tests.

### Shape test

```python
pred_repr, target_repr = model(
    images=images,
    context_indices=context_indices,
    target_indices=target_indices,
)

assert pred_repr.shape == target_repr.shape
```

### Gradient test

The online encoder should receive gradients.

```python
loss.backward()

online_has_grad = any(
    p.grad is not None
    for p in model.online_encoder.parameters()
)

assert online_has_grad
```

The target encoder should not receive gradients.

```python
target_has_grad = any(
    p.grad is not None
    for p in model.target_encoder.parameters()
)

assert not target_has_grad
```

### Mask overlap test

```python
assert_no_overlap(context_indices, target_indices)
```

### EMA movement test

After one optimizer step and EMA update, target parameters should move slightly toward online parameters.

```python
before = {
    name: p.detach().clone()
    for name, p in model.target_encoder.named_parameters()
}

update_ema(model.online_encoder, model.target_encoder, tau=0.996)

after = dict(model.target_encoder.named_parameters())

moved = any(
    not torch.allclose(before[name], after[name])
    for name in before
)

assert moved
```

These tests may feel basic, but they catch the most common JEPA implementation bugs.

---

## 2.0.14 What Success Looks Like

In the first experiment, success does not mean state-of-the-art performance.

Success means:

- the model runs end-to-end,
- masks are sampled correctly,
- context and target do not overlap,
- target encoder receives no gradients,
- EMA updates happen,
- loss decreases,
- representation variance does not collapse,
- prediction-target cosine similarity improves,
- nearest-neighbor retrieval looks non-random,
- linear probe performs above chance.

A healthy early training curve might show:

```text
loss:                    decreasing
pred_target_cosine:       increasing
target/std_mean:          stable, nonzero
pred/std_mean:            stable, nonzero
dead_dim_fraction:        near zero
```

Warning signs:

```text
loss rapidly goes to zero
target/std_mean goes to zero
dead_dim_fraction approaches one
pred_target_cosine is high but probes are poor
mask overlap is nonzero
target encoder has gradients
```

This chapter will give us the tools to detect these issues.

---

## 2.0.15 Chapter 2 Roadmap

The rest of Chapter 2 implements each component.

```text
2.1 Project Setup
2.2 Image Patchification
2.3 Positional Embeddings
2.4 Context and Target Mask Sampling
2.5 Minimal ViT Encoder
2.6 EMA Target Encoder
2.7 JEPA Predictor
2.8 Losses and Diagnostics
2.9 Training Loop
2.10 Evaluation: k-NN and Linear Probe
2.11 Running the First Experiment
```

The next section sets up the project structure and development environment.

---

## References and Further Reading

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.  
  <https://github.com/facebookresearch/ijepa>

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, **Masked Autoencoders Are Scalable Vision Learners**, 2021.  
  <https://arxiv.org/abs/2111.06377>

- Ashish Vaswani et al., **Attention Is All You Need**, 2017.  
  <https://arxiv.org/abs/1706.03762>

- Alexey Dosovitskiy et al., **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**, 2020.  
  <https://arxiv.org/abs/2010.11929>
