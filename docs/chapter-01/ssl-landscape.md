# 1.1 The Self-Supervised Learning Landscape

Self-supervised learning is the study of extracting supervision from the data itself.

In supervised learning, we are given pairs:

\[
(x, y)
\]

where \(x\) is an input and \(y\) is a human-provided label. We train a model \(h_\theta\) by minimizing:

\[
\mathcal{L}_{\mathrm{sup}}
=
\ell(h_\theta(x), y)
\]

In self-supervised learning, we usually do not have \(y\). Instead, we construct a learning problem from the structure of \(x\) itself.

For example, we may ask the model to:

- reconstruct a corrupted input,
- predict a missing part of the input,
- match two augmented views of the same input,
- distinguish related samples from unrelated samples,
- predict future observations,
- predict future representations.

The central design question is:

> What relationship should the model learn from the data?

Different answers give rise to different families of self-supervised learning methods.

This section gives a conceptual and implementation-oriented map of the landscape. The goal is not to cover every method historically. The goal is to understand the design space that leads naturally to JEPA.

---

## 1.1.1 Why Self-Supervised Learning Exists

The supervised learning recipe is powerful but limited.

A supervised classifier trained on ImageNet learns representations useful for recognizing ImageNet labels. But the labels only describe a small part of what is present in the image. An image labeled `dog` also contains pose, texture, depth, object boundaries, lighting, context, affordances, and relations to other objects.

A language model trained to predict text learns from the internal structure of sequences. A vision model can also learn from the internal structure of images and videos. A time-series model can learn from temporal continuity, missing values, and future prediction.

The motivation is simple:

\[
\text{data contains more structure than labels expose}
\]

Self-supervised learning attempts to use that structure.

Instead of relying on external labels, we construct related views or targets from the same sample:

\[
v_1 = T_1(x)
\]

\[
v_2 = T_2(x)
\]

and train a model so that its representation of \(v_1\) is useful for predicting, reconstructing, matching, or explaining \(v_2\).

A generic self-supervised objective can be written as:

\[
\mathcal{L}_{\mathrm{ssl}}
=
\ell(q_\theta(v_1), r_\phi(v_2))
\]

where:

- \(T_1, T_2\) are transformations, corruptions, masks, crops, or temporal shifts,
- \(q_\theta\) is the online branch,
- \(r_\phi\) is the target branch,
- \(\ell\) defines what relationship we want to enforce.

The entire field can be viewed as different choices of:

1. how to construct \(v_1\) and \(v_2\),
2. what target \(r_\phi(v_2)\) should be,
3. what loss \(\ell\) should enforce,
4. what prevents trivial solutions.

JEPA is one particular answer:

> Construct context and target regions from the same observation. Encode the target with a representation network. Train a predictor to infer the target representation from the context representation.

---

## 1.1.2 A Useful Taxonomy

For this tutorial, we will organize self-supervised methods into seven families:

1. reconstruction-based methods,
2. denoising and masked reconstruction methods,
3. contrastive joint-embedding methods,
4. non-contrastive joint-embedding methods,
5. redundancy-reduction methods,
6. masked predictive representation methods,
7. temporal and world-model methods.

The categories overlap, but the taxonomy clarifies what JEPA inherits and what it changes.

The table below gives a high-level map.

| Family | Input | Target | Loss Type | Predicts Pixels? | Main Risk |
|---|---|---|---|---:|---|
| Autoencoder | full input | same input | reconstruction | yes | low-level detail |
| Denoising AE | corrupted input | clean input | reconstruction | yes | shortcut learning |
| MAE | visible patches | masked pixels | reconstruction | yes | pixel bias |
| Contrastive SSL | augmented view | paired view | discrimination | no | false negatives |
| BYOL / SimSiam-style | augmented view | target representation | alignment | no | collapse |
| VICReg-style | augmented view | target representation + regularizers | alignment + variance/covariance | no | weak invariances |
| I-JEPA | context region | target-region representation | latent prediction | no | mask leakage, collapse |
| World model | past state | future state | prediction | optional | compounding error |
| Action world model | past state + action | future state | prediction/control | optional | causal confusion |

The key transition for this tutorial is:

\[
\text{reconstruct observations}
\quad \rightarrow \quad
\text{align representations}
\quad \rightarrow \quad
\text{predict representations}
\quad \rightarrow \quad
\text{predict future representations under actions}
\]

JEPA lives at the point where representation learning becomes explicitly predictive.

---

## 1.1.3 Reconstruction-Based Methods

The simplest self-supervised objective is reconstruction.

An autoencoder learns an encoder:

\[
z = f_\theta(x)
\]

and a decoder:

\[
\hat{x} = d_\phi(z)
\]

The loss asks the reconstruction \(\hat{x}\) to match the original input \(x\):

\[
\mathcal{L}_{\mathrm{AE}}
=
\left\|
d_\phi(f_\theta(x)) - x
\right\|_2^2
\]

In PyTorch, the core structure is minimal:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


def autoencoder_loss(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(x_hat, x)
```

A training step is similarly direct:

```python
def train_autoencoder_step(
    model: AutoEncoder,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    x_hat, _ = model(x)
    loss = autoencoder_loss(x_hat, x)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

The strength of this setup is that the supervision is dense. Every input dimension becomes a target. For images, every pixel contributes to the loss. For audio, every waveform sample can contribute. For time series, every channel and time step can be reconstructed.

The weakness is that reconstruction is not necessarily the same as understanding.

If the objective asks the model to reconstruct everything, the model may allocate capacity to details that are not useful for downstream reasoning. In images, this might mean fine texture, background clutter, or sensor noise. In cycling telemetry, it might mean transient measurement jitter rather than physiological state.

A reconstruction objective is therefore often too literal.

It says:

> Preserve the input.

But representation learning often wants something closer to:

> Preserve the information that matters.

This distinction becomes central when moving toward world models.

---

## 1.1.4 Denoising and Masked Reconstruction

A more useful reconstruction task corrupts the input and asks the model to recover the original.

Let:

\[
\tilde{x} = C(x)
\]

where \(C\) is a corruption process. The model learns:

\[
\hat{x} = d_\phi(f_\theta(\tilde{x}))
\]

with objective:

\[
\mathcal{L}_{\mathrm{denoise}}
=
\left\|
\hat{x} - x
\right\|_2^2
\]

For images, a common corruption is masking. We divide an image into patches, hide some of them, and ask the model to reconstruct the missing pixels.

If an image has \(N\) patches, we sample a visible set:

\[
\mathcal{V} \subset \{1,\dots,N\}
\]

and a masked set:

\[
\mathcal{M} = \{1,\dots,N\} \setminus \mathcal{V}
\]

The model receives only visible patches:

\[
x_\mathcal{V}
\]

and reconstructs masked patches:

\[
\hat{x}_\mathcal{M}
\]

The loss is:

\[
\mathcal{L}_{\mathrm{MAE}}
=
\left\|
\hat{x}_{\mathcal{M}} - x_{\mathcal{M}}
\right\|_2^2
\]

This is the basic idea behind Masked Autoencoders. MAE uses an asymmetric encoder-decoder design: the encoder processes only visible patches, while a lightweight decoder reconstructs the missing pixels. The original MAE paper also found that high masking ratios, such as 75%, produce a meaningful self-supervised task for vision models. See [He et al., 2021 — Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377).

A minimal random mask sampler looks like this:

```python
def sample_random_patch_mask(
    batch_size: int,
    num_patches: int,
    mask_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_masked = int(num_patches * mask_ratio)

    noise = torch.rand(batch_size, num_patches, device=device)
    shuffled = noise.argsort(dim=1)

    masked_indices = shuffled[:, :num_masked]
    visible_indices = shuffled[:, num_masked:]

    return visible_indices, masked_indices
```

A simplified MAE-style training loop looks like this:

```python
def train_mae_step(
    encoder: nn.Module,
    decoder: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    mask_ratio: float = 0.75,
) -> torch.Tensor:
    batch_size = images.size(0)
    num_patches = encoder.num_patches

    visible_idx, masked_idx = sample_random_patch_mask(
        batch_size=batch_size,
        num_patches=num_patches,
        mask_ratio=mask_ratio,
        device=images.device,
    )

    visible_tokens = encoder(images, visible_idx)
    pred_pixels = decoder(visible_tokens, masked_idx)

    target_pixels = patchify_and_gather(images, masked_idx)

    loss = F.mse_loss(pred_pixels, target_pixels)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

The function `patchify_and_gather` is intentionally left abstract here. Later implementation chapters will define this precisely.

The important point is the target:

```python
target_pixels = patchify_and_gather(images, masked_idx)
loss = F.mse_loss(pred_pixels, target_pixels)
```

MAE predicts missing pixels.

JEPA will keep the masked-prediction structure but change the target:

```python
target_repr = target_encoder(images, target_idx)
loss = representation_loss(pred_repr, target_repr)
```

This is the central difference.

MAE asks:

> What are the missing pixels?

JEPA asks:

> What is the latent representation of the missing region?

---

## 1.1.5 Contrastive Joint-Embedding Methods

Contrastive learning trains representations by comparing positive and negative pairs.

Given a sample \(x\), construct two augmented views:

\[
v_1 = T_1(x)
\]

\[
v_2 = T_2(x)
\]

These two views are treated as a positive pair. Views from different samples are treated as negatives.

An encoder produces representations:

\[
z_1 = f_\theta(v_1)
\]

\[
z_2 = f_\theta(v_2)
\]

A contrastive objective pulls \(z_1\) and \(z_2\) together while pushing representations from different samples apart.

A common objective is InfoNCE:

\[
\mathcal{L}_i
=
-
\log
\frac{
\exp(\mathrm{sim}(z_i, z_i^+) / \tau)
}{
\sum_j \exp(\mathrm{sim}(z_i, z_j) / \tau)
}
\]

where:

- \(z_i^+\) is the positive representation,
- \(z_j\) are candidate representations,
- \(\mathrm{sim}\) is usually cosine similarity,
- \(\tau\) is a temperature.

A minimal symmetric InfoNCE implementation is:

```python
def info_nce_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    temperature: float = 0.2,
) -> torch.Tensor:
    """
    z1, z2:
        Tensor of shape [batch_size, dim].

    Positive pairs:
        z1[i] and z2[i].

    Negatives:
        z2[j] for j != i, and symmetrically z1[j] for j != i.
    """
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits_12 = z1 @ z2.T
    logits_21 = z2 @ z1.T

    logits_12 = logits_12 / temperature
    logits_21 = logits_21 / temperature

    labels = torch.arange(z1.size(0), device=z1.device)

    loss_12 = F.cross_entropy(logits_12, labels)
    loss_21 = F.cross_entropy(logits_21, labels)

    return 0.5 * (loss_12 + loss_21)
```

A contrastive training step looks like:

```python
def train_contrastive_step(
    encoder: nn.Module,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    augment,
) -> torch.Tensor:
    view_1 = augment(images)
    view_2 = augment(images)

    z1 = encoder(view_1)
    z2 = encoder(view_2)

    loss = info_nce_loss(z1, z2)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

Contrastive learning directly shapes representation geometry.

It imposes:

\[
\mathrm{sim}(f(v_1), f(v_2)) \uparrow
\]

for positive pairs, and:

\[
\mathrm{sim}(f(v_i), f(v_j)) \downarrow
\]

for negatives.

This can produce strong representations. But it also introduces several difficulties:

- it often benefits from large batches or memory banks,
- it depends heavily on augmentation design,
- it can suffer from false negatives,
- it optimizes instance discrimination rather than explicit prediction.

For a world model, instance discrimination is not obviously the right objective. If the model is predicting a future state, the goal is not to distinguish that future state from other batch elements. The goal is to infer the latent consequences of the past and possibly of actions.

This motivates non-contrastive and predictive alternatives.

---

## 1.1.6 Non-Contrastive Joint-Embedding Methods

Non-contrastive joint-embedding methods remove explicit negative samples.

They still construct two related views:

\[
v_1 = T_1(x)
\]

\[
v_2 = T_2(x)
\]

But instead of pushing away other samples, they train one branch to predict or align with the representation of the other branch.

A simplified BYOL-style setup is:

\[
z_1 = f_\theta(v_1)
\]

\[
p_1 = q_\theta(z_1)
\]

\[
z_2 = f_{\bar{\theta}}(v_2)
\]

\[
\mathcal{L}
=
\left\|
\mathrm{norm}(p_1)
-
\mathrm{norm}(\mathrm{sg}(z_2))
\right\|_2^2
\]

BYOL uses an online network and a target network. The online network is optimized by gradient descent, while the target network is updated as an exponential moving average of the online network. See [Grill et al., 2020 — Bootstrap Your Own Latent](https://arxiv.org/abs/2006.07733).

A minimal BYOL-like model has four components:

```python
class BYOLLikeModel(nn.Module):
    def __init__(
        self,
        online_encoder: nn.Module,
        target_encoder: nn.Module,
        projector: nn.Module,
        predictor: nn.Module,
    ):
        super().__init__()
        self.online_encoder = online_encoder
        self.target_encoder = target_encoder
        self.projector = projector
        self.predictor = predictor

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        view_online: torch.Tensor,
        view_target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        online_repr = self.online_encoder(view_online)
        online_proj = self.projector(online_repr)
        online_pred = self.predictor(online_proj)

        with torch.no_grad():
            target_repr = self.target_encoder(view_target)
            target_proj = self.projector(target_repr)

        return online_pred, target_proj
```

A BYOL-style loss can be written as:

```python
def byol_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target.detach(), dim=-1)

    return 2.0 - 2.0 * (pred * target).sum(dim=-1).mean()
```

The EMA update is:

```python
@torch.no_grad()
def ema_update(
    online: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(
            p_online.data,
            alpha=1.0 - tau,
        )
```

The core training step is:

```python
def train_byol_step(
    model: BYOLLikeModel,
    images: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    augment,
    tau: float,
) -> torch.Tensor:
    view_1 = augment(images)
    view_2 = augment(images)

    pred, target = model(view_1, view_2)
    loss = byol_loss(pred, target)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    ema_update(model.online_encoder, model.target_encoder, tau)

    return loss.detach()
```

Non-contrastive methods raise an obvious question:

> If there are no negatives, why does the model not map every input to the same constant vector?

This is the collapse problem.

Different methods prevent collapse through different mechanisms:

- predictor asymmetry,
- stop-gradient,
- EMA target networks,
- normalization,
- architectural bias,
- explicit variance and covariance regularization.

JEPA inherits the non-contrastive joint-embedding idea of predicting target representations without using explicit negatives. But it changes the construction of the views.

BYOL predicts another augmented view of the same image.

I-JEPA predicts representations of target regions from context regions of the same image.

This makes the objective more spatially and semantically predictive.

---

## 1.1.7 Redundancy-Reduction Methods

Another important family avoids collapse through explicit statistical regularization.

VICReg is a representative example. It combines three terms:

1. invariance,
2. variance,
3. covariance.

Given two views \(v_1\) and \(v_2\), encode:

\[
z_1 = f_\theta(v_1)
\]

\[
z_2 = f_\theta(v_2)
\]

The invariance term aligns paired views:

\[
\mathcal{L}_{\mathrm{inv}}
=
\frac{1}{N}
\sum_i
\|z_{1,i} - z_{2,i}\|_2^2
\]

The variance term prevents each embedding dimension from collapsing:

\[
\mathcal{L}_{\mathrm{var}}
=
\frac{1}{D}
\sum_j
\max(0, \gamma - \sigma(z_{\cdot,j}))
\]

The covariance term discourages redundant dimensions:

\[
\mathcal{L}_{\mathrm{cov}}
=
\frac{1}{D}
\sum_{i \neq j}
\mathrm{Cov}(z)_{i,j}^2
\]

The full objective is:

\[
\mathcal{L}_{\mathrm{VICReg}}
=
\lambda \mathcal{L}_{\mathrm{inv}}
+
\mu \mathcal{L}_{\mathrm{var}}
+
\nu \mathcal{L}_{\mathrm{cov}}
\]

VICReg explicitly addresses collapse by requiring nonzero variance in embedding dimensions and discouraging feature redundancy. See [Bardes, Ponce, and LeCun, 2021 — VICReg](https://arxiv.org/abs/2105.04906).

A compact implementation is:

```python
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    batch_size, dim = z1.shape

    # Invariance
    repr_loss = F.mse_loss(z1, z2)

    # Variance
    std_z1 = torch.sqrt(z1.var(dim=0) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0) + eps)

    std_loss = 0.5 * (
        F.relu(1.0 - std_z1).mean()
        + F.relu(1.0 - std_z2).mean()
    )

    # Covariance
    z1 = z1 - z1.mean(dim=0)
    z2 = z2 - z2.mean(dim=0)

    cov_z1 = (z1.T @ z1) / (batch_size - 1)
    cov_z2 = (z2.T @ z2) / (batch_size - 1)

    cov_loss = (
        off_diagonal(cov_z1).pow(2).sum() / dim
        + off_diagonal(cov_z2).pow(2).sum() / dim
    )

    loss = (
        sim_coeff * repr_loss
        + std_coeff * std_loss
        + cov_coeff * cov_loss
    )

    return loss
```

This family is important for JEPA because it makes collapse diagnostics explicit.

Even if our JEPA implementation does not use a VICReg loss directly, we will borrow the habit of monitoring:

- feature standard deviation,
- covariance structure,
- embedding norm,
- dimensional usage.

For example:

```python
@torch.no_grad()
def representation_diagnostics(z: torch.Tensor) -> dict[str, float]:
    """
    z:
        [batch, tokens, dim] or [batch, dim]
    """
    if z.dim() == 3:
        z = z.flatten(0, 1)

    z = z.float()

    std_per_dim = z.std(dim=0)
    mean_norm = z.norm(dim=-1).mean()

    return {
        "repr/std_mean": std_per_dim.mean().item(),
        "repr/std_min": std_per_dim.min().item(),
        "repr/std_max": std_per_dim.max().item(),
        "repr/norm_mean": mean_norm.item(),
    }
```

In later chapters, this diagnostic function will become part of our JEPA training loop.

---

## 1.1.8 Masked Predictive Representation Learning

Masked predictive representation learning combines two ideas:

1. create a partial-observation problem,
2. predict a representation rather than raw data.

The model receives context:

\[
x_{\mathrm{ctx}}
\]

and predicts the representation of a target:

\[
z_{\mathrm{tgt}}
=
f_{\bar{\theta}}(x_{\mathrm{tgt}})
\]

The prediction is:

\[
\hat{z}_{\mathrm{tgt}}
=
g_\theta(f_\theta(x_{\mathrm{ctx}}), m_{\mathrm{tgt}})
\]

The loss is:

\[
\mathcal{L}_{\mathrm{JEPA}}
=
\left\|
\hat{z}_{\mathrm{tgt}}
-
\mathrm{sg}(z_{\mathrm{tgt}})
\right\|^2
\]

I-JEPA instantiates this idea for images. It predicts representations of target blocks from a context block within the same image. Its masking strategy is deliberately not the same as random MAE-style masking: target blocks should be large enough to encourage semantic prediction, and the context should be sufficiently informative. See [Assran et al., 2023 — I-JEPA](https://arxiv.org/abs/2301.08243).

A simplified JEPA-style model interface is:

```python
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

The training step is:

```python
def train_jepa_step(
    model: JEPAModel,
    images: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    tau: float,
) -> torch.Tensor:
    pred_repr, target_repr = model(
        images=images,
        context_indices=context_indices,
        target_indices=target_indices,
    )

    loss = F.smooth_l1_loss(
        pred_repr,
        target_repr.detach(),
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    ema_update(
        online=model.online_encoder,
        target=model.target_encoder,
        tau=tau,
    )

    return loss.detach()
```

The difference from MAE is visible in one line.

MAE:

```python
loss = F.mse_loss(pred_pixels, target_pixels)
```

JEPA:

```python
loss = F.smooth_l1_loss(pred_repr, target_repr.detach())
```

This substitution changes the semantics of the learning task.

MAE trains the model to recover observations.

JEPA trains the model to infer latent structure.

---

## 1.1.9 Why Mask Design Matters

Masking is not a minor implementation detail.

The mask defines the prediction problem.

If the target is too small, the model may solve the task using local texture continuation. If the target is too easy, the model may not need semantic understanding. If the context leaks too much information, the model may shortcut the task. If the context is too weak, the target may be impossible to predict.

A useful JEPA mask should create a problem that is:

1. difficult enough to require abstraction,
2. possible enough to learn from,
3. structured enough to avoid trivial shortcuts.

For images, I-JEPA uses block-based masking rather than purely independent random patch masking. The motivation is that target blocks should correspond to meaningful spatial regions, and context regions should contain enough information to support prediction.

A simplified block-mask sampler might look like:

```python
def sample_block_indices(
    grid_size: int,
    block_height: int,
    block_width: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Samples one rectangular block from a grid of patches.

    Returns:
        indices: [num_block_patches]
    """
    max_row = grid_size - block_height
    max_col = grid_size - block_width

    row = torch.randint(0, max_row + 1, (), device=device)
    col = torch.randint(0, max_col + 1, (), device=device)

    rows = torch.arange(row, row + block_height, device=device)
    cols = torch.arange(col, col + block_width, device=device)

    rr, cc = torch.meshgrid(rows, cols, indexing="ij")
    indices = rr * grid_size + cc

    return indices.flatten()
```

A batch version can sample multiple target blocks:

```python
def sample_target_blocks(
    batch_size: int,
    grid_size: int,
    num_blocks: int,
    block_height: int,
    block_width: int,
    device: torch.device,
) -> torch.Tensor:
    all_indices = []

    for _ in range(batch_size):
        blocks = [
            sample_block_indices(
                grid_size=grid_size,
                block_height=block_height,
                block_width=block_width,
                device=device,
            )
            for _ in range(num_blocks)
        ]

        indices = torch.cat(blocks).unique()
        all_indices.append(indices)

    # In a production implementation, we would handle variable lengths carefully.
    # For now, assume unique counts are equal or pad them.
    return torch.stack(all_indices, dim=0)
```

This is not yet a complete production sampler. It is a pedagogical starting point. Later we will implement:

- multiple target blocks,
- context exclusion,
- minimum context size,
- target/context overlap prevention,
- padding for variable target counts,
- visualization.

The core lesson is that the mask defines what “prediction” means.

For JEPA, better masks create better learning signals.

---

## 1.1.10 Temporal Self-Supervision and World Models

So far, we have discussed self-supervised learning mostly in images. But the world is temporal.

For a sequence of observations:

\[
x_1, x_2, \dots, x_T
\]

a temporal self-supervised model can learn from the relationship between past and future.

A predictive model may learn:

\[
\hat{x}_{t+k}
=
G_\theta(x_{\leq t})
\]

A latent predictive model learns:

\[
\hat{z}_{t+k}
=
F_\theta(z_{\leq t})
\]

where:

\[
z_t = f_\theta(x_t)
\]

An action-conditioned world model learns:

\[
\hat{z}_{t+k}
=
F_\theta(z_{\leq t}, a_{t:t+k})
\]

This is the form we ultimately want.

The JEPA template generalizes naturally:

| Setting | Context | Target |
|---|---|---|
| Image JEPA | visible image region | hidden region representation |
| Video JEPA | visible spatiotemporal region | hidden/future spatiotemporal representation |
| Time-series JEPA | past telemetry window | future telemetry representation |
| Action world model | past state + actions | future latent state |

For cycling data, a context might be:

\[
x_{t-L:t}
\]

a window of recent telemetry. A future target might be:

\[
x_{t:t+H}
\]

A JEPA-style temporal model would encode the future target using a target encoder:

\[
z_{t:t+H}^{\mathrm{target}}
=
f_{\bar{\theta}}(x_{t:t+H})
\]

and train a dynamics model to predict it:

\[
\hat{z}_{t:t+H}
=
F_\theta(f_\theta(x_{t-L:t}), a_{t:t+H})
\]

A skeleton PyTorch interface might be:

```python
class TemporalJEPAModel(nn.Module):
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

        for p in self.target_encoder.parameters():
            p.requires_grad = False

    def forward(
        self,
        past_observations: torch.Tensor,
        future_observations: torch.Tensor,
        future_actions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context_z = self.context_encoder(past_observations)

        with torch.no_grad():
            target_z = self.target_encoder(future_observations)

        pred_z = self.dynamics(
            context_z=context_z,
            future_actions=future_actions,
        )

        return pred_z, target_z
```

The training step is structurally identical to image JEPA:

```python
def train_temporal_jepa_step(
    model: TemporalJEPAModel,
    past_observations: torch.Tensor,
    future_observations: torch.Tensor,
    future_actions: torch.Tensor | None,
    optimizer: torch.optim.Optimizer,
    tau: float,
) -> torch.Tensor:
    pred_z, target_z = model(
        past_observations=past_observations,
        future_observations=future_observations,
        future_actions=future_actions,
    )

    loss = F.mse_loss(pred_z, target_z.detach())

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    ema_update(
        online=model.context_encoder,
        target=model.target_encoder,
        tau=tau,
    )

    return loss.detach()
```

This reveals the unifying idea:

```text
context → predicted target representation
```

The context may be an image region, a video prefix, or a physiological telemetry window. The target may be a hidden image block, a future video tube, or a future ride state. The machinery is similar.

---

## 1.1.11 Prediction, Abstraction, and Action

Self-supervised learning methods can be compared by the level at which prediction occurs.

Pixel reconstruction predicts at the observation level:

\[
x_{\mathrm{ctx}} \rightarrow x_{\mathrm{tgt}}
\]

Contrastive learning aligns views at the representation level:

\[
f(v_1) \leftrightarrow f(v_2)
\]

JEPA predicts at the representation level:

\[
f(x_{\mathrm{ctx}}) \rightarrow f(x_{\mathrm{tgt}})
\]

World models predict future representations:

\[
z_{\leq t} \rightarrow z_{t+k}
\]

Action-conditioned world models predict controllable futures:

\[
(z_{\leq t}, a_{t:t+k}) \rightarrow z_{t+k}
\]

This progression matters because action requires abstraction.

An agent usually cannot evaluate every possible future by reconstructing raw sensory observations in full detail. It needs to simulate consequences in a compressed state space.

For a cyclist, useful latent predictions might include:

- fatigue accumulation,
- heart-rate drift,
- likelihood of completing an interval,
- expected recovery cost,
- sustainable power over a future segment.

The relevant future is not the exact time series. The relevant future is the state that supports decisions.

That is the world-model interpretation of JEPA-style learning.

---

## 1.1.12 A Unified View in Code

We can express many self-supervised methods with the same abstract training template:

```python
class SelfSupervisedMethod(nn.Module):
    def make_views(self, x: torch.Tensor):
        raise NotImplementedError

    def encode_context(self, view):
        raise NotImplementedError

    def encode_target(self, view):
        raise NotImplementedError

    def predict(self, context_repr, target_metadata):
        raise NotImplementedError

    def loss(self, pred, target):
        raise NotImplementedError
```

A generic training step is:

```python
def train_ssl_step(
    method: SelfSupervisedMethod,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    context_view, target_view, metadata = method.make_views(x)

    context_repr = method.encode_context(context_view)

    with torch.no_grad():
        target_repr = method.encode_target(target_view)

    pred = method.predict(context_repr, metadata)
    loss = method.loss(pred, target_repr)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return loss.detach()
```

Different methods instantiate this differently.

For MAE:

```python
target_repr = target_pixels
pred = decoder(context_repr, mask_metadata)
loss = mse(pred, target_pixels)
```

For BYOL:

```python
context_view = augment_1(x)
target_view = augment_2(x)
pred = predictor(projector(online_encoder(context_view)))
target_repr = target_projector(target_encoder(target_view))
loss = byol_loss(pred, target_repr)
```

For I-JEPA:

```python
context_view = visible_patch_indices
target_view = target_patch_indices
pred = predictor(context_repr, target_positions)
target_repr = target_encoder(images, target_patch_indices)
loss = latent_prediction_loss(pred, target_repr)
```

For temporal JEPA:

```python
context_view = past_window
target_view = future_window
pred = dynamics(context_repr, future_actions)
target_repr = target_encoder(future_window)
loss = latent_prediction_loss(pred, target_repr)
```

This is the reason JEPA is such a good teaching vehicle. It is specific enough to implement concretely, but general enough to extend into world modeling.

---

## 1.1.13 Practical Lessons for Our Implementation

This landscape gives us several design lessons for the rest of the tutorial.

### Lesson 1: The prediction target determines the learned representation

Pixel targets encourage fidelity to observation space.

Representation targets encourage abstraction, but only if the target encoder produces useful abstractions.

### Lesson 2: Collapse is a constant threat

Any method that aligns or predicts representations without negatives must address collapse.

We will monitor:

- feature variance,
- representation norm,
- cosine similarity,
- covariance structure,
- predictor output statistics.

### Lesson 3: Masking defines the task

Random masks, block masks, temporal masks, and action-conditioned horizons create different learning problems.

For JEPA, mask design is part of the algorithm.

### Lesson 4: Prediction is the bridge to world models

The move from representation learning to world modeling is not a complete conceptual break. It is a change in what counts as context and what counts as target.

### Lesson 5: Implementation details are algorithmic details

EMA schedules, stop-gradient placement, positional embeddings, target normalization, and mask leakage are not minor details. They can determine whether the method works.

---

## 1.1.14 Summary

Self-supervised learning constructs learning signals from the structure of data itself.

Reconstruction methods train models to reproduce observations. Contrastive methods shape representation geometry using positive and negative pairs. Non-contrastive methods align related views without explicit negatives. Redundancy-reduction methods explicitly avoid collapse by regulating variance and covariance. Masked predictive representation methods, such as I-JEPA, train models to infer the latent representation of hidden regions from visible regions.

JEPA is especially relevant for world modeling because it is predictive and abstract. It does not merely reconstruct observations or align augmentations. It predicts representations of unobserved parts of the world.

This makes it a natural bridge from self-supervised image representation learning to temporal latent prediction and action-conditioned world models.

The next section will derive the JEPA objective more carefully.

---

## References and Further Reading

- Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick, **Masked Autoencoders Are Scalable Vision Learners**, 2021.  
  <https://arxiv.org/abs/2111.06377>

- Jean-Bastien Grill et al., **Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning**, 2020.  
  <https://arxiv.org/abs/2006.07733>

- Adrien Bardes, Jean Ponce, Yann LeCun, **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning**, 2021.  
  <https://arxiv.org/abs/2105.04906>

- Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas, **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture**, 2023.  
  <https://arxiv.org/abs/2301.08243>

- Facebook Research, **Official I-JEPA Codebase**.  
  <https://github.com/facebookresearch/ijepa>

- Yann LeCun, **A Path Towards Autonomous Machine Intelligence**, 2022.  
  <https://openreview.net/forum?id=BZ5a1r-kVsf>